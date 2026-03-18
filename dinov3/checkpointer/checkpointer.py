# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Suggested file structure:

output_dir/
|-- ckpt/
|   |-- 0/
|   |-- 99/
|   |-- 199/
|   |-- 199_keep/
|   |-- 299/
|   `-- ...
`-- eval/
    `-- 0/
    `-- 99/
        `-- ckpt/

Distributed checkpointer docs:
- https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- https://pytorch.org/docs/stable/distributed.checkpoint.html
"""

import logging
import re
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Set

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger("dinov3")


def _tensor_sample(values: torch.Tensor, count: int = 5) -> list[float]:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    return [float(x) for x in flat[: min(count, flat.numel())].cpu()]


def _remap_legacy_keys(state_dict: dict) -> dict:
    remapped_state = dict(state_dict)
    remap_count = 0
    gamma_pattern = re.compile(r"\.gamma_(1|2)$")

    for key in list(remapped_state.keys()):
        match = gamma_pattern.search(key)
        if match is None:
            continue
        suffix = ".ls1.gamma" if match.group(1) == "1" else ".ls2.gamma"
        new_key = gamma_pattern.sub(suffix, key)
        if new_key not in remapped_state:
            remapped_state[new_key] = remapped_state[key]
        del remapped_state[key]
        remap_count += 1

    for key in list(remapped_state.keys()):
        if not key.endswith("reg_token"):
            continue
        mask_key = f"{key[:-len('reg_token')]}mask_token"
        reg_token = remapped_state[key]
        if mask_key not in remapped_state and isinstance(reg_token, torch.Tensor):
            if reg_token.ndim == 3 and reg_token.shape[1] >= 1:
                remapped_state[mask_key] = reg_token[:, 0, :].clone()
                remap_count += 1
            elif reg_token.ndim == 2:
                remapped_state[mask_key] = reg_token.clone()
                remap_count += 1
        del remapped_state[key]
        remap_count += 1

    for key in list(remapped_state.keys()):
        if not key.endswith("attn.qkv.weight"):
            continue
        bias_key = key[:-len("weight")] + "bias"
        if bias_key in remapped_state:
            continue
        qkv_weight = remapped_state[key]
        if not isinstance(qkv_weight, torch.Tensor):
            continue
        if qkv_weight.ndim != 2:
            continue
        remapped_state[bias_key] = torch.zeros(
            qkv_weight.shape[0],
            dtype=qkv_weight.dtype,
            device=qkv_weight.device,
        )
        remap_count += 1

    if remap_count > 0:
        logger.info("Remapped %d legacy checkpoint keys", remap_count)
    return remapped_state


def adapt_patch_embed_input_channels(state_dict: dict, target_in_chans: int) -> dict:
    state_dict = _remap_legacy_keys(state_dict)
    patch_key = None
    for key in state_dict.keys():
        if key.endswith("patch_embed.proj.weight"):
            patch_key = key
            break
    if patch_key is None:
        return state_dict

    weight = state_dict[patch_key]
    if not isinstance(weight, torch.Tensor):
        return state_dict

    if weight.ndim != 4:
        raise ValueError("patch_embed.proj.weight must have 4 dimensions")

    source_in_chans = weight.shape[1]
    logger.info(
        "patch_embed.proj.weight found: shape=%s target_in_chans=%d mean=%.6f std=%.6f sample=%s",
        tuple(weight.shape),
        target_in_chans,
        float(weight.mean().item()),
        float(weight.std().item()),
        _tensor_sample(weight[0, : min(source_in_chans, 3), 0, 0], count=3),
    )
    if source_in_chans == target_in_chans:
        return state_dict

    if source_in_chans == 3 and target_in_chans == 5:
        expanded = torch.zeros(
            (weight.shape[0], target_in_chans, weight.shape[2], weight.shape[3]),
            dtype=weight.dtype,
            device=weight.device,
        )
        expanded[:, :3, :, :] = weight
        mean_rgb = weight.mean(dim=1, keepdim=True)
        expanded[:, 3:5, :, :] = mean_rgb.repeat(1, 2, 1, 1)
        state_dict[patch_key] = expanded
        logger.info(
            "Adapted %s to shape=%s extra_channel_mean=%.6f sample_c3=%s sample_c4=%s",
            patch_key,
            tuple(expanded.shape),
            float(expanded[:, 3:5, :, :].mean().item()),
            _tensor_sample(expanded[0, 3, :2, :2]),
            _tensor_sample(expanded[0, 4, :2, :2]),
        )
        return state_dict

    raise ValueError(
        f"Unsupported patch embedding adaptation from {source_in_chans} to {target_in_chans} channels"
    )


class CheckpointRetentionPolicy(Enum):
    ALL = "all"  # keep all checkpoints
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"
    NONE = "none"  # do not keep any checkpoints

    @property
    def keep_filters(self) -> Set[str]:
        """Files that match these patterns are not deleted by cleanup"""
        if self == CheckpointRetentionPolicy.LAST:
            return set(["final"])
        if self == CheckpointRetentionPolicy.BEST:
            return set(["best"])
        if self == CheckpointRetentionPolicy.LAST_AND_BEST:
            return set(["final", "best"])
        if self == CheckpointRetentionPolicy.ALL:
            return set()
        return set()

    @property
    def max_to_keep(self) -> int | None:
        """
        maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and not `save`. `None` for keep all
        """
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1


def save_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
):
    """Save a plain/DDP/FSDP/FSDP2 model, its optimizer, an integer iteration and other stateful objects."""
    rank = torch.distributed.get_rank(group=process_group)

    # Rank 0 checks if the checkpoint directory exists, but all ranks need to know if if exists,
    # so they can raise an error when overwrite is False. If overwrite is True, rank 0 will delete it
    # and other ranks wait for the deletion to finish.
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is not None:
        src_rank = torch.distributed.get_global_rank(group=process_group, group_rank=0)
    torch.distributed.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]
    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                logger.info(f"Deleted: {ckpt_dir}")
            torch.distributed.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    # Rank 0 creates a temporary directory for the checkpoint and broadcasts the name to all ranks.
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    torch.distributed.broadcast_object_list(ckpt_dir_tmp, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(ckpt_dir_tmp[0])

    to_save = {"iteration": iteration}
    to_save["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_save["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_save.update(others)
    dcp.save(
        to_save,
        storage_writer=dcpfs.FileSystemWriter(ckpt_dir_tmp),
        process_group=process_group,
    )

    # Rank 0 renames the temporary directory to the final checkpoint directory. All ranks wait for the rename.
    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)
    torch.distributed.barrier()

    logger.info(f"Saved: {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
) -> int | None:
    """
    Load a plain/DDP/FSDP/FSDP2 model, its optimizer, an integer iteration and other stateful objects.
    Can you take a checkpoint saved on N ranks and load it on M ranks? Sure you can!
    Activation checkpointing and torch-compile can also be different between save and load, no problem.
    """
    ckpt_dir = Path(ckpt_dir)
    to_load = {"iteration": None}
    to_load["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_load["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_load.update(others)
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(ckpt_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict_loading),
        process_group=process_group,
    )
    iteration = to_load["iteration"]
    dcpsd.set_model_state_dict(model, to_load["model"])
    if optimizer is not None:
        dcpsd.set_optimizer_state_dict(model, optimizer, to_load["optimizer"])
    logger.info(f"Loaded: {ckpt_dir}")
    return iteration


def register_dont_save_hooks(module: torch.nn.Module, dont_save: Sequence[str]):
    """
    Registers save/load state dict hooks such that the weights in `dont_save` are not persisted in the checkpoint.

    Typical use case: a classification model composed of a frozen backbone and a trainable head.
    If the frozen backbone is loaded from torch hub, it does't make sense to save a copy of it in each checkpoint.
    """

    def state_dict_post_hook(module, state_dict, prefix, local_metadata):
        # Remove frozen weights so they won't get saved.
        # If this module is not the top-level module, its weights will have a prefix in the state dict.
        nonlocal _dont_save
        for k in _dont_save:
            del state_dict[prefix + k]

    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This pre hook exists only to pass the prefix to the post hook when loading the state dict.
        nonlocal _prefix
        assert _prefix is None
        _prefix = prefix

    def load_state_dict_post_hook(module, incompatible_keys):
        # Remove the frozen weights from the missing keys so they don't raise an error.
        nonlocal _prefix
        assert _prefix is not None
        to_remove = []
        for missing_key in incompatible_keys.missing_keys:
            k = missing_key.removeprefix(_prefix)
            k = k.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing
            if k in _dont_save:
                to_remove.append(missing_key)
        for r in to_remove:
            incompatible_keys.missing_keys.remove(r)
        _prefix = None

    _dont_save = set(name.replace("_checkpoint_wrapped_module.", "") for name in dont_save)
    _prefix = None
    module.register_state_dict_post_hook(state_dict_post_hook)
    module.register_load_state_dict_pre_hook(load_state_dict_pre_hook)
    module.register_load_state_dict_post_hook(load_state_dict_post_hook)


def find_all_checkpoints(ckpt_dir: Path | str) -> list[Path]:
    """Find all checkpoints in a directory, i.e. subdirs with integer name. Sorted from first to last."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and _is_int(p.name)]
    checkpoints.sort(key=lambda p: int(p.name))
    return checkpoints


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Find the latest checkpoint in a directory, i.e. the subdir with the highest integer name."""
    checkpoints = find_all_checkpoints(ckpt_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None):
    """In a directory with integer-named subdirs, keep only the n subdirs with the highest number."""
    if n is None:
        return
    checkpoints = find_all_checkpoints(ckpt_dir)
    for ckpt_dir in checkpoints[:-n]:
        try:
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted: {ckpt_dir}")
        except Exception:
            logger.exception(f"Failed to delete: {ckpt_dir}")


def keep_checkpoint_copy(src: Path | str):
    """Copy a file/directory next to itself with a _keep suffix. Files are hardlinked."""
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    subprocess.check_output(["cp", "--recursive", "--link", src, dst])
    logger.info(f"Copied: {src} -> {dst}")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# Initialize a FSDP2 model from DCP or PyTorch standard checkpoint
def init_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: List[str] | None = None,
    keys_not_sharded: List[str] | None = None,
    process_group: dist.ProcessGroup = None,
):
    def _normalize_checkpoint_keys_for_model(raw_state: dict, target_model: torch.nn.Module) -> dict:
        normalized = {}
        for key, value in raw_state.items():
            if key.startswith("module."):
                normalized[key[len("module."):]] = value
            else:
                normalized[key] = value

        if not hasattr(target_model, "backbone"):
            return normalized

        roots = {k.split(".", 1)[0] for k in normalized.keys()}
        known_roots = {"backbone", "dino_head", "ibot_head", "dino_loss", "ibot_patch_loss", "teacher", "student", "model_ema"}
        backbone_like_prefixes = (
            "patch_embed.",
            "blocks.",
            "norm.",
            "cls_token",
            "mask_token",
            "reg_token",
            "rope_embed.",
            "pos_embed",
        )
        has_backbone_root = any(k.startswith("backbone.") for k in normalized.keys())
        looks_like_backbone_only = any(
            key.startswith(backbone_like_prefixes) for key in normalized.keys()
        )
        if (not has_backbone_root) and roots.isdisjoint(known_roots) and looks_like_backbone_only:
            normalized = {f"backbone.{k}": v for k, v in normalized.items()}

        return normalized

    if skip_load_keys is None:
        skip_load_keys = []
    if keys_not_sharded is None:
        keys_not_sharded = []
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        loaded = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(loaded, dict):
            if "teacher" in loaded and isinstance(loaded["teacher"], dict):
                chkpt = loaded["teacher"]
            elif "model" in loaded and isinstance(loaded["model"], dict):
                chkpt = loaded["model"]
            elif "student" in loaded and isinstance(loaded["student"], dict):
                chkpt = loaded["student"]
            elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                chkpt = loaded["state_dict"]
            else:
                chkpt = loaded
        else:
            raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
        chkpt = _normalize_checkpoint_keys_for_model(chkpt, model)
        chkpt = adapt_patch_embed_input_channels(chkpt, model.backbone.patch_embed.proj.weight.shape[1])
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        if process_group is None:
            world_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        else:
            world_mesh = DeviceMesh.from_group(process_group, "cuda")
        distributed_chkpt = {}
        for key, tensor in chkpt.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if any(key_not_sharded in key for key_not_sharded in keys_not_sharded):
                distributed_chkpt[key] = tensor
            else:
                distributed_chkpt[key] = torch.distributed.tensor.distribute_tensor(
                    tensor, world_mesh, src_data_rank=None
                )
        filtered_state = {
            key: tensor
            for key, tensor in distributed_chkpt.items()
            if not any(skip_load_key in key for skip_load_key in skip_load_keys)
        }
        model_state_keys = set(model.state_dict().keys())
        matched_keys = [k for k in filtered_state.keys() if k in model_state_keys]
        if len(matched_keys) == 0:
            raise RuntimeError(
                f"No checkpoint keys matched model keys for {checkpoint_path}. "
                "Checkpoint/model key mapping is invalid."
            )
        msg = model.load_state_dict(filtered_state, strict=False)
        missing_keys = sorted(msg.missing_keys)
        unexpected_keys = sorted(msg.unexpected_keys)
        logger.info(
            "Checkpoint load summary: matched=%d missing=%d unexpected=%d",
            len(matched_keys),
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            expected_missing = sorted(
                [
                    key
                    for key in missing_keys
                    if any(skip_load_key in key for skip_load_key in skip_load_keys)
                ]
            )
            unexpected_missing = sorted(
                [
                    key
                    for key in missing_keys
                    if not any(skip_load_key in key for skip_load_key in skip_load_keys)
                ]
            )
            logger.info(
                "Checkpoint missing keys (expected by skip rules): %s",
                expected_missing,
            )
            logger.info(
                "Checkpoint missing keys (not covered by skip rules): %s",
                unexpected_missing,
            )
        if unexpected_keys:
            logger.info("Checkpoint unexpected keys: %s", unexpected_keys)
    else:  # DCP checkpoint
        load_checkpoint(ckpt_dir=checkpoint_path, model=model, process_group=process_group)


# Initialize a standard non distributed PyTorch model from PyTorch standard checkpoint for evals
def init_model_from_checkpoint_for_evals(
    model: torch.nn.Module, pretrained_weights: str | Path, checkpoint_key: str = None
):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = adapt_patch_embed_input_channels(state_dict, model.patch_embed.proj.weight.shape[1])
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def cleanup_checkpoint(ckpt_dir: str, checkpoint_retention_policy: CheckpointRetentionPolicy):
    """
    ckpt_dir is the directory containing each individual checkpoint directories (either at iteration, best (validation performance) or final)
    |-- ckpt_dir/
    |   |-- 0/
    |       |--checkpoint.pth  or dcp_sharded_checkpoint_dir
    |   |-- 99/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 199/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- best/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 299/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- final/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoint_filters = checkpoint_retention_policy.keep_filters
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir()]
    for checkpoint in checkpoints:
        if checkpoint.name in checkpoint_filters:
            continue
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted: {checkpoint}")
        except Exception:
            logger.exception(f"Failed to delete: {checkpoint}")
