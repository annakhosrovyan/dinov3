# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import copy
import gc
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from torch.distributed._tensor import DTensor
from torch.distributed.elastic.multiprocessing.errors import record

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    register_dont_save_hooks,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.data import (
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.utils.mfu import compute_dino_flops_per_image, compute_mfu
from dinov3.utils.profiling import (
    build_profiler,
    enable_graph_break_logging,
    get_memory_stats,
    get_run_metadata,
    log_fragmentation_stats,
    log_phase_memory,
    make_nvtx,
    memory_profile_enabled,
)
from dinov3.train.cosine_lr_scheduler import CosineScheduler, WSDLRScheduler, linear_warmup_cosine_decay
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.train.ssl_meta_arch import SSLMetaArch

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = True

logger = logging.getLogger("dinov3")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_dino",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr_scheduler = str(cfg.optim.get("lr_scheduler", "cosine")).lower()
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    if lr_scheduler == "wsd":
        lr_schedule = WSDLRScheduler(
            base_value=cfg.optim["lr"],
            lr_min=cfg.optim["min_lr"],
            total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
            final_decay_pct=float(cfg.optim.get("wsd_final_decay_pct", 0.1)),
        )
        last_layer_lr_schedule = WSDLRScheduler(
            base_value=cfg.optim["lr"],
            lr_min=cfg.optim["min_lr"],
            total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
            final_decay_pct=float(cfg.optim.get("wsd_final_decay_pct", 0.1)),
        )
    elif lr_scheduler == "cosine":
        lr_schedule = CosineScheduler(**lr)
        last_layer_lr_schedule = CosineScheduler(**lr)
    else:
        raise ValueError(f"Unsupported optim.lr_scheduler={lr_scheduler}")
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )
    logger.info(f"Schedulers ready. lr_scheduler={lr_scheduler}")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def do_test(cfg, model, iteration, process_group, do_low_freq=False):
    # dump a sharded checkpoint
    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    if distributed.is_subgroup_main_process():
        eval_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.sharded_eval_checkpoint:
        ckpt_path = eval_dir / "sharded_teacher_checkpoint"
        if distributed.is_subgroup_main_process():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        teacher_backbone = model.model_ema
        save_checkpoint(
            ckpt_dir=ckpt_path, iteration=iteration, model=teacher_backbone, overwrite=True, process_group=process_group
        )
        if not distributed.is_subgroup_main_process():
            return
    else:
        new_state_dict = model.model_ema.state_dict()
        for k, tensor in list(new_state_dict.items()):
            if isinstance(tensor, DTensor):
                new_state_dict[k] = tensor.full_tensor()
        if not distributed.is_subgroup_main_process():
            return
        # save teacher checkpoint
        ckpt_path = eval_dir / "teacher_checkpoint.pth"
        torch.save({"teacher": new_state_dict}, ckpt_path)
        logger.info("Saved eval checkpoint: %s", ckpt_path)


def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    if cfg.multidistillation.enabled:
        assert cfg.multidistillation.global_batch_size % distributed.get_subgroup_size() == 0
        local_batch_size = cfg.multidistillation.global_batch_size // distributed.get_subgroup_size()
        dataloader_batch_size_per_gpu = (
            cfg.multidistillation.global_batch_size + (distributed.get_world_size() - 1)
        ) // distributed.get_world_size()
    else:
        local_batch_size = None  # will default to the standard local batch size matching the data batch size
        dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )
    batch_size = dataloader_batch_size_per_gpu
    num_workers = cfg.train.num_workers
    persistent_workers = bool(cfg.train.get("persistent_workers", True))
    prefetch_factor = int(cfg.train.get("prefetch_factor", 4))
    dataset_path = cfg.train.dataset_path
    augmentation = model.build_data_augmentation_dino(cfg)
    logger.info(
        "Data loader uses augmentation class=%s module=%s for dataset_path=%s",
        augmentation.__class__.__name__,
        augmentation.__class__.__module__,
        dataset_path,
    )
    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=augmentation,
        target_transform=lambda _: (),
    )

    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if cfg.train.cache_dataset else SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * dataloader_batch_size_per_gpu,
        drop_last=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    return data_loader


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size] if isinstance(cfg.crops.global_crops_size, int) else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size] if isinstance(cfg.crops.local_crops_size, int) else cfg.crops.local_crops_size
    )
    gram_teacher_crops_sizes = (
        [cfg.crops.gram_teacher_crops_size]
        if cfg.crops.gram_teacher_crops_size is None or isinstance(cfg.crops.gram_teacher_crops_size, int)
        else cfg.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (global_crops_size_i, local_crops_size_i, gram_teacher_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes)
    ):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
        cfg_i.train.seed = cfg.train.seed + increment + 1
        loaders.append(build_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))

    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=seed,
            name="MultiResDL",
        )
    return data_loader


def _parse_cpulist(text):
    """Parse a sysfs cpulist string like '0-47,56-103' into a set of ints."""
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _apply_rank_cpu_slice():
    """Phase 7.9: pin each local rank to a private NUMA-local slice of physical cores.

    Gated by DINOV3_RANK_CPU_SLICE=1 (off by default). Job 80711 showed episodic rank-level
    loader starvation when 8 ranks x 24 workers share one unpartitioned CPU mask: the
    scheduler periodically deschedules one rank's workers and its data_time jumps from
    ~0.3 ms to 0.2-0.75 s ([RANKDATA]), stalling every collective. A private slice removes
    cross-rank contention for the same cores.

    Slices are built from whole physical cores (both HT siblings via sysfs core_id) so no
    rank ends up holding only the hyperthread shadows of another rank's cores. Blocking is
    by PACKAGE (socket), not NUMA node, with equal-size chunks: job 87757 showed that on
    gpu08 Sub-NUMA Clustering exposes 4 NUMA domains with UNEQUAL allowed cores (28/20 per
    socket half), and per-node chunking gave ranks 6/7 ten-core slices that starved by
    construction (sustained 1.7-2.8 s data_time_max). Equal supply beats strict SNC
    locality: within a socket, cores are ordered by (numa_node, core_id), so at most one
    rank per socket straddles an SNC boundary (on gpu08's 28+20 split, rank 2 takes 4+8
    across domains while rank 3 lands fully on domain 1 — matching GPU3, which is the lone
    GPU on that domain; GPUs 0-2 sit on domain 0). Applied to the main process before
    DataLoader workers fork, so workers and OMP threads inherit it. Best-effort: any
    failure leaves affinity untouched.
    """
    if os.environ.get("DINOV3_RANK_CPU_SLICE", "0") != "1":
        return
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError):
        sys.stderr.write("[RANKSLICE] LOCAL_RANK/LOCAL_WORLD_SIZE not set — slice skipped\n")
        return
    try:
        allowed = os.sched_getaffinity(0)
        # DINOV3_SLICE_EXCLUDE_CPUS: cpulist of cores to keep OUT of every slice (they become a
        # housekeeping pool for IRQ/softirq, nsys, the launcher, and FS-client threads). Job
        # 88301 showed the storage NIC's IRQ + NET_RX softirq zone (gpu08: cores 24-29,32 +
        # HT siblings) landing inside one rank's exclusive slice -> that rank starved 5-12 s
        # per burst. Any core group intersecting the exclusion is dropped whole (both siblings).
        excl_env = os.environ.get("DINOV3_SLICE_EXCLUDE_CPUS", "").strip()
        excluded = _parse_cpulist(excl_env) if excl_env else set()
        # cpu -> NUMA node (for intra-socket core ordering)
        cpu_node = {}
        node_dir = "/sys/devices/system/node"
        for entry in sorted(os.listdir(node_dir)):
            if not (entry.startswith("node") and entry[4:].isdigit()):
                continue
            with open(os.path.join(node_dir, entry, "cpulist")) as f:
                for cpu in _parse_cpulist(f.read()) & allowed:
                    cpu_node[cpu] = int(entry[4:])
        # (package, core_id) -> sibling cpus, so slices hold whole physical cores
        cores = {}
        for cpu in allowed:
            topo = f"/sys/devices/system/cpu/cpu{cpu}/topology"
            with open(f"{topo}/physical_package_id") as f:
                pkg = int(f.read())
            with open(f"{topo}/core_id") as f:
                core = int(f.read())
            cores.setdefault((pkg, core), set()).add(cpu)
        packages = sorted({pkg for pkg, _ in cores})
        # Block assignment by socket: with 2 sockets and 8 local ranks, 0-3 -> pkg0, 4-7 -> pkg1.
        pkg_idx = min(local_rank * len(packages) // local_world, len(packages) - 1)
        pkg = packages[pkg_idx]
        ranks_on_pkg = [
            r for r in range(local_world) if min(r * len(packages) // local_world, len(packages) - 1) == pkg_idx
        ]
        pos = ranks_on_pkg.index(local_rank)
        # Order the socket's cores by (NUMA domain, core_id): contiguous equal chunks then
        # keep each rank inside one domain where the arithmetic allows it.
        pkg_cores = sorted(
            (k for k in cores if k[0] == pkg and not (cores[k] & excluded)),
            key=lambda k: (min(cpu_node.get(c, 0) for c in cores[k]), k[1]),
        )
        n, k = len(pkg_cores), len(ranks_on_pkg)
        if n < k:
            sys.stderr.write(
                f"[RANKSLICE] rank={local_rank}: fewer cores than ranks on package {pkg} — slice skipped\n"
            )
            return
        # Even chunking with remainder spread over the first (n % k) ranks.
        base, rem = divmod(n, k)
        start = pos * base + min(pos, rem)
        end = start + base + (1 if pos < rem else 0)
        my_cores = pkg_cores[start:end]
        slice_cpus = set().union(*(cores[c] for c in my_cores))
        os.sched_setaffinity(0, slice_cpus)
        domains = sorted({cpu_node.get(c, -1) for c in slice_cpus})
        sys.stderr.write(
            f"[RANKSLICE] rank={local_rank} package={pkg} numa_domains={domains} "
            f"cores={len(my_cores)} cpus={len(slice_cpus)} excluded={excl_env or 'none'} "
            f"list={','.join(map(str, sorted(slice_cpus)))}\n"
        )
        sys.stderr.flush()
    except Exception as e:  # never let the pinning experiment kill training
        sys.stderr.write(f"[RANKSLICE] failed, affinity untouched: {e}\n")


def _log_rank_topology():
    """Phase 7.8c: one-time per-rank CPU/NUMA topology dump (gated by DINOV3_PERRANK_DIAG).

    Pure host-side introspection — no CUDA kernels, no collectives. Lets us correlate the
    late-arriver rank (lowest exposed-NCCL in the nsys harness) with its CPU affinity / NUMA
    placement, to decide whether the straggler is loader/NUMA-bound (fixable by pinning) or
    H2D/PCIe-bound (a different fix). Best-effort: never lets diagnostics break training.
    """
    if os.environ.get("DINOV3_PERRANK_DIAG", "0") != "1":
        return
    try:
        rank = distributed.get_rank() if distributed.is_enabled() else 0
        local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "?"))
        try:
            aff = sorted(os.sched_getaffinity(0))
        except AttributeError:
            aff = []
        cpus_allowed = None
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list:"):
                        cpus_allowed = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        aff_str = f"{aff[0]}-{aff[-1]}" if aff else "?"
        sys.stderr.write(
            f"[RANKTOPO] rank={rank} local_rank={local_rank} "
            f"cuda_dev={torch.cuda.current_device()} "
            f"n_cpus={len(aff)} cpu_range={aff_str} cpus_allowed_list={cpus_allowed}\n"
        )
        sys.stderr.flush()
    except Exception as e:  # never let a diagnostic kill training
        logger.warning("[RANKTOPO] dump failed: %s", e)


def do_train(cfg, model, resume=False):
    _mem_profile = memory_profile_enabled()
    # Period for fragmentation tracking (every N iters). Default 25; override via env.
    _mem_profile_period = int(os.environ.get("DINOV3_MEMORY_PROFILE_PERIOD", "25"))
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _log_rank_topology()  # Phase 7.8c: per-rank NUMA/affinity (no-op unless DINOV3_PERRANK_DIAG=1)
    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    model.init_weights()
    if _mem_profile:
        log_phase_memory("post_init_weights")
    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    # Precompute MAC constant for MFU tracking (done once, outside the loop)
    num_gpus = distributed.get_world_size()
    # Unwrap DDP if present to access backbone attributes
    _backbone_raw = model.student.backbone
    if hasattr(_backbone_raw, "module"):
        _backbone_raw = _backbone_raw.module
    macs_per_image = compute_dino_flops_per_image(
        global_crop_size=cfg.crops.global_crops_size,
        local_crop_size=cfg.crops.local_crops_size,
        patch_size=cfg.student.patch_size,
        n_global_crops=2,
        n_local_crops=cfg.crops.local_crops_number,
        hidden_dim=_backbone_raw.embed_dim,
        num_layers=_backbone_raw.n_blocks,
        ffn_ratio=getattr(cfg.student, "ffn_ratio", 4.0),
        n_registers=cfg.student.n_storage_tokens,
        gram_enabled=cfg.gram.use_loss,
        head_overhead_pct=0.05,
    )
    logger.info(f"MFU tracking: {macs_per_image / 1e9:.1f} GMACs/image, {num_gpus} GPUs")

    # Build data loader
    data_loader = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    wandb_module = None
    wandb_run = None
    if cfg.wandb.enabled and distributed.is_main_process():
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("wandb logging is enabled but the wandb package is not installed.") from exc
        init_kwargs = {
            "project": cfg.wandb.project,
            "name": cfg.wandb.run_name or None,
            "group": cfg.wandb.group or None,
            "dir": cfg.train.output_dir,
            "resume": "allow",
        }
        if cfg.wandb.entity:
            init_kwargs["entity"] = cfg.wandb.entity
        if cfg.wandb.tags:
            init_kwargs["tags"] = list(cfg.wandb.tags)
        wandb_run = wandb_module.init(**init_kwargs)
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Profiling setup (gated by args.profiling passed via cfg)
    profiling_enabled = getattr(cfg.train, "_profiling", False)
    nvtx = make_nvtx(profiling_enabled)
    profiler = None
    if profiling_enabled:
        logger.info("Profiling mode enabled")
        profiler = build_profiler(
            output_dir=cfg.train.output_dir,
            warmup=int(getattr(cfg.train, "_profiler_warmup", 5)),
            active=int(getattr(cfg.train, "_profiler_active", 3)),
            repeat=int(getattr(cfg.train, "_profiler_repeat", 1)),
            rank=distributed.get_rank(),
        )
        if getattr(cfg.train, "_graph_break_log", False):
            enable_graph_break_logging()
        # Log static run metadata once
        run_meta = get_run_metadata(cfg)
        logger.info("Run metadata: %s", run_meta)
        # Pass nvtx factory to model for inner NVTX ranges
        if hasattr(model, "set_nvtx"):
            model.set_nvtx(nvtx)

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    consecutive_nan_count = 0
    # Phase 7.3: optionally run the cross-rank metric all-reduce / NaN-check (and the per-iter
    # loss logging that .item()-syncs) only every Nth iter. Under DDP+cudagraphs the f32 metric
    # collective is a large exposed barrier (~32% of wall, job 75082); gating it trades
    # NaN-detection latency for throughput. metrics_sync_period=1 keeps the original behavior.
    metrics_sync_period = max(1, int(cfg.train.get("metrics_sync_period", 1)))
    step_start_event = torch.cuda.Event(enable_timing=True)
    step_end_event = torch.cuda.Event(enable_timing=True)
    if profiler is not None:
        profiler.start()
    if _mem_profile:
        log_phase_memory("pre_training_loop")
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="Training",
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return

        # Whether to do the cross-rank metric sync + loss logging this iter. Always sync on the
        # final iter (loop runs iteration = start_iter .. max_iter-1) so the last logged numbers
        # are correct. With period>1, NaN detection only runs on sync iters: a blow-up is *seen*
        # up to (period) iters late, and since abort needs >2 consecutive NaN samples it can run
        # ~3 periods before aborting; transient NaNs between sync iters are missed entirely.
        do_metric_sync = (
            metrics_sync_period == 1 or (iteration + 1) % metrics_sync_period == 0 or (iteration + 1) == max_iter
        )

        # Garbage collection (trigger manually so it happens on all ranks at the same time)
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

        # Learning rates and other schedules
        with nvtx("schedule_update"):
            lr = lr_schedule[it]
            wd = wd_schedule[it]
            mom = momentum_schedule[it]
            teacher_temp = teacher_temp_schedule[it]
            last_layer_lr = last_layer_lr_schedule[it]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Forward backward
        step_start_event.record()
        optimizer.zero_grad(set_to_none=True)
        with nvtx("forward_backward"):
            try:
                total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)
            except BaseException:
                _r = distributed.get_rank()
                logger.error("[STEPDIAG] exception in forward_backward iter=%d rank=%d", it, _r)
                _items = list(data.items()) if isinstance(data, dict) else []
                for _k, _v in _items:  # CPU metadata first (no CUDA ops)
                    if torch.is_tensor(_v):
                        logger.error(
                            "[STEPDIAG]  %s shape=%s dtype=%s dev=%s", _k, tuple(_v.shape), _v.dtype, _v.device
                        )
                    else:
                        logger.error("[STEPDIAG]  %s = %r", _k, _v)
                for _k, _v in _items:  # CUDA stats second (guarded — may be wedged)
                    if torch.is_tensor(_v):
                        try:
                            _f = _v.detach().float()
                            logger.error(
                                "[STEPDIAG]  %s nan=%s inf=%s min=%.4g max=%.4g sum=%.4g",
                                _k,
                                bool(torch.isnan(_f).any()),
                                bool(torch.isinf(_f).any()),
                                _f.min().item(),
                                _f.max().item(),
                                _f.sum().item(),
                            )
                        except BaseException as _e:
                            logger.error("[STEPDIAG]  %s stats unavailable: %s", _k, _e)
                raise

        # Gradient clipping
        with nvtx("grad_clip"):
            if cfg.optim.clip_grad:
                for k, v in student.items():
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        v.parameters(),
                        max_norm=cfg.optim.clip_grad,
                    )
                    # Keep grad_norm as a tensor here. Calling .item() mid-step forces a
                    # GPU->CPU sync after backward but before optimizer.step()/the CUDA
                    # event record, which breaks compute/comm pipelining (see
                    # learnings/distributed_training.md). The value is overwritten by the
                    # all-reduced tensor in the metrics-reduction block below, and every
                    # downstream consumer (torch.as_tensor in that block, MetricLogger.update,
                    # the wandb block) already handles tensors; the scalar read happens at
                    # logging time, after step_end_event.synchronize().
                    metrics_dict[f"{k}_grad_norm"] = (
                        grad_norm.full_tensor()
                        if isinstance(grad_norm, torch.distributed.tensor.DTensor)
                        else grad_norm.detach()
                    )

        # Reduce total_loss to check for NaNs, reduce metrics for logging. Gated by
        # metrics_sync_period: on non-sync iters we skip the f32 collectives entirely and keep
        # the local per-rank total_loss / metrics_dict (used only for logging, not for the
        # backward that already ran), so no exposed metric all-reduce is issued this iter.
        if do_metric_sync:
            with nvtx("allreduce_metrics"):
                total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
                torch.distributed.all_gather_into_tensor(
                    total_loss_all_ranks,
                    total_loss.detach(),
                    group=distributed.get_process_subgroup(),
                )
                total_loss = total_loss_all_ranks.mean()
                metrics_values = torch.stack(
                    [
                        torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach()
                        for v in metrics_dict.values()
                    ]
                )
                torch.distributed.all_reduce(
                    metrics_values,
                    op=torch.distributed.ReduceOp.AVG,
                    group=distributed.get_process_subgroup(),
                )
            metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
            if total_loss_all_ranks.isnan().any():
                consecutive_nan_count += 1
                which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
                logger.warning("NaN loss detected on ranks: %s", which_ranks)
                logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
                metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
                logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
                if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                    msg = "Too many consecutive nans detected in loss, aborting..."
                    logger.error(msg)
                    if profiler is not None:
                        try:
                            profiler.stop()
                        except Exception:
                            pass
                    raise RuntimeError(msg)
            else:
                consecutive_nan_count = 0
        # Step optimizer
        with nvtx("optimizer_step"):
            optimizer.step()
        with nvtx("ema_update"):
            model.update_ema(mom)
        step_end_event.record()

        # Compute step time and MFU using CUDA events (GPU-synchronized timing)
        step_end_event.synchronize()
        step_time_ms = step_start_event.elapsed_time(step_end_event)
        images_per_sec = global_batch_size / (step_time_ms / 1000.0)
        mfu = compute_mfu(images_per_sec, macs_per_image, num_gpus)

        # Per-phase peak memory markers (only active when DINOV3_MEMORY_PROFILE=1)
        if _mem_profile:
            if iteration == start_iter:
                log_phase_memory("compile_warmup_iter0")
            elif iteration == start_iter + 10:
                log_phase_memory("steady_state")
            # Periodic fragmentation tracking: logs [MEMFRAG] every N iters.
            # Does NOT reset peak stats — tracks cumulative allocator fragmentation.
            # Key metric: alloc_retries + inactive_split_mb growing = fragmentation building.
            if (iteration + 1) % _mem_profile_period == 0:
                log_fragmentation_stats(iteration)

        # [GRAM] Update gram teacher when using gram teacher and frequent updates
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        # Loss/metrics logging .item()-syncs each tensor; only do it on sync iters so a
        # metrics_sync_period>1 run is free of per-iter loss-side device syncs too.
        if do_metric_sync:
            metric_logger.update(total_loss=total_loss, **metrics_dict)
        metric_logger.update(
            mfu=mfu * 100,
            images_per_sec=images_per_sec,
            step_time_ms=step_time_ms,
        )
        if wandb_run is not None:
            wandb_metrics = {
                "iteration": float(iteration),
                "lr": float(lr),
                "wd": float(wd),
                "mom": float(mom),
                "last_layer_lr": float(last_layer_lr),
            }
            if do_metric_sync:
                wandb_metrics["total_loss"] = float(total_loss.item())
                for key, value in metrics_dict.items():
                    if isinstance(value, DTensor):
                        wandb_metrics[key] = float(value.full_tensor().item())
                    elif torch.is_tensor(value):
                        wandb_metrics[key] = float(value.item())
                    else:
                        wandb_metrics[key] = float(value)
            wandb_metrics["mfu_pct"] = float(mfu * 100)
            wandb_metrics["images_per_sec"] = float(images_per_sec)
            wandb_metrics["step_time_ms"] = float(step_time_ms)
            wandb_module.log(wandb_metrics, step=iteration)

        # Extended memory metrics (logged every iteration when profiling)
        if profiling_enabled:
            mem_stats = get_memory_stats()
            metric_logger.update(**mem_stats)

        # Profiler step (must be called every iteration when profiler is active)
        if profiler is not None:
            profiler.step()

        # Submit evaluation jobs
        if (
            cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            if _mem_profile:
                log_phase_memory("pre_eval")
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()
            if _mem_profile:
                log_phase_memory("eval_complete")

        # Checkpointing
        if (iteration + 1) % cfg.checkpointing.period == 0:
            if _mem_profile:
                log_phase_memory("pre_checkpoint")
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))
            if _mem_profile:
                log_phase_memory("checkpoint_complete")

        iteration = iteration + 1
    # Always stop profiler, even if loop exits early or raises
    if profiler is not None:
        try:
            profiler.stop()
        except Exception:
            logger.warning("Profiler stop failed", exc_info=True)
    metric_logger.synchronize_between_processes()
    if wandb_run is not None:
        wandb_run.finish()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@record
def main(argv=None):
    # Phase 7.9: must run before CUDA init, DataLoader workers, and OMP pools so they all
    # inherit the slice. No-op unless DINOV3_RANK_CPU_SLICE=1.
    _apply_rank_cpu_slice()
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    if args.multi_distillation:
        print("performing multidistillation run")
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
        logger.info("setup_multidistillation done")
        assert cfg.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(
            "Config summary: arch=%s in_chans=%s dataset=%s batch_size_per_gpu=%s num_workers=%s",
            cfg.student.arch,
            cfg.student.in_chans,
            cfg.train.dataset_path,
            cfg.train.batch_size_per_gpu,
            cfg.train.num_workers,
        )
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    logger.info(f"Making meta arch {meta_arch.__name__}")
    with torch.device("meta"):
        model = meta_arch(cfg)
    model.prepare_for_distributed_training()
    # Fill all values with `nans` so that we identify
    # non-initialized values
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        model.init_weights()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    # Pass profiling flags through cfg so do_train can use them
    if args.profiling:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)
        cfg.train._profiling = True
        cfg.train._profiler_warmup = 5
        cfg.train._profiler_active = 3
        cfg.train._profiler_repeat = 1
        cfg.train._graph_break_log = True
        OmegaConf.set_struct(cfg, True)
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
