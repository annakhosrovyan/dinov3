"""
Unit tests for DDP checkpoint loading helpers in dinov3/checkpointer/checkpointer.py.

These cover the helpers added to support loading a PyTorch standard checkpoint into a
model whose sub-modules (backbone, dino_head, ibot_head) have each been individually
wrapped in DistributedDataParallel by prepare_for_distributed_training(). After DDP
wrapping, parameter keys gain a ".module." segment (e.g. backbone.module.patch_embed...),
and the raw backbone attributes (.patch_embed) are no longer directly accessible on the
wrapper — hence the unwrap + key-remap logic these tests exercise.

No real process group is required: DDP is mocked with a minimal _FakeDDP wrapper that
reproduces the only behaviour the helpers depend on — a `.module` attribute and the
resulting ".module." state_dict key prefix.

Run with: python -m pytest tests/test_ddp_checkpoint.py -v
"""
import torch
import torch.nn as nn
import pytest

from dinov3.checkpointer.checkpointer import (
    _unwrap_module,
    _get_backbone_in_chans,
    _state_dict_uses_ddp_prefix,
    _remap_checkpoint_keys_for_ddp,
    init_model_from_checkpoint_for_evals,
)


class _FakeDDP(nn.Module):
    """Minimal stand-in for DistributedDataParallel.

    Real DDP stores the wrapped module under `.module`, which makes state_dict keys
    gain a ".module." segment and makes attribute access (`.patch_embed`) fail on the
    wrapper. This mock reproduces both effects without needing a process group.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


class _PatchEmbed(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        # weight.shape == (out_chans, in_chans, kH, kW); shape[1] is in_chans.
        self.proj = nn.Conv2d(in_chans, 8, kernel_size=2)


class _Backbone(nn.Module):
    def __init__(self, in_chans: int = 5):
        super().__init__()
        self.patch_embed = _PatchEmbed(in_chans)
        self.norm = nn.LayerNorm(8)


def _keys(model: nn.Module) -> set:
    """The parameter-key set the loading path computes once and passes to the helpers."""
    return set(model.state_dict().keys())


class _Student(nn.Module):
    """Stand-in for the SSLMetaArch student container: holds a backbone that may or
    may not be DDP-wrapped."""

    def __init__(self, ddp: bool = False, in_chans: int = 5):
        super().__init__()
        backbone = _Backbone(in_chans)
        self.backbone = _FakeDDP(backbone) if ddp else backbone


class TestUnwrapModule:
    def test_unwraps_ddp(self):
        inner = _Backbone()
        wrapped = _FakeDDP(inner)
        assert _unwrap_module(wrapped) is inner

    def test_passthrough_when_not_wrapped(self):
        inner = _Backbone()
        assert _unwrap_module(inner) is inner


class TestGetBackboneInChans:
    def test_plain_backbone(self):
        model = _Student(ddp=False, in_chans=5)
        assert _get_backbone_in_chans(model) == 5

    def test_ddp_wrapped_backbone(self):
        model = _Student(ddp=True, in_chans=5)
        # Direct model.backbone.patch_embed would fail (wrapper has no patch_embed);
        # the helper must unwrap first.
        assert _get_backbone_in_chans(model) == 5

    def test_non_default_in_chans(self):
        model = _Student(ddp=True, in_chans=3)
        assert _get_backbone_in_chans(model) == 3

    def test_raises_without_backbone(self):
        class _NoBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)

        with pytest.raises(AttributeError):
            _get_backbone_in_chans(_NoBackbone())


class TestStateDictUsesDdpPrefix:
    def test_true_when_ddp_wrapped(self):
        model = _Student(ddp=True)
        assert _state_dict_uses_ddp_prefix(_keys(model)) is True
        # sanity: the wrapper really does inject ".module." into keys
        assert any(".module." in k for k in model.state_dict().keys())

    def test_false_when_plain(self):
        model = _Student(ddp=False)
        assert _state_dict_uses_ddp_prefix(_keys(model)) is False


class TestRemapCheckpointKeysForDdp:
    def test_inserts_module_segment_for_ddp_model(self):
        model = _Student(ddp=True)
        # A checkpoint saved from a non-DDP model: keys lack ".module.".
        state = {
            "backbone.patch_embed.proj.weight": torch.randn(8, 5, 2, 2),
            "backbone.patch_embed.proj.bias": torch.randn(8),
            "backbone.norm.weight": torch.randn(8),
            "backbone.norm.bias": torch.randn(8),
        }
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        for k in state:
            ddp_key = k.replace("backbone.", "backbone.module.", 1)
            assert ddp_key in remapped, f"expected {ddp_key} in remapped keys"
        # original (non-prefixed) keys should no longer be present
        assert "backbone.patch_embed.proj.weight" not in remapped

    def test_leaves_already_correct_keys_untouched(self):
        model = _Student(ddp=True)
        w = torch.randn(8)
        state = {"backbone.module.norm.weight": w}
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        assert "backbone.module.norm.weight" in remapped
        assert remapped["backbone.module.norm.weight"] is w

    def test_passthrough_for_keys_without_ddp_counterpart(self):
        model = _Student(ddp=True)
        state = {"extra.unrelated.weight": torch.randn(3)}
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        # No matching DDP key in the model → key is preserved unchanged.
        assert "extra.unrelated.weight" in remapped

    def test_no_keys_dropped_or_duplicated(self):
        model = _Student(ddp=True)
        state = {
            "backbone.patch_embed.proj.weight": torch.randn(8, 5, 2, 2),
            "backbone.patch_embed.proj.bias": torch.randn(8),
            "backbone.norm.weight": torch.randn(8),
            "backbone.norm.bias": torch.randn(8),
        }
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        assert len(remapped) == len(state)

    def test_values_preserved_by_reference(self):
        model = _Student(ddp=True)
        w = torch.randn(8, 5, 2, 2)
        state = {"backbone.patch_embed.proj.weight": w}
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        assert remapped["backbone.module.patch_embed.proj.weight"] is w

    def test_non_ddp_model_returns_unchanged(self):
        model = _Student(ddp=False)
        state = {
            "backbone.patch_embed.proj.weight": torch.randn(8, 5, 2, 2),
            "backbone.norm.weight": torch.randn(8),
        }
        remapped = _remap_checkpoint_keys_for_ddp(state, _keys(model))
        assert set(remapped.keys()) == set(state.keys())
        for k in state:
            assert remapped[k] is state[k]


class _EvalBackbone(nn.Module):
    """Stand-in for the teacher backbone that build_model_for_eval constructs.

    Mirrors the only structure init_model_from_checkpoint_for_evals touches: a
    top-level `.patch_embed.proj` conv whose weight.shape[1] is in_chans. Unlike
    the training student, the eval model is the backbone itself, so patch_embed
    is a *direct* top-level attribute.
    """

    def __init__(self, in_chans: int = 5):
        super().__init__()
        self.patch_embed = _PatchEmbed(in_chans)
        self.norm = nn.LayerNorm(8)


def _write_eval_checkpoint(path, in_chans: int = 5):
    """Write a consolidated PyTorch checkpoint shaped like the ones the eval loader
    consumes: a top-level dict keyed by 'teacher', whose values carry the
    'backbone.' prefix that the loader strips before load_state_dict."""
    sd = {
        "backbone.patch_embed.proj.weight": torch.randn(8, in_chans, 2, 2),
        "backbone.patch_embed.proj.bias": torch.randn(8),
        "backbone.norm.weight": torch.randn(8),
        "backbone.norm.bias": torch.randn(8),
    }
    torch.save({"teacher": sd}, str(path))
    return sd


class TestEvalLoaderIntegration:
    """End-to-end coverage of init_model_from_checkpoint_for_evals at its real
    call-site shape. The helper unit tests above prove the units; these exercise
    the wiring the units feed into — specifically the Commit 3 fix that unwraps a
    DDP-wrapped eval model before reading patch_embed.

    A real eval job never wraps this model (build_model_for_eval builds a plain
    teacher backbone), so the DDP branch of the fix is unreachable from any real
    run — only a constructed wrapped model exercises it. Hence this test.
    """

    def test_plain_eval_model_loads_weights(self, tmp_path):
        ckpt = tmp_path / "teacher.pth"
        sd = _write_eval_checkpoint(ckpt, in_chans=5)
        model = _EvalBackbone(in_chans=5)
        # The path every real eval job runs: an unwrapped backbone.
        init_model_from_checkpoint_for_evals(model, str(ckpt), "teacher")
        # Weights actually landed: the loader strips the "backbone." prefix, so the
        # checkpoint's backbone.patch_embed.proj.weight maps onto model.patch_embed...
        assert torch.equal(
            model.patch_embed.proj.weight.detach(),
            sd["backbone.patch_embed.proj.weight"],
        )

    def test_ddp_wrapped_eval_model_loads_weights(self, tmp_path):
        ckpt = tmp_path / "teacher.pth"
        sd = _write_eval_checkpoint(ckpt, in_chans=5)
        wrapped = _FakeDDP(_EvalBackbone(in_chans=5))
        # Regression guard for Commit 3, in two layers:
        # (1) before the _unwrap_module change, reading `model.patch_embed` on a
        #     DDP-wrapped module raised AttributeError exactly like the training-path
        #     bug Anna hit;
        # (2) an earlier version of the fix unwrapped only the in_chans *read* but
        #     still called load_state_dict on the wrapper — the checkpoint keys had
        #     `module.` stripped while the wrapper's own keys kept it, so strict=False
        #     silently matched ZERO keys and left the model at its init weights.
        # Asserting that the weights actually land catches both failure modes; a
        # no-exception check alone passes under (2).
        init_model_from_checkpoint_for_evals(wrapped, str(ckpt), "teacher")
        assert torch.equal(
            _unwrap_module(wrapped).patch_embed.proj.weight.detach(),
            sd["backbone.patch_embed.proj.weight"],
        )

    def test_ddp_wrapped_eval_model_loads_all_keys(self, tmp_path):
        # Stronger variant of the above: every tensor in the checkpoint must land in
        # the unwrapped module, not just patch_embed — guards against partial loads
        # that a single-weight equality check could miss.
        ckpt = tmp_path / "teacher.pth"
        sd = _write_eval_checkpoint(ckpt, in_chans=5)
        wrapped = _FakeDDP(_EvalBackbone(in_chans=5))
        init_model_from_checkpoint_for_evals(wrapped, str(ckpt), "teacher")
        loaded = _unwrap_module(wrapped).state_dict()
        for ckpt_key, value in sd.items():
            model_key = ckpt_key.replace("backbone.", "", 1)
            assert torch.equal(loaded[model_key], value), f"{model_key} did not load"

    def test_eval_loader_reads_in_chans_through_unwrap_for_non_default(self, tmp_path):
        # in_chans=3 checkpoint into an in_chans=3 model, DDP-wrapped: confirms the
        # unwrapped shape[1] (not the wrapper) drives adapt_patch_embed_input_channels,
        # and that the weights actually land (not merely that no exception was raised).
        ckpt = tmp_path / "teacher.pth"
        sd = _write_eval_checkpoint(ckpt, in_chans=3)
        wrapped = _FakeDDP(_EvalBackbone(in_chans=3))
        init_model_from_checkpoint_for_evals(wrapped, str(ckpt), "teacher")
        assert torch.equal(
            _unwrap_module(wrapped).patch_embed.proj.weight.detach(),
            sd["backbone.patch_embed.proj.weight"],
        )
