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
        assert _state_dict_uses_ddp_prefix(model) is True
        # sanity: the wrapper really does inject ".module." into keys
        assert any(".module." in k for k in model.state_dict().keys())

    def test_false_when_plain(self):
        model = _Student(ddp=False)
        assert _state_dict_uses_ddp_prefix(model) is False


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
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
        for k in state:
            ddp_key = k.replace("backbone.", "backbone.module.", 1)
            assert ddp_key in remapped, f"expected {ddp_key} in remapped keys"
        # original (non-prefixed) keys should no longer be present
        assert "backbone.patch_embed.proj.weight" not in remapped

    def test_leaves_already_correct_keys_untouched(self):
        model = _Student(ddp=True)
        w = torch.randn(8)
        state = {"backbone.module.norm.weight": w}
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
        assert "backbone.module.norm.weight" in remapped
        assert remapped["backbone.module.norm.weight"] is w

    def test_passthrough_for_keys_without_ddp_counterpart(self):
        model = _Student(ddp=True)
        state = {"extra.unrelated.weight": torch.randn(3)}
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
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
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
        assert len(remapped) == len(state)

    def test_values_preserved_by_reference(self):
        model = _Student(ddp=True)
        w = torch.randn(8, 5, 2, 2)
        state = {"backbone.patch_embed.proj.weight": w}
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
        assert remapped["backbone.module.patch_embed.proj.weight"] is w

    def test_non_ddp_model_returns_unchanged(self):
        model = _Student(ddp=False)
        state = {
            "backbone.patch_embed.proj.weight": torch.randn(8, 5, 2, 2),
            "backbone.norm.weight": torch.randn(8),
        }
        remapped = _remap_checkpoint_keys_for_ddp(state, model)
        assert set(remapped.keys()) == set(state.keys())
        for k in state:
            assert remapped[k] is state[k]
