# Torch Compile in This DINOv3 Fork - Phase 5 Context

Date: 2026-05-21

This note explains what `train.compile=true` means in this DINOv3 codebase, what it does and does not test, and how that relates to Armen's recommendation to use full-graph/static-shape compile, CUDA graphs, DDP, and no FSDP for the current ViT-B scale experiments.

## Short Answer

We have been running with PyTorch compile enabled.

In this repo, `train.compile=true` is not a raw PyTorch option named `torch.compile=true`. It is a DINOv3 Hydra config flag that routes selected modules through `nn.Module.compile()`, which is PyTorch's module-level wrapper around `torch.compile`.

The current default is:

```yaml
train:
  compile: true
  compile_mode: null
  cudagraphs: false
```

Evidence:

- `dinov3/configs/ssl_default_config.yaml:80-82` sets `compile: true`, `compile_mode: null`, and `cudagraphs: false`.
- `dinov3/fsdp/ac_compile_parallelize.py:191` gates compile application on `if cfg.train.compile:`.
- `dinov3/fsdp/ac_compile_parallelize.py:80-83` maps `compile_mode: null` or `"default"` to `None`.
- `dinov3/fsdp/ac_compile_parallelize.py:72-76` calls one of:
  - `module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})`
  - `module.compile(mode=compile_mode)`
  - `module.compile()`

So the Phase 5 default path is plain `module.compile()`, not full-graph CUDA graph capture.

## What Gets Compiled

The compile entry point is `dinov3/fsdp/ac_compile_parallelize.py`.

For transformer backbones, DINOv3 compiles each transformer block individually:

```python
def compile_transformer(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    assert isinstance(model.blocks, nn.ModuleList)
    for block_id, block in enumerate(model.blocks):
        model.blocks[block_id] = wrap_compile_block(
            block,
            cfg.train.cudagraphs,
            is_backbone_block=True,
            compile_mode=compile_mode,
        )
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:97-101`.

For ConvNeXt-style backbones, DINOv3 compiles at stage/downsample-layer granularity:

```python
def compile_convnext(cfg, model: nn.Module):
    compile_mode = _get_compile_mode(cfg)
    assert isinstance(model.stages, nn.ModuleList)
    for stage_id, stage in enumerate(model.stages):
        model.stages[stage_id] = wrap_compile_block(...)
    assert isinstance(model.downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = wrap_compile_block(...)
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:86-94`.

For other top-level modules in the `ModuleDict`, DINOv3 wraps the whole module:

```python
model[k] = wrap_compile_block(
    model[k],
    use_cuda_graphs=False,
    is_backbone_block=False,
    compile_mode=compile_mode,
)
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:198`.

That matters for the SSL model because `SSLMetaArch` builds a `ModuleDict` containing `backbone`, `dino_head`, and `ibot_head`:

- `dinov3/train/ssl_meta_arch.py:83` attaches `student_model_dict["dino_head"]`.
- `dinov3/train/ssl_meta_arch.py:127` attaches `student_model_dict["ibot_head"]`.
- `dinov3/train/ssl_meta_arch.py:132` creates `self.student = nn.ModuleDict(student_model_dict)`.

It also passes inference-only models through the same helper:

- `dinov3/train/ssl_meta_arch.py:845-852` passes `self.model_ema`, optional `self.gram_teacher`, and `self.teacher` as inference-only models.

## What `compile_mode: null` Means

`compile_mode: null` is the PyTorch default compile path.

The local helper implements:

```python
def _get_compile_mode(cfg) -> str | None:
    """Return the compile mode string, or None for PyTorch default."""
    mode = getattr(cfg.train, "compile_mode", None)
    return None if (mode is None or mode == "default") else str(mode)
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:80-83`.

Then the wrapper does:

```python
elif compile_mode is not None:
    module.compile(mode=compile_mode)
else:
    module.compile()
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:73-76`.

So `compile_mode: null` and `compile_mode: default` are behaviorally the same in this code: both call `module.compile()` with no explicit mode argument.

This is different from setting:

- `compile_mode=max-autotune`
- `compile_mode=max-autotune-no-cudagraphs`
- `compile_mode=reduce-overhead`
- `fullgraph=True`
- `dynamic=False`
- explicit CUDA graph capture

## What `cudagraphs: false` Means

The fullgraph/static CUDA graph branch is only taken when both conditions are true:

1. `cfg.train.cudagraphs` is true
2. the wrapped module is marked as a backbone block

The code is:

```python
if use_cuda_graphs and is_backbone_block:
    module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
elif compile_mode is not None:
    module.compile(mode=compile_mode)
else:
    module.compile()
```

Evidence: `dinov3/fsdp/ac_compile_parallelize.py:65-76`.

In current defaults, `cudagraphs: false`, so transformer blocks use plain `module.compile()` rather than:

```python
module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
```

Evidence: `dinov3/configs/ssl_default_config.yaml:82`.

Important nuance: this code's CUDA graph branch is only wired for `is_backbone_block=True`. For ViT blocks, `compile_transformer()` passes `is_backbone_block=True`. For heads and other top-level modules, the fallback passes `is_backbone_block=False`, so they do not use this CUDA graph branch.

## What Phase 5 Actually Tested

Phase 5 tested a lot of runs with `train.compile=true`, but mostly with the default compile mode and CUDA graphs disabled.

Recent example:

- Job 48312 was `DDP bs=96, no AC, compile=true, no expandable_segments`.
- `docs/phase5_perf_plan.md:391` records the run description.
- `docs/phase5_perf_plan.md:407-436` records the result context.
- `docs/claude-html-files/status.html:546` records iter 400-999: mean `1,387 img/s`, median `1,379`, MFU mean `7.94%`, `step_ms 545`, peak alloc `25.8 GB`.
- `docs/claude-html-files/status.html:574` labels it `DDP bs=96 no AC, compile=true, no ES`.

Archived short DDP screening also used compile:

- `docs/claude-html-files/status.html:623` lists job 9631 as `DDP bs=128 no AC, compile=true`, but explicitly flags it as a 100-iter archived screening run.
- `docs/claude-html-files/status.html:627` says these archived DDP rows are not current performance targets and contrasts them with job 48312.
- `docs/phase5_perf_plan.md:380` says the archived DDP convention question remains unresolved until rerunning the old recipe on the current codebase.

The safe interpretation is:

- We did test "PyTorch compile enabled."
- We did not fully test Armen's recommendation of static-shape, full-graph CUDA graph compile.
- Current `compile=true` means per-block or per-module compile wrappers, usually plain `module.compile()`.

## Dynamic Shapes and iBOT

The project docs say the problematic dynamic shape is the stochastic iBOT masked-token count.

Prior compile-mode screening concluded:

- `docs/perf_experiment_log.md:646-650` says a compiled kernel for one masked-token shape, such as shape `7503`, would fail on the next iteration's different shape and concludes `max-autotune-no-cudagraphs` is fundamentally incompatible.
- `CLAUDE.md:286` says `compile_mode: null` is the only viable path in the current setup because `max-autotune-no-cudagraphs` is incompatible with iBOT's dynamic masked-token count.

This is the core disagreement or open question behind Armen's feedback:

- Current code assumes stochastic iBOT mask counts make static/fullgraph compile hard.
- Armen is asking whether we should instead force the training shapes to be static, likely via fixed resizing plus padding/bucketing of masked-token tensors, so the compiler can use fullgraph/static assumptions.

## Armen's Recommendation, Interpreted Against This Codebase

Armen's advice has five separate parts:

1. Set `fullgraph=True` for DINO compile.
2. Make shapes static, or explain why they are not static if images are resized.
3. Remove FSDP for the 85M-100M parameter model.
4. Use DDP instead of FSDP2.
5. Turn off `reshard_after_forward`, because it is useful mainly for much larger models.

How that maps to this repo:

| Armen recommendation | Current Phase 5 state | What would need to change |
|---|---|---|
| `fullgraph=True` | Not used in default path. Only used if `train.cudagraphs=true` and only for backbone blocks. | Add a deliberate experiment with `train.cudagraphs=true`, or add a separate `fullgraph` flag that can be tested independently. |
| Static shapes | Images may be resized, but iBOT masked-token counts are stochastic and shape-varying. | Audit the exact tensor whose shape changes, then test padding/bucketing masked-token tensors to fixed sizes. |
| Remove FSDP | Phase 5 includes both FSDP2 and DDP comparisons. | Keep DDP as the main ViT-B candidate if matched long-soak DDP beats FSDP2 after replication. |
| Use DDP | Job 48312 is the current matched DDP bs=96 calibration. | Replicate same-node/same-date, and rerun the archived DDP bs=128 recipe on current code. |
| Turn off reshF/reshT issue | Phase 5 tested `reshard_after_forward` variants under FSDP2. | If using DDP, this setting becomes irrelevant; if using FSDP2, keep it in the matrix. |

## Key Takeaway

Do not say "we did not use torch.compile." We did.

The precise statement is:

> We used DINOv3's `train.compile=true`, which compiles transformer blocks and other modules through `nn.Module.compile()`. In Phase 5, that generally meant PyTorch's default compile path with `compile_mode: null` and `cudagraphs: false`. We have not yet validated the stronger Armen proposal: static shapes plus `fullgraph=True` / CUDA-graph-style compilation under DDP.

## Next Experiment That Would Answer Armen Cleanly

A clean test should separate three variables:

1. DDP vs FSDP2
2. dynamic default compile vs static/fullgraph compile
3. bs=96 vs bs=128

Suggested first experiment:

```bash
# Baseline, current default behavior
sbatch scripts/screening/ddp_bs96_calibration.sh

# New targeted variant, after adding/confirming static shape padding:
# DDP, bs=96, no AC, no ES, train.compile=true, train.cudagraphs=true,
# and either padded/bucketed iBOT mask tensors or a reduced config that proves
# fullgraph capture can run.
```

Acceptance criteria:

- It must run for at least 1000 iterations, not just 100.
- Report instantaneous `step_time_ms`, `images_per_sec`, and MFU over iter 400-999.
- Log recompiles or graph breaks if possible.
- If `fullgraph=True` fails, preserve the exact error because that identifies the Python operation or dynamic shape blocking capture.
- If padding is introduced, log padded vs real masked-token counts so the overhead is measurable.

## Evidence Index

- `dinov3/configs/ssl_default_config.yaml:80-82` - default compile flags.
- `dinov3/fsdp/ac_compile_parallelize.py:65-76` - wrapper that chooses CUDA graphs, explicit mode, or plain `module.compile()`.
- `dinov3/fsdp/ac_compile_parallelize.py:80-83` - `compile_mode` mapping.
- `dinov3/fsdp/ac_compile_parallelize.py:86-94` - ConvNeXt stage/downsample compile path.
- `dinov3/fsdp/ac_compile_parallelize.py:97-101` - transformer block compile path.
- `dinov3/fsdp/ac_compile_parallelize.py:191-198` - compile applied to top-level modules when `cfg.train.compile` is true.
- `dinov3/train/ssl_meta_arch.py:83,127,132` - student `dino_head`, `ibot_head`, and `ModuleDict`.
- `dinov3/train/ssl_meta_arch.py:845-852` - inference-only teacher/EMA models passed through the same parallelize helper.
- `docs/phase5_perf_plan.md:391` - job 48312 recipe.
- `docs/claude-html-files/status.html:546` - job 48312 measured result.
- `docs/claude-html-files/status.html:623-627` - archived DDP short-run caveat.
- `docs/perf_experiment_log.md:646-650` - dynamic iBOT mask shape vs `max-autotune-no-cudagraphs`.
