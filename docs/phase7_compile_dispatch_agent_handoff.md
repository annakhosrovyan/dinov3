# Phase 7 torch.compile Dispatch Handoff

Date: 2026-06-11

Purpose: give another AI agent the current, code-derived understanding of how `run_ddp.sh` routes config into `dinov3/fsdp/ac_compile_parallelize.py`, and where the prior Phase 4/6 compile-mode story is stale.

## Bottom line

The other agent's branch-order claim is correct, but the useful framing needs tightening.

- `run_ddp.sh` leaves `train.compile_mode` unset/null and overrides `train.cudagraphs=true`, `train.compile=true`, and `train.distributed_strategy=ddp`.
- Config merge order is default config, selected config file, then CLI overrides. The `run_ddp.sh` CLI values win.
- `ac_compile_parallelize()` compiles first, then applies DDP or FSDP wrapping.
- Under current `run_ddp.sh`, ViT backbone blocks use:
  `module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})`.
- Under current `run_ddp.sh`, `dino_head` and `ibot_head` use plain `module.compile()` because heads are passed with `use_cuda_graphs=False` and `compile_mode=None`.
- DDP wrappers do not bypass or replace the compile dispatcher. They are applied afterward to the student submodules only.

## Code evidence

### Config source and overrides

- Defaults: `dinov3/configs/ssl_default_config.yaml:78-83`
  - `train.checkpointing=false`
  - `train.compile=true`
  - `train.compile_mode=null`
  - `train.cudagraphs=false`
  - `train.distributed_strategy=fsdp2`
- DDP script overrides: `run_ddp.sh:111-138`
  - `train.compile=true`
  - `train.cudagraphs=true`
  - `train.checkpointing=false`
  - `train.checkpointing_full=false`
  - `train.distributed_strategy=ddp`
  - no `train.compile_mode=...` override, so it remains null.
- Merge order: `dinov3/configs/config.py:71-90`
  - `overrides = [*args.opts]`
  - `opts_cfg = OmegaConf.from_cli(overrides)`
  - `cfg = OmegaConf.merge(default_cfg, cfg, opts_cfg)`

### Compile dispatch

`dinov3/fsdp/ac_compile_parallelize.py:65-77`:

```python
def wrap_compile_block(module, use_cuda_graphs, is_backbone_block, compile_mode=None):
    if use_cuda_graphs and is_backbone_block:
        module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
    elif compile_mode is not None:
        module.compile(mode=compile_mode)
    else:
        module.compile()
    return module
```

This means branch order is load-bearing:

- If `cudagraphs=true` and the module is a backbone block, branch 1 wins before `compile_mode` is consulted.
- If `compile_mode` is set and branch 1 does not apply, branch 2 uses `module.compile(mode=compile_mode)`.
- If neither applies, branch 3 uses default `module.compile()`.

### What gets compiled

`dinov3/fsdp/ac_compile_parallelize.py:217-233` compiles all model dictionaries in:

```python
all_models = [trained_model] + inference_only_models
```

For each dictionary:

- key `backbone` calls architecture-specific compile function.
- non-backbone keys, including `dino_head` and `ibot_head`, call:
  `wrap_compile_block(model[k], use_cuda_graphs=False, is_backbone_block=False, compile_mode=compile_mode)`.

Important precision: "inference-only" does not mean uncompiled. EMA/teacher model dictionaries are included in the compile loop. It means they are not DDP-wrapped later.

### DDP wrapper ordering

`dinov3/fsdp/ac_compile_parallelize.py:164-176` documents wrapper order:

1. Activation checkpointing.
2. Compile blocks.
3. FSDP blocks and global model, or DDP when `cfg.train.distributed_strategy == "ddp"`.

`dinov3/fsdp/ac_compile_parallelize.py:235-242` calls the DDP helper after the compile loop.

`dinov3/fsdp/ac_compile_parallelize.py:316-357` then:

- moves all model dictionaries to CUDA.
- casts params to `cfg.compute_precision.param_dtype`.
- DDP-wraps each student submodule.
- leaves inference-only models unwrapped because they have no gradients.

## Effective behavior matrix

| Scenario | Backbone blocks | Heads | Meaning |
|---|---|---|---|
| Current `run_ddp.sh`: `cudagraphs=true`, `compile_mode=null` | branch 1: `fullgraph=True`, `dynamic=False`, `options={"triton.cudagraphs": True}` | branch 3: `module.compile()` | Champion baseline. No max-autotune. |
| Add only `train.compile_mode=max-autotune-no-cudagraphs` while keeping `cudagraphs=true` | still branch 1, `compile_mode` shadowed | branch 2: `mode="max-autotune-no-cudagraphs"` | Not a backbone autotune test. It reaches non-backbone heads only. |
| Set `cudagraphs=false` and `compile_mode=max-autotune-no-cudagraphs` | branch 2: `mode="max-autotune-no-cudagraphs"` | branch 2 | Config-only autotune isolation test, but not the champion path. |
| Edit branch 1 to `options={"max_autotune": True, "triton.cudagraphs": True}` | branch 1 with both options | unchanged unless separately edited | Actual combined autotune plus graph replay test for the backbone. |

## Review of the contested AI response

Confirmed:

- It is correct that branch 1 in `wrap_compile_block()` wins before `compile_mode` is consulted.
- It is correct that a config-only `compile_mode=max-autotune-no-cudagraphs` override under `cudagraphs=true` would not autotune the backbone.
- It is correct that `fullgraph+cudagraphs` and `max_autotune` are different optimization axes: graph replay targets launch overhead; autotune targets matmul kernel choice.

Needs precision:

- "Only autotune the heads" should be read as "only non-backbone modules in the compile loop." That includes student heads and inference-only/EMA/teacher heads, not just the student heads.
- The direct payoff is still expected to be small because the documented head matmul share is about 2.8 percent, but that is a matmul-MAC estimate, not measured wall time.
- A config-only `max-autotune-no-cudagraphs` test with `cudagraphs=false` can measure autotune in isolation, but it necessarily abandons the champion's graph-replay win.

Stale or contradictory context to avoid:

- `learnings/compile_modes.md:137-156` retracts the "dynamic K" story.
- `learnings/compile_modes.md:158-164` still says never use `max-autotune-no-cudagraphs` because iBOT dynamic shapes cause SIGABRT. That operating rule conflicts with the correction above and should be treated as stale until the file is cleaned up.
- `collate.py:63` says the post-`nonzero()` shape is variable. Current code contradicts that at fixed per-GPU batch size.

## iBOT K shape correction

For the default fixed per-GPU batch-size path:

- `dinov3/data/collate.py:42-56` deterministically chooses `n_samples_masked`, `probs`, and the target patch count `int(N * prob_max)` for each masked sample.
- `dinov3/data/masking.py:87-93` calls `complete_mask_randomly()`, which force-tops-up every mask to exactly the requested `num_masking_patches`.
- Therefore the number of `True` entries in `collated_masks` is constant for a fixed local batch size. Positions vary; count does not.
- At `bs=128`, the current Phase 7 doc says empirical trials produced `K = 7503` every iteration.

Caveats:

- Changing per-GPU batch size changes K and may trigger a separate compile/cache key.
- Multidistillation and some Gram token-selection paths can change shape behavior. They are not enabled in the current champion DDP config.

## PyTorch terminology mapped to this repo

From the PyTorch 2.10 `torch.compile` API:

- `fullgraph=True`: require the whole called function/module frame to be capturable as one graph; otherwise raise.
- `dynamic=False`: specialize to static shapes instead of generating dynamic kernels.
- `mode="max-autotune-no-cudagraphs"`: use max-autotune mode without CUDA graph capture.
- `mode="max-autotune"`: max-autotune mode and CUDA graphs by default on GPU.
- `options={"max_autotune": True}`: explicitly enable profiling to pick a matmul configuration.
- `options={"triton.cudagraphs": True}`: explicitly enable CUDA graph replay for supported regions.

Repo-specific trap: because branch 1 uses `options=...` directly and is selected before `compile_mode`, mode strings are not the way to tune the champion backbone path.

## Recommended Phase 7 ladder

1. Run nsys residual attribution first. Measure the actual head/loss wall share rather than inferring it from MACs.
2. If testing autotune, run the config-only isolation test with `cudagraphs=false` and `compile_mode=max-autotune-no-cudagraphs`, using matched per-GPU batch size and a single-rank warmup to populate the Triton/Inductor cache.
3. Only if the isolation test shows a real step-time gain, edit branch 1 to combine `max_autotune` with `triton.cudagraphs` and benchmark against the current DDP+cudagraphs bs=128 champion.
4. Treat head CUDA-graph/fullgraph work as conditional and likely low ROI unless nsys shows dispatch overhead in that path.

## One-line candidate patch for the gated test

Do not apply this as a default. It is the gated Phase 7.3 experiment:

```python
module.compile(
    fullgraph=True,
    dynamic=False,
    options={"max_autotune": True, "triton.cudagraphs": True},
)
```

Target location: `dinov3/fsdp/ac_compile_parallelize.py:71-72`.

## Short version

`run_ddp.sh` selects DDP plus backbone fullgraph/CUDA graphs, not max-autotune. `compile_mode` is still null in the champion script. If you set a compile mode while keeping `cudagraphs=true`, the backbone ignores it because branch 1 wins first; only heads reach the mode path. To combine max-autotune with the champion backbone path, edit branch 1 options and test it as a gated experiment.
