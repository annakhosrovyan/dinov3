# torch.compile Mode Screening — DINOv3

What compile modes mean in this repo, what was tested, and the key multi-rank autotuning hazard.

---

## Background: torch.compile modes relevant here

| Mode | What it does | Compile time | Expected steady-state gain |
|------|-------------|-------------|--------------------------|
| `default` (null) | Fast Triton codegen, no autotuning | ~35s (iter-0) | baseline |
| `max-autotune-no-cudagraphs` | Triton kernel autotuning (tile sizes, warp config) without CUDA graph capture | ~18+ min first run, cache reuse on subsequent | 5–20% faster matmuls in compiled blocks |
| `max-autotune` | Same + CUDA graphs | N/A for this repo | Confirmed broken (multi-crop architecture) |
| `reduce-overhead` | CUDA graphs only | N/A for this repo | Confirmed broken |

`max-autotune-no-cudagraphs` is the only available autotuning path that doesn't require CUDA graphs.
It finds the best Triton tile configuration for each matmul shape in the backbone blocks.

---

## Config key

`train.compile_mode` in `ssl_default_config.yaml` (default: `null` = PyTorch default mode).

CLI override: `train.compile_mode=max-autotune-no-cudagraphs`

Code path: `dinov3/fsdp/ac_compile_parallelize.py:_get_compile_mode()` → `wrap_compile_block()`.
`null` and `"default"` both map to `module.compile()` with no mode arg (zero behavior change).

---

## Default mode results (2026-04-07, job 16719)

Config: DDP bs=128, 8×H100, 200 iters.

Steady-state MFU (iters 20–199): **~23.1%**, range 22.9–23.5%.
~4040 img/s. Consistent with DDP bs=128 results from earlier screening runs.

---

## max-autotune-no-cudagraphs: multi-rank OOM hazard (2026-04-07, job 16720)

**What happened**: Job started at 13:31, crashed at 13:49 — 18 minutes into iter-0 compile.
Zero training log lines produced. SIGABRT on rank 2.

**Root cause**: `max-autotune-no-cudagraphs` benchmarks every Triton kernel variant by running
it on the GPU with test tensors. When 8 ranks all autotune concurrently (each on its own GPU),
each rank's GPU carries the model weights PLUS temporary benchmark allocations. The combined
per-GPU peak exceeds 80 GB, triggering OOM → SIGABRT.

This is NOT a steady-state memory issue — training at DDP bs=128 uses only ~34 GB. It is an
autotuning-time memory spike: Triton's benchmarking allocates additional buffers for each of
the ~8 kernel variants it tries per matmul shape.

**Stderr evidence**: 214K of `triton_mm_XX` benchmarking output, all from the iter-0 compile
phase. The last logged Triton candidates before crash were still exploring BLOCK configs for
an early backbone block. All 12 blocks × multiple ops = many concurrent benchmarks.

---

## Fix: single-rank Triton cache warmup before multi-rank launch

Triton caches autotuning results in `~/.cache/torch/inductor/`, keyed on:
- Op shape (same for nproc=1 and nproc=8 when per-GPU bs is identical)
- Hardware type (H100)
- PyTorch + Triton version

**Warmup procedure** (implemented in `scripts/screening_compile_modes.sh`):

1. **Phase 1**: `CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 ... train.OFFICIAL_EPOCH_LENGTH=2`
   — triggers compilation and autotuning on a single GPU. Only one rank benchmarks at a time,
   staying well under the 80 GB ceiling. Cache is written to NFS home dir.

2. **Phase 2**: Full `torchrun --nproc_per_node=8` run. Inductor finds existing cache entries,
   skips benchmarking, uses the pre-tuned kernel configs immediately. No OOM, normal compile
   warmup time (~35s like default mode).

**Why cache is valid across nproc=1 and nproc=8**: The compiled backbone blocks operate on
per-GPU tensors (shape driven by local batch size + token count). With the same `batch_size_per_gpu`,
the tensor shapes are identical regardless of world size. DDP's gradient all-reduce happens
outside the compiled regions (in the DDP wrapper), so it doesn't affect the cache keys.

---

## Root cause: iBOT dynamic masked token shapes (2026-04-08, jobs 16720 + 17824)

Two runs, same SIGABRT. The crash is architectural, not a configuration issue.

**The last autotuned op before crash:**
```
AUTOTUNE mm(2048x7503, 7503x768)   ← non-power-of-2, dynamic size
SingleProcess AUTOTUNE benchmarking takes 6.8508 seconds for 20 choices
[SIGABRT on rank 0]
```

**What `2048x7503` is**: iBOT loss operates on masked patch tokens gathered across the global
batch. With 8 GPUs × bs=128 = 1024 images × 2 global crops = 2048 crops. The `7503` is the
total number of masked patches across those 2048 crops — determined dynamically each iteration
by the stochastic masking process (mask_ratio_min_max=[0.1, 0.5]).

**Why this breaks max-autotune**: `max-autotune-no-cudagraphs` benchmarks every Triton kernel
variant for a specific shape (7503), selects the best tile config, and compiles that config.
The next iteration produces a different masked token count (e.g., 7489 or 7621). The compiled
kernel either:
1. Crashes on the shape mismatch (no guard — `dynamic=False` was assumed during tuning)
2. Triggers a CUDA assertion from the Triton kernel's tile config on the non-aligned dimension

The default compile mode handles this correctly by treating shapes symbolically (`dynamic=True`
implicitly), generating guards and recompilation as needed. `max-autotune` cannot be combined
with dynamic shapes through the simple `mode=` string API.

**Why the single-rank warmup fix also failed (job 17824)**:
Phase 1 (nproc=1, global_batch=128) caches backbone matmuls (fixed shapes — correctly cached),
but the iBOT global-batch op has size proportional to global batch size: nproc=1 gives ~939
masked tokens per 256 global crops; nproc=8 gives ~7503 per 2048 crops. **Different shape →
cache miss → fresh benchmarking in Phase 2 → same crash.**

**Conclusion: max-autotune-no-cudagraphs is fundamentally incompatible with iBOT's dynamic
masked token count.** This is not fixable by configuration — the mode requires static shapes,
but iBOT produces variable shapes at a global-batch granularity that changes with world size
AND with the stochastic masking per iteration.

---

## Current status (confirmed 2026-04-08)

| Mode | Result | MFU at bs=128 |
|------|--------|--------------|
| `default` (null) | ✓ Complete | **~23.1%** steady-state |
| `max-autotune-no-cudagraphs` | ✗ Incompatible — dynamic iBOT shapes | N/A |

**max-autotune-no-cudagraphs is deprioritized.** The default compile mode is the correct
and only viable torch.compile path for this model. Do not attempt further max-autotune runs.

---

## Operating rule for compile modes in this repo

1. Use `train.compile_mode=null` (default). It is stable, correct, and already contributing
   to the 23.9% MFU at DDP bs=256+ES (production config).
2. Never set `max-autotune` (with graphs) — CUDA graphs are broken here (see cuda_graphs.md).
3. Never set `max-autotune-no-cudagraphs` — iBOT dynamic shapes cause SIGABRT at iter-0.
4. If PyTorch adds a `dynamic=True` + `max-autotune` combination in a future version, retest.
