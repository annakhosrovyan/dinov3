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

## Current status (2026-04-08)

- Default mode: **measured** — ~23.1% MFU at bs=128
- max-autotune-no-cudagraphs: **pending rerun** with single-rank warmup fix
- Comparison delta (default vs max-autotune): **unknown until rerun completes**

Expected: if max-autotune finds better tile configs, a 5–15% improvement in kernel throughput
would raise MFU from ~23.1% to ~24–26% at bs=128, or from 23.9% to ~25–27% at bs=256.
If the backbone's matmul shapes (seq_len=197, hidden=768) already happen to align well with
PyTorch's default Triton config, the gain may be smaller (2–5%).

---

## Operating rule for max-autotune in this repo

1. Always run the single-rank warmup before launching the 8-rank job.
2. The warmup takes ~18–25 min (once per hardware/PyTorch version combo).
3. Subsequent runs on the same node reuse the cache — warmup not needed again.
4. If PyTorch version changes, clear `~/.cache/torch/inductor/` and re-warm.
5. Never set `train.compile_mode=max-autotune` (with graphs) — CUDA graphs are broken here.
