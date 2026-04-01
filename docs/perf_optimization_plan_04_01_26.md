# DINOv3 Performance Operating Plan

**Date**: 2026-04-01  
**Scope**: Post-MFU-baseline execution plan for profiling, optimization search, run tracking, and documentation hygiene.

## Current State

* **MFU tracking** is implemented in the training loop and logged to both console output and `training_metrics.json`.
* The **MFU denominator** has already been corrected to dense BF16 H100 peak (`989 TFLOPS`) in `dinov3/utils/mfu.py`.
* **Step timing** is measured with CUDA events around the core training step in `dinov3/train/train.py`.
* The repo already has **repeatable baseline scripts** for 2-GPU validation and 8-GPU real-data runs in `scripts/mfu_validation_run.sh` and `scripts/mfu_8gpu_real_data.sh`.

---

## Working Model

Treat optimization search as a constrained causal search over exposed `step_time_ms`. Use this order:

1.  Verify the metric.
2.  Get one good steady-state trace.
3.  Build a bottleneck ledger from that trace.
4.  Change only the knobs that hit the exposed bottleneck.
5.  Re-profile after each meaningful win.

### Key Interpretations for this Repo
* **MFU around 11-13%** on 8x H100 is plausible for this workload and is not obviously a measurement bug.
* This model is **not comparable** to large decoder-only LLM MFU numbers. Multi-crop ViT SSL has more sequential passes, smaller attention problems, and more overhead-heavy teacher work.
* Attention already goes through **SDPA** in `dinov3/layers/attention.py:106`.
* `torch.compile` is enabled and **CUDA graphs** are plumbed but disabled by default in `dinov3/configs/ssl_default_config.yaml:78` and `dinov3/fsdp/ac_compile_parallelize.py:65`.
* Memory numbers currently logged are **allocated tensor memory**, not full reserved VRAM, in `dinov3/logging/helpers.py:113`.

---

## What To Do Next

### Phase 0: Close the MFU Baseline Cleanly
**Goal**: Leave the repo with one trusted baseline and one trusted explanation.

* Re-run one short 8-GPU real-data baseline with the current corrected MFU code as the canonical baseline.
* Record three windows: post-compile steady-state, later thermal steady-state, and whole-run average.
* Confirm `images_per_sec ~= global_batch_size / (step_time_ms / 1000)`.
* Record both allocated and reserved memory to clarify headroom.

### Phase 1: Build Profiling Infrastructure
**Goal**: Make profiling cheap enough that an agent can run it repeatedly.

* Enable `--profiling` in `dinov3/train/train.py:91`.
* Add a profiling mode featuring:
    * Warmup-skip window.
    * PyTorch profiler trace export.
    * NVTX ranges (data wait, forward/backward, optimizer step, EMA update).
    * `torch._dynamo` graph-break logging.
* Extend JSON metrics to include `memory_reserved_mb`, `node_name`, `world_size`, and enabled flags (`compile`, `cudagraphs`, etc.).

### Phase 2: Run a Single Baseline Profiling Pass
**Goal**: Determine the exposed limiter before searching combinations.

1.  **PyTorch profiler** on one short 8-GPU steady-state run.
2.  **Nsight Systems** if the trace suggests exposed idle, CPU launch, or NCCL.
3.  **Nsight Compute** only for the top 2-3 kernels from the timeline.

### Phase 3: Search Plausible Knobs
Start with existing knobs: `batch_size_per_gpu`, `checkpointing`, `checkpointing_full`, and `cudagraphs`.
* Use checkpointing only as an enabler for a larger batch.
* Test CUDA graphs only after confirming shape stability.
* If NCCL is not exposed, do not spend time on communication overlap yet.

### Phase 4: Code-Changing Optimizations
These are secondary moves: batched multi-crop, DDP vs FSDP2 for ViT-B, or custom fused kernels.

---

## Branch Strategy

* **`mfu-tracking-baseline`**: Keep as the measurement baseline branch.
* **`perf-profiling-foundation`**: Branch here for profiling infrastructure.
* **Feature Branches**: Branch from profiling into focused experiments (e.g., `perf-batch-scaling`, `perf-cudagraphs`).

---

## Run Tracking Standard

Every run should leave a compact record. Suggested per-run note block:

```text
Run: 2026-04-01-baseline-rerun
Branch: <branch>
Commit: <sha>
Job: <slurm job id>
Node: <node>
Config delta: batch_size_per_gpu=64, compile=true, cudagraphs=false
Output dir: <path>
Window used for summary: iters X-Y
Images/sec: 
Step time ms: 
MFU %: 
Allocated mem MB: 
Reserved mem MB: 
Trace artifacts: <none|paths>
Bottleneck classification: 
Decision: 
```

---

## Immediate Recommended Sequence

1.  Reconfirm the 8-GPU corrected baseline as the canonical reference.
2.  Add profiling mode plus better memory/run metadata logging.
3.  Run one short baseline PyTorch-profiler job.
4.  Run one Nsight Systems job if the trace suggests exposed non-kernel time.
5.  Choose the first 2-3 factors for a small screening design from existing knobs.
