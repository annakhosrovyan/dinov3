# Profiling Workflow — AI Systems Perf Eng Ch13

Notes distilled from `~/knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch13.md`.
Tailored to DINOv3 8×H100 / FSDP2 / torch.compile setup.

---

## Tool Selection & Recommended Order

| Tool | Scope | When to Use |
|------|-------|-------------|
| **PyTorch Profiler** | Op-level (CPU+GPU) | First pass — which ops are hot; validate FLOP formula |
| **Nsight Systems** | System-wide timeline | After PyTorch profiler — end-to-end gaps, NCCL overlap, data stalls |
| **Nsight Compute** | Per-kernel deep-dive | After nsys identifies hot kernels — roofline, occupancy, stall reasons |
| **Linux perf** | CPU side | Catch Python overhead, GIL, I/O, host-side bottlenecks |
| **HTA** | Multi-GPU distributed | Merge rank traces — load imbalance, communication overlap |

**Rule**: Always warm-up 5–10 iterations before profiling. Skip iters 1–10 for DINOv3 (torch.compile JIT = ~35s first iter).

---

## 1. PyTorch Profiler — Exact Setup

```python
from torch import profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=50, warmup=5, active=5),  # skip first 55 iters
    on_trace_ready=profiler.tensorboard_trace_handler(
        "/mnt/weka/adovlatyan/profiler_traces"),
    record_shapes=True,    # shape-dependent bottlenecks
    profile_memory=True,   # per-op memory usage
    with_stack=True,       # full call stacks for root cause
    with_flops=True,       # cross-check against compute_dino_flops_per_image()
) as prof:
    for step, batch in enumerate(loader):
        train_step(...)
        prof.step()

# Print top-10 ops by GPU time
print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=10,
    fields=["self_cuda_time_total", "calls"]))

prof.export_chrome_trace("/mnt/weka/adovlatyan/profiler_traces/trace.json")
# View at: https://ui.perfetto.dev/
```

**What to look for:**
- `aten::matmul` / `aten::linear` dominating → compute-intensive, focus on Tensor Core utilization
- `dispatch`/`combine`/`scatter`/`gather` → communication or MoE overhead
- Layer norm, softmax, activations prominent → fuse with adjacent ops via torch.compile
- Many small ops in sequence → good candidate for torch.compile fusion
- Does `with_flops=True` estimate agree with `compute_dino_flops_per_image()`? If not, formula needs revisiting.

---

## 2. NVTX Markers — Instrument DINOv3 Training Loop

Add to `dinov3/train/train.py` before profiling runs. These make Nsight Systems traces readable.

```python
# Key locations to wrap (train.py):
torch.cuda.nvtx.range_push("dataload")
batch = next(data_iter)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("H2D")
# collate_data_and_cast — moves to GPU
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("teacher_fwd")
# teacher forward (@no_grad, ssl_meta_arch.py:391-398)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("student_fwd")
# student forward (global + local, ssl_meta_arch.py:400-407)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("losses_backward")
# compute_losses + backprop_loss, ssl_meta_arch.py:421-433
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("optimizer_ema")
# optimizer.step + model.update_ema, train.py:606-607
torch.cuda.nvtx.range_pop()
```

Low overhead when no profiler attached — safe to leave in permanently.

---

## 3. Nsight Systems — Commands

```bash
# Capture steady-state iters (skip warmup externally or use cudaProfilerStart/Stop)
nsys profile -t cuda,nvtx -o /mnt/weka/adovlatyan/nsys_traces/dinov3 \
    torchrun --nproc_per_node=8 dinov3/train/train.py ...

# Get NVTX summary table (no GUI needed)
nsys stats --report=nvtx_gpu_proj_sum /mnt/weka/adovlatyan/nsys_traces/dinov3.nsys-rep
```

**What to look for:**
- Is backward compute overlapping with FSDP AllGather / AllReduce? (look for NCCL regions alongside compute kernels)
- GPU idle gaps before each iteration → data loading bottleneck
- One rank consistently finishing later → load imbalance (dataset, masking?)
- `NCCL` taking >30% of step time → communication bottleneck

---

## 4. Nsight Compute — Roofline on Hot Kernels

```bash
# After nsys identifies hot kernel name:
ncu \
  --target-processes all \
  --kernel-name-regex "matmul" \
  --metrics \
    gpu__time_duration.avg,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
  --csv full \
  -o /mnt/weka/adovlatyan/ncu_reports/matmul_roofline \
  torchrun --nproc_per_node=1 dinov3/train/train.py ...  # single GPU for ncu
```

**Roofline interpretation:**
- `gpu__dram_throughput >> FLOPS%` → **memory-bound** → increase arithmetic intensity (batch size, fuse kernels)
- `FLOPS% >> dram_throughput` → **compute-bound** → focus on overlap (communication, data)
- Low occupancy (<25%) AND `Stall: Not Selected` high → insufficient parallelism

---

## 5. torch.compile — Check for Graph Breaks

```bash
# Log all graph breaks and inductor decisions
TORCH_LOGS="+dynamo,+inductor" \
    torchrun --nproc_per_node=8 dinov3/train/train.py ... 2>&1 | tee /mnt/weka/adovlatyan/logs/compile_debug.log

# Or in Python:
torch._dynamo.explain(model)
```

Each graph break = re-enter Python interpreter = overhead. FSDP boundaries cause breaks (expected, correct). Other breaks are optimization opportunities.

**Compile modes:**
| Mode | Best For |
|------|---------|
| `default` | General; tight memory |
| `reduce-overhead` | Small batches, dynamic shapes |
| `max-autotune` | Long training runs, stable shapes — **use this** |

Cache compiled artifacts between runs:
```bash
export TORCHINDUCTOR_CACHE_DIR=/mnt/weka/adovlatyan/triton_cache
```

---

## 6. NCCL / FSDP Communication Profiling

```bash
# Verbose NCCL logging (use for debug, not production):
NCCL_DEBUG=INFO torchrun ... 2>&1 | tee nccl_debug.log
# Look for: "NET/IB" (NVLink/InfiniBand path); "NET/Socket" = fallback = bad
# Also: collective timings in the log

# NVLink PMU events (confirms NVLink is doing the work):
perf stat -a -e nvidia_nvlink_c2c0_pmu_0/cycles/ torchrun ...
# High NVLink cycles = communication is NVLink-routed (expected on H100 cluster)
```

**FSDP overlap**: `SHARD_GRAD_OP` strategy should overlap AllGather with forward compute.
Nsight Systems timeline should show interleaved compute kernels and NCCL operations, not sequential.
If they're sequential → sync point causing serialization → investigate `wait_event` calls.

---

## 7. Multi-GPU Profiling with HTA

```bash
# Collect per-rank traces (set TORCH_PROFILER_TRACE_DIR or use on_trace_ready above)
# Then:
hta analyze --trace-dir /mnt/weka/adovlatyan/profiler_traces --output-dir ./hta_output
```

HTA reveals load imbalance (one rank finishes forward much later), idle time per rank, AllReduce gaps.

---

## 8. Single-GPU Profiling Checklist (do this first before 8-GPU)

```
[ ] 5+ warm-up iters (skip torch.compile JIT) — for DINOv3 skip first 10 iters
[ ] PyTorch Profiler with record_shapes, profile_memory, with_stack, with_flops
[ ] Review top-10 ops by self_cuda_time_total
    [ ] Dominated by matmul/linear? → Tensor Core utilization issue
    [ ] Dominated by NCCL? → This is multi-GPU, go to multi-GPU checklist
[ ] Run nsys for timeline view (5 steady-state iters sufficient)
[ ] For top-3 hot kernels, run ncu for roofline
    [ ] Memory-bound? → batch size, kernel fusion
    [ ] Compute-bound? → communication overlap issue
[ ] Check graph breaks: TORCH_LOGS="+dynamo"
[ ] Check MFU formula cross-validation: with_flops=True should approximate
    2 × compute_dino_flops_per_image() × imgs_per_sec / (GPUs × 989e12)
```

---

## 9. CUDA Graph Capture (if considering)

```python
# 1. Preallocate static tensors
# 2. Warm-up to finalize allocations
# 3. Capture with torch.cuda.CUDAGraph()
# 4. Replay with g.replay() + static_input.copy_(new_data) each iter

# Gotchas:
# - NO dynamic shapes
# - NO new tensor allocations during capture
# - ALWAYS sync before reading results
# - Weights updated outside graph won't be in replayed graph
# DINOv3: multi-crop has variable seq lengths (197 vs 37) → cannot capture both in one graph
```

---

## 10. Key Diagnostic Questions (from Ch13 decision tree)

| Question | Tool | What it reveals |
|----------|------|-----------------|
| % of step in NCCL/FSDP comms? | Nsight Systems | If >30% → comms bottleneck |
| Are NCCL and compute overlapping? | Nsight Systems timeline | FSDP SHARD_GRAD_OP should overlap |
| Graph breaks in torch.compile? | `TORCH_LOGS` / `_dynamo.explain` | Each break = Python overhead |
| Do FLOP estimates match formula? | PyTorch Profiler `with_flops=True` | Cross-validates 226 GMACs/image |
| Are FFN GEMMs memory-bound? | Nsight Compute Roofline | Yes → increase batch size; No → algo change |
| Is one rank slower than others? | HTA / Nsight Systems | Dataset I/O imbalance? Masking? |
