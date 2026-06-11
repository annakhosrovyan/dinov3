# DINOv3 Performance Learnings

High-signal optimization notes distilled from local measurements and `~/knowledge-base`.

## Current Priorities

1. Treat the old DDP+ES bs=256 notes as polluted screening evidence, not production proof. Short 70-100 step runs and limited 500-iter soaks are not enough to validate real training memory spikes or OOM risk.
2. Treat `989 TFLOPS` as the MFU reporting denominator, but use `~794.5 TFLOPS` H100 BF16 MAMF as the realistic matmul ceiling when reasoning about remaining headroom.
3. Use MFU mainly for relative comparisons inside this repo. BF16-mixed training and activation checkpointing make absolute MFU comparisons noisy; cross-check FLOP math with `FlopCounterMode`.
4. Assume the data pipeline is already reasonably tuned unless profiling shows GPU idle gaps. `num_workers=20`, `pin_memory=True`, `non_blocking=True`, `persistent_workers=True`, and `prefetch_factor=8` already cover the obvious wins.
5. If multi-GPU runs show periodic throughput dips, suspect Python GC stragglers before chasing kernels. Disable automatic GC and collect manually on a fixed interval.
6. Do not over-index on batch-size alignment. For Tensor Core efficiency, sequence length and hidden dimension matter much more.
7. Prefer FSDP2 as the long-run distributed platform. Validate any candidate batch size with real-duration training coverage, not short profiling windows.

## Topic Notes

- `gpu_performance.md`: realistic H100 ceilings, tensor-shape alignment, GC stragglers, FFN-vs-attention cost, memory headroom, thermal drift.
- `cuda_graphs.md`: what CUDA graphs means in this repo, what a backbone block is, why upstream left the flag off by default, and why the current multi-crop architecture breaks CUDAGraph trees.
- `compile_modes.md`: torch.compile mode options, default vs max-autotune-no-cudagraphs results, the multi-rank Triton OOM hazard, and the single-rank cache warmup fix.
- `terminology.md`: ongoing glossary of repo-specific terms and recurring performance jargon with code-grounded examples.
- `distributed_training.md`: communication overlap, NCCL pitfalls, FSDP2/DDP mode trade-offs, and allocator caveats.
- `profiling_workflow.md`: MFU vs HFU, `FlopCounterMode`, profiler order of operations, Nsight and `perf` workflows.
- `data_pipeline.md`: DataLoader tuning, H2D overlap, measured `num_workers` and `pin_memory` gains, when to escalate to DALI or GDS.
- `kernel_optimization.md`: roofline framing, Tensor Core behavior, fusion, arithmetic intensity, and batch-size effects.
