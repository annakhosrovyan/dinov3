# DINOv3 Performance Learnings

High-signal optimization notes distilled from local measurements and `~/knowledge-base`.

## Current Priorities

1. Benchmark DDP vs FSDP2 on single-node ViT-B. The model fits trivially on 80GB H100, and ZeRO-3/FSDP2 adds 50% more communication volume than DDP.
2. Treat `989 TFLOPS` as the MFU reporting denominator, but use `~794.5 TFLOPS` H100 BF16 MAMF as the realistic matmul ceiling when reasoning about remaining headroom.
3. Use MFU mainly for relative comparisons inside this repo. BF16-mixed training and activation checkpointing make absolute MFU comparisons noisy; cross-check FLOP math with `FlopCounterMode`.
4. Assume the data pipeline is already reasonably tuned unless profiling shows GPU idle gaps. `num_workers=20`, `pin_memory=True`, `non_blocking=True`, `persistent_workers=True`, and `prefetch_factor=8` already cover the obvious wins.
5. If multi-GPU runs show periodic throughput dips, suspect Python GC stragglers before chasing kernels. Disable automatic GC and collect manually on a fixed interval.
6. Do not over-index on batch-size alignment. For Tensor Core efficiency, sequence length and hidden dimension matter much more.

## Topic Notes

- `gpu_performance.md`: realistic H100 ceilings, tensor-shape alignment, GC stragglers, FFN-vs-attention cost, memory headroom, thermal drift.
- `cuda_graphs.md`: what CUDA graphs means in this repo, what a backbone block is, why upstream left the flag off by default, and why the current ViT-B satellite recipe is still a plausible candidate.
- `ddp_vs_fsdp2.md`: focused single-node guidance for when DDP should beat FSDP2, when FSDP2 still matters, and how the local ViT-B results should be interpreted.
- `terminology.md`: ongoing glossary of repo-specific terms and recurring performance jargon with code-grounded examples.
- `distributed_training.md`: communication overlap, NCCL pitfalls, DDP vs ZeRO/FSDP trade-offs, why ViT-B may not want FSDP2 on one node, and when `expandable_segments` helps or hurts.
- `profiling_workflow.md`: MFU vs HFU, `FlopCounterMode`, profiler order of operations, Nsight and `perf` workflows.
- `data_pipeline.md`: DataLoader tuning, H2D overlap, measured `num_workers` and `pin_memory` gains, when to escalate to DALI or GDS.
- `kernel_optimization.md`: roofline framing, Tensor Core behavior, fusion, arithmetic intensity, and batch-size effects.
