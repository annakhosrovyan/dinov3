# DDP vs FSDP2 on a Single Node

Distilled from `~/knowledge-base/ml-engineering/training/model-parallelism/README.md` and
`~/knowledge-base/ml-engineering/training/performance/README.md`, then narrowed to the DINOv3
single-node ViT-B case using the local measurements in this repo.

This note is about a narrow but common decision: one training job, one host, several local GPUs.
The answer is not "FSDP2 is newer, so use FSDP2." The real trade is memory savings versus extra
communication and wrapping complexity.

---

## Bottom Line

- If the model, gradients, optimizer states, and activations fit comfortably on each GPU at the
  target batch size, start with DDP.
- If DDP OOMs, or only fits at a batch size that is too small to be useful, switch to FSDP2.
- If sharding is required, prefer FSDP2 over legacy FSDP1 in this codebase.

For single-node training, DDP is often the throughput-first choice for small and mid-sized models.
FSDP2 is the memory-first choice when fitting the model is the real problem.

---

## Why This Matters In This Repo

- `run.sh` is a single-node job (`#SBATCH --nodes=1`) on `8x H100`.
- The default config still sets `train.distributed_strategy: fsdp2` in
  `dinov3/configs/ssl_default_config.yaml:82`.
- The codebase supports both strategies in `dinov3/fsdp/ac_compile_parallelize.py:143-188`.
- The DDP path explicitly notes that ViT-B is only about `172 MB` in BF16 and fits trivially on an
  `80 GB` H100 in `dinov3/fsdp/ac_compile_parallelize.py:268-271`.

That makes this a real decision, not a hypothetical one. For ViT-B on one H100 node, FSDP2 is not
required just to make the model fit.

---

## Rule Of Thumb

| Constraint | Prefer DDP | Prefer FSDP2 |
|---|---|---|
| Model + optimizer state fit per GPU with headroom | Yes | No |
| Single-node throughput is the main goal | Yes | Only if memory forces it |
| Model or batch size OOMs under replication | No | Yes |
| You need sharded params, grads, and optimizer state | No | Yes |
| Small / mid-sized model that already fits | Usually | Usually not |
| Much larger future model or tighter VRAM budget | Maybe not | Usually |

The key question is not "single node or multi node?" It is "does replication still fit
comfortably?" On one node, the communication penalty of FSDP2 is smaller than on multi-node jobs,
but it is still real.

---

## Why DDP Can Win On One Node

DDP keeps a full replica of the trainable model on every rank and synchronizes gradients with an
all-reduce during backward. FSDP2 shards parameters, gradients, and optimizer states, but it pays
for that memory reduction with repeated all-gather and reduce-scatter collectives around wrapped
modules.

For a model that already fits, DDP avoids:
- Per-block parameter materialization and resharing.
- Extra all-gather traffic in forward and backward.
- Some of the tuning and allocator sensitivity that shows up with sharded execution.

On a single NVLink or PCIe node, those collectives are much cheaper than on a network, but they are
not free. That is why "single node" does not automatically mean "FSDP2 is fine either way."

---

## Why FSDP2 Still Matters

FSDP2 exists for the cases where DDP replication is the bottleneck:
- The model does not fit at all under DDP.
- The optimizer state or gradients, not just the weights, blow up memory usage.
- Activation checkpointing and BF16 are still not enough to reach a useful batch size.
- A future backbone, teacher stack, or head configuration grows beyond ViT-B-scale assumptions.

If memory is the active constraint, FSDP2 is the right tool. If memory is not the active
constraint, FSDP2 should not be assumed to be neutral.

---

## Repo-Specific Observations

Local notes already point in the same direction for single-node ViT-B on `8x H100`:

| Config | MFU | img/s | Note |
|---|---|---|---|
| DDP bs=256 + `expandable_segments:True` | **24.5%** | **4229** | Best measured local result |
| FSDP2 bs=256 | 23.5% | 4106 | Close, but slower |
| FSDP2 bs=256 + `expandable_segments:True` | 21.8% | 3809 | Allocator setting hurt FSDP2 |

These numbers come from the experimental summary in `learnings/distributed_training.md`.

Two practical implications follow:
- DDP is already competitive or better for the current single-node ViT-B regime.
- Allocator tuning is strategy-specific in this repo; `expandable_segments` helped DDP and hurt
  FSDP2, so do not assume a memory tweak transfers cleanly between both strategies.

---

## Practical Recommendation For DINOv3

1. For single-node ViT-B runs, benchmark DDP first.
2. Keep FSDP2 available for cases where memory headroom becomes the blocker.
3. Compare both strategies under the same batch size, compile settings, and data pipeline.
4. Re-evaluate the choice if model size, optimizer choice, checkpointing policy, or node topology
   changes.

For the current setup, "DDP first, FSDP2 if memory forces it" is the right default framing.

---

## Related Notes

- `learnings/distributed_training.md` for communication overlap, NCCL notes, and allocator results.
- `learnings/gpu_performance.md` for memory headroom and utilization framing.
- `learnings/terminology.md` for MACs, DDP, FSDP2, sharding, and collective terms.
