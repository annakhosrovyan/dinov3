# CUDA Graphs Learnings - DINOv3

Current repo-specific meaning of `train.cudagraphs`, including stale Phase 4
conclusions that should no longer be used as agent context.

Last corrected: 2026-06-11.

---

## Current operating rule

1. `train.cudagraphs=true` is viable and validated for the DDP ViT-B path after
   the Phase 6 fixes.
2. The current DDP champion path is `train.compile=true`,
   `train.cudagraphs=true`, `train.compile_mode=null`,
   `train.distributed_strategy=ddp`, and per-GPU `bs=128`.
3. `run.sh` remains the conservative FSDP2 path. `run_ddp.sh` is the faster DDP
   + CUDA-graphs path.
4. CUDA graphs in this repo are not a whole-training-step capture. They are an
   Inductor option applied to compiled backbone blocks.
5. The old Phase 4 negative verdict is stale. It predates the Phase 6 fixes
   listed below.

See:

- `run_ddp.sh:111-138`
- `dinov3/fsdp/ac_compile_parallelize.py:65-77`
- `dinov3/fsdp/ac_compile_parallelize.py:114-127`
- `dinov3/fsdp/ac_compile_parallelize.py:217-233`
- `dinov3/train/ssl_meta_arch.py:379-387`
- `dinov3/layers/block.py:146-177`
- `docs/phase6_perf_plan.md`
- `docs/phase7_compile_dispatch_agent_handoff.md`

---

## What "compiled backbone blocks use cudagraphs" means here

Implementation scope:

- `train.compile=true` enables the repo's compile loop during model setup.
- For ViT backbones, `compile_transformer()` iterates over
  `model.blocks`.
- If `train.cudagraphs=true`, each backbone block takes this branch:

  ```python
  module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
  ```

- Non-backbone modules, including `dino_head` and `ibot_head`, are compiled via
  the head path with `use_cuda_graphs=False`.
- DDP wrapping happens after compile.

Decision implication:

- This knob mainly reduces launch overhead inside repeated transformer blocks.
- It does not graph the full trainer, losses, optimizer step, logging,
  checkpointing, DDP all-reduce, or Sinkhorn collectives.

---

## What a backbone block is here

In this repo:

- `SSLMetaArch` builds `student["backbone"]`, `teacher["backbone"]`, and heads
  such as `dino_head` and `ibot_head`.
- For `student.arch=vit_base`, the backbone is a 12-block ViT-B/16.
- A backbone block is one `SelfAttentionBlock` inside
  `DinoVisionTransformer.blocks`.

Practical implication:

- CUDA graphs target the repeated ViT trunk first.
- Any head-side CUDA-graph work is secondary and expected to have lower ROI
  unless profiling shows dispatch overhead there.

---

## Why upstream left the flag off by default

Best-supported reading:

- The strict branch uses `fullgraph=True` and `dynamic=False`.
- The broader DINOv3 codebase supports recipes with less stable execution shape.
- The code has reset hooks for feature changes such as FP8 and sparsity.

Therefore:

- Plain compile is the robust baseline.
- `cudagraphs=true` is a recipe-level benchmark/training choice for stable
  repeated shapes, not a universal default for all recipes.

---

## Phase 6 fixes that changed the verdict

The early multi-crop failure was not the final state of this branch. Phase 6
made two load-bearing changes:

1. `torch.compiler.cudagraph_mark_step_begin()` is called at the top of
   `SSLMetaArch.forward_backward()`. This tells the Inductor CUDA graph tree
   that a new training step is starting and prevents unsafe buffer aliasing
   across repeated graph invocations.
2. Advanced indexing in the block sample-drop path was replaced with
   `torch.index_select`, making the compiled/cudagraph path compatible with the
   active block code.

Current implication:

- Do not cite the early Phase 4 CUDA-graph failure result as current.
- The DDP+cudagraphs path is a measured Phase 6.A winner for single-node ViT-B
  throughput.

---

## Current caveats

CUDA graphs still have narrower applicability than plain compile:

- The code only applies the explicit cudagraph option to backbone blocks.
- Training still includes Python orchestration, EMA update, losses, optimizer,
  DDP all-reduce, and logging outside the graphed region.
- Changing per-GPU batch size or recipe shape can require new capture/cache
  work.
- FSDP2 is still a different risk profile because sharding and parameter
  materialization complicate capture.
- Whole-step graphing remains a separate research item and is bounded by
  collectives and H2D/data boundaries.

---

## Short version

Use this file's current rule, not the old Phase 4 result: CUDA graphs are viable
for the DDP ViT-B champion path and are already part of `run_ddp.sh`. In this
repo, `train.cudagraphs=true` means backbone-block fullgraph/static-shape
Inductor CUDA graph replay. It does not mean full-step capture, and it does not
make `compile_mode` control the backbone when the cudagraph branch is active.
