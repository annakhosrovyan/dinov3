# GPT-5 Pro Prompt: Phase 6 DDP + CUDA Graphs Performance Ideation

Use this after pasting `compressed_phase6_perf_plan.md`, plus any selected source excerpts from
`../shared/`. This prompt supersedes the older FSDP2 / bs=128 prompt for the current performance
direction.

## Current Rough Direction

We have pivoted away from "make FSDP2 bs=128 work" as the primary performance path.

The current measured champion is:

- Single node, 8x H100.
- DDP, not FSDP2.
- `train.compile=true`.
- `train.cudagraphs=true`.
- `train.batch_size_per_gpu=128`.
- No activation checkpointing.
- Comparator job: `53708`.
- Observed screen result: about `2,394 img/s`, `13.70% MFU`, `34.1 GB` peak VRAM.
- Full long-soak validation job `60959` completed `4000/4000` iterations; it passed the old
  rank-6 crash point and showed flat VRAM across all checkpoint/eval events.

The next high-value direction is not more broad knob twiddling. It is to find places where the
current DDP + cudagraph path still leaks performance because parts of the training step remain
dynamic, launch-heavy, host-bound, communication-exposed, or poorly batched.

The strongest known Phase 6.B candidate is:

- Make the DINO / iBOT head and loss path more static-shape and graphable.
- In particular, address dynamic iBOT masked-token count (`n_masked`) so heads/losses can be
  pulled closer to the fullgraph CUDA-graph regime already achieved for the backbone blocks.

Other plausible areas:

- Wall-clock vs GPU-event gap in soaks.
- Host-side DataLoader / page-cache / collate / H2D overlap.
- Redundant teacher/student or global/local crop work.
- iBOT masked patch gather/head/loss organization.
- DINO/iBOT center all-reduce launch and overlap.
- EMA update and optimizer overhead outside captured graph.
- Multi-node future: DDP all-reduce bucketization, overlap, NCCL topology, and whether the
  single-node DDP+cudagraph assumptions survive network collectives.

Do not assume 30% MFU is realistic for ViT-B + DINO/iBOT + 10 crops on one 8x H100 node. Treat
15-25% as a more plausible local ceiling unless evidence says otherwise. The goal is to find
days-scale, DINOv3-specific performance experiments that preserve representation quality.

## Prompt To Paste Into GPT-5 Pro

I want you to act as a frontier-lab style systems/performance research partner for a
satellite-specialized DINOv3 self-supervised vision training fork.

I will paste compact context, current Phase 6 evidence, and selected source excerpts. Your task
is not generic GPU tuning. Your task is to search for FlashAttention-like opportunities in the
current Phase 6 direction:

- DDP on one 8x H100 node.
- `torch.compile` enabled.
- CUDA graphs enabled for the backbone blocks.
- Per-GPU batch size 128 is the current champion.
- Activation checkpointing is not the throughput path for this model/scale.
- FSDP2 remains the conservative stability baseline, but it is not the current performance
  champion.

Do not answer until I paste all context and say:

`END CONTEXT, BEGIN IDEATION`

Expected outcome:

Generate ranked DINOv3-specific ideas for improving delivered training efficiency with minimal
expected accuracy loss. Each idea must include:

- mechanism;
- why it may preserve training semantics;
- expected performance lever;
- implementation sketch;
- accuracy/stability risk;
- smallest falsifying experiment;
- whether the idea is single-node only, multi-node relevant, or both.

Scope:

- Focus on DDP + CUDA graphs, static-shape execution, DINO/iBOT heads and losses, multi-crop ViT
  execution, iBOT masked-patch gather/head/loss structure, teacher/student asymmetry, host-side
  wall-clock gaps, and DDP collective overlap.
- You may discuss FSDP2 only as contrast or for future multi-node implications.
- You may discuss attention kernels, but do not limit yourself to attention. The bigger remaining
  bottleneck may be dynamic shape overhead, head/loss launch overhead, gather/scatter memory
  traffic, loss collectives, CPU-side step overhead, or bad crop batching structure.
- Prefer ideas testable in this repo within days.
- Do not propose changing many knobs at once.
- Do not optimize benchmark-only GPU-event throughput at the cost of likely wall-clock regressions
  or representation-quality collapse.

Current observed facts to respect:

- Backbone CUDA graphs were a real win after replacing advanced indexing with `index_select` in
  the block path.
- DDP + cudagraphs + bs=128 no-AC is the measured throughput champion so far.
- AC=full greatly reduces VRAM but costs about 31% throughput at bs=96/128; it is a VRAM-budget
  mode, not the current speed path.
- bs=192 attempts failed due to host-RAM / Slurm cgroup behavior, not CUDA VRAM.
- Loader memory accounting via cgroup `memory.current` is page-cache contaminated; do not infer a
  fixed per-worker memory model from cgroup limit plateaus.
- A deterministic rank-6 / iter-1310 crash was root-caused to corrupt NAIP tile error logging into
  a read-only dataset directory; it is fixed and is not a CUDA graph or iBOT failure.
- The 60959 soak completed 4000/4000 and showed flat VRAM, but its honest sustained throughput is
  lower than the optimistic 53708 short-screen number.
- MetricLogger `images_per_sec` is GPU-event step throughput, not necessarily delivered wall-clock
  throughput. Treat wall-clock gaps as first-class evidence.

Evidence rules:

- Separate observed facts from inference.
- Use `HYPOTHESIS:` for causal claims not directly proven by pasted evidence.
- If a proposed idea changes the SSL objective, mask distribution, crop policy, teacher/student
  semantics, or loss weighting, label it higher-risk.
- If an idea only improves a benchmark metric but may not improve wall-clock throughput, say so.
- If an idea is mostly useful for multi-node rather than single-node, say exactly what new evidence
  would be needed before coding.

Output format:

1. Executive conclusion: where the best opportunity likely is now.
2. Ranked idea table with payoff, accuracy risk, implementation difficulty, first experiment, and
   single-node vs multi-node relevance.
3. Deep dive on the top 5 ideas.
4. Evidence that supports or weakens each idea.
5. Exact first experiment plan for the best idea.
6. Additional profiling evidence wanted before coding.
7. "Do not do yet" list: ideas that are tempting but premature or likely to confound the next
   experiment.

## Evidence To Paste Before The Prompt

Minimum useful bundle:

1. `compressed_phase6_perf_plan.md`
   - compressed Phase 6.A scoreboard.
   - compressed Phase 6.B sustainability soaks.
   - 6.B.4 rank-6 root cause and fix.
   - 6.B.5 jobs 60590/60959 validation, with 60959 as the full completion run.
2. `minimum_useful_bundle_source_excerpts.md`
   - one consolidated markdown file with cropped excerpts from:
     `dinov3/fsdp/ac_compile_parallelize.py`,
     `dinov3/train/ssl_meta_arch.py`,
     `dinov3/data/collate.py`,
     `dinov3/loss/ibot_patch_loss.py`,
     `dinov3/loss/dino_clstoken_loss.py`,
     and `dinov3/train/train.py`.
   - covers compile/cudagraph gating, DDP wrapping, `forward_backward()`,
     `get_student_output()`, `compute_losses()`, dynamic iBOT mask construction,
     loss/center-update behavior, CUDA-event timing, MetricLogger boundary, and
     checkpoint/eval syncs.

Optional if asking for multi-node ideas:

1. Current or planned multi-node launch script.
2. NCCL environment and topology evidence.
3. Any nsys / torch profiler trace showing all-reduce timing and overlap under DDP.
4. Bucket size / gradient order / parameter size summary for the DDP model.

## Seed Hypotheses To Encourage, Not Bias Toward

These are not conclusions. They are prompts for where to look.

1. **Static-shape iBOT padding may unlock end-to-end graph capture.**
   HYPOTHESIS: padding masked patch rows to a fixed upper bound preserves the SSL objective while
   trading some extra head/loss compute for fewer graph breaks, fewer launches, and more stable
   cudagraph replay.

2. **Fixed-K masking may be the faster second-stage variant.**
   HYPOTHESIS: once padding proves the performance lever, fixed-K masking removes pad waste but
   changes the mask distribution enough that it needs a convergence check.

3. **The iBOT gather/head/loss path may be HBM-traffic bound.**
   HYPOTHESIS: repeated index/gather of masked patch tokens plus separate teacher/student head
   passes may be reorganizable to improve locality or reduce materialization without changing the
   mathematical loss.

4. **The DINO/iBOT center updates may be launch/collective exposed.**
   HYPOTHESIS: async all-reduce exists, but the sequencing may still expose small collectives or
   host launches on the critical path. Profiling should verify whether they overlap with useful
   compute in the DDP+cudagraph regime.

5. **Global/local multi-crop execution may leave batching efficiency on the table.**
   HYPOTHESIS: global and local crops have different sequence lengths, but there may be a better
   way to batch or schedule the local-crop-heavy work, especially outside the backbone graph.

6. **GPU-event MFU may hide a wall-clock bottleneck.**
   HYPOTHESIS: job 60590-style wall gaps may come from host-side H2D launch, collate/cast, Python
   bookkeeping, DataLoader queue behavior, checkpoint/eval phases, or uncaptured optimizer/EMA
   work. Use wall-clock throughput as a separate objective from GPU-event throughput.

7. **DDP single-node wins may not transfer cleanly to multi-node.**
   HYPOTHESIS: on multi-node, gradient all-reduce and loss collectives become more visible, so the
   next experiment should separate single-node graph/static-shape wins from multi-node NCCL overlap
   and bucketization work.

## First Experiments The Answer Should Consider

Ask GPT-5 Pro to choose, but these are likely good candidates:

1. Static-shape iBOT padding A/B:
   - same config as 53708/60590;
   - only change masked-token padding / loss masking;
   - compare GPU-event throughput, wall-clock iter time, graph breaks/recompiles, and loss parity
     on a short run;
   - if promising, rerun a 1000-iter screen.

2. Wall-clock decomposition profile:
   - run a short DDP+cudagraph bs=128 profile with NVTX ranges around DataLoader wait, H2D,
     student/teacher forward, heads/loss, backward, optimizer, EMA, checkpoint/eval disabled;
   - answer whether the wall↔GPU gap is real steady-state overhead or soak-phase accounting.

3. Head/loss profiler:
   - isolate DINO/iBOT head and loss time under current dynamic shapes;
   - record shape distribution for `n_masked`;
   - estimate pad-to-max overhead before coding.

4. DDP collective overlap profile:
   - on one node first, measure all-reduce timing and overlap under the champion config;
   - only after that decide whether bucket tuning or comm hooks are worth multi-node prep.

## Short Version

The current direction is: keep DDP + cudagraphs + bs=128 no-AC as the speed path, validate it
with a full-length soak, and search next for static-shape head/iBOT and wall-clock bottlenecks.
The best GPT-5 Pro prompt should ask for DINOv3-specific graph/static-shape/memory-traffic ideas,
not generic NCCL or FSDP2 tuning.
