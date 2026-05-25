# GPT-5 Pro Context Pack: Why bs=128 Is Not Significantly Faster Than bs=96

Purpose: this is the file list and prompt scaffold I would give to a deep reasoning model to analyze the DINOv3 Phase 5 puzzle:

> On one DGX H100 node with 8x H100 GPUs, why does per-GPU batch size 128 not materially outperform per-GPU batch size 96 under the current FSDP2/selective-activation-checkpointing setup?

The goal is not to get a generic "maybe communication bottleneck" answer. The goal is to force the model to integrate the actual code path, hardware topology assumptions, run ledger, profiler summaries, memory behavior, and current experimental controls.

## Core Files To Attach

Attach these first. They should be enough for a serious first pass.

1. `docs/gpt5-pro-bs128-context-pack.md`
2. `docs/yerevann-dinov3-perf-optimizations-project-dossier.md`
3. `docs/phase5_perf_plan.md`
4. `docs/perf_bottleneck_ledger.md`
5. `docs/perf_experiment_log.md`
6. `docs/upstream-pr-findings-2026-05-20.md`
7. `01_architecture_and_training.md`
8. `02_performance_and_systems.md`
9. `03_open_questions_and_experiments.md`
10. `04_architecture_and_mfu_teaching_guide.md`

## Experiment Scripts To Attach

These define the actual knobs, Slurm shape, env vars, node pinning, and isolation assumptions.

1. `run.sh`
2. `scripts/fsdp2/fsdp2_bs128_reshT_sel_resoak.sh`
3. `scripts/fsdp2/fsdp2_bs128_reshF_sel.sh`
4. `scripts/fsdp2/fsdp2_bs128_ac_selective.sh`
5. `scripts/fsdp2/fsdp2_bs128_ac_full.sh`
6. `scripts/profiling/fsdp2_bs128_memprofile.sh`
7. `scripts/fsdp2/fsdp2_bs128_singlegpu_sel.sh`
8. `scripts/fsdp2/fsdp2_bs96_ac_selective.sh`
9. `scripts/fsdp2/fsdp2_bs96_ac_full.sh`
10. `scripts/fsdp2/fsdp2_bs96_reshT_noAC.sh`
11. `scripts/fsdp2/fsdp2_bs96_reshF_sel.sh`
12. `scripts/fsdp2/fsdp2_bs96_reshF_full.sh`
13. `scripts/fsdp2/fsdp2_bs96_reshF_noAC.sh`
14. `scripts/fsdp2/fsdp2_bs96_singlegpu_sel.sh`
15. `scripts/fsdp2/fsdp2_bs96_ncclsweep_stageA.sh`
16. `scripts/fsdp2/fsdp2_ncclsweep_stageB.sh`
17. `scripts/fsdp2/fsdp2_ncclsweep_stageC.sh`
18. `scripts/profiling/nsys_profile.sh`
19. `scripts/profiling/nsys_dinov3_summary.py`

## Code Files To Attach

These are the minimum source files needed to reason from the measured result back to the implementation.

1. `dinov3/train/train.py`
2. `dinov3/train/ssl_meta_arch.py`
3. `dinov3/fsdp/ac_compile_parallelize.py`
4. `dinov3/models/vision_transformer.py`
5. `dinov3/layers/block.py`
6. `dinov3/configs/config.py`
7. `dinov3/configs/ssl_default_config.yaml`
8. `dinov3/utils/mfu.py`
9. `dinov3/utils/profiling.py`
10. `dinov3/logging/helpers.py`
11. `dinov3/data/loaders.py`
12. `dinov3/data/collate.py`
13. `dinov3/data/datasets/mixed_satlas_dataset.py`
14. `dinov3/data/datasets/channel_utils.py`
15. `dinov3/loss/dino_clstoken_loss.py`
16. `dinov3/loss/ibot_patch_loss.py`
17. `dinov3/loss/koleo_loss.py`
18. `dinov3/train/cosine_lr_scheduler.py`

Why these matter:

- `train.py`: schedules, optimizer step, EMA update, CUDA-event timing, MFU/images/sec logging, eval/checkpoint sync points.
- `ssl_meta_arch.py`: H2D transfers, teacher/student forward split, multi-crop student pass, loss computation, backward, EMA.
- `ac_compile_parallelize.py`: FSDP2 wrapping, selective AC, `torch.compile`, `reshard_after_forward`.
- `vision_transformer.py` and `block.py`: multi-resolution `forward_features_list()` and transformer block cost/shape behavior.
- `mfu.py` and `profiling.py`: what the reported throughput and memory metrics actually mean.
- data/loss files: rule out or quantify dataloader/H2D/masking/loss-shape explanations rather than hand-waving.

## Learning Notes To Attach If Budget Allows

Attach these if the model has a very large context window or if you want it to understand prior closed paths and terminology.

1. `learnings/README.md`
2. `learnings/distributed_training.md`
3. `learnings/gpu_performance.md`
4. `learnings/data_pipeline.md`
5. `learnings/compile_modes.md`
6. `learnings/cuda_graphs.md`
7. `learnings/profiling_workflow.md`
8. `learnings/terminology.md`

## External Run Evidence To Include

If possible, include these external files as attachments or pasted excerpts. They are not all repo files.

1. `/mnt/weka/adovlatyan/nsys_profiles/2026-05-08/39140/dinov3-fsdp2-bs128-39140.summary.md`
2. `/mnt/weka/adovlatyan/nsys_profiles/2026-05-08/39141/dinov3-fsdp2-bs96-39141.summary.md`
3. `/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshF-sel-47554.out`
4. `/mnt/weka/adovlatyan/logs/fsdp2-bs128-reshT-sel-resoak-51069.out`, once job 51069 finishes
5. Relevant stdout logs for jobs 45280, 45363, 45364, 45366, 45367, 45368, 45370, 45476, and 45477, if still available

Minimum excerpts to include from logs:

- final and steady-window `images_per_sec`, `mfu`, `step_time_ms`, `data`, and max memory lines
- all `[MEMPROFILE]` and `[MEMFRAG]` lines
- any OOM, allocator retry, eval, or checkpoint phase-boundary messages
- compile warmup region and the first stable window after warmup

## Facts The Prompt Should Pin Down

Use these as givens unless the model finds a contradiction in the attached evidence.

- Hardware target is one DGX H100-class node with 8 H100 GPUs.
- Current production-safe candidate remains `bs=96`, FSDP2, selective AC, `train.fsdp_reshard_after_forward=true`.
- `bs=128` is investigational because a researcher-reported long-training FSDP2 OOM has not been fully explained.
- `bs=96 reshT selective AC` job 45280 on gpu07 reached about `1351 img/s`, `7.73% MFU`, about `14.3 GB` allocated.
- `bs=128 reshT selective AC` job 45366 on gpu07 reached about `1276 img/s`, `7.30% MFU`, about `18.9 GB` allocated.
- `bs=128 reshF selective AC` job 47554 on gpu05 reached about `1341 img/s` mean, `1312 img/s` median, `7.67% MFU`, with eval/checkpoint disabled. This comparison crosses node and date relative to the gpu07 bs=96 anchor.
- Job 51069 is the paired same-node `bs=128 reshT selective AC` re-soak on gpu05 intended to isolate reshT vs reshF against job 47554.
- Single-GPU disambiguator on gpu07 showed bs=128 faster locally: job 45476 bs=96 at about `448 img/s/GPU`, `20.49% MFU`; job 45477 bs=128 at about `478 img/s/GPU`, `21.88% MFU`.
- The sign flips at 8 GPUs: bs=128 is not materially better, and often worse, once distributed communication is present.
- Existing nsys summaries for jobs 39140 and 39141 report NCCL as the dominant trace signal, around 70-80% of kernel time with low compute/NCCL overlap.
- Dataloader is probably not the main 8-GPU bottleneck in current bs=128 logs because `data` is near zero in the relevant steady regions, but the single-GPU bs=128 run had a host-side data stall and should not be overinterpreted without separating CUDA-event throughput from wall time.
- `compile_mode: null` is the viable torch.compile path; max-autotune variants are not currently viable because dynamic iBOT masked-token counts break the assumptions.
- Do not propose `expandable_segments=True` for FSDP2 as a default. It was a DDP/large-batch screening knob and is considered off-limits for current FSDP2 production experiments unless deliberately isolated.

## The Actual Prompt I Would Give GPT-5 Pro

You are analyzing a satellite-specialized DINOv3 training fork on one 8x H100 DGX-class node. Your task is to explain, from first principles and the attached repo evidence, why per-GPU batch size 128 is not significantly faster than per-GPU batch size 96 under the current Phase 5 FSDP2 setup.

Do not give generic performance advice. Build a ranked hypothesis tree that distinguishes observed facts from inference. For each hypothesis, explain:

1. the exact mechanism,
2. which attached files or log evidence support it,
3. which evidence argues against it,
4. what one clean experiment would falsify it,
5. whether it predicts the observed 1-GPU vs 8-GPU sign flip,
6. whether it predicts the reshT vs reshF behavior,
7. whether it predicts the memory/OOM history.

Pay special attention to:

- why single-GPU bs=128 improves throughput but 8-GPU bs=128 does not;
- whether additional local compute from bs=128 should hide NCCL better, and why it apparently does not;
- whether FSDP2 all-gather/reduce-scatter timing, `reshard_after_forward`, activation checkpointing, or bucketization could make bs=128 communication-exposed rather than compute-saturated;
- whether multi-crop shapes, dynamic iBOT masks, loss communication, or `forward_features_list()` create synchronization or launch patterns that do not scale with batch size;
- whether measured `images_per_sec` and MFU are CUDA-event compute-window metrics or wall-clock end-to-end metrics, and how that changes interpretation;
- whether node heterogeneity can explain the existing bs=96 vs bs=128 comparisons;
- what job 51069 would prove under each possible outcome.

Output format:

1. Executive conclusion, maximum 10 bullets.
2. Ranked hypotheses table.
3. Evidence ledger with exact file names and job IDs.
4. Interpretation of job 51069 outcomes.
5. Minimal next-experiment plan, limited to three experiments.
6. Instrumentation changes that would most reduce uncertainty.

Constraints:

- Keep bs=128 promotion separate from bs=128 diagnosis. Even if bs=128 can be made faster, long-run memory safety is still unresolved.
- Do not recommend changing multiple knobs at once unless explicitly labeling it as a later stack experiment.
- Treat cross-node and cross-date comparisons as weak evidence.
- Treat same-node paired runs and nsys traces as stronger evidence.
- Use "HYPOTHESIS:" for causal claims not directly proven by logs or code.

## What I Want The Model To Notice

The strongest answer should probably focus on a communication/exposure story, but it should not stop there. The important subtlety is that "more batch means more compute to hide communication" is only true if the added work overlaps the communication on the right streams and if the fixed/distributed overheads do not move onto the critical path. In this training loop, the extra bs=128 local compute may be real, as the single-GPU run suggests, while the 8-GPU shape remains dominated by NCCL/FSDP/loss synchronization and low overlap.

The model should also respect the measurement boundaries:

- CUDA-event `images_per_sec` is not the same as wall-clock throughput if host `data` stalls are present.
- Eval and checkpoint are disabled in some probes and active in production.
- Memory "survived 1000 iters" is not equivalent to long-training safety.
- `bs=128 reshF sel` at 1341 img/s is interesting, but not yet a clean win over the bs=96 default because the best bs=96 comparator is on a different node.

## Files I Would Not Attach Initially

Skip generated HTML unless the model needs a visual status page:

- `docs/claude-html-files/*.html`
- `docs/codex-html-files/*.html`

Skip unrelated eval configs and downstream task code unless a hypothesis unexpectedly points there.

