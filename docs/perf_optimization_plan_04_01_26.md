# DINOv3 Performance Optimization Plan

**Date**: 2026-04-02  
**Status**: Active  
**Owner Branch**: `mfu-tracking-baseline`  
**Starting Point**: Step 1 is complete. This document begins at Step 2.

## Purpose

This is the controlling document for MFU-adjacent performance work in this repo.

It is meant to be good enough that a later Codex session can be told:

`Read docs/perf_optimization_plan_04_01_26.md and start on Step X.`

The document therefore does four things:

1. States the current understanding of the codebase and measured baseline.
2. Defines the execution order for profiling and optimization work.
3. Defines branch, documentation, and artifact-management rules.
4. Defines the deliverables and exit criteria for each step.

## Current Understanding

- MFU tracking is implemented and the corrected dense H100 denominator is already in use.
- Step timing is measured with CUDA events around the core training step.
- The existing baseline result is directionally credible: dense hardware MFU around `11-13%` on `8x H100` is plausible for this multi-crop ViT SSL workload.
- This workload should not be benchmarked mentally against large decoder-only LLM MFU numbers.
- Attention already uses SDPA.
- `torch.compile` is already enabled by default.
- CUDA graphs are already wired in the codebase but disabled by default.
- For the current canonical ViT-B satellite recipe, CUDA graphs are a viable screening candidate because the active training path is relatively fixed-shape; the gain is still unmeasured and should not be assumed.
- Existing memory logging is not yet sufficient for optimization decisions because the logs emphasize allocated memory and do not fully expose reserved-memory headroom.

## Recommendation Summary

Starting from Step 2, stay on `mfu-tracking-baseline`.

That is the right choice for now because:

- the remaining profiling work is tightly coupled to the MFU implementation
- unbundling profiling from MFU later is easier if profiling code is additive and flag-gated
- the branch is clean now, so we can keep history readable with disciplined commits

Do not branch off yet just because profiling is being added.

Branch off later only when:

- an optimization changes training behavior rather than measurement or observability
- a line of work is speculative enough that it may be abandoned
- two optimization lines would otherwise become hard to disentangle

## Operating Principles

Use this framework for all decisions:

1. Trace the exposed step time.
2. Classify the limiter.
3. Estimate the step-time share.
4. Change only the knobs that attack that exposed bottleneck.
5. Re-profile after each meaningful win.

Three questions must always be answered before broad optimization work:

1. Is the target on the exposed critical path?
2. Is it compute-bound, memory-bound, communication-bound, or host-bound?
3. If fixed perfectly, could it move end-to-end throughput enough to matter?

## Branch Policy

### Working branch

Use `mfu-tracking-baseline` for Steps 2 through 5.

This branch now covers:

- MFU implementation
- profiling infrastructure
- baseline profiling runs
- bottleneck analysis
- first-pass low-risk optimization screening over already-wired knobs

### When to split

Create a new branch only when starting one of these categories:

- distributed-strategy changes such as `DDP vs FSDP2`
- architectural work such as batched multi-crop or sequence packing
- changes likely to affect training quality or semantics
- multiple competing optimization directions that should be evaluated independently

Reasonable future branch names:

- `perf-ddp-vs-fsdp`
- `perf-batch-scaling`
- `perf-cudagraphs`
- `perf-multicrop-packing`

### Commit discipline on this branch

Keep commits narrow and intentional.

Preferred pattern:

1. profiling instrumentation
2. profiling job scripts
3. parsing or ledger tooling
4. documentation updates
5. small optimization knob changes

Avoid mixing instrumentation and optimization in one commit unless the instrumentation is inseparable from the change.

## Rules For Keeping MFU and Profiling Bundled Safely

The user concern is correct: it can become annoying to separate profiling work from MFU work later.

The way to avoid that is not to split branches early; it is to impose strict code-shape rules:

1. Profiling code must be additive, not invasive.
2. The default training path must remain unchanged when profiling is off.
3. Profiling-only behavior must be behind a flag or explicit config.
4. Baseline scripts must remain reproducible and not silently change semantics.
5. Instrumentation must write extra artifacts, not alter training logic.

Concrete implementation rules:

- `--profiling` must gate profiling behavior.
- NVTX ranges, profiler hooks, trace export, and graph-break logging must be opt-in.
- New profiling scripts should be separate from baseline scripts rather than rewriting them in place.
- If a script changes experiment semantics, create a new script rather than mutating the canonical baseline script.

## Documentation Policy

### `docs/`

Use `docs/` for current project-state documents and operational documents.

Recommended files:

- `docs/perf_optimization_plan_04_01_26.md`
  This file. It is the controller document.
- `docs/perf_experiment_log.md`
  Append-only run ledger.
- `docs/perf_bottleneck_ledger.md`
  Current performance picture and decision state.
- `docs/mfu-results-2026-03-30.md`
  Historical MFU validation record. Update only when clarifying or superseding baseline interpretation.

### `learnings/`

Use `learnings/` only for durable conclusions that survived repeated measurement or trace-backed analysis.

Do not put tentative one-run observations into `learnings/`.

Good candidates for `learnings/`:

- stable profiler workflow decisions
- repeated bottleneck patterns
- reliable hardware or cluster-specific caveats
- conclusions that are likely still true after the current branch ends

### Update rules

After each meaningful run or code change:

1. update the experiment log
2. update the bottleneck ledger if the bottleneck picture changed
3. update `learnings/` only if the conclusion now looks durable
4. update this controller doc only if the plan itself changed

## Artifact Policy

Do not store large runtime artifacts in the git repo.

Keep these outside the repo:

- raw profiler traces
- Nsight reports
- large training output directories
- binary exports

Suggested location pattern on Weka:

- `/mnt/weka/adovlatyan/perf_runs/<date>/<run_name>/`
- `/mnt/weka/adovlatyan/profiler_traces/<date>/<run_name>/`

Each run directory should contain:

- copied config or command
- `training_metrics.json`
- primary log file
- parsed summary text
- pointers to any trace artifacts

## Required Per-Run Record

Every submitted run must leave a compact note in the experiment log.

Use this template:

```text
Run Name:
Date:
Branch:
Commit:
Step:
Job ID:
Node:
Goal:
Command or Script:
Config Delta:
Output Directory:
Trace Artifacts:
Summary Window:
Images/sec:
Step Time ms:
MFU %:
Allocated Mem MB:
Reserved Mem MB:
Observed Bottleneck:
Decision:
Next Action:
```

## Step Structure

Each step below includes:

- objective
- work items
- required artifacts
- exit criteria

Do not skip exit criteria.

## Step 2: Build Profiling Foundation

### Objective

Make profiling cheap, reproducible, and safe to run repeatedly from this branch.

### Work items

1. Make the existing `--profiling` flag actually enable a profiling mode.
2. Add PyTorch profiler support with trace export and a warmup/active schedule.
3. Add optional NVTX ranges around:
   - data wait
   - forward/backward
   - optimizer step
   - EMA update
4. Add optional graph-break diagnostics for `torch.compile`.
5. Extend training metrics and logs to include:
   - reserved memory
   - max reserved memory
   - node name
   - world size
   - compile enabled
   - cudagraphs enabled
   - checkpointing enabled
6. Add one dedicated profiling Slurm script instead of overloading the baseline script.

### Required artifacts

- code changes for profiling mode
- one profiling run script
- one short note in the experiment log describing the new profiling path

### Exit criteria

- profiling can be turned on and off without changing default training behavior
- a profiling run emits trace artifacts plus normal logs
- the run metadata is rich enough to analyze a result later without guessing the config

## Step 3: Run One Baseline Profiling Pass

### Objective

Get one trustworthy baseline trace for the existing 8-GPU real-data configuration.

### Work items

1. Run one short PyTorch-profiler baseline job on the canonical config.
2. Capture a steady-state window that skips compile warmup.
3. Export a compact textual summary alongside the trace artifacts.
4. Record the result in the experiment log.

### Required artifacts

- trace output directory
- parsed summary text
- experiment-log entry

### Exit criteria

- one profiling run exists that can be handed to a later agent without extra context
- the trace window and exact config are documented
- the result is sufficient to classify the bottleneck at a first pass

## Step 4: Build the Bottleneck Ledger

### Objective

Translate the baseline trace into a working causal model for step time.

### Work items

1. Create or update `docs/perf_bottleneck_ledger.md`.
2. For each meaningful exposed cost, record:
   - phase or kernel family
   - approximate exposed share of step time
   - likely bound type
   - candidate knobs
   - prerequisites
   - reasons to defer it
3. Explicitly classify whether exposed time is dominated by:
   - GPU compute
   - CPU launch or host gaps
   - NCCL
   - input stalls
   - skew or thermal drift

### Required artifacts

- bottleneck ledger

### Exit criteria

- the current dominant bottleneck is written down explicitly
- there is a ranked list of plausible next knobs
- at least one category of tempting but low-value work has been ruled out

## Step 5: Small Screening Design Over Existing Knobs

### Objective

Search the already-wired configuration space without brute-force chaos.

### Allowed first factors

- `train.batch_size_per_gpu`
- `train.checkpointing`
- `train.checkpointing_full`
- `train.cudagraphs`

### Work items

1. Choose `2-4` factors based on the bottleneck ledger, not by habit.
2. Use a small screening design rather than a broad sweep.
3. Keep the baseline script untouched and create separate profiling or experiment scripts as needed.
4. Record every run in the experiment log.
5. Update the bottleneck ledger after meaningful result shifts.

### Interpretation rules

- checkpointing is an enabler unless it directly lowers exposed step time at the same batch
- CUDA graphs are only worth time if launch or CPU gaps are exposed and shapes are stable
- On the current fixed `224/96` ViT-B recipe, `train.cudagraphs=true` is a reasonable experiment and is likely stackable with later non-semantic optimizations because it targets execution overhead rather than changing training semantics
- if NCCL is not exposed, do not prioritize overlap work
- if batch scaling improves MFU and throughput without exposing a worse bottleneck, keep pushing batch first

### Required artifacts

- experiment-log entries for all screening runs
- updated bottleneck ledger
- short summary of main effects and strong interactions

### Exit criteria

- one or two knobs emerge as clearly more promising than the rest
- at least one low-value direction is explicitly deprioritized

## Step 6: Decide Whether To Stay on This Branch or Split

### Objective

Choose the right branch boundary after the screening results are in.

### Stay on this branch if

- the work is still profiling-heavy
- the changes are still mostly additive instrumentation
- the next moves are still low-risk config experiments

### Split to a new branch if

- the next move changes distributed strategy
- the next move changes model structure or crop processing
- the next move is risky enough that it may need to be abandoned cleanly

### Exit criteria

- the branch decision is written down in the experiment log or bottleneck ledger

## Step 7: Promote Durable Knowledge

### Objective

Keep the repo understandable when the user is away.

### Work items

1. Move stable conclusions into the appropriate `learnings/*.md` file.
2. Keep one paragraph of current status at the top of the bottleneck ledger.
3. Keep the experiment log append-only.
4. If an older doc has stale numbers or conclusions, correct it rather than letting contradictions accumulate.

### Exit criteria

- the repo documents tell a consistent story
- a returning human can understand what happened without reading raw Slurm logs

## Specific Guidance On The Branch Question

Yes, Steps 2 through 5 should stay on this branch.

That is the recommended default unless the work suddenly becomes a separate optimization project rather than measurement and profiling work.

The practical reason is simple:

- profiling is part of the MFU story in this repo
- the code overlap is high
- default-off instrumentation is easy to keep logically separate
- the branch is clean and can remain understandable if commit discipline is enforced

The wrong reason to split now would be abstract neatness.

The right reason to split later would be real semantic divergence.

## What A Later Codex Run Should Do

When handing this off later, use a prompt like:

`Read docs/perf_optimization_plan_04_01_26.md and start on Step 2. Follow the branch, documentation, and artifact rules exactly.`

Or:

`Read docs/perf_optimization_plan_04_01_26.md and start on Step 4 using the current profiling artifacts. Update the experiment log and bottleneck ledger as you go.`

## Immediate Next Action

Start on Step 2 on `mfu-tracking-baseline`.

The first concrete deliverable is not a new optimization result. It is a profiling foundation that makes all later optimization work easier to run, easier to compare, and easier to document.
