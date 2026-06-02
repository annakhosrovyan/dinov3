# Upstream DINOv3 PR Findings: FSDP2 Resharding and Python GC

Date: 2026-05-20

This note records two upstream DINOv3 pull requests that materially affect how we
should read our Phase 5 performance notes:

- Upstream PR #324: <https://github.com/facebookresearch/dinov3/pull/324>
- Upstream PR #340: <https://github.com/facebookresearch/dinov3/pull/340>

This document is an addendum to `docs/phase5_perf_plan.md`, `CLAUDE.md`, and the
HTML summaries under `docs/*-html-files/`. When those older docs conflict with
this note, this note should be treated as the newer interpretation.

## Executive Summary

### `train.fsdp_reshard_after_forward`

Upstream PR #324 shows that public upstream DINOv3 did not originally expose a
working FSDP2 `reshard_after_forward` training knob. Its FSDP2 wrapping path
hardcoded `reshard_after_forward=True` in the relevant `fully_shard(...)` calls,
so older upstream config fields such as `compute_precision.sharding_strategy`
did not change FSDP2 wrapping behavior.

Our fork appears to already contain the important behavioral part of PR #324:
our FSDP2 wrapping path reads `cfg.train.fsdp_reshard_after_forward` and passes
that value into block/head `fully_shard(...)` calls for trained models. Therefore
our Phase 5 `reshF` runs are probably not invalidated by PR #324.

However, our docs and config vocabulary still contain misleading legacy language
around `compute_precision.sharding_strategy=SHARD_GRAD_OP`. That key should not
be used to reason about FSDP2 behavior in this fork. The authoritative knob is:

```text
train.fsdp_reshard_after_forward
```

### `gc.disable()`

Upstream PR #340 changes our interpretation of Python GC. Our older docs treated
`gc.disable()` plus periodic manual `gc.collect()` as a partial mitigation for
periodic GC straggler stalls. PR #340 presents evidence that disabling automatic
cycle GC interacts badly with `torchvision.transforms.v2`: short-lived reference
cycles can accumulate in DataLoader workers, holding tensor storage references
and shared-memory file descriptors. The PR reports worker anonymous RSS growth
of roughly `~1.25 GB/hour/worker` in practice and a controlled benchmark showing
`~3.4 MB/iter` worker RSS growth with `gc.disable()` versus `~0.01 MB/iter` with
cycle GC enabled.

Therefore, we should stop treating `gc.disable()` as a safe default mitigation.
The better interpretation is:

- Keep periodic rank-synchronized `gc.collect()` if it helps predictable pauses.
- Do not disable automatic cycle GC by default.
- If straggler spikes are suspected, instrument GC events and worker RSS first.
- Long-run memory growth should now include `gc.disable()` as a suspect.

## Details: Upstream PR #324

Source: <https://github.com/facebookresearch/dinov3/pull/324>

PR #324 is titled "Add FSDP2 reshard_after_forward training config". Its summary
says it adds:

```text
train.fsdp_reshard_after_forward
```

and wires it through the SSL training FSDP wrapping path.

The PR states that previously:

- `compute_precision.sharding_strategy` existed in configs.
- That config did not control FSDP2 wrapping behavior.
- Relevant `fully_shard(...)` calls in `dinov3/fsdp/ac_compile_parallelize.py`
  were hardcoded with `reshard_after_forward=True`.
- `ssl_meta_arch.py` only asserted `SHARD_GRAD_OP`; it did not map that setting
  to FSDP2 behavior.

The upstream fix:

- Adds `train.fsdp_reshard_after_forward` to train configs.
- Removes `compute_precision.sharding_strategy` from configs.
- Passes `cfg.train.fsdp_reshard_after_forward` into SSL FSDP wrapping.
- Keeps inference-only models always resharded after forward.
- Adds migration handling so legacy `compute_precision.sharding_strategy` is
  warned about and removed before strict config merge.

## What This Means For Our Phase 5 Runs

Our fork has a `train.fsdp_reshard_after_forward` key and our Phase 5 scripts use
it directly. The important local behavior to verify is:

```text
cfg.train.fsdp_reshard_after_forward
  -> ac_compile_parallelize.py
  -> fully_shard(..., reshard_after_forward=<that value>)
```

Based on local inspection during this review, our fork already follows that
pattern for trained models. That means jobs such as:

```text
45368: bs=96 reshF noAC
45369: bs=96 reshF selective AC
45370: bs=96 reshF full AC
```

should be interpreted as real `train.fsdp_reshard_after_forward=false` attempts,
not as accidental `reshard_after_forward=true` reruns.

Important nuance: `reshard_after_forward=false` is "DDP-like", but it is not
guaranteed to match DDP performance. It still runs through FSDP2 wrapping,
FSDP2 prefetch scheduling, parameter unshard/reshard bookkeeping, and different
communication ordering. Our Phase 5 observation that `bs=96 reshF noAC` was
worse than `bs=96 reshT noAC` is surprising, but PR #324 alone does not prove it
was caused by a non-working knob.

## Documentation Corrections From PR #324

Older docs should be read with these corrections:

- "SHARD_GRAD_OP" is not the operative FSDP2 behavior knob in our Phase 5 work.
- `compute_precision.sharding_strategy` is legacy vocabulary and should not be
  used to explain Phase 5 `reshT` or `reshF` behavior.
- `train.fsdp_reshard_after_forward=true` means reshard after forward, i.e.
  ZeRO-3-like behavior.
- `train.fsdp_reshard_after_forward=false` means keep parameters unsharded
  through backward, i.e. no-release/DDP-like behavior.
- Inference-only models may still force immediate resharding for memory
  predictability; this does not invalidate trained-model `reshF` tests.

## Details: Upstream PR #340

Source: <https://github.com/facebookresearch/dinov3/pull/340>

PR #340 is titled "Re-enable cycle GC in training loop to prevent worker anon RSS
leak". It removes `gc.disable()` from the training loop while keeping the
periodic manual `gc.collect()`.

The PR's mechanism:

```text
torchvision.transforms.v2
  -> pytree.tree_flatten
  -> short-lived reference cycles
  -> cycle GC disabled, so cycles are not collected
  -> tensor storage refs and shared-memory FDs accumulate
  -> DataLoader worker anonymous RSS grows over long runs
```

The PR reports:

- With `gc.disable()`: worker RssAnon growth around `~3.4 MB/iter`.
- With cycle GC enabled: worker RssAnon growth around `~0.01 MB/iter`.
- Practical observed growth around `~1.25 GB/hour/worker` with `num_workers=16`.
- No measurable iteration-time impact from re-enabling automatic cycle GC in the
  benchmark cited by the PR.

## What This Means For Our GC Notes

Several of our docs say, in effect:

```text
gc.disable() + manual gc.collect() every ~100/150 iters partially mitigates GC
straggler effects.
```

That statement is now too strong and potentially backwards. A safer statement is:

```text
Periodic manual gc.collect() can still be useful for predictable rank-synchronized
pauses, but disabling automatic cycle GC can cause long-run DataLoader worker RSS
growth when torchvision.transforms.v2 creates reference cycles.
```

In other words:

- `gc.disable()` may reduce automatic-GC timing noise in short runs.
- It can also create a long-run memory leak mechanism.
- For our long-running training jobs, the memory risk matters more.
- Future runs should prefer automatic cycle GC enabled, plus periodic manual
  collection if needed.

## Practical Follow-Ups

### Code / Config Cleanup

1. Remove or deprecate `compute_precision.sharding_strategy` in our configs.
2. Remove stale asserts that require `cfg.compute_precision.sharding_strategy ==
   "SHARD_GRAD_OP"` if they no longer map to real FSDP2 behavior.
3. Add a short startup log line that prints:

```text
train.distributed_strategy
train.fsdp_reshard_after_forward
train.checkpointing
train.checkpointing_full
```

This prevents future ambiguity in performance logs.

### FSDP2 Verification

Add a cheap read-only or smoke-test check that inspects trained-model FSDP states
and confirms whether `reshard_after_forward` is set as requested. This is better
than inferring behavior from throughput alone.

Suggested check:

```text
Run tiny FSDP2 init with train.fsdp_reshard_after_forward=false.
Inspect backbone block FSDPState objects.
Confirm trained blocks use false while inference-only modules may force true.
```

### GC / Memory Verification

1. Remove `gc.disable()` from the training loop or gate it behind an explicit
   experimental flag.
2. Keep periodic manual `gc.collect()`.
3. For long soaks, log DataLoader worker RSS periodically if practical.
4. Reinterpret old long-run OOM suspicions with this mechanism in mind,
   especially if RSS grew outside CUDA allocator metrics.

## Impact On Current Performance Interpretation

These upstream PRs do not require throwing away Phase 5.

The revised interpretation is:

- The `reshF` runs probably did exercise the intended no-release FSDP2 mode in
  our fork.
- The old claim "FSDP2 no-release is basically DDP" should remain conceptual,
  not a performance guarantee.
- The DDP-vs-FSDP2 gap may still be real for this ViT-B-on-H100 regime, but a
  matched DDP bs=96 run and a small FSDP state audit would make that conclusion
  much stronger.
- Any docs recommending `gc.disable()` as a safe mitigation are stale. Treat
  that as a risk until changed in code.

## Sources

- Upstream DINOv3 PR #324, "Add FSDP2 reshard_after_forward training config":
  <https://github.com/facebookresearch/dinov3/pull/324>
- Upstream DINOv3 PR #340, "Re-enable cycle GC in training loop to prevent
  worker anon RSS leak": <https://github.com/facebookresearch/dinov3/pull/340>
- PyTorch vision issue referenced by PR #340:
  <https://github.com/pytorch/vision/issues/9242>
- Secondary PyTorch vision issue referenced by PR #340:
  <https://github.com/pytorch/vision/issues/6437>
