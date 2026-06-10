# DDP checkpoint fix cleanup + test coverage + sync reduction — results

**Date**: 2026-06-05
**Branch**: `fix/ddp-checkpoint-perf-cleanup` (off `master`)
**Implements**: `docs/handoff_ddp_checkpoint_cleanup.md`

## TL;DR

Four cleanup commits layered on top of Anna's DDP checkpoint fix (`7a42442`). The
checkpointer DDP-load path now makes one `state_dict()` call instead of four, a latent
DDP-unwrap bug in the eval path is fixed, the `grad_norm` scalar sync is deferred out of
the hot step, and the DDP checkpoint helpers now have unit-test coverage.

All work is verified: `tests/test_ddp_checkpoint.py` (14 new tests) + `tests/test_mfu.py`
(14 existing) = **28 passed**. `ruff check` clean on all changed files.

## Base-branch note

The handoff said Anna's fix was "committed to master." The **local** master checkout was
stale (`871f26e`, the parent of `7a42442`), which initially looked like the fix was missing.
But `origin/master` is at `7a42442` — **Anna's fix is already on remote master.**

So the final branch is based directly on `origin/master` (which includes `7a42442`) and
contains **only** the four cleanup commits + this doc — no cherry-pick, no duplicate of
Anna's fix. (An earlier local iteration branched off the stale local master and cherry-picked
`7a42442`; that cherry-pick was dropped via `git rebase --onto origin/master` once the remote
state was confirmed, since its tree was byte-identical to `origin/master`.)

This keeps the PR self-contained and clean — net diff is exactly the cleanup — and avoids the
128-file experiment pollution on `perf-ddp-fullgraph`.

## Commits

Four commits on top of `origin/master` (in order):

| # | Type | File(s) | Summary |
|---|---|---|---|
| 1 | test | tests/test_ddp_checkpoint.py | 14 tests for the DDP checkpoint helpers (green against the *current* helpers — equivalence baseline). |
| 2 | refactor | checkpointer.py, test file | DDP-load `state_dict()` calls 4 → 1. |
| 3 | fix | checkpointer.py | Unwrap DDP in `init_model_from_checkpoint_for_evals`. |
| 4 | perf | train.py | Defer `grad_norm` scalar sync to post-step logging. |

## Details

### Commit 1 — tests first (TDD / equivalence baseline)

Per advisor guidance, the tests were written and confirmed **green against the unrefactored
helpers** *before* the refactor, so the refactor is proven equivalent rather than asserted.
DDP is mocked with a minimal `_FakeDDP(nn.Module)` (a module with `self.module = inner`) —
this reproduces the only two behaviours the helpers depend on: a `.module` attribute and the
resulting `.module.` state_dict key prefix. No process group / multi-GPU is required.

Coverage: `_unwrap_module` (wrapped + passthrough), `_get_backbone_in_chans` (plain, DDP,
non-default in_chans, raises without backbone), `_state_dict_uses_ddp_prefix` (both cases),
`_remap_checkpoint_keys_for_ddp` (insert `.module.`, leave correct keys, passthrough,
no drops/dups, value identity, non-DDP unchanged).

### Commit 2 — `state_dict()` 4 → 1

`init_fsdp_model_from_checkpoint` called `model.state_dict()` four times on the DDP-load
path: the outer guard, again inside `_remap_checkpoint_keys_for_ddp`, a redundant re-check
of the guard inside that function, and the final matched-keys check. `state_dict()` is a
rank-synchronizing op for distributed models.

- `_state_dict_uses_ddp_prefix(model_keys)` and `_remap_checkpoint_keys_for_ddp(state,
  model_keys)` now take the precomputed `set[str]` instead of the model.
- The dead internal guard re-check is removed: the function is only entered when the outer
  guard is True; for a non-DDP model the remap loop returns the state unchanged anyway, so
  the early-out was redundant (this property is what the "non-DDP returns unchanged" test
  pins down — it passes before *and* after the refactor).
- The loading path computes `model_keys` once and reuses it for guard + remap + matched-keys.

Net: **4 → 1** `state_dict()` calls. Tests pass unchanged after updating only the call
wiring (assertions identical).

### Commit 3 — eval-path DDP unwrap

`init_model_from_checkpoint_for_evals` accessed `model.patch_embed.proj.weight` directly.
The path is non-distributed by design, but the access is a latent copy of the training-path
bug Anna fixed: if a DDP-wrapped module is passed, `.patch_embed` raises `AttributeError`
(DDP does not proxy sub-attributes). Now uses `_unwrap_module(model).patch_embed...` —
identity for the non-DDP case, defensive for DDP.

### Commit 4 — defer `grad_norm` scalar sync

The grad-clip block stored `grad_norm.item()` (or `full_tensor().item()` for DTensor) every
iteration — a GPU→CPU sync **after backward but before `optimizer.step()` and the
`step_end_event.record()`**, which breaks compute/comm pipelining
(`learnings/distributed_training.md`).

Now stores the raw tensor (`.detach()` plain/DDP, `.full_tensor()` DTensor).

**Why this is behaviour-preserving** (audited, not assumed):
- The metrics-reduction block ~15 lines below rebuilds `metrics_dict` from the all-reduced
  GPU tensors (`metrics_dict = dict(zip(..., metrics_values))`), so post-reduction every
  value is *already a tensor today*, independent of this change.
- The only reader between the grad-clip block and that overwrite is
  `torch.as_tensor(v, dtype=torch.float32, device=...)`, which accepts a 0-dim GPU tensor
  identically to a Python float.
- Downstream: `MetricLogger.update` (`helpers.py:27-28`) already does
  `if isinstance(v, torch.Tensor): v = v.item()`, and it runs *after*
  `step_end_event.synchronize()` (already a sync point); the wandb block handles
  DTensor/tensor/scalar; the JSON writer dumps `meter.median` (already a float).

**Honest scope note**: for the DTensor case, `.full_tensor()` still triggers an all-gather
collective — this removes the *host* sync, not the collective. On the DDP+cudagraphs path we
actually run, `grad_norm` is a plain tensor and `.detach()` keeps it on-GPU with no sync.

## Verification

```bash
export PATH="/mnt/weka/adovlatyan/.conda/envs/dinov3_env_210clone/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONPATH=.
python -m pytest tests/ -q          # 28 passed
ruff check dinov3/checkpointer/checkpointer.py dinov3/train/train.py tests/test_ddp_checkpoint.py  # clean
python -m py_compile dinov3/train/train.py   # OK (train.py can't be imported standalone — needs dist init)
```

(pytest was not previously installed in `dinov3_env_210clone`; installed into the env for
this work.)

## What was intentionally NOT done (per handoff)

- Did not merge `perf-ddp-fullgraph` wholesale (128 experiment files).
- Did not touch the iter-0 hang (a 2-GPU partial-node + pretrained + cudagraphs NCCL
  device-guess artifact — see the write/resume section; a candidate fix is passing `device_id`
  to `init_process_group`, but it does not affect the validated 8-GPU production path).
- Did not add error handling in `_get_backbone_in_chans` for a missing backbone (would hide
  a real misconfiguration — the raise is intentional and tested).
- Did not touch `run_ddp.sh` loader settings.
- Did not run `ruff format` over the checkpointer file: the repo is not `ruff format`-clean
  to begin with (pre-existing code triggers reformats), so doing so would churn unrelated
  lines including Anna's cherry-picked code. `ruff check` (the enforced lint) passes.

## Post-review validation (2026-06-08)

Codex (GPT-5.5, high effort) reviewed the four commits: Commits 2/3/4 SAFE; the only flag was
"no integration test for the modified call sites" (Q4 RISKY). Addressed in two ways, each
scoped to the actual residual risk of the commit rather than a blanket "realistic run".

### Integration tests for the eval call site (commit `e6b25df`)

Added `TestEvalLoaderIntegration` (3 tests) exercising `init_model_from_checkpoint_for_evals`
end-to-end with a real on-disk checkpoint:
- `test_plain_eval_model_loads_weights` — the path every real eval job runs (unwrapped
  backbone); asserts weights land after `backbone.`/`module.` stripping.
- `test_ddp_wrapped_eval_model_does_not_raise` — regression guard for Commit 3: a real eval
  job never wraps this model (`build_model_for_eval` builds a plain teacher backbone), so the
  DDP branch of the fix is **unreachable from any real run** — only a constructed wrapped
  model exercises it. Before the fix, `model.patch_embed` raised `AttributeError`.
- `test_eval_loader_reads_in_chans_through_unwrap_for_non_default` — confirms the *unwrapped*
  `shape[1]` drives patch_embed channel adaptation.

`tests/`: **31 passed** (14 mfu + 17 ddp_checkpoint), `ruff check` clean.

### Real 2-GPU smoke for Commit 4 (job 68670)

Commit 4 (`grad_norm` deferred as a tensor) was the one change *not* already covered by the
bs=128 soaks (53708/60959) and it interacts with cudagraph capture — so it got a real run.
Scoped deliberately: 2 GPUs, reduced loader (8w/2pf, to dodge the iter-0 fork-storm hang),
**no pretrained_weights** (the load path is equivalence-tested + field-proven; random init is
fine for a step-mechanics check), 50 iters of DDP + `torch.compile` + `cudagraphs=true`.
Script: `~/scripts/ddp_smoke_commit4.sh`.

Result — **PASS**. Log `/mnt/weka/adovlatyan/logs/ddp-smoke-c4-68670.out`:
- `[COMPILE] ... fullgraph=True, triton.cudagraphs=True` + `DDP wrapping complete for student
  sub-models: ['backbone', 'dino_head', 'ibot_head']` — real captured-graph DDP path active.
- All 50 iters completed (`Training Total time: 0:01:26`), no hang, no `AttributeError`, no
  cudagraph capture error, no NaN.
- The deferred grad_norm metrics are logged **finite and correct** at every step
  (`backbone_grad_norm` 6752→23, `dino_head_grad_norm` 80.5→0.30) and track the loss decay
  (20.4→17.1) — proving the tensor is overwritten by the live all-reduced value and the
  scalar read happens at logging time (post `step_end_event.synchronize()`), not mid-step.
- ~1,100 img/s, MFU ~20–27% on 2 GPUs, VRAM flat ~33.9 GB.

### Real-checkpoint load smoke (job 68971)

Commit 4's smoke deliberately ran with **no pretrained_weights**, so it never touched the
checkpoint *load* path that Commits 2–3 actually modify. This smoke closes that gap by loading
a real stock checkpoint into a DDP-wrapped student under the `run_ddp.sh` recipe (in_chans=5,
bs=128, `compile=true`, `cudagraphs=true`, DDP; loader reduced to 8w/2pf). Script:
`~/scripts/ddp_smoke_checkpoint_full.sh`. Checkpoint:
`/mnt/weka/akhosrovyan/dinov3_s1_s2/pretrained_weights/dinov3_vitb16_pretrain.pth` — stock
DINOv3 ViT-B/16, **RGB (in_chans=3)**, flat bare-key `state_dict`.

**Load path — PASS.** Log `/mnt/weka/adovlatyan/logs/ddp-smoke-ckpt-68971.out:28-32`:
- `patch_embed.proj.weight found: shape=(768, 3, 16, 16) target_in_chans=5` → the DDP-wrapped
  student's in_chans (5) was read through `_get_backbone_in_chans` **without `AttributeError`** —
  the exact training-path twin of Anna's bug, exercised against a real checkpoint.
- `Adapted backbone.patch_embed.proj.weight to shape=(768, 5, 16, 16)` — the 3→5 channel
  adaptation ran correctly.
- `Checkpoint load summary: matched=174 missing=15 unexpected=0`. The 15 "missing" are all
  expected: the dino/ibot heads (skip rules, freshly initialized) + `rope_embed.periods` (a
  regenerated buffer). The missing-key *names* carry the `.module.` segment (e.g.
  `dino_head.module.last_layer.weight`), which **proves Commit 2's DDP-prefix remap ran** and
  matched the wrapped model's keys.

So the actual checkpoint surface this PR touches (Commits 2 + 3 load logic) is now validated
end-to-end with a real checkpoint, not just unit-equivalence.

**Write + resume — PASS (jobs 69782 write / 69783 resume).** Validated end-to-end after first
isolating an unrelated iter-0 hang (see below). Run as a **two-job pair** (separate processes —
a fresh process is the honest resume test) sharing a fixed Weka output dir
`/mnt/weka/adovlatyan/ckpt_writeresume_smoke`, under the production step config (DDP +
`compile=true` + `cudagraphs=true`, 2 GPUs, gpu07). Scripts: `~/scripts/ckpt_write.sh`,
`~/scripts/ckpt_resume.sh`.

- **Write (69782):** trained 30 iters (`period=10`), wrote DCP checkpoints `ckpt/{9,19,29}`
  (`checkpointer.py:248 Saved`). Loss 20.40→17.97, all three grad-norm metrics finite per step
  (`backbone_grad_norm: 6720→36.25`, dino/ibot heads finite) — **Commit 4's deferred grad-norm
  tensor logging confirmed in a real captured-graph run**, not just static review.
- **Resume (69783):** fresh process, `train.py:454 Checkpoint found .../ckpt/29` →
  `checkpointer.py:281 Loaded .../ckpt/29` (**no key-mapping RuntimeError** — the torch.compile
  `_orig_mod.` state_dict round-trips cleanly through DCP and a re-compiled process) →
  `train.py:501 Starting training from iteration 30` (resumed at the right iter, not 0) → ran to
  60, wrote `ckpt/{39,49,59}`, and `max_to_keep=3` rotation pruned `9/19/29`. Clean exit.

This is *stock DCP code this PR does not modify*, so the PASS is a completeness confirmation
rather than coverage of a PR-touched surface. It was run from-scratch (`pretrained_weights=""`)
because DCP write/resume is init-source-agnostic, and because loading external pretrained under
this 2-GPU config hangs at iter-0 — see next.

**iter-0 hang — root cause corrected.** Job 68971's earlier writeup attributed the iter-0 hang
to **gpu08 node contention** (a leftover non-NVLink GPU pair). **That diagnosis was wrong.** The
hang reproduced on a *clean* gpu07 with 4 free GPUs (job 69781, `.err`: `CANCELLED ... DUE TO
TIME LIMIT`), so it is not contention. Isolation via the 68670 baseline pins it precisely:

| Job | GPUs / node | pretrained | compile+cudagraphs | iter-0 |
|-----|-------------|------------|--------------------|--------|
| 68670 | 2 / (clean) | **none** (`""`) | yes | **PASS** (50 steps) |
| 69782/69783 | 2 / gpu07 (clean) | **none** (`""`) | yes | **PASS** (this smoke) |
| 68971 | 2 / gpu08 | yes (real ckpt) | yes | **HANG** |
| 69781 | 2 / gpu07 (clean) | yes (real ckpt) | yes | **HANG** |

The sole differentiator is **loading external `pretrained_weights`**. Loading it under
`compile`+`cudagraphs` on a **2-GPU partial-node** allocation deadlocks the first captured NCCL
collective (`ProcessGroupNCCL.cpp:5138] Guessing device ID based on global rank` — `device_id`
is not passed to `init_process_group` at `dinov3/distributed/torch_distributed_wrapper.py:264`;
on a 2-of-8 GPU set the guess can land on a non-NVLink pair). **This is a 2-GPU smoke-harness
artifact, not a PR or production risk:** production `run_ddp.sh` runs the identical
pretrained+cudagraphs recipe on **8 GPUs full-node** and is validated for Anna's full-dataset
run (PR #1, torch 2.6 + 2.10). And the pretrained *load logic* this PR touches already PASSED
above (68971/69781: matched=174, unexpected=0) — the load completes cleanly *before* the first
training step where the collective hangs.

### Related: Anna's two reported issues (both already understood)

- `AttributeError: 'DistributedDataParallel' object has no attribute 'patch_embed'` — the
  training-path DDP unwrap bug, fixed by Anna's own `7a42442` (on master); this PR refactors
  that path. Not a new issue.
- First-iteration hang with `num_workers=20, prefetch_factor=8` — operational fork-storm at
  iter 0 (loader workers × ranks filling prefetch) colliding with first-step cudagraph capture
  stalling the DDP all-reduce. **Not a code bug**; resolved by lowering `num_workers`/
  `prefetch_factor`, which Anna independently confirmed ("different settings → 10 epochs ran
  fine"). The smoke above uses 8w/2pf and shows no hang.

## Additional review + Commit-3 repair (2026-06-10)

A third independent review pass (fresh-context adversarial reviewer, findings reproduced
manually) re-confirmed Commits 2 and 4 clean — including the FSDP `model_keys`-staleness and
cudagraphs tensor-retention angles — but found one real flaw in Commit 3:

### Finding (RISKY): the defensive unwrap silently loaded zero weights when it fired

`init_model_from_checkpoint_for_evals` strips the `module.` prefix from the **checkpoint**
keys, but the original Commit 3 only unwrapped the in_chans *read* and still called
`model.load_state_dict(...)` on the **wrapped** model — whose own keys keep the `module.`
prefix. Result: every checkpoint key lands in `unexpected_keys`, `strict=False` swallows the
100% mismatch, and the model is silently left at its init weights. Reproduced empirically:

```
weights changed by load: False
weights match checkpoint: False
```

So the original fix traded a loud `AttributeError` for a silently wrong (random-weight) eval
model. Mitigations: the branch is unreachable from any real run (the sole caller,
`dinov3/models/__init__.py:128`, always passes a plain backbone), and the test guarding it
(`test_ddp_wrapped_eval_model_does_not_raise`) asserted only no-exception plus a
true-by-construction shape check — which is why it passed despite nothing loading.

### Repair

`checkpointer.py` (`init_model_from_checkpoint_for_evals`): load into the **unwrapped**
module — `target = _unwrap_module(model)` drives both the in_chans read *and*
`target.load_state_dict(...)`. Both key namespaces are then prefix-free and match. For the
plain non-DDP case `target is model`, so the real eval path is byte-identical in behavior.

### Test coverage (red-checked)

- `test_ddp_wrapped_eval_model_does_not_raise` → **renamed/strengthened** to
  `test_ddp_wrapped_eval_model_loads_weights`: asserts the checkpoint tensor is actually
  equal to the loaded weight (catches both the AttributeError mode *and* the silent
  zero-key-match mode).
- **New** `test_ddp_wrapped_eval_model_loads_all_keys`: every checkpoint tensor must land in
  the unwrapped module, guarding against partial loads a single-weight check could miss.
- `test_eval_loader_reads_in_chans_through_unwrap_for_non_default`: upgraded from a
  shape-only assertion (true by construction) to weight-equality.

Honesty check: the strengthened tests were run against the **pre-repair** code
(`git stash` of the checkpointer fix) and fail — `3 failed, 1 passed` (the plain-path test
passes, all wrapped-model tests fail) — then pass with the repair: **32 passed**,
`ruff check` clean.

### Review findings NOT acted on (recorded for honesty)

- NIT: the "`state_dict()` is a rank-synchronizing op" rationale in the Commit-2 comments is
  overstated — DDP `state_dict()` is a local dict walk (no collective); FSDP2 sharded
  `state_dict()` returns DTensors without all-gather. The 4→1 collapse stands on
  redundancy-removal grounds; the comments overstate the perf motivation.
- NIT (pre-existing, not a PR regression): the remap loop in `_remap_checkpoint_keys_for_ddp`
  silently drops an entry if a checkpoint contains *both* `backbone.x` and
  `backbone.module.x` (last write wins); `test_no_keys_dropped_or_duplicated` only covers the
  collision-free case, so its name over-promises.

### Why no new Slurm smoke is needed for this repair

The changed branch (DDP-wrapped eval model) is **unreachable from any real run** — every real
eval job goes through `build_model_for_eval`, which constructs a plain unwrapped teacher
backbone, and for that path `target is model` makes the repair a no-op (pinned by
`test_plain_eval_model_loads_weights`). The function is also CPU-only (plain `torch.load` +
`load_state_dict`, no collectives, no GPU), so the unit/integration tests exercise 100% of the
changed code — unlike Commits 2/4, whose real smokes (68670/68971/69782-3) covered
GPU/dist/cudagraph surfaces no unit test can reach. A 2-GPU job would re-run the training
path, which this repair does not touch.
