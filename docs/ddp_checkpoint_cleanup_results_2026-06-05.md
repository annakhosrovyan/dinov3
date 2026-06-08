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
- Did not add `barrier()` for the iter-0 hang (operational, not a code bug).
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

### Related: Anna's two reported issues (both already understood)

- `AttributeError: 'DistributedDataParallel' object has no attribute 'patch_embed'` — the
  training-path DDP unwrap bug, fixed by Anna's own `7a42442` (on master); this PR refactors
  that path. Not a new issue.
- First-iteration hang with `num_workers=20, prefetch_factor=8` — operational fork-storm at
  iter 0 (loader workers × ranks filling prefetch) colliding with first-step cudagraph capture
  stalling the DDP all-reduce. **Not a code bug**; resolved by lowering `num_workers`/
  `prefetch_factor`, which Anna independently confirmed ("different settings → 10 epochs ran
  fine"). The smoke above uses 8w/2pf and shows no hang.
