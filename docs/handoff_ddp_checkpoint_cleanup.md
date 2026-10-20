# Handoff: DDP checkpoint fix cleanup + test coverage + sync reduction

**For**: New Opus 4.8 / 1M-context session  
**Branch to create**: `fix/ddp-checkpoint-perf-cleanup` off `master`  
**Working directory**: `/home/adovlatyan/dinov3-performance-optimizations/dinov3`

---

## What happened (full context)

Anna (colleague, works on `master`) reported two bugs after the DDP+cudagraphs work was cherry-picked into her branch:

**Bug 1 — `AttributeError: 'DistributedDataParallel' object has no attribute 'patch_embed'`**

Root cause: `init_fsdp_model_from_checkpoint` in `dinov3/checkpointer/checkpointer.py` was written for FSDP2. After `prepare_for_distributed_training()` runs (which wraps each student sub-module individually in DDP), `model.backbone` is `DDP(backbone)`, not the raw backbone. DDP does not proxy arbitrary sub-attributes, so `.patch_embed` access fails.

The training call order is:
1. `train.py:841` → `model.prepare_for_distributed_training()` (DDP wraps `backbone`, `dino_head`, `ibot_head` individually)
2. `train.py:450` → `model.init_weights()` → `ssl_meta_arch.py:316` → `init_fsdp_model_from_checkpoint(self.student, ...)` ← crash here

**Bug 2 — First training iteration never finishes (hang) with num_workers=20, prefetch_factor=8**

Not a code bug. Operational: with `num_workers × prefetch_factor × 8 ranks` all forking at iter 0 simultaneously + per-block CUDA graph capture (first invocation), some ranks stall in prefetch while others enter DDP all-reduce, which waits for all ranks → apparent deadlock. Resolves by reducing num_workers/prefetch_factor. No code fix needed.

---

## Anna's fix (committed to master, merged into perf-ddp-fullgraph)

Commit `7a42442` — `fix DDP checkpoint loading for patch_embed channel adaptation`

She added four helpers and bifurcated the loading path in `init_fsdp_model_from_checkpoint`:

```python
def _unwrap_module(module):          # strip DDP .module wrapper
def _get_backbone_in_chans(model):   # unwraps backbone before .patch_embed access
def _state_dict_uses_ddp_prefix(model):  # detects DDP by checking .module. in state_dict keys
def _remap_checkpoint_keys_for_ddp(state, model):  # inserts .module. into keys where DDP expects it
```

In the loading path, she branches on DDP vs FSDP2:
- **DDP path**: remap keys + plain `.cuda()` (correct — DDP doesn't need DTensor distribution)
- **FSDP2 path**: original `distribute_tensor` logic unchanged

Codex (GPT-5.5, high effort) reviewed the fix and confirmed it is **functionally correct for Bug 1**. Bug 2 is not addressed (expected).

---

## Problems remaining — what this session should fix

### Problem 1 — Redundant `state_dict()` calls (performance)

`model.state_dict()` is called **4 times** during checkpoint loading for a DDP model. For a distributed model, this is a rank-synchronizing operation. While it happens only once at startup (not hot path), it is still wasteful and signals a design issue:

| Call site | File:line | Why |
|---|---|---|
| `_state_dict_uses_ddp_prefix(model)` outer guard | `checkpointer.py:481` | Needed — guards the branch |
| `model_keys = set(model.state_dict().keys())` inside `_remap_checkpoint_keys_for_ddp` | `checkpointer.py:401` | Needed — for key lookup |
| `_state_dict_uses_ddp_prefix(model)` inside `_remap_checkpoint_keys_for_ddp` | `checkpointer.py:402` | **REDUNDANT** — this re-enters the guard, but the function is only called when the outer guard is True. This is dead defensive code that costs an extra `state_dict()` call. |
| `model_state_keys = set(model.state_dict().keys())` for matched_keys check | `checkpointer.py:515` | Needed — but could reuse the keys already computed inside `_remap_checkpoint_keys_for_ddp` |

**Fix**: Refactor `_remap_checkpoint_keys_for_ddp` to accept `model_keys: set[str]` as a parameter instead of computing it internally. Remove the internal `_state_dict_uses_ddp_prefix` re-check (caller already guarantees DDP). Pass the same `model_keys` set to the final `matched_keys` check at line 515.

Net result: 2 `state_dict()` calls total (outer guard + one for model_keys, shared across remap + final check).

**Even cleaner alternative**: Compute `is_ddp = isinstance(model.backbone, nn.parallel.DistributedDataParallel)` directly, avoiding `state_dict()` for the guard entirely. Reduces to 1 `state_dict()` call.

### Problem 2 — Latent DDP unwrap bug in eval path (correctness)

`init_model_from_checkpoint_for_evals` at `checkpointer.py:572`:

```python
# Current (BROKEN if model is DDP-wrapped):
state_dict = adapt_patch_embed_input_channels(state_dict, model.patch_embed.proj.weight.shape[1])
```

Anna didn't fix this. The docstring says "non distributed", but the bug is latent — if anyone passes a DDP-wrapped backbone, it crashes identically to Bug 1. Fix: use `_unwrap_module(model).patch_embed` (the helper is now available as a module-level function).

One-line fix:
```python
state_dict = adapt_patch_embed_input_channels(
    state_dict, _unwrap_module(model).patch_embed.proj.weight.shape[1]
)
```

### Problem 3 — `grad_norm.item()` sync inside the training step (performance)

In `train.py` lines 638–642, inside `with nvtx("grad_clip"):`:
```python
metrics_dict[f"{k}_grad_norm"] = (
    grad_norm.full_tensor().item()   # line 639 — DTensor path
    if isinstance(grad_norm, torch.distributed.tensor.DTensor)
    else grad_norm.item()            # line 641 — DDP path
)
```

This happens **every iteration**, after backward but before `optimizer.step()` and before `step_end_event.record()`. The `.item()` forces a GPU→CPU sync mid-step, breaking pipelining overlap.

**Context from KB** (`learnings/distributed_training.md:78`, `knowledge-base/ai_systems_perf_engineering/ai_systems_perf_eng_ch04.md:217`):
> `.item()` in the training loop forces GPU-to-CPU transfer and kills DDP compute/communication overlap.

**Note on `.item()` calls after `step_end_event.synchronize()` (line 688)**:  
The wandb `.item()` calls at lines 735–741 happen AFTER `step_end_event.synchronize()`, which is already an explicit sync. Those are NOT an additional concern — the GPU is already flushed at that point.

**The real concern is line 639-641** — it fires mid-step, before the CUDA event is recorded.

**Fix**: Keep `grad_norm` as a tensor in `metrics_dict` (don't call `.item()` immediately). The all-reduce at line 656 already handles converting metrics to tensors. The scalar read happens at logging time (inside `metric_logger.update` and wandb block), after `step_end_event.synchronize()`.

For the DDP case (`grad_norm.item()` path), the tensor is already on GPU, so this is straightforward. For the DTensor case (`full_tensor().item()`), deferral requires calling `.full_tensor()` later too — doable since `metrics_dict` values are handled uniformly at lines 737–741.

Concretely: change lines 638–642 to store the raw tensor:
```python
metrics_dict[f"{k}_grad_norm"] = (
    grad_norm.full_tensor()
    if isinstance(grad_norm, torch.distributed.tensor.DTensor)
    else grad_norm.detach()
)
```
The existing metrics-aggregation path at lines 653–661 (`torch.stack`, `all_reduce`, `metrics_values`) already handles tensors, so no other changes needed there.

---

## Proposed commit plan (modular, on new branch `fix/ddp-checkpoint-perf-cleanup` off `master`)

### Commit 1 — `refactor(checkpointer): eliminate redundant state_dict() calls in DDP loading path`

Files: `dinov3/checkpointer/checkpointer.py`

Changes:
1. Refactor `_remap_checkpoint_keys_for_ddp(state, model)` → `_remap_checkpoint_keys_for_ddp(state, model_keys)` — accept precomputed `set[str]` instead of recomputing internally. Remove the internal `_state_dict_uses_ddp_prefix` re-check.
2. In `init_fsdp_model_from_checkpoint`, compute `model_keys` once before the branch, pass to `_remap_checkpoint_keys_for_ddp`, reuse for the final `matched_keys` check.
3. Alternatively: replace `_state_dict_uses_ddp_prefix` with a direct `isinstance` check on `model.backbone`.

### Commit 2 — `fix(checkpointer): unwrap DDP in init_model_from_checkpoint_for_evals`

Files: `dinov3/checkpointer/checkpointer.py`

Changes:
1. Line 572: `model.patch_embed...` → `_unwrap_module(model).patch_embed...`
2. Add a brief comment noting this is the eval path, non-distributed by design, but unwrap is defensive.

### Commit 3 — `test(checkpointer): DDP checkpoint loading correctness`

Files: `tests/test_ddp_checkpoint.py` (new file)

Test cases:
1. `_unwrap_module` — returns `.module` when DDP-wrapped, returns self when not
2. `_get_backbone_in_chans` — works on a mock DDP-wrapped model; works on a plain model; raises on model without backbone
3. `_state_dict_uses_ddp_prefix` — True on a module whose state_dict has `.module.` keys, False otherwise (or replace with isinstance test if that's the approach taken in Commit 1)
4. `_remap_checkpoint_keys_for_ddp` — given a checkpoint with `backbone.layer.weight`, maps to `backbone.module.layer.weight` when target model has DDP keys; leaves already-correct keys untouched; passthrough for keys with no DDP counterpart
5. Key count: no keys dropped or duplicated by remapping

Use `torch.nn.Linear` wrapped in `nn.parallel.DistributedDataParallel` in `cpu` device for mock — does NOT require multi-GPU; DDP can wrap in single-process mode with `dist.init_process_group("gloo", ...)` for testing. Look at `tests/test_mfu.py` for existing test conventions in this repo.

### Commit 4 — `perf(train): defer grad_norm scalar sync to post-step logging`

Files: `dinov3/train/train.py`

Changes:
1. Lines 638–642: store raw tensor instead of calling `.item()` immediately.
2. Verify that the existing all-reduce path at lines 653–661 handles the tensor correctly (it does — `torch.as_tensor(v, ...)` already handles plain tensors).
3. Add a brief inline comment explaining WHY: `.item()` mid-step forces GPU→CPU sync before optimizer.step(); defer to logging path which is already post-sync.

---

## Key file locations

| File | What matters |
|---|---|
| `dinov3/checkpointer/checkpointer.py:383–416` | Anna's four helpers |
| `dinov3/checkpointer/checkpointer.py:479–558` | `init_fsdp_model_from_checkpoint` loading path (DDP branch at :481) |
| `dinov3/checkpointer/checkpointer.py:560–574` | `init_model_from_checkpoint_for_evals` — latent bug at :572 |
| `dinov3/fsdp/ac_compile_parallelize.py:345–354` | DDP wrapping loop (each submodule wrapped individually) |
| `dinov3/train/ssl_meta_arch.py:303–306` | `_unwrap()` — existing unwrap pattern already in the class |
| `dinov3/train/ssl_meta_arch.py:308–327` | `init_weights()` — calls `init_fsdp_model_from_checkpoint(self.student, ...)` |
| `dinov3/train/train.py:630–642` | `grad_norm.item()` sync site (inside nvtx grad_clip) |
| `dinov3/train/train.py:644–661` | all-reduce metrics path (handles tensors correctly already) |
| `dinov3/train/train.py:685–688` | `step_end_event.record()` then `.synchronize()` — explicit sync point |
| `dinov3/train/train.py:728–747` | wandb `.item()` calls — SAFE (after step_end_event.synchronize) |
| `tests/test_mfu.py` | Existing test file — follow its conventions |

---

## Notes for implementation

- `master` is the base (not `perf-ddp-fullgraph`). Confirm with `git log master | head -5`.
- The `_unwrap_module` helper is now a module-level function in `checkpointer.py` (added by Anna) — use it in both commits.
- For the test file, single-process DDP init via `dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)` is the standard pattern for unit tests. Can also mock `nn.parallel.DistributedDataParallel` with a simple wrapper class `class _FakeDDP(nn.Module): def __init__(self, m): super().__init__(); self.module = m` to avoid needing a real process group.
- The `_state_dict_uses_ddp_prefix` detector has a known fragility: it checks string content, not module type. The `isinstance(model.backbone, nn.parallel.DistributedDataParallel)` approach is strictly more robust. Codex flagged this. If refactoring, prefer the isinstance check.
- The KB confirms `.item()` mid-step is a real concern for DDP: `learnings/distributed_training.md:78` and `ai_systems_perf_eng_ch04.md:217`. The CUDA graph wiki (`concept-cuda-graph-capture-debugging.md:63-70`) also flags it for the fullgraph direction even though we're doing per-block captures today.

---

## What NOT to do

- Do not merge `perf-ddp-fullgraph` wholesale — that branch has 128 changed files, most are experiment scripts and docs not intended for mainline.
- Do not add `barrier()` calls to "fix" the iter-0 hang — that's operational, not a code bug.
- Do not add error handling in `_get_backbone_in_chans` for the case where backbone doesn't have `patch_embed` — that would hide a real misconfiguration.
- Do not modify `run_ddp.sh` num_workers/prefetch_factor as part of this task — those are tuned for the specific memory budget.
