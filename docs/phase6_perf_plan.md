# Phase 6 — DDP + `torch.compile(fullgraph=True)` Plan

| Field | Value |
|---|---|
| Branch | `perf-ddp-fullgraph` (cut 2026-05-21 from `perf-fsdp2-pipeline` head) |
| Date opened | 2026-05-21 |
| Predecessor | Phase 5 — FSDP2 + Data/Compute Pipeline (closed 2026-05-21, `docs/phase5_perf_plan.md`) |
| Trigger | External advisor input from **Armen Aghajanyan** (2026-05-21): drop FSDP for ViT-B, use DDP, set `fullgraph=True`, fix static shapes, use CUDA graphs. |
| Strategy | **DDP only.** No FSDP2, no `reshard_after_forward`. Push throughput through `torch.compile` + CUDA graphs on top of the existing DDP path. |
| Current baseline to beat | **DDP bs=96, compile=true, no AC, no ES, no CUDA-graphs**: job 48312, iter 400–999 mean **1,387 img/s, 7.94% MFU, step_ms 545 ms, peak alloc 25.8 GB** on gpu03. |
| Aspirational target | Lab-wide internal target of **30% MFU** (~5.2 k img/s at bs=96, 8×H100). Armen believes a model this small "definitely benefits from CUDA graphs." |
| Goal of Phase 6 | Determine empirically what fraction of the 7.94 % → 30 % gap can be closed by `fullgraph=True` + `triton.cudagraphs=true` on DDP, and what work is required to unblock those flags. |

---

## 1. Why this phase (and why it's a U-turn from Phase 5)

Phase 5 spent 2 weeks comparing FSDP2 reshard variants and chasing a bs=128 OOM
that may not have been a real long-run signal. The wall-clock cost: a wrap-up
2×2 grid that we cancelled before reading the result.

Armen's review on 2026-05-21 reframed the problem entirely:

1. **FSDP2 is the wrong tool for this scale.** ViT-B is ~85 M params, ~172 MB in
   BF16. There is no shard-vs-fit problem to solve. The all-gather / reduce-scatter
   pattern adds communication that DDP's single gradient all-reduce avoids. Phase 5
   nsys evidence (jobs 39140 / 39141) was already pointing here: NCCL AllGather alone
   was 69.5 % of kernel time with 11 % overlap. We optimized within the wrong family.
2. **`torch.compile` was on, but in its weakest configuration.** `train.compile=true`
   in this repo routes through `nn.Module.compile()` per transformer block with
   `compile_mode: null` and `cudagraphs: false`. The fullgraph + CUDA-graph branch
   in `dinov3/fsdp/ac_compile_parallelize.py:72` was wired but never the default.
   Armen's diagnosis: for ViT-B at this scale the Python / launch overhead per
   kernel is the budget that CUDA graphs eliminate.
3. **Dynamic shapes are a fixable design choice, not a law of nature.** The iBOT
   stochastic-mask-token count was treated as immutable in Phase 4 (closed
   `max-autotune-no-cudagraphs` as "fundamentally incompatible"). Armen pushed back:
   *"why are the shapes not static, are you not resizing into a fixed shape?"*
   Phase 6 takes that seriously.

See `docs/torch-compile-dinov3-phase5-2026-05-21.md` for the full code-level
walkthrough of how compile is wired today.

---

## 2. Current operating point — full anchor

| Knob | Value | Where |
|---|---|---|
| Distributed strategy | DDP | `dinov3/fsdp/ac_compile_parallelize.py:281–322`, `train.distributed_strategy=ddp` |
| `train.compile` | `true` | `dinov3/configs/ssl_default_config.yaml:80` |
| `train.compile_mode` | `null` (== `nn.Module.compile()` plain) | `dinov3/configs/ssl_default_config.yaml:81` |
| `train.cudagraphs` | `false` | `dinov3/configs/ssl_default_config.yaml:82` |
| `train.checkpointing` | `false` (no AC) | default |
| `PYTORCH_CUDA_ALLOC_CONF` | unset (no expandable_segments) | job 48312 script |
| Per-GPU bs | 96 | `scripts/screening/ddp_bs96_calibration.sh` |
| World size | 8 × H100 (one node) | `gpu03` for 48312 |
| DDP wrapping | `gradient_as_bucket_view=True`, `static_graph=True`, per sub-module (`backbone`, `dino_head`, `ibot_head`) | `dinov3/fsdp/ac_compile_parallelize.py:310–319` |
| Param dtype / reduce dtype | bf16 / bf16 (DDP, single dtype after `.to(param_dtype)`) | `dinov3/fsdp/ac_compile_parallelize.py:301–308` |
| MFU formula | `2 × MACs/image` over `H100_BF16_TFLOPS = 989.0` | `dinov3/utils/mfu.py`, `CLAUDE.md` |
| MACs/image (ViT-B, 5ch, 2 global + 8 local) | 226.4 G | `dinov3/utils/mfu.py`, validated in `tests/test_mfu.py` |

**Job 48312 — windowed steady state, iter 400–999 (n = 61):**

| Statistic | Value |
|---|---|
| Mean img/s | 1,387 |
| Median img/s | 1,379 |
| CV(img/s) | 8.2 % |
| Mean step_ms | 545 |
| Mean MFU (dense, %) | 7.94 |
| Trend (300-window means) | 1,310 → 1,373 → 1,419 img/s (monotonic up; still warming) |
| Peak alloc / reserved | 25.8 GB / 26.7 GB |

This is what every Phase 6 number is measured *against*, not against archived 100-iter
screening from April. The archived ~4 k img/s figures (job 9631 family) are explicitly
treated as a separate open question — see §6.1.

---

## 3. Why fullgraph and CUDA graphs are the next lever (not data, not comms, not AC)

Three orthogonal facts decide the lever:

1. **Data is not bottleneck.** `num_workers=20`, `pin_memory=True`, `persistent_workers=True`,
   `prefetch_factor=8`, `non_blocking=True`. The Phase 5 nsys traces (39140 / 39141) put
   H2D at ~1 % of GPU time. Building a copy_stream pipeline does not help here.
2. **NCCL overhead is a DDP-vs-FSDP2 axis we already moved on.** DDP has one all-reduce
   per backward; FSDP2 had AllGather at 69.5 % kernel time. Switching to DDP collapses
   the comm category. What's left after that switch is *per-kernel launch overhead* and
   *Python overhead per step*.
3. **At ViT-B scale, per-step Python is a real fraction of the step.** 545 ms / step
   with thousands of small kernels (12 blocks × multi-crop × heads × loss reductions)
   means each kernel launch is ~hundreds of microseconds in CPU time. CUDA graphs are
   designed exactly for this: capture the full kernel-launch sequence once and replay
   without Python in the path. `module.compile()` (no `triton.cudagraphs`) lowers
   *individual* kernels but does not eliminate the per-launch CPU overhead.

So the Phase 6 hypothesis is concrete:

> **Hypothesis H6.** Enabling `fullgraph=True` + `triton.cudagraphs=true` on the
> transformer backbone under DDP, with iBOT's dynamic-shape path resolved or worked
> around, closes a substantial portion of the 7.94 % → 30 % MFU gap on ViT-B at bs=96.

Phase 6 succeeds if H6 is confirmed or falsified with clear measurement.

---

## 4. The dynamic-shape blocker — precise diagnosis

This is the technical heart of Phase 6. Without resolving it, fullgraph either fails
to capture or recompiles every iteration (which is what Phase 4 saw at
`max-autotune-no-cudagraphs`).

**Where the dynamism originates — full verified code chain:**

```python
# dinov3/data/collate.py:43
probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
# ^ stochastic mask-ratio schedule — different tokens masked each iter

# dinov3/data/collate.py:62
collated_masks = torch.stack(masks_list).flatten(1)
# ^ shape (B, N_patches) — STATIC (B = n_global_crops * 2, N = 196 for ViT-B/16)

# dinov3/data/collate.py:63
mask_indices_list = collated_masks.flatten().nonzero().flatten()
# ^ shape (K,) — DYNAMIC; K = number of True entries, varies per batch

# dinov3/data/collate.py:65
masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
# ^ shape (K,) — ALSO DYNAMIC; boolean indexing [collated_masks] produces same variable K

# dinov3/data/collate.py:74
"n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long)
# ^ records K as a scalar for downstream loss

# dinov3/train/ssl_meta_arch.py:585
masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
# ^ source g_patch.flatten(0,1) is shape (B*n_global, N_patches, D) — STATIC
# ^ result masked_patches_pre_head is shape (K, D) — DYNAMIC first dim

# dinov3/loss/ibot_patch_loss.py:115
loss = loss[:n_masked_patches]
# ^ dynamic slice; K changes → different loss tensor shape each iter
```

Key clarification from the HTML explainer (`docs/dinov3-training-pipeline-ibot-compile-explainer-2026-05-22.html`):
- `collated_masks` is a **fixed rectangular tensor** — `(B, N_patches)` boolean. Armen's question
  "are you not resizing into a fixed shape?" applies here: the mask *grid* is fixed. The variable
  shape is introduced by `nonzero()` which **materializes only the True positions** into a compact
  1D index list.
- The ViT backbone source (`g_patch`) is **also fully static** — `(B * n_global_crops, N_patches, D)`.
  Only the *result* of `index_select` is variable. This means the backbone compute path is not
  directly affected by the dynamic-K problem.
- **Both `mask_indices_list` and `masks_weight` are variable-K.** Option B padding must handle both.

**Downstream effects (chain of varying-shape tensors):**

1. `masked_patches_pre_head` — shape `(K, D)` — gathered via `index_select` from student backbone
   output at `ssl_meta_arch.py:585`.
2. `teacher_patch_tokens` — same gather on teacher side.
3. `masks_weight` — shape `(K,)` — loss normalization weights, same K as above.
4. The iBOT loss computes over `(K, D)` logit space; `loss[:n_masked_patches]` at `ibot_patch_loss.py:115`
   slices the variable-K dimension.

`torch.compile` will trace this branch with one specific `n_masked`. Next iter, n_masked
changes, and either: (a) Inductor recompiles (with `dynamic=False` and `max-autotune` this
costs seconds-to-minutes and may OOM), or (b) the trace expects `dynamic=True` and emits
guard checks that prevent CUDA-graph capture.

**The backbone is NOT directly affected** — it sees `(B, 197, dim)` for global crops and
`(B*L, 37, dim)` for local crops, both with fixed `B` and fixed seq lens. So:

> **Important nuance.** Enabling `cudagraphs=true` *today* captures only backbone blocks
> (`is_backbone_block=True` gating at `dinov3/fsdp/ac_compile_parallelize.py:71`). That
> path may already work without touching iBOT. The dino_head, ibot_head, and loss path
> are wrapped with plain `module.compile()` and never go through the CUDA-graph branch.

This splits Phase 6 into two essentially independent sub-questions:

- **6.A — Backbone CUDA graphs:** flip `cudagraphs=true` on the current DDP path, see
  what fraction of the gap that alone closes. Low-risk, no code change.
- **6.B — End-to-end fullgraph:** make iBOT shapes static (padding, fixed-K, or bucketed),
  then extend the CUDA-graph capture to heads and loss. Higher lift, larger potential gain.

---

## 5. Design decisions to settle before coding

The shape-fix strategy is a real design call with throughput / memory / fidelity
trade-offs. Settling this before code prevents the Phase-4-style "we tried max-autotune
and it broke" loop.

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **A. Fixed-K masking** | Force `mask_generator` to produce exactly K true positions per masked image (replace `torch.linspace(*mask_ratio_tuple, n_samples_masked+1)` with a constant K). Total `n_masked = K * n_samples_masked = K * floor(B * mask_probability)` — constant. | Cleanest fix. Zero wasted compute. Fully static downstream. | Changes the SSL recipe — researcher needs to sign off that fixed-K vs sampled ratio doesn't hurt convergence. |
| **B. Padding to upper bound** | Keep stochastic mask, but always pad `student_patch_tokens_masked` / `teacher_patch_tokens_masked` / `masks_weight` to `K_max = ceil(mask_ratio_tuple[1] * B * N)`. Multiply loss by a 0/1 weight to zero out the pad rows. | Preserves the SSL recipe exactly. Static shapes everywhere. | Wastes compute on pad rows (~10–20 % overhead). Need to verify backward through the masked-out rows is genuinely zero. |
| **C. Bucketed shapes** | Round `n_masked` up to the nearest bucket in a small set, e.g., 8 fixed sizes. Use option B-style padding within each bucket. | Lower compute waste than B. | Forces 8× more compiled graphs / CUDA-graph captures (one per bucket). Cache-friendliness suffers. |

`★ Insight ─────────────────────────────────────`
- Option B (padding to upper bound) is the most defensible *first* experiment because
  it changes nothing about the SSL math the researcher cares about — the loss numerically
  matches the original. The compute overhead is bounded and measurable.
- Option A is the *best* answer if you can afford a parallel convergence study, because
  it has no overhead at all. The right time to switch from B to A is after the throughput
  win is real and the question is "can we squeeze the last 10–15 %."
- Option C is rarely the right answer in practice — you pay the compile-cache cost of
  B without saving enough compute to justify it.
`─────────────────────────────────────────────────`

**Decision needed from Aram:** which of A / B / C is Phase 6.B's first attempt? I'll
default to **B (padding)** unless told otherwise, because it lets the throughput question
be answered without entangling it with a convergence question.

A second smaller decision: when we measure `cudagraphs=true` (sub-question 6.A), do we
do it on a fresh branch / fresh script, or extend `scripts/screening/ddp_bs96_calibration.sh` with
a config override? I'll default to a fresh `scripts/screening/ddp_bs96_cudagraphs_backbone.sh` so
the calibration baseline stays pristine.

---

## 6. Experiment ladder

Each rung is gated on the previous. No rung is run until the previous one has been read.

### 6.1 — Diagnostic: what compile is actually doing today (no new runs needed)

**Goal:** measure recompiles / graph breaks on the *current* DDP + `compile=true` +
`cudagraphs=false` path, so the baseline isn't a black box.

**Method.** Re-submit the 48312 recipe with environment vars added:

```bash
export TORCH_LOGS="recompiles,graph_breaks,dynamic"
export TORCHDYNAMO_VERBOSE=1
```

100–200 iterations is enough. Look for:
- recompile counter per compiled module per iter,
- which guards triggered (this is where `mask_indices_list.shape[0]` will show up),
- any "graph break" messages.

**Why this is the first thing to do.** It will tell us whether the current compile path
is already eating recompile cost we don't see in the metric log. It also gives us the
exact symbol names that bound any future static-shape work.

### 6.2 — Backbone-only CUDA graphs (no shape work yet)

**Goal:** test the cheap half of H6: does flipping `train.cudagraphs=true` on the
backbone alone change the 7.94 % MFU number?

**Recipe.** `scripts/screening/ddp_bs96_cudagraphs_backbone.sh` — identical to
`scripts/screening/ddp_bs96_calibration.sh` except:

```yaml
train.cudagraphs=true   # backbone blocks go through fullgraph + triton.cudagraphs branch
```

Everything else (compile_mode null, no AC, no ES, bs=96, 1000 iters) matches.

**Expected outcomes and what each means:**

| Outcome | Interpretation |
|---|---|
| Throughput up ≥ 15 % vs 48312 (~1,600+ img/s) | Backbone launch overhead was a real fraction. Strong support for proceeding to 6.B. |
| Throughput up 0–15 % | Some win, but the heads / loss are non-trivial fraction of step. 6.B definitely needed. |
| Throughput flat | Backbone wasn't the launch bottleneck. CPU contention or DDP-bucket cost dominates. Re-examine before going further. |
| Compile fails / crashes | `dynamic=False` is incompatible with some op in the ViT block (likely RoPE or stochastic depth). Fix or document, do not paper over. |

**Acceptance.** Same window convention as 48312: iter 400–999, mean / median / CV /
step_ms / MFU. Report peak alloc — `dynamic=False` plus `triton.cudagraphs=true` will
likely raise memory because Inductor allocates fixed workspaces.

### 6.3 — Static-shape pilot (no fullgraph yet)

**Goal:** implement Option B (padding) in isolation and measure its overhead under the
current `cudagraphs=false`, `compile_mode=null` recipe. We want to know the *cost* of
padding *before* we know the *benefit* of fullgraph — so we can attribute correctly.

**Code change scope (single PR):**

- `dinov3/data/collate.py:63`: after computing `mask_indices_list`, pad it to
  `K_max = ceil(mask_ratio_tuple[1] * N) * n_samples_masked` using e.g.
  `F.pad(mask_indices_list, (0, K_max - K))`. The pad values can be 0 (any valid index;
  loss masking makes them inert).
- `dinov3/data/collate.py:65`: `masks_weight` is also shape `(K,)` — pad it to `K_max`
  with zeros so padded-token loss contributions are zeroed out automatically.
- `dinov3/data/collate.py:74`: `n_masked_patches` can remain K (real count) for the
  loss normalization denominator — we need it so we can normalize correctly over real
  tokens only.
- `dinov3/loss/ibot_patch_loss.py:97-117`: replace `loss[:n_masked_patches]` with a
  multiplication by the (now `K_max`-length) `masks_weight` which already has 0s on
  pad positions. Normalization denominator uses the stored real K.
- A flag `train.static_ibot_shapes=true` to gate the change (needed for A/B against the
  stochastic baseline to verify numeric equivalence in §6.3).

**Recipe.** Same as 48312 except `train.static_ibot_shapes=true`. 1000 iters.

**Acceptance.** Loss curve should be numerically *equivalent* (up to padded-row weighting)
to the stochastic baseline. Throughput should be flat or slightly down (5–15 %) due to the
extra compute on padded rows. If throughput is significantly *up* in this rung alone,
something is wrong — call it out and investigate.

### 6.4 — End-to-end CUDA graphs (the H6 test)

**Goal:** combine 6.2 and 6.3. `cudagraphs=true`, `static_ibot_shapes=true`. This is the
configuration that maps to Armen's recommendation.

**Possible additional change:** extend `wrap_compile_block` to pass `fullgraph=True` for
the heads as well, not just `is_backbone_block`. With static shapes this should now work.

**Acceptance criteria — these are the numbers that decide Phase 6:**

- Steady-state MFU on iter 400–999, same convention as 48312.
- Per-iter step_ms median + CV (CUDA graphs should *also* reduce within-run variance,
  not just mean step time — verify both).
- Peak alloc and reserved (CUDA graphs increase memory; if this OOMs at bs=96 we have
  a separate problem to solve).
- Throughput vs 6.2 and vs 48312 — both deltas needed to attribute the gain.

If the combined number does not clear ~15 % MFU (~2× the 48312 baseline), the H6
hypothesis is partially falsified and we re-examine before continuing.

### 6.5 — Headroom probe: bs=128 on the fullgraph path

**Goal:** the FSDP2 bs=128 OOM was a Phase 5 worry. DDP doesn't reshard, so memory math
is different (full params + activations + gradients per rank). At bs=96 we saw 25.8 GB
peak; bs=128 is ~34 GB linear estimate (still fits an 80 GB H100). Try it once we have
the static-shape recipe.

Gate: only run if 6.4 results justify it.

---

## 7. Items intentionally NOT in Phase 6

So we don't slide back into Phase 5's "more knobs always help" failure mode:

| Item | Why not |
|---|---|
| FSDP2 anything | Closed by Armen + Phase 5 evidence. |
| Activation checkpointing | bs=96 fits without it; AC raises HFU but not MFU. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | Only helped DDP+ES bs=256 screening; not load-bearing at bs=96. |
| Packed multi-crop attention (NestedTensor / FA2 varlen) | High-effort structural change. Defer until Phase 6 numbers are known. |
| Copy_stream pipeline | Phase 5 trace put H2D at ~1 % of GPU time. Not the lever. |
| NCCL bucket tuning, NVLS, etc. | Phase 5 covered. With DDP single all-reduce, less load-bearing. Revisit only if 6.4 traces say so. |

---

## 8. Risks & failure modes to call out up front

| Risk | Mitigation |
|---|---|
| `dynamic=False` + ViT trips on RoPE or stochastic-depth dynamism | 6.2 finds this first. If it does, isolate to the offending op and document precise error. Don't `try/except` it away. |
| Padding overhead in 6.3 wipes the 6.2 win | 6.3 measures cost in isolation. If padded overhead is > expected, switch to Option A (fixed-K) before 6.4. Requires researcher sign-off on convergence. |
| CUDA-graph workspace memory pushes us OOM at bs=96 | Check 6.2 peak alloc first. If close to 80 GB, bs=128 (6.5) is off the table; if blown out, may need to capture per crop-type instead of whole step. |
| Loss numerics drift between stochastic and padded path | 6.3 explicitly checks this. Loss must match to ~1e-4 over equivalent batches before we trust throughput numbers. |
| Compile cache thrash on cluster restarts | Set `TORCHINDUCTOR_CACHE_DIR` on Weka so the cache persists across jobs — drops warmup from ~5 min to seconds. |
| Convergence regression at full scale | Phase 6 measures throughput, not final accuracy. After the throughput question is settled, run a true long soak with researcher-validated checkpoints before promoting to `run.sh`. |

---

## 9. What "done" means for Phase 6

Phase 6 is complete when:

1. We have a measured DDP + fullgraph + CUDA-graph MFU number on iter 400–999 at bs=96,
   either confirming H6 (≥ 15 % MFU, ~2× baseline) or falsifying it with clean attribution.
2. The dynamic-shape blocker is either *fixed* (Option A or B implemented and validated)
   or *bounded* (we know exactly what fraction of the win is gated on it).
3. A small follow-up plan exists for promoting the winning config to `run.sh`, including
   a researcher-side convergence validation.
4. The Phase 5 archived-DDP convention question (§6.1 diagnostic in this plan) is resolved
   to "fair upper bound" or "measurement artifact." The DDP family's "true ceiling" matters
   because it tells us whether 6.4 is approaching it or still leaving table on it.

---

## 10. Open questions for Aram before we start coding/queueing

These shape Phase 6 substantially; please confirm or redirect before I touch code.

1. **Shape-fix strategy:** A (fixed-K) vs B (padding) vs C (bucket) for the first
   static-shape pilot. Default: B. (See §5.)
2. **Branch:** new `perf-ddp-fullgraph` branched from `perf-fsdp2-pipeline` head, or
   continue on `perf-fsdp2-pipeline`. Default: new branch.
3. **Researcher sign-off bandwidth:** Option A requires a parallel convergence study with
   Anna/the researcher. If that channel is open we can pursue A directly after B's throughput
   number is known. If not, B is the terminal Phase 6 recipe.
4. **Ordering within 6.A vs 6.B:** plan above runs 6.2 (cheap, no code) before 6.3 (code).
   Confirm that's the order you want, vs. doing the code work first and testing combined.
5. **Diagnostic-first vs experiment-first:** §6.1 requires re-running the 48312 recipe
   with TORCH_LOGS env vars. Cheap (~10 min if it queues) but gates everything. Confirm OK
   to spend that first.

---

## 11. Living evidence index

- `docs/dinov3-training-pipeline-ibot-compile-explainer-2026-05-22.html` — **verified visual explainer**
  of the full iBOT dynamic-K code chain (all 7 code refs from `collate.py:43` through
  `ibot_patch_loss.py:115`), DINO vs iBOT patch-token diagrams, and torch.compile timeline
  showing recompile storm. Primary reference for §4 and §6.3 code-change scope.
- `docs/torch-compile-dinov3-phase5-2026-05-21.md` — code walkthrough of `train.compile=true`.
- `docs/phase5_perf_plan.md` — closed predecessor; preserved for nsys / NCCL / OOM history.
- `dinov3/fsdp/ac_compile_parallelize.py:65–76` — `wrap_compile_block` (the fullgraph / CUDA-graph branch lives here).
- `dinov3/fsdp/ac_compile_parallelize.py:281–322` — DDP wrapping (`gradient_as_bucket_view`, `static_graph`).
- `dinov3/configs/ssl_default_config.yaml:80–82` — `compile`, `compile_mode`, `cudagraphs` defaults.
- `dinov3/data/collate.py:62–74` — `mask_indices_list` dynamic shape source.
- `dinov3/loss/ibot_patch_loss.py:96–117` — `loss[:n_masked_patches]` dynamic slice.
- `scripts/screening/ddp_bs96_calibration.sh` — the recipe job 48312 ran.
- `scripts/screening/ddp_bs96_cudagraphs.sh` — Phase 6.2 script.
- `/mnt/weka/adovlatyan/logs/ddp-bs96-calib-48312.out` — baseline log to beat.

---

## 12. Job log and experiment results

Chronological record of Phase 6 runs. Append a new entry for every job.

---

### 6.A.1 — First `cudagraphs=true` attempt (job 53655, 2026-05-23, FAILED)

**Script:** `scripts/screening/ddp_bs96_cudagraphs.sh`  
**Node:** gpu01  
**Status:** CRASHED during first training step (~4 min after start)

**[COMPILE] log confirmed** the right path ran:
```
[COMPILE] torch.compile enabled — cudagraphs=True, compile_mode=None
[COMPILE] compile_transformer: 12 backbone blocks → path: fullgraph=True, dynamic=False, triton.cudagraphs=True
[COMPILE] head 'dino_head' → path: default (module.compile(), dynamic=True)
[COMPILE] head 'ibot_head' → path: default (module.compile(), dynamic=True)
```

**Root cause — Inductor CUDA graph tree buffer aliasing:**
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run.
...block.py:194: x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
To prevent overwriting, clone the tensor outside of torch.compile() or call
torch.compiler.cudagraph_mark_step_begin() before each model invocation.
Stack: torch/_inductor/cudagraph_trees.py:1625 → _allocate_and_copy_recording_inputs
```

**Why it happened:** each compiled transformer block receives `[global_crops_tokens, local_crops_tokens]` as a list and loops over both inside `_forward_list`. Inductor's CUDA graph tree captures a separate sub-graph per input shape (197 tokens vs 37 tokens). Without `cudagraph_mark_step_begin()`, the tree doesn't know that the two sub-graphs belong to the *same* step — it allows output buffers from the first shape's graph to be aliased/overwritten by the second shape's graph. Additionally, teacher and student each call the compiled backbone, making it 4+ compiled-function invocations per step.

**Fix applied:** added `torch.compiler.cudagraph_mark_step_begin()` at the top of `SSLMetaArch.forward_backward()` in `dinov3/train/ssl_meta_arch.py`. This call is a documented no-op when cudagraphs are not active, so it has zero cost on default runs.

**Note on prior compile history:** pre-Phase 6, every submitted job used `train.cudagraphs=false` (yaml default). The `fullgraph=True, triton.cudagraphs=True` branch in `wrap_compile_block` had *never* run before job 53655. All MFU numbers prior to Phase 6 reflect `module.compile()` default mode — dynamic=True, no fullgraph, no CUDA graphs.

---

### 6.A.2 — `cudagraphs=true` with mark_step_begin fix (job 53672, 2026-05-23, COMPLETED — REGRESSION)

**Script:** `scripts/screening/ddp_bs96_cudagraphs.sh`  
**Fix:** `torch.compiler.cudagraph_mark_step_begin()` at top of `forward_backward()`  
**Node:** gpu01  
**Status:** Ran to completion 1000 iters, no crash. Throughput *worse* than baseline.

**Steady-state results (iter 400–999, n = 80 logged points):**

| Run | Strategy | img/s | MFU | step_ms | Peak alloc |
|---|---|---|---|---|---|
| Job 48312 (baseline) | DDP, compile=true, cudagraphs=false | 1,387 | 7.94 % | 545 | 25.8 GB |
| **Job 53672** | **DDP, compile=true, cudagraphs=true** | **1,244** | **7.12 %** | **599** | **~25 GB** |
| **Delta** | | **−10.3 %** | **−0.82 pp** | **+9.9 %** | flat |

**Root cause of the regression — `index_put_` with `accumulate=True`:**

stderr shows 8 instances of:
```
skipping cudagraphs due to index put with accumulate. Found from :
    x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]
```

That call sits in the **indexed branch of `_forward_list` in `dinov3/layers/block.py`**. The advanced-indexing pattern `x[indices_1]` in forward produces `index_put_(..., accumulate=True)` in backward — an op Inductor **refuses to capture into CUDA graphs**. Inductor falls back to eager for that section, but the run still pays:

1. `cudagraph_mark_step_begin()` per step,
2. Cudagraph tree partitioning and decision overhead,
3. `_copy_inputs_and_remove_from_src` input-copy cost per captured sub-graph,
4. Twice the compile time (197-token graph + 37-token graph instead of one default-mode graph).

With the broken-graph path taking eager fallback inside an otherwise capture-attempted region, the net effect is **negative** rather than positive.

**This is informative, not a dead end:**
- The fullgraph branch *traces* cleanly (compilation succeeds, no graph breaks at the Dynamo level).
- The `cudagraph_mark_step_begin()` fix is correct — runtime no longer crashes on the multi-shape `_forward_list` aliasing.
- The remaining blocker is one specific autograd-time op (`index_put_` with `accumulate=True`) coming from one specific access pattern in `block.py`.

**6.A verdict:** `cudagraphs=true` flipped naively on the current backbone code is a **regression**. To unlock the CUDA-graph win, the indexed-access pattern in `block.py:_forward_list` must be rewritten — see §6.A.3 below.

---

### 6.A.3 — Replace `x[indices]` with `torch.index_select` (job 53681, 2026-05-23, ✅ BIG WIN)

**Goal:** eliminate the `index_put_(accumulate=True)` backward op so `triton.cudagraphs=True` can actually capture the backbone.

**Code change (commit `cda29e2`):** `dinov3/layers/block.py` — replaced 5 occurrences of `x[indices]` advanced indexing with `torch.index_select(x, 0, indices)`. Sites: `_maybe_index_rope` (sin/cos), `_forward` (x_subset_1, x_subset_2), `_forward_list` (x_subset_1_list, x_subset_2_list). All inside the stochastic-depth `sample_drop_ratio > 0` branch.

**Mathematical equivalence:** indices come from `torch.randperm(b)[:k]` — guaranteed unique, so accumulate-vs-non-accumulate is moot. Forward output identical; backward identical up to float reduction order.

**Results (job 53681, iter 400–999, n = 80):**

| Run | img/s | MFU | step_ms | Δ vs 48312 baseline |
|---|---|---|---|---|
| Job 48312 (cudagraphs=false) | 1,387 | 7.94 % | 545 | — |
| Job 53672 (cudagraphs=true, broken `index_put`) | 1,244 | 7.12 % | 599 | **−10.3 % img/s** |
| **Job 53681 (cudagraphs=true + `index_select` fix)** | **2,009** | **11.50 %** | **379** | **+44.8 % img/s** |

**Verification:**
- `skipping cudagraphs due to index put with accumulate` warnings: **0** (down from 8 in 53672).
- Loss curve looks healthy: total_loss ~14.04 at iter 999, matching the trajectory of the baseline.
- `[COMPILE]` log confirmed the fullgraph + triton.cudagraphs branch ran on all 12 backbone blocks.
- No crashes, no graph breaks, no recompiles after warmup (would need TORCH_LOGS to confirm formally).

**What this means:** Phase 6 hypothesis H6 is *partially confirmed*. Closing the 7.94 % → 30 % gap by 44.8 % in img/s (from one-knob change) is a strong signal that the per-step Python / launch overhead Armen flagged was real. We've moved from `7.94 %` to `11.50 %` MFU. The remaining 18.5 pp gap to 30 % is now the question for 6.B (heads / iBOT static shapes) and 6.A.4 (AC + bigger batch).

**Caveats for next steps:**
- Per-iter MFU variance is wider than baseline (range 8.96 %–14.26 % in the last 200 iters). The peaks suggest more headroom; the troughs suggest a remaining bottleneck — likely the heads (still on `module.compile()` dynamic default).
- Memory delta not yet measured (cudagraph workspaces add). Need to grep peak_alloc from the log.

**Gate cleared.** 6.B (full static-shape iBOT + extend fullgraph to heads) is now worth the larger code lift, and 6.A.4 (AC + bigger batch + cudagraphs) is queued.

---

### 6.A.4 — AC + larger batch + cudagraphs (QUEUED, post-6.A.3)

**Hypothesis (Aram, 2026-05-23):** with selective activation checkpointing on, DDP memory headroom opens up so per-GPU batch can grow (bs=128, possibly 192). Larger batch amortizes the fixed per-step Python / launch overhead — same lever cudagraphs targets, but via a different mechanism. Combined with cudagraphs, the two wins should be at least partly additive: bigger batch reduces *frequency* of step overhead, cudagraphs reduces *cost* of each step.

**Why this is queued, not immediate:**
1. AC introduces graph breaks at checkpoint boundaries by design (the `checkpoint_wrapper` is a control-flow construct). Combining AC with `triton.cudagraphs=True` is a different compatibility question than 6.A.3. We want the 6.A.3 result first so we know whether the cudagraph path is fixable at all.
2. bs=128 OOM under FSDP2 (Phase 5) was unexplained. Under DDP without AC at bs=96 we saw 25.8 GB / 80 GB; bs=128 linear estimate is ~34 GB. AC should bring that down substantially. But cudagraph workspaces add memory too, so the prediction is non-trivial.
3. AC raises HFU but not MFU on its own; the win here is *only* if larger batch is unlocked. The experiment design must therefore be a 2×2 matrix at minimum: {AC on/off} × {cudagraphs on/off}, all at the new larger batch, with the AC-off legs being the OOM/throughput controls.

**Experiment matrix (post-6.A.3 — 6.A.3 cleared, queue open):**

| ID      | bs  | AC  | cudagraphs | Status | Job | Purpose |
|---------|-----|-----|------------|--------|-----|---------|
| 6.A.4.a | 128 | off | true       | **DONE — WIN** (53708) | 53708 | bs=128: 2,394 img/s, 13.70% MFU, 34.1 GB peak. +19.2% img/s vs 53681. No OOM. Loss matches baseline. |
| 6.A.4.d | 192 | off | true       | **OOM** (53739) | 53739 | Predicted ~51 GB linear; actual exceeded 80 GB. cudagraph workspace scales non-linearly past bs=128. **AC is now the path to bs≥192**, not a deprioritized option. |
| 6.A.4.b | 128 | sel | true       | superseded by 6.A.5.c | — | (Moved into the 6.A.5 AC sweep below.) |
| 6.A.4.c | 128 | off | false      | TBD (diagnostic) | — | Isolates batch-amortization win from cudagraphs win. Only worth running if we need to attribute deltas precisely. |

Acceptance: 6.A.4.a clears job 53681 (2,009 img/s) by ≥ 10 % — **cleared at +19.2%**.

---

### 6.A.5 — Activation-checkpointing sweep (RUNNING, 2026-05-23)

**Hypothesis:** AC's value at this point is no longer about fitting bs=128 (already fits with cudagraphs, 34.1 GB) — it's about fitting **bs≥192**, which 53739 showed needs memory headroom we don't have without AC. Before scaling batch under AC, characterize the throughput cost of `sel` vs `full` at both bs=96 and bs=128.

**2x2 matrix (cudagraphs=true for all) — final:**

| ID      | bs  | AC mode | Status | Job | img/s | MFU | step ms | peak GB | Δ vs no-AC |
|---------|-----|---------|--------|-----|-------|-----|---------|---------|-----------|
| 6.A.5.a | 96  | sel     | DONE   | 53740 | 1,380 | 7.89% | 695 | 14.9 | **-31.3% img/s vs 53681** (2,009 / 11.50%); mem **-42%** (25.8 → 14.9 GB) |
| 6.A.5.b | 96  | full    | DONE (re-run) | 53999 | 1,378 | 7.89% | 689 | **9.6** | **-31.4% img/s vs 53681**; mem **-63%** (25.8 → 9.6 GB) |
| 6.A.5.c | 128 | sel     | DONE   | 53742 | 1,628 | 9.32% | 980 | 19.4 | **-32.0% img/s vs 53708** (2,394 / 13.70%); mem **-43%** (34.1 → 19.4 GB) |
| 6.A.5.d | 128 | full    | DONE (re-run) | 54000 | 1,654 | 9.46% | 874 | **12.4** | **-30.9% img/s vs 53708**; mem **-64%** (34.1 → 12.4 GB) |

**Config bug discovered (2026-05-23):** in `ac_compile_parallelize.py:205`, AC is gated by `cfg.train.checkpointing` alone. `cfg.train.checkpointing_full` only switches the *policy* (full-block `checkpoint_wrapper` vs `create_selective_checkpoint_contexts`) **after** AC is enabled. Setting `checkpointing=false, checkpointing_full=true` silently runs without AC. Correct config for full recompute: **both** flags must be `true`. Original 53741/53743 results were invalidated by this bug; re-runs 53999/54000 above used the corrected config and were confirmed by the "using selective checkpointing on backbone with full checkpointing policy" log line.

**Findings:**

1. **`full` AC dominates `sel` AC here — same throughput, ~35% less memory.** Contrary to textbook intuition. Reason: this codebase's selective save list (`mm`, `_scaled_mm`, both SDPA variants, `reduce_scatter`) covers the *expensive-to-recompute* ops but not the *bulky-to-store* activations (intermediate norms, layer-scale, residual buffers, dropout state). So selective saves the right things for recompute speed but doesn't save much memory. Full saves a lot more memory for nearly the same recompute cost.

2. **AC throughput cost is ~31% across the board, batch-independent.** This is the recompute overhead — extra forward-pass work in the backward — and it scales linearly with batch.

3. **Fullgraph + triton.cudagraphs survives BOTH AC modes.** `[COMPILE]` logs confirm all 12 backbone blocks still take the `fullgraph=True, dynamic=False, triton.cudagraphs=True` branch. The compatibility risk we flagged earlier did not materialize.

4. **Steady-state (post-warmup) deltas are narrower than running averages.** Late-iter instant MFU at bs=128 full hits 13.6%, nearly matching 53708's running 13.70%. Running averages are pulled down by the first ~50 iters of dynamic-shape compile + cudagraph capture.

**Batch-scaling extrapolation using AC=full (the better variant):**

Per-batch slope from 53999→54000: `(1654 − 1378) / 32 = 8.625 img/s per unit batch`. Memory slope: `(12.4 − 9.6) / 32 = 0.0875 GB/batch-unit`.

| bs | Predicted img/s | Predicted mem GB | vs 53708 (2,394 img/s) |
|---|---|---|---|
| 160 | ~1,930 | ~16 | -19% (loses) |
| 192 | ~2,206 | ~19 | -8% (loses) |
| **224** | **~2,482** | **~22** | **+4% (first break-even)** |
| **256** | **~2,758** | **~25** | **+15%** |
| 384 | ~3,862 | ~37 | +61% (linearity likely breaks) |
| 512 | ~4,966 | ~50 | +107% (highly speculative) |

**Why linearity may not hold to bs=512:** at bs=192 *without* AC, cudagraph workspace bloat alone OOM'd at >80 GB (job 53739, well above the 51 GB linear prediction). With AC reducing activation memory by 64% the headroom is better, but the cudagraph-tree growth pattern is non-linear and not yet characterized at these batches. Empirical testing required, no clean prediction.

**Decision:**
- Stay at **bs=128 no-AC (53708)** as the production high-water mark for now: 2,394 img/s, 13.70% MFU, 34.1 GB.
- AC=sel is dominated by AC=full and dropped from future work.
- Probe AC=full at progressively larger batch with bs=192 first (Phase 6.A.6) — start small to characterize the cudagraph workspace memory slope before risking another bs=256 OOM.

**Important nuance on AC + DDP single-node (Aram, 2026-05-24):** AC's value normally shows up in regimes the satellite ViT-B fork *doesn't* live in — large models where activations dominate memory, or FSDP2/multi-node where the sharded budget makes AC the only path to a usable batch. For DDP single-node + ViT-B, AC only pays off if the batch can grow enough to amortize the ~31% throughput tax. With AC=full freeing ~64% of memory, that *might* be reachable at bs≥256 — but the math is tight and depends on linearity that hasn't been tested past bs=128.

**Realism check on the 30% MFU target (Aram, 2026-05-24):** ViT-B + DINO+iBOT + RoPE + 10 crops on a single H100 node is the *wrong shape* for 30% MFU. Published MFU for similarly-sized models with rich SSL pipelines typically sits in the 15-25% range. Hitting 30% likely requires (a) a much larger model where matmul ops dominate the FLOP budget, (b) multi-node + FSDP2 to amortize comms, or (c) static-shape heads (Phase 6.B). The lab target may be aspirational rather than achievable on this hardware/model combination.

---

### 6.A.6 — bs=192 + AC=full + cudagraphs (DONE — Outcome C, job 56131)

**Result: OOM during compile/cudagraph capture.** Signal 9 (SIGKILL) from Slurm cgroup OOM killer; AC engaged correctly per log; no training iterations completed; no `max mem` logged.

**Corrected diagnosis (supersedes the earlier cudagraph-workspace theory):**

The OOM is **Slurm cgroup OOM = host RAM**, NOT CUDA VRAM. Evidence:
- Slurm reports `oom_kill event in StepId=56131.batch` — that's the Linux cgroup OOM killer (host memory).
- No `CUDA out of memory` Python exception in stderr — PyTorch was never given a chance to raise one.
- SIGKILL came from the kernel, not a graceful CUDA failure.
- Identical crash signature to 53739 (bs=192 no-AC). **AC made zero difference because activation VRAM was never the bottleneck.**

**Why host RAM is the ceiling at bs=192:** the DataLoader keeps `num_workers=20` × `prefetch_factor=8` = 160 batches in flight per rank, times 8 ranks = 1,280 batches. Each bs=192 batch is 10 crops × 192 samples (2× 224²·5ch + 8× 96²·5ch ≈ 670 MB raw float32 before normalization). The aggregate working set scales superlinearly with batch and overruns Slurm's cgroup memory allocation. AC reduces *activation memory inside the model* — irrelevant to this bottleneck.

**Phase 6.A AC exploration is closed:**
- AC=full strictly dominates AC=sel at this scale (same throughput, ~35% less activation memory).
- But AC's value is moot on this configuration — VRAM was never the binding constraint; **host RAM is**.
- Both bs=192 attempts (with and without AC) hit the same wall.
- **bs=128 no-AC (53708) remains the production champion**: 2,394 img/s, 13.70% MFU, 34.1 GB VRAM peak.

**If we wanted to probe bs>128**, the lever would be reducing `num_workers` or `prefetch_factor` to free host RAM, accepting some risk of data-pipeline stalls. That's a non-trivial experiment (would need NVTX profiling to characterize the loader↔step overlap) and lower expected value than Phase 6.B.

**Recommended pivot: Phase 6.B (static-shape heads / iBOT).** The DINO and iBOT heads still compile via the default `module.compile()` dynamic-shape path. Late-iter MFU variance in 53708/54000 (instant MFU 12.9–17.0% at bs=128) is consistent with a heads-bound bottleneck. Static-shape heads + fullgraph would close that gap without touching the batch.

---

## Phase 6 — Final scoreboard (post-6.A)

| Phase | Run | bs | Config | img/s | MFU | Peak VRAM |
|---|---|---|---|---|---|---|
| 6.A.0 baseline | 48312 | 96  | DDP, compile=true, cg=false | 1,387 | 7.94% | 25.8 GB |
| 6.A.3 cudagraphs win | 53681 | 96  | + cudagraphs=true (+ block.py index_select fix) | 2,009 | 11.50% | ~25.8 GB |
| **6.A.4 production champ** | **53708** | **128** | **+ bs=128** | **2,394** | **13.70%** | **34.1 GB** |
| 6.A.5.b | 53999 | 96  | + AC=full | 1,378 | 7.89% | 9.6 GB |
| 6.A.5.d | 54000 | 128 | + AC=full | 1,654 | 9.46% | 12.4 GB |
| 6.A.6 OOM | 56131 | 192 | + AC=full | — | — | host-RAM OOM |
| 6.A.4.d OOM | 53739 | 192 | no AC | — | — | host-RAM OOM |

**Total Phase 6.A gain over Phase 5 baseline (job 48312):** +72.6% img/s, +5.76 pp MFU absolute (+72.5% relative). Achieved with two compiler-correctness fixes (`mark_step_begin` + `x[idx]→index_select`) and one config flip (`cudagraphs=true`) + one batch bump.
