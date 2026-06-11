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
| 6.A.4.d | 192 | off | true       | **OOM** (53739) | 53739 | ~~Predicted ~51 GB linear; actual exceeded 80 GB; cudagraph workspace scales non-linearly.~~ **DISPROVEN by §6.B.6 (job 64979): bs=192 VRAM is only 56.4 GB — this was a host-RAM cgroup OOM, not a CUDA OOM.** VRAM scales linearly; the bottleneck is the DataLoader host footprint. |
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

---

## Phase 6.B — Sustainability Soaks

**Why 6.B exists.** Phase 6.A was throughput screening: 1000-iter runs to find the fastest config. What it didn't answer: *can the champion survive real training?* The FSDP2 bs=128 precedent (passed a 500-iter memprofile at 36.3 GB; OOM'd in a real long run) proves that short screens miss allocator fragmentation, late eval/checkpoint memory spikes, and host-RAM growth over thousands of iterations.

Phase 6.B validates that the Phase 6.A winner is production-viable, and — as a secondary objective — explores whether bs=192 becomes viable once the host-RAM bottleneck is addressed.

### What makes a soak different from a screen

| Property | 6.A screen | 6.B soak |
|---|---|---|
| Duration | 1000 iters (~8 min) | 4000 iters (~38 min) |
| Checkpoint saves | 2 | 8 (period=500) |
| Eval runs | 1 | 4 (period=1000) |
| VRAM monitoring | peak from log | `[MEMPROFILE]` at every phase boundary |
| Host RAM monitoring | none | Shell sidecar → JSONL every 10s |
| Fragmentation tracking | none | `[MEMFRAG]` every 50 iters |
| Loader config | num_workers=20, pf=8 | same (or reduced for 6.B.3) |

The key instrument is the combination of `DINOV3_MEMORY_PROFILE=1` (fires `[MEMPROFILE]` markers at `pre_checkpoint`, `checkpoint_complete`, `pre_eval`, `eval_complete`) and a shell background process that samples `/proc/meminfo` + `nvidia-smi` every 10s to a JSONL file. Together they answer: does peak VRAM grow across eval/checkpoint events (fragmentation), and does host RAM grow monotonically (loader leak)?

### Experiment ladder

**6.B.1** `scripts/soak/ddp_bs128_cg_soak.sh` — **MUST RUN FIRST**

| Property | Value |
|---|---|
| Config | DDP, bs=128, cudagraphs=true, AC=off |
| Script | `scripts/soak/ddp_bs128_cg_soak.sh` |
| Comparator | job 53708 (screen): 2394 img/s, 13.70% MFU, 34.1 GB |
| Iters | 4000 |
| Pass criterion | No OOM; max_reserved stable across 8 checkpoint events |
| Fail criterion | VRAM creep > 2 GB over run, or any OOM |
| Status | TBD |

**6.B.2** `scripts/soak/ddp_bs128_cg_ac_full_soak.sh`

| Property | Value |
|---|---|
| Config | DDP, bs=128, cudagraphs=true, AC=full |
| Script | `scripts/soak/ddp_bs128_cg_ac_full_soak.sh` |
| Comparator | job 54000 (screen): 1654 img/s, 9.46% MFU, 12.4 GB |
| Iters | 4000 |
| Purpose | Validate AC=full as the safe-mode config (12 GB vs 34 GB) |
| Status | TBD |

**6.B.3** `scripts/soak/ddp_bs192_cg_soak.sh` — *exploratory, run after 6.B.1*

| Property | Value |
|---|---|
| Config | DDP, bs=192, cudagraphs=true, AC=off |
| Loader | **num_workers=12, prefetch_factor=4** (reduced from 20/8 — see below) |
| Script | `scripts/soak/ddp_bs192_cg_soak.sh` |
| Iters | 2000 (exploratory; promote to 4k if clean) |
| Purpose | Find whether bs=192 is viable once host-RAM bottleneck removed |
| Status | TBD |

**Why reduce the loader for 6.B.3:** the 6.A.4.d and 6.A.6 OOMs were Slurm cgroup (host RAM), not CUDA. The host RAM working set is `8 ranks × num_workers × prefetch_factor × batch_size × per-sample bytes`. At bs=192 with 20×8=160 batches/rank, this exceeds the cgroup allocation. Reducing to 12×4=48 batches/rank (3.3× less) should clear the ceiling. The risk: if the loader can't feed 8 GPUs at bs=192, `data_time` will grow relative to `step_time` — watch the data_time column in the output. If data stalls appear, the loader-reduction fix costs more throughput than the batch bump gains.

### Decision tree post-6.B.1

```
6.B.1 passes (VRAM stable, no OOM)
  → champion confirmed for production
  → run 6.B.2 in parallel to validate AC=full safe-mode path
  → then run 6.B.3 (bs=192 exploratory)

6.B.1 OOM (VRAM)
  → cudagraph workspace grows late; need to characterize with nsys
  → Phase 6.C: nsys trace at iters 500, 2000, 4000 to catch late growth

6.B.1 OOM (host RAM, same signature as 6.A)
  → reduce prefetch_factor for bs=128 too (12×4), then retest
  → note: this trades throughput; quantify with data_time metric
```

### Results

| Exp | Job | Config | Wall img/s | MFU (GPU) | Peak VRAM | cgroup peak | Verdict |
|---|---|---|---|---|---|---|---|
| 6.B.1 | 57299 | bs=128, no-AC, 20w/8pf, --mem=512G | ~2,020 (partial, MetricLogger) | 11.5% avg | 36.3 GB (stable) | unknown* | partial (1300/4000 iters), rank-6 crash |
| 6.B.2 attempt 1 | 57799 | bs=128, AC=full, 20w/8pf, --mem=380G | — | — | — | — | OOM during compile (cgroup strict) |
| 6.B.2 attempt 2 | 58188 | bs=128, AC=full, **12w/4pf**, --mem=300G | **1,158 wall** (2,558 GPU-only) | 14.6% GPU-only | 14.6 GB (AC=full) | **300.0 GB (= limit)** | loader-bound, rank-6 crash at iter ~1310 (gpu03) |
| 6.B.2.b | 59273 | bs=128, AC=full, **16w/4pf**, --mem=430G | ~940–980 wall (~2,070 GPU-only) | — | 12.7 GB (AC=full) | **430.0 GB (= limit)** | loader-bound, rank-6 crash at iter ~1310 (gpu07) |
| 6.B.2.c | 59274 | bs=128, AC=full, **16w/8pf**, --mem=540G | **~1,110–1,150 wall** (~2,070 GPU-only) | — | 12.7 GB (AC=full) | **539.8 GB (= limit)** | data fully hidden (data_time≈0), rank-6 crash at iter ~1310 (gpu07) |
| 6.B.6 | 64978 | bs=160, no-AC, **16w/8pf** (128/rank) | — (OOM in warmup) | — | 44.8 GB | **512 GB (= cap)** | **host-RAM OOM ~5 min, pre-iter-50.** See §6.B.6 |
| 6.B.6 | 64979 | bs=192, no-AC, **12w/8pf** (96/rank) | ~1,260 wall (~3,500 GPU-only) | ~20% GPU-only | **56.4 GB** | **512 GB (= cap)** | ran 1260 iters then **host-RAM OOM**; loader-starved (data≫step). See §6.B.6 |
| **6.B.7** | **67153** | bs=160, no-AC, **24w/8pf**, **--mem=1500G --cpus=192** | **~1,810 wall** (3,410 GPU-only) | **22.0% GPU-only** | **48.3 GB** | anon peak **188.6 GB** (cap 1500) | **✅ FULL 2000/2000, 0 OOM/retry.** Wall flat vs bs=128. See §6.B.7 |
| **6.B.7** | **67154** | bs=192, no-AC, **24w/8pf**, **--mem=1500G --cpus=192** | **~1,830 wall** (3,740 GPU-only) | **22.6% GPU-only** | **56.6 GB** | anon peak **218.6 GB** (cap 1500) | **✅ FULL 2000/2000, 0 OOM/retry.** Wall flat vs bs=128. See §6.B.7 |
| 6.B.5 (fix validation, interim) | 60590 | bs=128, no-AC, 20w/8pf, **save_error_path fix applied** | ~1,940 avg / ~2,000–2,600 steady | 11.1% avg | 37.9 GB reserved (flat iter 49→3770) | ~570 GB host | crash GONE — reached iter 3770, killed by TIME LIMIT (walltime too short). See §6.B.5 |
| **6.B.5 (fix validation, FULL)** | **60959** | bs=128, no-AC, 20w/8pf, fix applied, **--time=4h** | **1,830 run-avg / 2,055 late-median (max 2,658)** | **10.5% avg / 11.8% late** | **36.3 GB reserved (dead-flat iter 49→3999, frag 0.040, 0 OOM/retry)** | 613 GB peak, **+15 GB/hr drift** | **✅ FULL 4000/4000 — fix validated end-to-end. See §6.B.5** |

*6.B.1 memlog read node-wide `/proc/meminfo`, not cgroup memory — see 6.B.2 memlog fix.

> **Headline (2026-05-27):** every cgroup-aware soak pegs `memory.current` at *exactly* its
> `--mem` (300/300, 430/430, 540/540). The cgroup number is dominated by **reclaimable page
> cache that expands to fill the limit** — it is NOT a measurement of the job's true memory
> need. The "1.75 GB/worker, 300 GB exact" model from 58188 was an artifact and is retracted
> below. Separately, all four bs=128 soaks (57299, 58188, 59273, 59274) died at **rank 6,
> iter ~1310, across two nodes (gpu03 + gpu07)** — a data-deterministic crash that is now the
> #1 blocker: no bs=128 soak has ever passed 1310/4000 iters.
>
> **UPDATE (2026-05-28): the rank-6 crash is SOLVED and fixed.** Capture job 59847 (with
> `@record` + per-rank logs) caught the real traceback: a corrupt NAIP PNG triggers the
> dataset's bad-tile error logger, which tried to write *into akhosrovyan's read-only Weka
> dir* → `PermissionError` killed the worker → killed rank 6. Not an iBOT/CUDA-graph issue.
> Fixed by redirecting the error log to a writable dir + a non-fatal guard. See §6.B.4 below;
> full write-up in `scripts/debug/rank6_root_cause.md`.
>
> **VALIDATED END-TO-END (2026-05-30): job 60959 completed the FULL 4000/4000 iterations**
> (4h walltime). The single `libpng error: Read Error` event self-heals instead of killing
> rank 6; VRAM is dead-flat at **36.3 GB reserved** across all 8 checkpoints + 4 evals
> (`frag=0.040`, `alloc_retries=0`, `num_ooms=0` the entire run). The 6.B sustainability
> question is now closed: **the champion config is production-sustainable.** The interim run
> 60590 reached iter 3770 but was killed by a too-short walltime; 60959 supersedes it. See §6.B.5.
>
> **Throughput correction (Codex-reviewed):** sustained throughput is **~1,830 img/s run-avg
> (–23.5% vs the 53708 1000-iter screening "champion" of 2,394)**; only the late-stage median
> (~2,055 img/s, iter 3000–3999) approaches that range. The screening number was a favorable
> short-burst window — the honest long-run figure is lower. Throughput *improves* over the run
> (early ~1,636 → late ~2,055 img/s) as CUDA graphs stabilize and caches warm; it does **not**
> thermally throttle.

---

### 6.B.6 — bs=160 / bs=192 memory-validation probe (DONE, jobs 64978/64979) — both host-RAM OOM

**Question (Aram, 2026-06-05):** the bs=128 champion peaks at only **36 GB on an 80 GB H100** —
can the idle ~44 GB buy a bigger batch? `scripts/soak/ddp_memval_soak.sh` (2000 iters, 8 ckpt +
4 eval events = same event count as the 4000-iter soak 60959; **cgroup-aware sidecar** that logs
the job's own `memory.stat` `anon`/`file` + per-GPU VRAM, fixing the node-`/proc/meminfo` confound
of 6.B.1 and the page-cache-inflated `memory.current` of 6.B.2) probed bs=160 and bs=192 with
loaders reduced to keep the in-flight buffer pool near the validated bs=128 point.

| Job | bs | Loader (batches/rank) | Peak VRAM | cgroup current peak | Outcome |
|---|---|---|---|---|---|
| 64978 | 160 | 16w/8pf (128) | **44.8 GB** | 512 GB (= cap) | **host-RAM OOM in warmup** (~5 min, pre-iter-50) |
| 64979 | 192 | 12w/8pf (96)  | **56.4 GB** | 512 GB (= cap) | ran 1260 iters, **host-RAM OOM** (3 oom_kill events) |

**Three things are now settled:**

1. **The bs>128 ceiling is host RAM, definitively — not VRAM, and not cudagraph workspace.**
   These are the FIRST real VRAM numbers above bs=128 (the prior bs=192 runs OOM'd *before*
   PyTorch reported VRAM). bs=160 → 44.8 GB, bs=192 → 56.4 GB, both with **24–35 GB of headroom**
   on the 80 GB card. VRAM scales ~linearly from bs=128's 34 GB (≈0.28 GB per sample-of-batch),
   `fragmentation_ratio=0.015`, `num_ooms=0`, `alloc_retries=0` throughout. **The old "cudagraph
   workspace exceeds 80 GB / scales non-linearly" theory (6.A.4.d row, job 53739) is disproven
   and retired** — the corrected §6.A.6 host-RAM diagnosis is now confirmed with measured VRAM.

2. **Loader in-flight working set, not batch size, sets the host-RAM ceiling.** Counter-
   intuitively the *bigger* batch lasted longer: bs=160 @ 128/rank OOM'd during loader warmup
   (all prefetch buffers fill at once) while bs=192 @ 96/rank survived 1260 iters. The ordering
   tracks `w×pf×batch` (128×160=20480 vs 96×192=18432), confirming the prefetch buffer pool — not
   the model — fills the 512 GB cgroup. (Correction to the "hold w×pf×batch ≈ const" heuristic:
   even 18432 OOMs eventually; the validated bs=128 point at 20×8×128=20480 fits only because
   no-AC bs=128 also has the slowest host-RAM drift, and even it ran at +15 GB/hr in 60959.)

3. **Where bs=192 fit memory, it was loader-starved — a net throughput LOSS.** GPU-compute
   throughput was excellent (CUDA-event ~3,500–4,000 img/s, MFU ~20% — far above bs=128's 13.7%),
   but `data_time` ≈ 0.7 s vs `step_time` ≈ 0.4 s even at steady state: the 12-worker pool cannot
   decode fast enough at bs=192. **Wall throughput collapsed to ~1,260 img/s — below the bs=128
   champion's honest ~1,830 run-avg.** With 64 CPUs / 8 ranks = 8 decode cores/rank, you cannot
   both (a) cut workers enough to fit host RAM at bs≥160 and (b) keep 8 H100s fed.

**Conclusion (under the 512 GB / 64 CPU constraint): bs=128 @ 20w/8pf stays the champion.**
Bigger batch is VRAM-feasible but host-RAM-bound *at this provisioning*, and every loader
reduction that fits 512 GB starves the GPU worse than the batch bump helps. **This conclusion is
constraint-bound, not fundamental** — see §6.B.7, which re-ran bs=160/192 at **1500 GB / 192 CPU
/ 24w/8pf** and cleared the OOM entirely (full 2000/2000, 0 OOM, VRAM 48/57 GB). The host-RAM
ceiling was a provisioning artifact. **What §6.B.7 then revealed is the *real* limit: wall
throughput is flat across bs=128/160/192 (~1,820 img/s) because the iteration is gated by a
batch-independent per-iter CPU overhead, not by compute, data, or memory.** So the lever for
higher throughput is **not** batch size — it is the compute-side **static-shape heads / fullgraph**
work (which cuts per-iter dispatch/launch overhead). This supersedes the deprioritized 6.B.3 plan.

> **Instrumentation note (refines the §6.B.5 retraction).** The new cgroup sidecar confirmed
> `memory.current` pegs at the 512 GB cap in BOTH runs while cgroup `anon` peaked at only ~135 GB.
> The ~380 GB remainder is page cache + DataLoader **shared-memory IPC buffers** (non-reclaimable
> while a batch is referenced in flight). So `memory.current` is page-cache-inflated *at rest*
> (the §6.B.5 point stands), but **under the loader's allocation burst that pool becomes
> unreclaimable and triggers the OOM-kill** — total cgroup occupancy approaching cap, not `anon`
> alone, is the OOM predictor. To lower it, shrink `w×pf`; to raise the ceiling without starving,
> you must shrink per-batch host bytes (crops/dtype/caching), not just reshuffle workers.

---

### 6.B.7 — bs=160 / bs=192 re-test with proper provisioning (DONE, jobs 67153/67154) — both PASS, but wall throughput is flat

**Question (Aram, 2026-06-05):** §6.B.6 capped host RAM at 512 GB (= 64 CPU × DefMemPerCPU 8 GB,
no `--mem` set) and gated decode at 8 cores/rank. But a DGX H100 node has ~2 TB RAM and 224 CPUs.
Re-run bs=160/192 with the loader **scaled up** (not down) and host RAM no longer the ceiling:
`--mem=1500G --cpus-per-task=192`, **24w/8pf** (≈ 24 decode cores/rank). Same instrument as §6.B.6
(cgroup-aware sidecar + `[MEMPROFILE]` markers), same 2000-iter / 8-ckpt / 4-eval schedule.

All metrics below are derived from the **median per-iter** `time:` (wall) and `step_time_ms`
(CUDA-event) over iters ≥50, so they self-rederive: wall img/s = `global_batch / median(time)`,
GPU-only = `global_batch / median(step_time)`, MFU from GPU-only (226.4 GMACs/img, 989 TFLOPS).

| Job | bs | Loader | Outcome | Peak VRAM (max/min GPU) | Peak cgroup anon | Wall img/s (median) | GPU-only img/s | MFU (GPU) |
|---|---|---|---|---|---|---|---|---|
| 67153 | 160 | 24w/8pf | **✅ FULL 2000/2000, 0 OOM, 0 alloc_retries** | 48.3 / 46.0 GB | 188.6 GB (cap 1500) | **~1,730** (1280/0.741s) | ~3,500 (1280/0.365s) | **~20%** |
| 67154 | 192 | 24w/8pf | **✅ FULL 2000/2000, 0 OOM, 0 alloc_retries** | 56.6 / 54.3 GB | 218.6 GB (cap 1500) | **~1,810** (1536/0.849s) | ~3,720 (1536/0.413s) | **~21%** |

bs=128 references (both torch **2.6**, NOT directly comparable to each other): job 53708 = 1000-iter
*screen*, 2,394 img/s / 13.7% MFU; job 60959 = 4000-iter *soak*, ~1,830 wall run-avg / **~11.8% late
MFU**. The right same-kind comparison for these soaks is **60959's ~11.8%**; job 67639 (bs=128 on
torch 2.10, same 2000-iter soak) is in flight to remove the torch-version confound entirely.

**What is now settled:**

1. **bs>128 passed a 2000-iter / 8-ckpt / 4-eval memory soak under 1500 GB / 192 CPU — the §6.B.6
   host-RAM OOM was a provisioning artifact at 512 GB / 64 CPU, removed by proper provisioning.**
   (Deliberately *not* "proven safe": the FSDP2 bs=128 precedent OOM'd only in a real *long* run,
   so a 2000-iter soak lowers risk but is not a full-duration proof — see caveats.) Both ran clean
   to 2000/2000: `num_ooms=0`, `alloc_retries=0`, `fragmentation_ratio≈0.015`, VRAM dead-flat across
   all 8 ckpt + 4 eval events (48/57 GB peak, 24–34 GB of headroom on the 80 GB card). Host RAM was
   comfortable *in this soak*: cgroup `anon` peaked at only **188 GB (bs=160) / 219 GB (bs=192)**
   with no upward trend over 2000 iters (the verdict's "projected anon" is a degenerate
   negative-slope fit — ignore that number; the signal is peak ≈ 200 GB, ~1300 GB of margin). Note
   the *prior* OOM mechanism was page-cache + shmem IPC buffers, not `anon` — so low `anon` lowers
   but does not eliminate long-run memory risk. The 24-worker pool keeps the GPU fed: `data_time`
   median **0.8 ms (bs=160) / 1.3 ms (bs=192)** — the loader is no longer the gate.

2. **But wall throughput is ~FLAT across bs=160 / 192 (~1,730 / ~1,810 img/s) and near the bs=128
   soak's ~1,830.** This is the load-bearing correction. The verdict's `images_per_sec` (~3,500–3,720)
   and `mfu` (~20–21%) are **CUDA-event compute-only** metrics (`global_batch / step_time_ms`); they
   rise with batch because the GPU is genuinely better utilized per GPU-busy-ms at large batch. But
   the **wall** clock (`time:` per iter, the models/hour metric) is flat: median-derived **~1,730
   (bs=160) / ~1,810 (bs=192)**, both ≈ the bs=128 60959 soak's **~1,830 run-avg** (cross-torch, so
   indicative not exact — 67639 will pin it). Enlarging the batch does **not** increase models/hour.

3. **There is a large per-iter RESIDUAL of ~0.35–0.43 s (`wall − step_time − data_time`) that scales
   WITH batch — but it is a residual, not an attributed phase.** Decomposing per logged iteration
   (medians over iters ≥50): bs=160 wall 0.741 s = step **0.365 s** + residual **0.349 s** + data
   ~0.001 s; bs=192 wall 0.849 s = step **0.413 s** + residual **0.434 s** + data ~0.001 s.
   **Caveat on the method (per Codex GPT-5.5 review):** subtracting a CUDA-*event* `step_time` from a
   wall `time:` does **not** cleanly isolate "non-GPU" work — the residual can include GPU work not
   bracketed by the events, the `step_end_event.synchronize()` (outside the event interval but inside
   wall), CPU running ahead on async launches, and H2D transfer whose placement is ambiguous. So the
   only *proven* statements are: (a) the residual is large (~half of wall), (b) it is **not**
   data-loading (`data_time≈1 ms`), and (c) it **grows with batch** (0.349→0.434 s, +24% for +20%
   batch), which is why per-image wall barely moves (0.579→0.553 ms/img) and wall img/s stays flat
   while compute-MFU rises. **Hypothesis (NOT yet proven):** the residual is dominated by CPU-side
   per-iter dispatch + syncs (grad-clip `.item()`, metric all-reduce, `synchronize()`, schedule,
   periodic `gc`), which static-shape heads + end-to-end fullgraph capture would cut. **This must be
   confirmed by an nsys run** using the NVTX ranges already wired in train.py (`schedule_update`,
   `forward_backward`, `grad_clip`, `allreduce_metrics`, `optimizer_step`, `ema_update`) before any
   "recoverable ceiling" is claimed — the compute-only ~3,500–3,720 img/s is an *upper bound* the
   wall could approach only if the residual proves removable without changing GPU/collective time.

**Caveats / action items:**
- **Torch-version confound (biggest):** the bs=160/192 compute-MFU (~20–21%) is on torch **2.10**;
  the bs=128 soak reference (60959, ~11.8% late) is on torch **2.6** (~10% faster between versions).
  Do **not** attribute the full ~12%→~21% compute-MFU rise to batch size — part is the torch bump,
  part is genuine large-batch tensor-core efficiency. Job 67639 (bs=128 on 2.10, same soak) is the
  control that splits these. The **wall**-throughput comparison (flat) is the robust takeaway and
  does not depend on resolving this.
- **2000 iters ≠ full run:** this soak (8 ckpt + 4 eval, ~30 min) is not a full-duration memory
  proof. The FSDP2 bs=128 precedent OOM'd only in a real long run. Treat bs>128 as
  "memory-soak-passed under 1500G/192CPU," not "OOM-safe."
- **Convergence unvalidated:** bs=192 → global batch **1536** (vs 1024 at bs=128). LR auto-scaling
  (`apply_scaling_rules_to_cfg`) handles the scalar, but large-batch convergence/quality is a
  research call for Anna, not a throughput fact. Frame bs>128 as a **throughput-neutral,
  memory-soak-passed candidate pending convergence sign-off**, not a drop-in champion.
- **`run_ddp.sh` is load-bearing:** if bs>128 is ever used in production, the run script **must**
  carry `--mem≈1500G` and `--cpus-per-task=192` (24w/8pf) — otherwise it reproduces the §6.B.6
  512 GB OOM (64978/64979) exactly. The current `run_ddp.sh` is bs=128 and does not need this;
  flag it before any bs bump.

**Conclusion:** bs=128 remains the production recommendation **on throughput grounds** (bs>128
buys no models/hour in this soak), now for a *better-understood* reason: not "bigger batch OOMs"
(it doesn't, with 1500 GB) but "bigger batch doesn't improve wall throughput." §6.B.6 was right
operationally but wrong on mechanism (it blamed loader-starvation under a 512 GB cap); §6.B.7
shows the binding factor is a large per-iter residual independent of memory or loader. The leading
**hypothesis** is that the residual is CPU-side dispatch/sync overhead, pointing to compute-side
fullgraph as the next lever — **but that attribution is unproven and must be confirmed by an nsys
profile before it drives the next experiment.**

---

### 6.B.2 attempt 2 — what we actually learned about loader sizing (job 58188)

**Cgroup memory truth (cgroup-aware memlog, fixed in this run):**

> **⚠️ The per-worker memory breakdown below was RETRACTED on 2026-05-27** after 16w/4pf and
> 16w/8pf also pegged at their (different) `--mem` limits. cgroup `memory.current` is
> page-cache-dominated and fills whatever limit you set; it does not reveal a fixed per-worker
> anon cost. See "RETRACTED: the corrected loader-sizing model" below. The `--mem`-peak fact
> (300.0 GB = limit) is real; the inferred decomposition into per-worker / per-batch terms is not.

| Metric | Value |
|---|---|
| `--mem` request (cgroup limit) | 300 GB |
| **Actual cgroup peak** | **300.0 GB (307,183 MB)** — pinned to the limit (page-cache fill) |
| ~~Per-worker overhead (inferred)~~ | ~~~2 GB resident per Python worker process~~ — retracted |
| ~~Per-batch buffers (inferred)~~ | ~~~46 GB~~ — retracted |
| ~~Worker process overhead (inferred)~~ | ~~~192 GB~~ — retracted |

**Throughput truth (GPU was idle 61% of the wall time):**

| Metric | p10 | p50 | p90 |
|---|---|---|---|
| step_time (GPU event, ms) | 323 | 346 | 382 |
| data_time (s) | 0.001 | 0.099 | 0.297 |
| wall iter_time (s) | 0.644 | 0.884 | 1.036 |
| data / iter ratio | 0.1% | **10.8%** | **34.1%** |

At p50 the GPU is doing 346 ms of compute and the iter takes 884 ms — the GPU is *idle 538 ms* (61%) waiting for the loader, NCCL, GC, or framework overhead. p90 data ratio of 34% says the loader is the visible cause for a large fraction of iters.

**Wall-clock throughput: 1,158 img/s. MetricLogger reported 2,558.** The MetricLogger field reports `1024 / step_time_ms_only`, i.e. *what throughput would be if the GPU never waited.* It overstates real throughput by ~2× under loader stall conditions.

This invalidates earlier MFU comparisons that relied on MetricLogger numbers across different loader configs — they were comparing peak GPU rates, not delivered rates.

### RETRACTED: "the corrected loader-sizing model" (58188)

The 58188 write-up claimed a fixed `~2 GB × workers × 8 ranks` anonymous-memory cost and
concluded "20w/8pf is throughput-optimal, the cgroup is just too small." **The 16w/4pf and
16w/8pf runs disprove the memory half of that.**

**What actually happens to cgroup memory.** Three runs, three different `--mem` limits, each
peaks at *exactly* its limit:

| Job | Config | `--mem` | cgroup peak | OOM? |
|---|---|---|---|---|
| 58188 | 12w/4pf | 300 GB | 300.0 GB | no |
| 59273 | 16w/4pf | 430 GB | 430.0 GB | no |
| 59274 | 16w/8pf | 540 GB | 539.8 GB | no |
| 57799 | 20w/8pf | 380 GB | — | **yes, during compile warmup** |

If the job had a fixed anonymous footprint (e.g. 300 GB), the 430 GB and 540 GB runs would
have peaked near ~300 GB, not at their limits. They peaked at the limit because
`cgroup.memory.current` counts **reclaimable file-backed page cache** (the HDF5/tile reads
off Weka), and page cache expands to fill whatever room you give it, then gets reclaimed
under pressure instead of triggering OOM. So the per-worker formula was reverse-fit from one
coincidence (300 GB happened to equal the limit) and is **retracted**.

**What we genuinely know about the true (anonymous, unreclaimable) floor:**

- 12w/4pf anon < 300 GB · 16w/4pf anon < 430 GB · 16w/8pf anon < 540 GB (none OOM'd)
- 20w/8pf anon > 380 GB (OOM'd during compile warmup, before any iter)

That's all the data constrains. The 16w configs are unbounded *below* — we never probed how
low they can go. **To find the real floor we need `memory.stat` (anon vs file split) or a
step-down `--mem` sweep** (e.g. 16w/4pf at 250G, 200G, 150G until it OOMs). Until then, do
not quote a specific GB requirement for any 16w config — the old "16w/4pf needs ~387 GB"
prediction was the same artifact and is also retracted.

> **`sacct MaxRSS` is unusable here.** It reported 883 GB (59273) and 988 GB (59274) — both
> far above the 430/540 GB cgroup limits — because it sums per-task RSS and double-counts
> copy-on-write fork pages and shared mmaps across 130+ worker processes. Use cgroup
> `memory.current` (or `memory.stat`), never sacct MaxRSS, for this accounting.

### What the loader sweep DID establish (throughput)

Caveat: every run crashed at iter ~1310, so these are short steady-state windows
(iters ~1000–1300) and the cross-config orderings are noisy. The robust signals:

| Config | data_time (loader stall) | Wall img/s (p50) | GPU-only img/s |
|---|---|---|---|
| 12w/4pf (58188) | visible (~10–30%+) | ~1,158 | 2,558 |
| 16w/4pf (59273) | **high — loader-bound** | ~940–980 | ~2,070 |
| 16w/8pf (59274) | **≈0 (p50 ~2 ms) — fully hidden** | ~1,110–1,150 | ~2,070 |

1. **Prefetch depth, not worker count, is what hid the loader.** Both 16w runs have the same
   worker count; only the pf=8 run drove data_time to ~0. At 16 workers, pf=4 is too shallow
   (queue drains, GPU stalls); pf=8 keeps it full. This is the opposite of the retracted
   "worker count is the binding knob" claim.
2. **AC=full is compute-capped near ~1,110–1,150 wall img/s at bs=128.** In 59274 the loader
   is no longer the bottleneck (data_time≈0) yet throughput tops out there — that's the AC
   recompute tax (~31%), the price for 12.7 GB VRAM vs ~34 GB no-AC. For raw throughput the
   no-AC Phase 6.A champion (53708) remains the path; AC is the VRAM-budget mode.
3. **A large wall↔GPU gap remains even with data_time≈0.** In 59274, GPU-event step ≈430 ms
   but wall iter ≈890–920 ms — roughly 2×. With the loader excluded, this ~450 ms/iter is
   CPU-side per-iter overhead (collate/cast, H2D launch, EMA, GC, Python, cudagraph replay).
   **This is the next optimization lever once the loader is non-binding** — and MetricLogger's
   GPU-only img/s hides it entirely (overstates delivered throughput ~2×).

### Recommended next experiments

1. ~~**FIX THE RANK-6 CRASH FIRST — blocks all of Phase 6.C.**~~ **✅ DONE (2026-05-28) — see
   §6.B.4 below.** Root cause was a corrupt NAIP PNG whose bad-tile error-logger tried to write
   into a read-only dataset dir (`PermissionError`), not a data-value/HDF5 issue. Fixed in
   `dinov3/data/datasets/satlas_datasets.py`. **Next: re-run a clean bs=128 soak to validate
   it now passes iter 1310 and completes 4000 iters.**

2. **Find the real memory floor** — step-down `--mem` sweep on one config (e.g. 16w/8pf at
   450G, 400G, 350G) or add `memory.stat` anon/file logging to the sidecar. This replaces the
   retracted per-worker model with a measured number.

3. **Production loader candidate: 16w/8pf** is the only config that drove data_time to ~0.
   Once rank-6 is fixed, confirm it sustains over 4000 iters; if its true anon floor turns out
   to fit a node's free RAM, it's the AC=full production config.

4. **Phase 6.B.3 (bs=192) stays deprioritized.** Its motivation was the loader-reduction
   trick; with the loader story now resolved (prefetch depth, page-cache-bound cgroup), and
   bs=128 still unable to complete a soak, bs=192 is not worth running until rank-6 is fixed.

---

### 6.B.4 — Rank-6 / iter-~1310 crash: ROOT CAUSE + FIX (job 59847, 2026-05-28) ✅

**How it was caught.** The four prior soaks reported `error_file: <N/A>` with no traceback
because `main()` was not decorated with torchelastic's `@record`. Capture job **59847** added
`@record` + `torchrun --redirects=3 --tee=3 --log-dir` (per-rank stdout/stderr) and reproduced
the crash at iter ~1310. Rank 6's per-rank log
(`/mnt/weka/adovlatyan/logs/rank6-capture-59847-perrank/none_okz52y0h/attempt_0/6/stderr.log`)
finally held the real exception.

**Root cause — a corrupt NAIP PNG + a read-only error-log write (NOT iBOT / CUDA graphs):**

```
libpng error: Read Error
PermissionError: Caught PermissionError in DataLoader worker process 11.
  mixed_satlas_dataset.py:169   self.datasets[i][base_idx]
  satlas_datasets.py:342        self.save_error_path(tci_path)
  satlas_datasets.py:58         with open(save_path, "a") as f:
PermissionError: [Errno 13] Permission denied:
  '/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip_error_paths.txt'
```

Chain: a corrupt/truncated NAIP PNG → `cv2.imread` returns `None` (`satlas_datasets.py:338`) →
the dataset's self-heal recovery calls `save_error_path()` (`:342`) → which appended the bad
path to a log file **inside akhosrovyan's dataset directory** (`:56-59`). `adovlatyan` has
read-but-not-write access there → uncaught `PermissionError` → kills the DataLoader worker →
kills rank 6 → kills the job. The dataset *wanted* to skip the bad tile and continue; the only
fatal element was the log write into a read-only dir.

**Why deterministic (rank 6, iter ~1310, 4 jobs / 2 nodes):** observed fact. Mechanism: the SSL
loader supplies no `worker_init_fn` (`loaders.py:210,250`), so forked workers inherit the rank's
main-process numpy state seeded by `fix_random_seeds(seed+rank)` (`config.py:165,206`). The
`_sample_ok_index` `np.random.randint` walk (`satlas_datasets.py:111-114`) is thus reproducible
run-to-run and reaches the same corrupt tile at the same global step. (Fragile to fork→spawn or
torch-version changes; the crash *class* is robust, the exact iter is not.)

**Why the CPU repro (job 59814) gave a false negative:** it replayed only the outer
`ShardedInfiniteSampler` index stream and called `ds[idx]`. It did not model the NAIP internal
*ok-index pool* remapping (`__getitem__:332-333`) nor the per-worker numpy RNG state, so it
sampled different tiles and never hit the corrupt PNG. Lesson: reproducing the sampler stream ≠
reproducing the dataset's internal stochastic index selection.

**Independent review:** Codex reviewed the analysis (PARTIALLY AGREE — it confirmed the chain
from source but could not read the Weka traceback). It corrected my determinism wording (no
per-worker numpy seeding; it's fork-inheritance of the rank seed) and flagged a latent
`while True` recovery-loop DDP-deadlock risk if a whole shard is unreadable (`satlas_datasets.py`
`:179` Sen1, `:257` Sen2, `:334` NAIP). That loop-bound (Codex's "fix C") is **not yet
implemented** — tracked as a follow-up.

**The fix (commit pending):** `dinov3/data/datasets/satlas_datasets.py`, `save_error_path()` —
the method is defined **once** on the base `SatlasDataset` and inherited by Sen1/Sen2/NAIP, so a
single edit covers all three. Two changes:
- **Redirect** the bad-tile log to a writable dir (`_ERROR_LOG_DIR`, env-overridable via
  `DINOV3_ERROR_LOG_DIR`, default `/mnt/weka/adovlatyan/logs/dataset_errors`). Aram-specific and
  documented as such in the source — the dataset owner can set `DINOV3_ERROR_LOG_DIR=""` to
  restore in-place logging.
- **Guard** the write in `try/except OSError` (warn + continue) so bad-tile *logging* can never
  again be fatal to training.

Regression test: `scripts/debug/test_save_error_path_fix.py` (3/3 pass — reproduces the original
crash, proves the redirect, proves the guard survives an unwritable target). No GPU needed.

**Next:** ~~re-run a clean bs=128 soak (no-AC, 53708 recipe at 4000 iters) to confirm it now
passes iter 1310 and completes~~ — ✅ **DONE (job 60590, 2026-05-29). The crash is gone; see
§6.B.5 below.** Follow-ups: (a) implement Codex's loop-bound (C) to remove the shard-unreadable
deadlock risk; (b) the wall↔GPU ~2× gap (finding #3 above) is the next throughput lever once a
soak completes.

### 6.B.5 — Fix validation soak (jobs 60590 interim + 60959 FULL, 2026-05-29/30) ✅ CLOSED

Re-ran the 53708 recipe at 4000 iters (bs=128, no-AC, DDP, cudagraphs, 20w/8pf) with the
`save_error_path` fix in the working tree (`config.py` logged `sha: 55014c6, status: has
uncommited changes` — confirms the uncommitted fix was live). Script:
`scripts/soak/ddp_bs128_cg_soak.sh`. Logs: `/mnt/weka/adovlatyan/logs/soak-bs128-cg-60590.{out,err}`.

**Verdict: the rank-6 / iter-~1310 crash is SOLVED in a real run.**

1. **It passed the crash point.** Every prior soak (57299/58188/59273/59274) died at iter
   ~1310. This run reached **iter 3770/4000** before Slurm killed it on `TIME LIMIT` — *not* a
   fault, not an OOM, no traceback, no `[STEPDIAG]` trip.
2. **The smoking gun: `libpng error: Read Error` appears once in `60590.err` and training
   continues.** That is the *same* corrupt-NAIP-tile event class that used to raise the fatal
   `PermissionError` at `satlas_datasets.py:342→:58`. With the fix, the bad tile is logged to the
   redirected writable path (or silently skipped if even that fails), `_invalidate_index` +
   resample run, and the loop continues. This is the §6.B.4 fix executing end-to-end.
3. **Memory is rock-stable** — the FSDP2-style "short screen passes, long run OOMs" failure mode
   does **not** reproduce for DDP+cudagraphs. `[MEMFRAG] current_reserved_mb=37908` is *flat* from
   iter 49 → 3749; `fragmentation_ratio=0.032` flat; `alloc_retries=0`, `num_ooms=0` throughout.
   `[MEMPROFILE]` shows `max_reserved_mb=37908` constant across all 8 checkpoint saves and 4 eval
   runs (`steady_state` peak `max_alloc_mb=34097`). **SUCCESS CRITERION met** (no OOM,
   max_reserved non-increasing).
4. **Host RAM plateaus.** The 10s sidecar (`soak_ddp_bs128_cg-60590-memlog.jsonl`, 533 samples)
   ramps to ~560–590 GB and holds (q0.25=564, q0.5=560, q0.75=572, q1.0=577 GB of 2 TB node
   RAM) — page-cache/prefetch working set, not unbounded growth.
5. **Throughput consistent with the champion.** Running avg ~1,940 img/s over 4000 iters,
   11.1% MFU avg; individual steady iters spike to 2,000–2,600 img/s. The average is below the
   53708 1000-iter screen (2,394 img/s) as expected — a 4000-iter soak amortizes compile warmup
   over more iters but also pays 8 checkpoint saves + 4 eval runs that the short screen never hit.

**Caveat (interim run 60590) — did not *complete* 4000 iters.** Walltime was `01:30:00`; the
full run needs ~1h54m. Re-submitted at `--time=04:00:00` → job 60959 below.

---

#### FULL COMPLETION — job 60959 (2026-05-30), Codex-reviewed ✅ 6.B CLOSED

Re-ran identical config with a 4h walltime. **Completed the FULL 4000/4000 iterations**
(`Training Total time: 1:53:44`, gpu06). Logs: `/mnt/weka/adovlatyan/logs/soak-bs128-cg-60959.{out,err}`,
memlog `soak_ddp_bs128_cg-60959-memlog.jsonl`. A Codex adversarial second-opinion pass
confirmed all findings below and corrected two of my initial numbers (folded in).

1. **Crash fix validated end-to-end.** 4000/4000 iters, no crash, no OOM, no NaN, no worker
   restart, no traceback. Exactly **one** `libpng error: Read Error` in `60959.err` — it
   self-heals and training continues (the §6.B.4 fix in production). Prior soaks all died at
   iter ~1310; this passed it cleanly and ran to completion.
2. **VRAM dead-flat and sustainable.** `[MEMPROFILE] max_reserved_mb=36266` identical across
   all 8 checkpoint saves + 4 eval runs (one-time warmup spike to 36454 at iter 0, then 36266
   for the rest). `[MEMFRAG] fragmentation_ratio=0.040`, `alloc_retries=0`, `num_ooms=0` at
   every marker iter 49→3999. The FSDP2 "short screen passes / long run OOMs" failure mode
   does **not** reproduce for DDP+cudagraphs. **SUCCESS CRITERION met** (reserved
   non-increasing, zero OOM).
3. **Throughput — honest sustained number is below the screening champion.** Run-avg
   **~1,830 img/s / 10.5% MFU** (median over all steady iters ≥100). This is **–23.5% vs the
   53708 1000-iter screen (2,394 img/s / 13.70%)** — the screening "champion" was a favorable
   short-burst window, not a sustained rate. Throughput *improves* over the run as CUDA graphs
   stabilize and caches warm — it does **not** thermally throttle:

   | window | median img/s | median MFU | vs 2,394 |
   |---|---|---|---|
   | early (iter 200–1000) | 1,636 | 9.4% | –31.7% |
   | mid (iter 1500–2500) | 1,857 | 10.6% | –22.4% |
   | late (iter 3000–3999) | **2,055** | **11.8%** | –14.2% |
   | all steady (≥100) | 1,830 | 10.5% | –23.5% |

   Per-iter step_time ranges 375–575ms+. Codex's slow-iter analysis (53 iters >650ms):
   only 5 are data stalls (a NAIP cache-fill burst around iter 240–300, `data_s>0.1`); the
   other 48 are compute stalls **concentrated before iter 1000** (32/53) and nearly absent
   after iter 2000 — i.e. early CUDA-graph/thermal ramp, **not** iBOT mask-count variance
   (which would be uniform). The late-stage **~2,055 img/s is the truest sustained number.**
4. **Host RAM — not a flat plateau; a slow drift.** Peak **612.8 GB** of a ~2 TB node. The
   stable window (t+10→t+115min) regresses at **+15 GB/hr** (1.8× the 8 GB page-cache noise
   floor) — almost certainly `train.cache_dataset=true` filling the dataset cache, not an
   active leak. At this rate an 8h real run reaches ~695 GB — **safe headroom, but worth
   monitoring** on multi-hour production runs. **Open question, not a blocker.**
5. **Convergence note (not a soak-validity issue):** `koleo_loss` goes slightly negative late
   (iter 3999 instantaneous –0.117, window-avg +0.196). KoLeo is a diversity regularizer;
   negative values indicate representations spreading out — likely healthy learning, flag for
   convergence analysis on real training, not a stability concern.

**6.B verdict: CLOSED.** DDP + bs=128 + cudagraphs (no AC) is **production-sustainable** — VRAM
flat at 36.3 GB, no crash over a full 4000-iter soak with the `save_error_path` fix. The honest
sustained throughput is **~1,830 img/s run-avg / ~2,055 late-steady (10.5–11.8% MFU)**, below
the optimistic 2,394 screening number. One open watch-item: host-RAM +15 GB/hr drift
(`cache_dataset=true`) on runs ≫4000 iters.
