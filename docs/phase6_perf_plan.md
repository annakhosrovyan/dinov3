# Phase 6 — DDP + `torch.compile(fullgraph=True)` Plan

| Field | Value |
|---|---|
| Branch | TBD (recommend `perf-ddp-fullgraph`, branched from current `perf-fsdp2-pipeline` head) |
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
| Per-GPU bs | 96 | `scripts/ddp_bs96_calibration.sh` |
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

**Where the dynamism originates:**

```python
# dinov3/data/collate.py:62-63
collated_masks = torch.stack(masks_list).flatten(1)        # shape (B, N) — STATIC
mask_indices_list = collated_masks.flatten().nonzero().flatten()
                                                           # shape (n_masked,) — DYNAMIC
```

- `collated_masks` itself is shape-stable: `(B, N)` where `B = global_batch * 2` global crops
  and `N = 196` patch positions (ViT-B / patch 16 / 224 px). Same every iteration.
- `mask_indices_list` is the result of `.nonzero()` on the flattened mask, so its
  length is the total number of True entries — varies stochastically per iter via
  `mask_ratio_tuple` and `mask_probability`.
- `n_masked_patches` is then stored as a scalar tensor and used at
  `dinov3/loss/ibot_patch_loss.py:115` as `loss = loss[:n_masked_patches]`.

**Downstream effects (chain of varying-shape tensors):**

1. `student_patch_tokens_masked` — gathered from student features using `mask_indices_list`
   (`dinov3/train/ssl_meta_arch.py` student forward).
2. `teacher_patch_tokens_masked` — same gather on teacher side.
3. The iBOT loss math operates on tensors of shape `(n_masked, dim)`.

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
do it on a fresh branch / fresh script, or extend `scripts/ddp_bs96_calibration.sh` with
a config override? I'll default to a fresh `scripts/ddp_bs96_cudagraphs_backbone.sh` so
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

**Recipe.** `scripts/ddp_bs96_cudagraphs_backbone.sh` — identical to
`scripts/ddp_bs96_calibration.sh` except:

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

- `dinov3/data/collate.py:62-74`: after computing `mask_indices_list`, pad it (and
  related `masks_weight`, the gathered student/teacher tensors downstream) to
  `K_max = ceil(mask_ratio_tuple[1] * N) * n_samples_masked`. Store `n_real_masked` so
  the loss path can compute the correct denominator.
- `dinov3/loss/ibot_patch_loss.py:97-117`: replace `loss[:n_masked_patches]` with a
  multiplication by a 0/1 weight, and use `n_real_masked` for normalization.
- A flag `train.static_ibot_shapes=true` to gate it (we want to be able to A/B against
  the stochastic path).

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

- `docs/torch-compile-dinov3-phase5-2026-05-21.md` — code walkthrough of `train.compile=true`.
- `docs/phase5_perf_plan.md` — closed predecessor; preserved for nsys / NCCL / OOM history.
- `dinov3/fsdp/ac_compile_parallelize.py:65–76` — `wrap_compile_block` (the fullgraph / CUDA-graph branch lives here).
- `dinov3/fsdp/ac_compile_parallelize.py:281–322` — DDP wrapping (`gradient_as_bucket_view`, `static_graph`).
- `dinov3/configs/ssl_default_config.yaml:80–82` — `compile`, `compile_mode`, `cudagraphs` defaults.
- `dinov3/data/collate.py:62–74` — `mask_indices_list` dynamic shape source.
- `dinov3/loss/ibot_patch_loss.py:96–117` — `loss[:n_masked_patches]` dynamic slice.
- `scripts/ddp_bs96_calibration.sh` — the recipe job 48312 ran.
- `/mnt/weka/adovlatyan/logs/ddp-bs96-calib-48312.out` — baseline log to beat.
