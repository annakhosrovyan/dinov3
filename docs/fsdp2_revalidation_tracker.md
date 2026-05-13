# FSDP2 Revalidation Experiments — Tracker

**Branch**: `perf-ddp-vs-fsdp`  
**Submitted**: 2026-04-24  
**Closed (DDP vs FSDP2 question)**: 2026-04-25

**Context**: Prior analysis showed the "DDP wins 2×" narrative was misleading — the gain came
almost entirely from batch-size scaling (bs=64 → bs=256), not from DDP vs FSDP2. At matched
bs=256, FSDP2 (23.5%) and DDP+ES (23.9%) are within 0.4 pp MFU.

**Closure**: Tim Darcet (DINOv2/v3 co-author) confirmed 2026-04-25 that both are fine at this
scale. He uses FSDP2 exclusively since 2022 with `reshard_after_forward=False` (DDP-like
communication, gather before fwd instead of after bwd, with FSDP2 state sharding). The 0.4 pp gap
we measured is from `reshard_after_forward=True` (ZeRO-3) overhead — not a fundamental DDP vs
FSDP2 difference.
See `learnings/distributed_training.md` for the full conclusion.

Experiments 1 and 2 were cancelled as a result. Experiment 3 (fragmentation soak) is retained
as an independent operational question unrelated to the DDP vs FSDP2 choice.

---

## Summary Table

| # | Experiment | Jobs | Submitted | Status | Key Result |
|---|-----------|------|-----------|--------|------------|
| 1a | FSDP2 bs=272 memprofile | 29695 | 2026-04-24 16:59 | **CANCELLED** | Not needed — see closure above |
| 1b | FSDP2 bs=288 memprofile | 29696 | 2026-04-24 16:59 | **CANCELLED** | Not needed — see closure above |
| 2a | FSDP2 bs=256 no-ES × 3 | 29697, 29698, 29699 | 2026-04-24 16:59 | **CANCELLED** | Not needed — see closure above |
| 2b | FSDP2 bs=256 +ES × 3 | 29700, 29701, 29702 | 2026-04-24 16:59 | **CANCELLED** | Not needed — see closure above |
| 3 | FSDP2 bs=128 long soak 5000 iters | 29703 | 2026-04-24 16:59 | PENDING | — |

---

## Experiment 1: FSDP2 bs=272 and bs=288 — Can FSDP2 unlock higher batch sizes?

**Hypothesis**: FSDP2 shards params+grads+optimizer state → saves ~1.3 GB/GPU vs DDP.
At bs=256, DDP uses 67.3 GB and FSDP2 uses 66 GB. The 1.3 GB difference could enable
bs=272 (~0.25 GB per extra image at bs=256 activations). DDP OOMs at bs=320;
if FSDP2 runs cleanly at bs=272–288, that's a net throughput win even with equal per-bs MFU.

**Closed outcome**: Cancelled. Even if bs=272/288 fit, that would be a batch-size/headroom
optimization, not evidence that ZeRO-3 FSDP2 is categorically better than DDP. Tim Darcet's
clarification settles the strategic question: both are fine when both fit, and FSDP2 no-release is
the right next baseline to measure.

**Scripts**: `memprofile_fsdp2.sh 272/288 false false ""`  
**Log files**: `/mnt/weka/adovlatyan/logs/memprof-fsdp2-29695.out` (bs=272)  
               `/mnt/weka/adovlatyan/logs/memprof-fsdp2-29696.out` (bs=288)

**How to check results**:
```bash
# bs=272
grep '\[MEMPROFILE\]' /mnt/weka/adovlatyan/logs/memprof-fsdp2-29695.out | grep 'rank=0'
# bs=288
grep '\[MEMPROFILE\]' /mnt/weka/adovlatyan/logs/memprof-fsdp2-29696.out | grep 'rank=0'
# Did it OOM? (no Training lines = OOM or crash)
grep 'Training \[' /mnt/weka/adovlatyan/logs/memprof-fsdp2-29695.out | head -3
grep 'Training \[' /mnt/weka/adovlatyan/logs/memprof-fsdp2-29696.out | head -3
```

**Historical decision criteria, superseded**:
- If max_reserved_mb < 78000 and training ran: bs fits → FSDP2 may be useful for extra batch headroom
- If OOM at compile/iter0: memory savings do not translate → DDP+ES remains production

| Sub-exp | Job | Status | OOM? | max_reserved_mb | Verdict |
|---------|-----|--------|------|----------------|---------|
| bs=272 | 29695 | PENDING | — | — | — |
| bs=288 | 29696 | PENDING | — | — | — |

---

## Experiment 2: FSDP2 bs=256 ±ES replicate (3× each) — How reliable is the ES penalty?

**Hypothesis**: Prior runs (9653/9654/9655) showed ES hurts FSDP2 by 7–12%. But each was
a single 100-iter run where compile-warmup contamination and thermal ramp can move MFU by
~1 pp. 3 replicates per condition gives a mean ± std and distinguishes signal from noise.

**Expected outcome**: If ES consistently hurts FSDP2 at all three replicates, the 7–12%
penalty is real. If runs are noisy and overlap, the effect is smaller than advertised.

**Scripts**:  
- No-ES: `screening_run.sh 256 false false` (default fsdp2 strategy, no ES set)  
- +ES:   `screening_fsdp2_expandseg.sh 256 false`

**Log files**:
```
/mnt/weka/adovlatyan/logs/screen-29697.out  (FSDP2 no-ES rep1)
/mnt/weka/adovlatyan/logs/screen-29698.out  (FSDP2 no-ES rep2)
/mnt/weka/adovlatyan/logs/screen-29699.out  (FSDP2 no-ES rep3)
/mnt/weka/adovlatyan/logs/fsdp2-es-29700.out  (FSDP2+ES rep1)
/mnt/weka/adovlatyan/logs/fsdp2-es-29701.out  (FSDP2+ES rep2)
/mnt/weka/adovlatyan/logs/fsdp2-es-29702.out  (FSDP2+ES rep3)
```

**How to check results** (steady-state = iters 50–99):
```bash
for f in screen-29697 screen-29698 screen-29699; do
  echo "=== $f ==="
  grep 'mfu=' /mnt/weka/adovlatyan/logs/${f}.out | tail -5
done
for f in fsdp2-es-29700 fsdp2-es-29701 fsdp2-es-29702; do
  echo "=== $f ==="
  grep 'mfu=' /mnt/weka/adovlatyan/logs/${f}.out | tail -5
done
```

| Rep | Strategy | Job | Status | MFU (iters 50–99 avg) | img/s |
|-----|----------|-----|--------|----------------------|-------|
| 1 | FSDP2 no-ES | 29697 | PENDING | — | — |
| 2 | FSDP2 no-ES | 29698 | PENDING | — | — |
| 3 | FSDP2 no-ES | 29699 | PENDING | — | — |
| 1 | FSDP2+ES | 29700 | PENDING | — | — |
| 2 | FSDP2+ES | 29701 | PENDING | — | — |
| 3 | FSDP2+ES | 29702 | PENDING | — | — |
| **mean** | FSDP2 no-ES | — | — | — | — |
| **mean** | FSDP2+ES | — | — | — | — |

---

## Experiment 3: FSDP2 bs=128 long soak — 5000 iters

**Hypothesis**: A colleague hit OOM at FSDP2 bs=128 in a real training run. Our 500-iter
fragmentation study (jobs 16750/16751) showed zero alloc_retries and flat inactive_split_mb.
The OOM could occur at longer horizons: 3750+ iters between checkpoints. This run goes to
5000 iters with 10 eval cycles and 5 checkpoint cycles to stress the allocator.

**Script**: `fsdp2_long_soak_bs128.sh`  
**Log file**: `/mnt/weka/adovlatyan/logs/fsdp2-longsoak-29703.out`  
**Expected runtime**: ~25 min training + eval/ckpt cycles ≈ 40 min total

**Untested hypotheses not covered by this run**:
- (a) Gram loss enabled — if colleague had `cfg.gram` set, adds ~172 MB extra; not tested here
- (b) Pretrained weight loading double-buffer at init — not tested here (use `pretrained_weights=path`)

**How to check results**:
```bash
# Fragmentation over time (rank=0, every 50 iters):
grep '\[MEMFRAG\]' /mnt/weka/adovlatyan/logs/fsdp2-longsoak-29703.out | grep 'rank=0' \
  | awk '{for(i=1;i<=NF;i++) if($i~/iter=|alloc_retries=|inactive_split_mb=/) printf $i " "; print ""}'

# MEMPROFILE phase markers (eval + checkpoint peaks):
grep '\[MEMPROFILE\]' /mnt/weka/adovlatyan/logs/fsdp2-longsoak-29703.out | grep 'rank=0'

# Did it OOM?
tail -30 /mnt/weka/adovlatyan/logs/fsdp2-longsoak-29703.out
```

| Metric | Job | Status | Value at iter 500 | Value at iter 2500 | Value at iter 5000 | Verdict |
|--------|-----|--------|------------------|-------------------|-------------------|---------|
| alloc_retries | 29703 | PENDING | — | — | — | — |
| inactive_split_mb | 29703 | PENDING | — | — | — | — |
| max_reserved_mb | 29703 | PENDING | — | — | — | — |

---

## Open Hypotheses Not Addressed by These Experiments

| Hypothesis | Status |
|------------|--------|
| Gram loss enabled → +172 MB OOM risk at FSDP2 bs=128 | Not tested — submit `memprofile_fsdp2.sh 128 false false <pretrained_path>` with Gram enabled if reproducible |
| Pretrained weight double-buffering at init | Not tested — submit `memprofile_fsdp2.sh 128 false false /auto/home/anna.khosrovyan/dinov3/pretrained_weights/dinov3_vitb16_pretrain.pth` |
| Torch 2.7+ auto overlap/bucketing for FSDP2 | Deferred — needs PyTorch upgrade |

---

## Closed Decision

The DDP vs FSDP2 strategy question is closed for single-node ViT-B:

- DDP+ES remains the production path until FSDP2 no-release is measured because it is already
  soak-tested in this repo.
- FSDP2 is also valid when configured appropriately. With `reshard_after_forward=False`, it is
  DDP-like in communication while retaining FSDP2 state sharding; with `reshard_after_forward=True`,
  it trades a small throughput cost for lower parameter residency.
- Do not use sub-1 pp MFU differences to justify more strategy churn.
- The next distributed-runtime step is FSDP2 no-release. The next larger recipe/runtime threads are
  crop count, GC/rank-straggler control, and allocator evidence.

Experiment 3 remains useful only as an operational check for the reported FSDP2 long-run OOM:

```
Exp 3 (long soak):
  → alloc_retries or inactive_split_mb growing by iter 3000+?
      YES → fragmentation accumulation is real; reproduce colleague's OOM; need fix
      NO  → colleague's OOM is config-specific (Gram/pretrained init); our config is safe
```
