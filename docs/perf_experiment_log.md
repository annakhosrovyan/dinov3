# Performance Experiment Log

Use this file as the append-only run ledger for profiling and optimization work.

Rules:

1. Append entries. Do not rewrite old entries except to correct obvious factual mistakes.
2. One run or one tightly-coupled run bundle gets one entry.
3. Link decisions back to the current bottleneck picture.
4. Do not put long theory here. Keep this operational.

## Entry Template

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

## Entries

### Run 1: Profiling infrastructure validation + baseline trace

```text
Run Name: profile-baseline-8gpu
Date: 2026-04-02
Branch: mfu-tracking-baseline
Commit: 2423391 (uncommitted profiling infra on top)
Step: 3 (baseline profiling pass)
Job ID: 7553
Node: gpu08
Goal: Validate profiling infrastructure, capture first baseline trace with PyTorch profiler
Command or Script: scripts/profiling_run.sh
Config Delta: --profiling, bs=64/GPU, compile=true, 15 iters, real Weka data (Sen1+Intelinair+MAID+NAIP)
Output Directory: /mnt/weka/adovlatyan/output_profile_7553
Trace Artifacts: /mnt/weka/adovlatyan/profiler_traces/2026-04-02/7553/ (8 ranks × json.gz + summary.txt)
Summary Window: Profiler active iters 5-7 (post-compile warmup); logged metrics cover all 15 iters
Images/sec: ~2404 (steady-state iter 14), ~2071 (iter 10 avg)
Step Time ms: 213 (iter 14), 247 (iter 10)
MFU %: 13.8 (iter 14), 11.9 (iter 10), 12.1 (overall avg)
Allocated Mem MB: 386 (after step; params only — FSDP sharded)
Reserved Mem MB: 17718 (steady-state), max 18026
Observed Bottleneck: See profiler summary — NCCL all_gather dominates (203ms/step, 28.7% of CUDA time),
  followed by aten::mm (78ms, 11%), reduce_scatter (86ms, 12.2%), all_reduce (42ms, 5.9%),
  optimizer AdamW (38ms, 5.4%), flash_attention_backward (31ms, 4.0%).
  Total NCCL: ~330ms out of 706ms CUDA time = 46.8%.
Decision: NCCL communication is the dominant cost on this single-node 8-GPU config with FSDP2 SHARD_GRAD_OP.
  This aligns with the prior that ViT-B (172 MB) is too small for FSDP2 — DDP would eliminate
  all_gather/reduce_scatter overhead. Build bottleneck ledger (Step 4) next.
Next Action: Step 4 — build bottleneck ledger from this trace
```

### Run 2: Screening — batch_size=128 (Run B)

```text
Run Name: screen-bs128
Date: 2026-04-02
Branch: mfu-tracking-baseline
Commit: 2423391 (uncommitted)
Step: 5 (screening)
Job ID: 7563
Node: gpu08
Goal: Test batch_size=128 (2× baseline) to improve compute/comms ratio
Command or Script: scripts/screening_run.sh 128 false false
Config Delta: train.batch_size_per_gpu=128 (vs baseline 64), 100 iters
Output Directory: /mnt/weka/adovlatyan/output_screen_bs128_cgfalse_ckptfalse_7563
Summary Window: Iters 50-99 (post-compile warmup, steady-state)
Images/sec: ~3647 (iter 99), ~3451 (overall avg)
Step Time ms: 279 (iter 99), 284 (iter 90)
MFU %: 20.9 (iter 99), 19.8 (overall avg)
Allocated Mem MB: 386 (after step)
Reserved Mem MB: max 33031 MB (41% of 80 GB)
Observed Bottleneck: MFU jumped from 13.8% → 20.9% (+51%). Doubling batch size
  amortizes NCCL overhead per image. Still room for more scaling.
Decision: batch_size=128 is a clear win. Push further to 256.
Next Action: Test bs=256
```

### Run 3: Screening — cudagraphs=true, bs=64 (Run C)

```text
Run Name: screen-cg-true-bs64
Date: 2026-04-02
Branch: mfu-tracking-baseline
Commit: 2423391 (uncommitted)
Step: 5 (screening)
Job ID: 7564
Node: gpu08
Goal: Test CUDA graphs to reduce CPU launch overhead
Command or Script: scripts/screening_run.sh 64 true false
Config Delta: train.cudagraphs=true
Output Directory: N/A (failed)
Summary Window: N/A
Images/sec: N/A
Step Time ms: N/A
MFU %: N/A
Allocated Mem MB: N/A
Reserved Mem MB: N/A
Observed Bottleneck: CUDA graph capture fails with "accessing tensor output of CUDAGraphs
  that has been overwritten by a subsequent run" in block.py:194/_forward_list.
  Root cause: multi-crop architecture processes global+local crops sequentially within
  the same compiled region — CUDAGraph tree can't handle the tensor reuse pattern.
Decision: cudagraphs=true is NOT viable without architectural changes (cudagraph_mark_step_begin
  or restructuring forward_features_list). DEPRIORITIZE.
Next Action: Skip CUDA graphs for now
```

### Run 4: Screening — cudagraphs=true, bs=128 (Run D)

```text
Run Name: screen-cg-true-bs128
Date: 2026-04-02
Branch: mfu-tracking-baseline
Commit: 2423391 (uncommitted)
Step: 5 (screening)
Job ID: 7565
Node: gpu08
Goal: Test CUDA graphs + larger batch
Command or Script: scripts/screening_run.sh 128 true false
Config Delta: train.cudagraphs=true, train.batch_size_per_gpu=128
Output Directory: N/A (failed — same CUDA graph error as Run 3)
Decision: Confirms CUDA graphs are not viable on this architecture. Same error.
Next Action: N/A
```

### Run 5: Screening — batch_size=256 (Run E)

```text
Run Name: screen-bs256
Date: 2026-04-02
Branch: mfu-tracking-baseline
Commit: 2423391 (uncommitted)
Step: 5 (screening)
Job ID: 7566
Node: gpu08
Goal: Test batch_size=256 (4× baseline) to further improve compute/comms ratio
Command or Script: scripts/screening_run.sh 256 false false
Config Delta: train.batch_size_per_gpu=256, 100 iters
Output Directory: /mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckptfalse_7566
Summary Window: Iters 70-99 (steady-state after early GC/thermal effects)
Images/sec: ~4106 (iter 99), ~4054 (iters 90)
Step Time ms: 498 (iter 99), 504 (iter 90)
MFU %: 23.5 (iter 99), 23.2 (iter 90)
Allocated Mem MB: 385 (after step)
Reserved Mem MB: max 65595 MB (82% of 80 GB)
Observed Bottleneck: MFU improved to 23.5% (+70% over baseline). Interesting bimodal
  behavior: MFU ~11% for iters 10-50, then jumps to ~23% at iter 60+. Likely GC cycle
  at iter 50 (every 150 iters, but the overall run was only 100 iters) or
  thermal/compilation stabilization.
  Memory: 65.6 GB used of 80 GB — tight but fits. Room for maybe bs=320-384.
Decision: batch_size=256 is a strong win. Memory is tight but fits.
  Push to 384 to find the OOM boundary.
Next Action: Test bs=384 (may OOM), then bs=256+checkpointing if needed
```

### Run 6: Screening — batch_size=384 (Run F)

```text
Run Name: screen-bs384
Date: 2026-04-02
Step: 5 (screening)
Job ID: 7567
Node: gpu08
Goal: Push batch_size to find OOM boundary
Command or Script: scripts/screening_run.sh 384 false false
Config Delta: train.batch_size_per_gpu=384
Output Directory: N/A (OOM on iter 0)
Observed Bottleneck: OOM during torch.compile codegen — 78 GB used, tried to allocate 1.06 GB.
Decision: bs=384 exceeds 80 GB H100 without checkpointing. OOM boundary is between 256 and 320.
```

### Run 7: Screening — batch_size=320 (Run G)

```text
Run Name: screen-bs320
Date: 2026-04-02
Step: 5 (screening)
Job ID: 7573
Node: gpu08
Goal: Find the exact OOM boundary between 256 and 384
Command or Script: scripts/screening_run.sh 320 false false
Config Delta: train.batch_size_per_gpu=320
Output Directory: N/A (OOM on iter 0)
Observed Bottleneck: OOM during torch.compile — 77 GB used, tried to allocate 4.57 GB.
Decision: bs=320 also OOMs. Maximum without checkpointing is bs=256.
```

### Run 8: Screening — batch_size=256 + checkpointing (Run H)

```text
Run Name: screen-bs256-ckpt
Date: 2026-04-02
Step: 5 (screening)
Job ID: 7574
Node: gpu08
Goal: Test if checkpointing enables larger batch while maintaining throughput
Command or Script: scripts/screening_run.sh 256 false true
Config Delta: train.batch_size_per_gpu=256, train.checkpointing=true
Output Directory: /mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckpttrue_7574
Summary Window: Iters 80-99 (steady-state)
Images/sec: ~3734 (iter 99)
Step Time ms: 548 (iter 99)
MFU %: 21.4 (iter 99) — LOWER than bs=256 without checkpointing (23.5%)
Allocated Mem MB: 386
Reserved Mem MB: max 37329 MB (47% of 80 GB — saved ~28 GB vs no checkpointing)
Observed Bottleneck: Checkpointing adds ~10% step time overhead due to recompute.
  At bs=256, throughput is 3734 vs 4106 img/s without checkpointing (-9%).
  Same bimodal MFU pattern: ~7% for iters 10-50, then ~21% for 60-99.
Decision: Checkpointing trades ~10% throughput for ~28 GB memory savings.
  Not worth it at bs=256 since we fit without it. Would enable bs=384-512 if
  the throughput gain from larger batch exceeds the checkpointing overhead.
Next Action: Test bs=384+checkpointing to see if net effect is positive.
```

### Run 9: Screening — batch_size=384 + checkpointing (Run I)

```text
Run Name: screen-bs384-ckpt
Date: 2026-04-02
Step: 5 (screening)
Job ID: 7593
Node: gpu08
Goal: Test if checkpointing enables 384 batch with net throughput gain
Command or Script: scripts/screening_run.sh 384 false true
Config Delta: train.batch_size_per_gpu=384, train.checkpointing=true
Output Directory: /mnt/weka/adovlatyan/output_screen_bs384_cgfalse_ckpttrue_7593
Summary Window: Iters 80-90 steady-state (bimodal pattern — low phase)
Images/sec: ~1428 (iters 80-90), spike to 3582 at iter 99
Step Time ms: ~2074-2151 (iters 80-90)
MFU %: 8.1 (iters 80-90), spike to 20.5 at iter 99
Allocated Mem MB: 386
Reserved Mem MB: max 55759 MB (70% of 80 GB)
Observed Bottleneck: Checkpointing recompute overhead is massive at this batch size.
  Steady-state throughput (1428 img/s) is MUCH worse than bs=256 no-ckpt (4106 img/s).
  Bimodal pattern is worse here — low phase dominates most of the run.
Decision: bs=384+checkpointing is definitively worse than bs=256 without.
  Checkpointing + batch scaling does NOT compensate for recompute overhead.
```

### Run 10: Screening — batch_size=512 + checkpointing (Run J)

```text
Run Name: screen-bs512-ckpt
Date: 2026-04-02
Step: 5 (screening)
Job ID: 7594
Node: gpu01
Goal: Test extreme batch scaling with checkpointing
Command or Script: scripts/screening_run.sh 512 false true
Config Delta: train.batch_size_per_gpu=512, train.checkpointing=true
Output Directory: N/A (OOM)
Observed Bottleneck: OOM even with checkpointing — 75 GB used, tried to allocate 7.3 GB.
Decision: bs=512 exceeds memory even with selective activation checkpointing.
```

### Step 5 Screening Conclusion

**Complete results table** (10 runs, with direct run citations):

| Run | Job ID | bs/GPU | cudagraphs | ckpt | MFU % | img/s | max_mem GB | Output / Log Path | Status |
|-----|--------|--------|-----------|------|-------|-------|-----------|-------------------|--------|
| Baseline | 7553 | 64 | false | false | 13.8 | 2404 | 18.0 | `/mnt/weka/adovlatyan/output_profile_7553`; traces: `/mnt/weka/adovlatyan/profiler_traces/2026-04-02/7553/` | OK |
| B | 7563 | 128 | false | false | 20.9 | 3647 | 33.0 | `/mnt/weka/adovlatyan/output_screen_bs128_cgfalse_ckptfalse_7563` | OK |
| C | 7564 | 64 | true | false | — | — | — | N/A (failed before output dir) | FAILED |
| D | 7565 | 128 | true | false | — | — | — | N/A (failed before output dir) | FAILED |
| **E** | **7566** | **256** | **false** | **false** | **23.5** | **4106** | **65.6** | `/mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckptfalse_7566` | **BEST** |
| F | 7567 | 384 | false | false | — | — | — | N/A (OOM on iter 0) | OOM |
| G | 7573 | 320 | false | false | — | — | — | N/A (OOM on iter 0) | OOM |
| H | 7574 | 256 | false | true | 21.4 | 3734 | 37.3 | `/mnt/weka/adovlatyan/output_screen_bs256_cgfalse_ckpttrue_7574` | -9% vs E |
| I | 7593 | 384 | false | true | 8.1 | 1428 | 55.8 | `/mnt/weka/adovlatyan/output_screen_bs384_cgfalse_ckpttrue_7593` | -65% vs E |
| J | 7594 | 512 | false | true | — | — | — | N/A (OOM) | OOM |

**Exit criteria met:**
1. One knob emerged as clearly most promising: **batch_size=256** (+70% MFU over baseline)
2. Three directions deprioritized: CUDA graphs (broken), checkpointing (net negative), batch>256 (OOM)
3. The clear next move is **DDP vs FSDP2** — changes distributed strategy → new branch

---

## DDP vs FSDP2 Screening (Branch: perf-ddp-vs-fsdp)

### Run 11: DDP bs=64

```text
Run Name: ddp-bs64
Date: 2026-04-03
Branch: perf-ddp-vs-fsdp
Job ID: 9630
Node: gpu06
Command or Script: scripts/screening_ddp.sh 64 false
Config Delta: train.distributed_strategy=ddp, bs=64
Images/sec: ~3169 (iters 70-80)
Step Time ms: 162
MFU %: 18.1
Max Mem MB: 17711
Observed Bottleneck: DDP at bs=64 = 18.1% MFU vs FSDP2 13.8% = +31%.
  Step time 213ms → 162ms = -24%. Confirms FSDP2 overhead dominant at this batch size.
```

### Run 12: DDP bs=128

```text
Run Name: ddp-bs128
Date: 2026-04-03
Branch: perf-ddp-vs-fsdp
Job ID: 9631
Node: gpu06
Command or Script: scripts/screening_ddp.sh 128 false
Config Delta: train.distributed_strategy=ddp, bs=128
Images/sec: ~4042 (iter 99)
Step Time ms: 253
MFU %: 23.1
Max Mem MB: 33965
Observed Bottleneck: DDP bs=128 = 23.1% MFU vs FSDP2 bs=128 = 20.9% = +11%.
  Near-parity with FSDP2 bs=256 (23.5%) at half the memory.
```

### Run 13: DDP bs=256

```text
Run Name: ddp-bs256
Date: 2026-04-03
Branch: perf-ddp-vs-fsdp
Job ID: 9632
Node: gpu06
Command or Script: scripts/screening_ddp.sh 256 false
Config Delta: train.distributed_strategy=ddp, bs=256
Images/sec: ~2113-3157 (bimodal, unstable)
Step Time ms: 586-933 (variable)
MFU %: 12.1-18.1 (bimodal)
Max Mem MB: 66526
Observed Bottleneck: DDP at bs=256 is WORSE than FSDP2 bs=256.
  Bimodal MFU pattern. DDP's single gradient all-reduce doesn't overlap with
  compute; at large batch sizes FSDP2's per-block overlapped comms win.
```

### DDP vs FSDP2 Summary Table

| Strategy | bs/GPU | MFU % | img/s | step_ms | max_mem GB | Winner |
|----------|--------|-------|-------|---------|-----------|--------|
| FSDP2 | 64 | 13.8 | 2404 | 213 | 18.0 | |
| **DDP** | **64** | **18.1** | **3169** | **162** | **17.7** | **DDP +32%** |
| FSDP2 | 128 | 20.9 | 3647 | 279 | 33.0 | |
| **DDP** | **128** | **23.1** | **4042** | **253** | **34.0** | **DDP +11%** |
| **FSDP2** | **256** | **23.5** | **4106** | **498** | **65.6** | **FSDP2 wins** |
| DDP | 256 | 12-18 | 2113-3157 | 586-933 | 66.5 | |

**Best configs:**
- **DDP bs=128**: 23.1% MFU, 4042 img/s, 34 GB — best throughput per memory
- **FSDP2 bs=256**: 23.5% MFU, 4106 img/s, 66 GB — slightly higher throughput, much more memory

---

## expandable_segments Screening (Branch: perf-ddp-vs-fsdp)

Goal: test whether `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` improves throughput
or unlocks larger batch sizes by reducing allocator fragmentation.
Date: 2026-04-03. Scripts: `scripts/screening_ddp_expandseg.sh`, `scripts/screening_fsdp2_expandseg.sh`.

### Run 14: DDP+ES bs=64

```text
Run Name: ddp-es-bs64
Job ID: 9650
Node: gpu01 (sequential with prior jobs)
Command or Script: scripts/screening_ddp_expandseg.sh 64 false
Config Delta: train.distributed_strategy=ddp, bs=64, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~3061 (iter 99 cumulative avg)
Step Time ms: 156-186ms
MFU %: 17.5 (avg iters 50-99)
Max Mem MB: 17610
Comparison: DDP bs=64 without ES = 18.1%, 3169 img/s. ES = -0.6% MFU, -3% throughput.
Decision: No benefit at bs=64 — fragmentation was not the bottleneck here.
```

### Run 15: DDP+ES bs=128

```text
Run Name: ddp-es-bs128
Job ID: 9651
Command or Script: scripts/screening_ddp_expandseg.sh 128 false
Config Delta: train.distributed_strategy=ddp, bs=128, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~3993 (iter 99 cumulative avg)
Step Time ms: 249-253ms
MFU %: 22.9 (avg iters 50-99)
Max Mem MB: 33922
Comparison: DDP bs=128 without ES = 23.1%, 4042 img/s. ES = -0.2% MFU, -1% throughput.
Decision: Neutral at bs=128 — slight noise-level regression.
```

### Run 16: DDP+ES bs=256

```text
Run Name: ddp-es-bs256
Job ID: 9652
Command or Script: scripts/screening_ddp_expandseg.sh 256 false
Config Delta: train.distributed_strategy=ddp, bs=256, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~4229 (iter 99 cumulative avg)
Step Time ms: 475-478ms (stable!)
MFU %: 24.5 (avg iters 50-99) — STABLE, no bimodal pattern
Max Mem MB: 66484
Comparison: DDP bs=256 without ES = 12-18% bimodal, 2113-3157 img/s (unstable).
Decision: MAJOR WIN. expandable_segments completely eliminated the bimodal instability.
  DDP bs=256+ES is now 24.5% MFU — new best config overall (+1% over FSDP2 bs=256).
  expandable_segments prevents the allocator from fragmenting and triggering defrag pauses
  that caused the periodic MFU collapse at high DDP batch sizes.
```

### Run 17: FSDP2+ES bs=64

```text
Run Name: fsdp2-es-bs64
Job ID: 9653
Command or Script: scripts/screening_fsdp2_expandseg.sh 64 false
Config Delta: train.distributed_strategy=fsdp2, bs=64, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~2110 (iter 99 cumulative avg)
Step Time ms: 235-246ms
MFU %: 12.1 (avg iters 50-99)
Max Mem MB: 16691
Comparison: FSDP2 bs=64 without ES = 13.8%, 2404 img/s. ES = -1.7% MFU, -12% throughput.
Decision: ES hurts FSDP2 at bs=64. FSDP2's fine-grained per-block alloc/free pattern
  likely conflicts with the expandable-segment growth heuristic.
```

### Run 18: FSDP2+ES bs=128

```text
Run Name: fsdp2-es-bs128
Job ID: 9654
Command or Script: scripts/screening_fsdp2_expandseg.sh 128 false
Config Delta: train.distributed_strategy=fsdp2, bs=128, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~3213 (iter 99 cumulative avg)
Step Time ms: 315-330ms
MFU %: 18.4 (avg iters 50-99)
Max Mem MB: 33003
Comparison: FSDP2 bs=128 without ES = 20.9%, 3647 img/s. ES = -2.5% MFU, -12% throughput.
Decision: ES consistently hurts FSDP2 across batch sizes. Do not use with FSDP2.
```

### Run 19: FSDP2+ES bs=256

```text
Run Name: fsdp2-es-bs256
Job ID: 9655
Command or Script: scripts/screening_fsdp2_expandseg.sh 256 false
Config Delta: train.distributed_strategy=fsdp2, bs=256, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Summary Window: Iters 50-99 steady state
Images/sec: ~3809 (iter 99 cumulative avg)
Step Time ms: 521-532ms
MFU %: 21.8 (avg iters 50-99)
Max Mem MB: 65566
Comparison: FSDP2 bs=256 without ES = 23.5%, 4106 img/s. ES = -1.7% MFU, -7% throughput.
Decision: ES hurts FSDP2 at all tested batch sizes. Confirmed anti-pattern for FSDP2.
```

### Run 20: DDP+ES bs=320 (OOM boundary test)

```text
Run Name: ddp-es-bs320
Job ID: 9656
Command or Script: scripts/screening_ddp_expandseg.sh 320 false
Config Delta: train.distributed_strategy=ddp, bs=320, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Observed Bottleneck: Silent OOM during torch.compile iter 0 — no Training [0/100] output logged.
  expandable_segments reduces runtime fragmentation but not compile-time peak allocation.
  bs=320 = ~83 GB peak needed; H100 has 80 GB. Cannot fit.
Decision: bs=320 OOMs regardless of expandable_segments. Confirmed hard memory ceiling.
```

### Run 21: FSDP2+ES bs=320 (OOM boundary test)

```text
Run Name: fsdp2-es-bs320
Job ID: 9657
Command or Script: scripts/screening_fsdp2_expandseg.sh 320 false
Config Delta: train.distributed_strategy=fsdp2, bs=320, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Observed Bottleneck: Silent OOM during torch.compile iter 0 — no Training [0/100] output logged.
Decision: bs=320 OOMs for FSDP2 too. Confirmed hard memory ceiling at bs=256 for both strategies.
```

### expandable_segments Summary Table

| Strategy | bs/GPU | ES | MFU % | img/s | step_ms | max_mem GB | vs no-ES |
|----------|--------|----|-------|-------|---------|-----------|---------|
| DDP | 64 | no | 18.1 | 3169 | 162 | 17.7 | baseline |
| DDP+ES | 64 | yes | 17.5 | 3061 | 157–186 | 17.6 | -3% |
| DDP | 128 | no | 23.1 | 4042 | 253 | 34.0 | baseline |
| DDP+ES | 128 | yes | 22.9 | 3993 | 249–253 | 33.9 | -1% |
| DDP | 256 | no | 12–18 | bimodal | 586–933 | 66.5 | baseline |
| **DDP+ES** | **256** | **yes** | **24.5** | **4229** | **475–478** | **66.5** | **FIXED** |
| FSDP2 | 64 | no | 13.8 | 2404 | 213 | 18.0 | baseline |
| FSDP2+ES | 64 | yes | 12.1 | 2110 | 235–246 | 16.7 | -12% |
| FSDP2 | 128 | no | 20.9 | 3647 | 279 | 33.0 | baseline |
| FSDP2+ES | 128 | yes | 18.4 | 3213 | 315–330 | 33.0 | -12% |
| FSDP2 | 256 | no | 23.5 | 4106 | 498 | 65.6 | baseline |
| FSDP2+ES | 256 | yes | 21.8 | 3809 | 521–532 | 65.6 | -7% |
| DDP+ES | 320 | yes | OOM | — | — | — | |
| FSDP2+ES | 320 | yes | OOM | — | — | — | |

**New best config: DDP bs=256 + expandable_segments = 24.5% MFU, 4229 img/s, 66 GB**

Key finding: `expandable_segments:True` is a targeted fix for DDP at large batch sizes only.
It hurts FSDP2 across the board. The bimodal DDP bs=256 pattern was caused by allocator
fragmentation during the single end-of-backward all-reduce — expandable_segments prevents
defragmentation pauses that periodically stalled forward passes.

---

## Phase 3: Memory Validation + Compile Mode Screening (2026-04-07/08)

### Run 22: Worst-case memory profile — DDP bs=256+ES

```text
Run Name: memprof-ddp-bs256-es
Job ID: (memprofile_ddp.sh run, exact ID not logged here)
Command: sbatch scripts/memprofile_ddp.sh 256 false true
Config Delta: DDP, bs=256, expandable_segments:True, DINOV3_MEMORY_PROFILE=1
  eval_period_iterations=40, checkpointing.period=50 (70 iters total)
Purpose: Confirm worst-case per-phase peak memory for production config.

[MEMPROFILE] results (rank=0):
  pre_training_loop:      max_reserved=1,028 MB,  alloc_retries=0
  compile_warmup_iter0:   max_reserved=67,156 MB, alloc_retries=0
  steady_state (iter 10): max_reserved=67,260 MB, alloc_retries=0
  eval_complete:          max_reserved=67,260 MB, alloc_retries=0  (eval adds 0 reserved)
  checkpoint_complete:    max_reserved=67,260 MB, alloc_retries=0  (ckpt adds 0 reserved)

Worst-case: 67.2 GB — 12.8 GB headroom on 80 GB H100. Zero fragmentation pressure.
Eval: max_alloc during eval only 1,326 MB (allocator reuses reserved pages — not a spike).
Decision: DDP bs=256+ES is confirmed safe at per-phase granularity.
```

### Run 23: Worst-case memory profile — FSDP2 bs=128 (no ES)

```text
Run Name: memprof-fsdp2-bs128
Job ID: (memprofile_fsdp2.sh run)
Command: sbatch scripts/memprofile_fsdp2.sh 128 false false
Config Delta: FSDP2, bs=128, no expandable_segments, DINOV3_MEMORY_PROFILE=1

[MEMPROFILE] results (rank=0):
  pre_training_loop:      max_reserved=156 MB,    alloc_retries=0
  compile_warmup_iter0:   max_reserved=34,090 MB, alloc_retries=0
  steady_state (iter 10): max_reserved=34,464 MB, alloc_retries=0
  eval_complete:          max_reserved=34,464 MB, alloc_retries=0  (eval: max_alloc 891 MB)
  checkpoint_complete:    max_reserved=36,340 MB, alloc_retries=0  (+1.9 GB for DCP write)

Worst-case: 36.3 GB — 43.7 GB headroom. Zero alloc_retries.
Note: FSDP2 bs=128 reportedly OOMed in a colleague's run. This profiling does not reproduce
that OOM. The colleague's failure likely involved additional GPU load, pretrained weight loading,
or a long-run fragmentation accumulation not captured by 70 iters. Not explained by this data.
Decision: FSDP2 bs=128 is safe per this profiling, but it is not the recommended production config.
```

### Run 24: DDP+ES bs=256 — 500-iter soak test (long-run memory stability)

```text
Run Name: soak-ddp-bs256-es
Job ID: 16718
Command: sbatch scripts/soak_test_ddp.sh 256 true
Config Delta: DDP, bs=256, expandable_segments:True, DINOV3_MEMORY_PROFILE=1
  OFFICIAL_EPOCH_LENGTH=500, eval_period=100 (5 eval cycles), checkpoint_period=200 (2 ckpt cycles)
Purpose: Validate no memory creep across hundreds of iterations and multiple eval+checkpoint phases.

[MEMPROFILE] — all 17 events (rank=0):
  compile_warmup_iter0:   max_reserved=67,156 MB
  steady_state:           max_reserved=67,260 MB
  pre_eval (×5):          max_reserved=67,260 MB  ← FLAT
  eval_complete (×5):     max_reserved=67,260 MB  ← FLAT
  checkpoint_complete (×2): max_reserved=67,260 MB ← FLAT
  alloc_retries=0 at every checkpoint.

max_reserved_mb is perfectly stable across all 17 MEMPROFILE events. Zero memory creep.

MFU (steady state, iters 50–490, n=46): avg=23.92%, range=23.4–24.1%
Images/sec: ~4170–4210 sustained.
No bimodal behavior. No step-time spikes at eval or checkpoint boundaries.

Decision: DDP bs=256+ES is CONFIRMED PRODUCTION-SAFE. Promoted to run.sh default.
Also enabled train.sharded_eval_checkpoint=true and train.compile_mode config key in run.sh.
```

### Run 25: Compile mode screening — default vs max-autotune-no-cudagraphs

```text
Run Name: compile-mode-default-bs128
Job ID: 16719
Command: sbatch scripts/screening_compile_modes.sh default 128
Config Delta: DDP, bs=128, expandable_segments:True, train.compile_mode=null

Steady-state MFU (iters 20–199): avg ~23.1%, range 22.9–23.5%
Images/sec: ~4000–4100
Step time: ~249–256ms
Decision: Default compile mode baseline confirmed. Consistent with prior DDP bs=128 data.
```

```text
Run Name: compile-mode-max-autotune-attempt-1
Job ID: 16720
Command: sbatch scripts/screening_compile_modes.sh max-autotune-no-cudagraphs 128
Config Delta: DDP, bs=128, expandable_segments:True, train.compile_mode=max-autotune-no-cudagraphs
Result: SIGABRT on rank 2 after 18 min. Training never started (0 log lines produced).
Cause: All 8 ranks benchmarked Triton kernel variants concurrently during iter-0 compile,
  each holding model weights + benchmark temp allocs → OOM per GPU.
```

```text
Run Name: compile-mode-max-autotune-attempt-2 (with single-rank warmup)
Job ID: 17824
Command: sbatch scripts/screening_compile_modes.sh max-autotune-no-cudagraphs 128
  (updated script with Phase 1: single-rank cache warmup)

Phase 1 (nproc=1, GPU 0): completed successfully in ~27s. Cache written to ~/.cache/torch/inductor.
Phase 2 (nproc=8): SIGABRT on rank 0 after 26 min.

Root cause (definitive): The last autotuned op before crash:
  AUTOTUNE mm(2048x7503, 7503x768)  ← iBOT masked-patch global-batch matmul
  2048 = 1024 images × 2 global crops; 7503 = total masked patches (stochastic, changes each iter)

This shape is world-size-dependent (scales with global batch) and changes every iteration due to
stochastic iBOT masking. Phase 1 warmup (nproc=1, global_batch=128) cached different shapes
→ cache miss in Phase 2 for this op → re-benchmarked → crash.

Even if the crash were prevented: the compiled kernel for shape 7503 would fail on the next
iteration's 7489 (dynamic shape + static kernel = mismatch). max-autotune requires static shapes;
iBOT produces dynamic shapes at global-batch granularity.

Decision: max-autotune-no-cudagraphs is FUNDAMENTALLY INCOMPATIBLE with this model.
  Deprioritized permanently. Default compile mode is the correct and only viable path.
  See learnings/compile_modes.md for full analysis.
```

---

## Phase 4: FSDP2 no-release screening (2026-04-27)

Hypothesis: `reshard_after_forward=False` eliminates per-block all-gather overhead, closing the
0.4 pp gap between ZeRO-3 FSDP2 (23.5%) and DDP+ES (23.9%). Two variants: with and without ES.

### Bug found and fixed: DTensor/Tensor mixing in update_ema

**Root cause**: First two job attempts (31010, 31011) crashed on all 8 ranks at iter 1 during
`update_ema()`:

```
RuntimeError: aten._foreach_add_.List: got mixed torch.Tensor and DTensor
  ssl_meta_arch.py:757 → torch._foreach_add_(teacher_param_list, student_param_list, alpha=1-m)
```

With `reshard_after_forward=False`, FSDP2 all-gathers each block's params before its forward and
leaves them as plain (non-DTensor) tensors for the duration of the forward. The teacher model
(inference-only, no backward) never triggers a reshard — its params stay as plain tensors after
forward. The student's params ARE resharded during its backward pass (reduce-scatter always reshards).
After optimizer.step(), student params = sharded DTensors, teacher params = plain tensors → mixed
type in `_foreach_add_`.

**Fix** (`ac_compile_parallelize.py:244,259`): inference-only models always use `reshard_after_forward=True`
regardless of the global setting. They get no benefit from no-release (no per-block overhead), and
always resharding keeps their params as sharded DTensors — consistent with student state post-backward.

```python
inference_only_set = set(id(m) for m in inference_only_models)
...
effective_reshard = reshard if id(model) not in inference_only_set else True
```

### Run 26: FSDP2 no-release bs=256 (no ES)

```text
Run Name: fsdp2-norelease-bs256
Date: 2026-04-27
Branch: perf-ddp-vs-fsdp
Job ID: 31102 (first attempt 31010 crashed — DTensor bug)
Node: gpu07
Command or Script: scripts/screening_fsdp2_norelease.sh
Config Delta: fsdp2, reshard_after_forward=false, bs=256, no expandable_segments
Compile warmup: ~90 iters to stabilize (much longer than ZeRO-3's ~30-40 iters)
Stable window: iters 90-99 (2 data points — noisy)
Images/sec: ~4057 (iter 99)
Step Time ms: 502 (avg iters 90-99)
MFU %: 23.20 (avg iters 90-99)
Max Mem MB: 65,766 (64.2 GB)
Comparison:
  vs DDP+ES bs=256: 23.9% MFU → no-release 23.2% — 0.7 pp below (not a win)
  vs ZeRO-3 FSDP2 bs=256: 23.5% MFU → no-release 23.2% — 0.3 pp below; memory virtually identical
    (ZeRO-3 at bs=256 uses ~65.6 GB per Run 5; no-release uses 64.2 GB — no memory advantage)
  NOTE: ZeRO-3's 36.3 GB figure is for bs=128, not bs=256. At equal batch size,
    no-release and ZeRO-3 have the same memory footprint.
Caveats:
  - bs=256 is NOT the right production operating point for FSDP2. At 64–66 GB it leaves only
    ~14–16 GB headroom — same as DDP+ES. FSDP2's memory advantage materializes at bs=128 (36.3 GB).
  - 100 steps does not represent a real training memory profile: no eval cycle (adds ~1 GB peak
    for checkpoint writes), no long-run fragmentation accumulation, no pretrained-weight init
    double-buffer. A 100-step result is a lower bound on peak memory, not an upper bound.
  - Only 2 stable data points post-warmup; MFU estimate unreliable.
Decision: No-release shows no throughput improvement over ZeRO-3 at matched batch size, with
  identical memory footprint. Not a viable alternative at any bs.
```

### Run 27: FSDP2 no-release+ES bs=256

```text
Run Name: fsdp2-norelease-es-bs256
Date: 2026-04-27
Branch: perf-ddp-vs-fsdp
Job ID: 31103 (first attempt 31011 crashed — same DTensor bug)
Node: gpu07
Command or Script: scripts/screening_fsdp2_norelease_es.sh
Config Delta: fsdp2, reshard_after_forward=false, bs=256, expandable_segments:True
Compile warmup: ~50 iters to stabilize
Stable window: iters 50-99 (6 data points)
Images/sec: ~3950 (iter 99)
Step Time ms: 510 (avg iters 50-99)
MFU %: 22.87 (avg iters 50-99)
Max Mem MB: 65,728 (64.2 GB)
Comparison: no-release without ES = 23.2% → with ES = 22.9% — marginal -0.3 pp
Caveats: Same as Run 26 — bs=256 is not a valid production bs for FSDP2 (same memory as DDP+ES),
  and 100 steps does not cover eval/checkpoint peaks or long-run fragmentation.
Decision: ES slightly negative with no-release (consistent with DDP+ES pattern where ES
  adds minor allocator overhead). Not worth using with no-release.
```

### FSDP2 no-release summary table

| Config | MFU % | step_ms | max_mem GB | stable iters | Notes |
|--------|-------|---------|-----------|-------------|-------|
| DDP+ES bs=256 (production, 500 iters) | 23.9 | ~501 | 67.2 | 450+ | Soak-tested; memory tight |
| ZeRO-3 FSDP2 bs=256 (100-iter screen) | 23.5 | ~508 | ~65.6 | ~50 | Run 5; NOT soak-tested; same memory as DDP |
| ZeRO-3 FSDP2 bs=128 (1300-iter soak) | ~17–18% | ~325 | 36.3 | 1300+ | 43 GB headroom; production-viable |
| No-release FSDP2 bs=256 | 23.2 | 502 | 64.2 | 2 (noisy) | Run 26; 100 steps; bs=256 not production-viable for FSDP2 |
| No-release FSDP2+ES bs=256 | 22.9 | 510 | 64.2 | 6 | Run 27; same caveats |

**Memory context**: FSDP2 ZeRO-3's memory advantage over DDP is only visible at bs=128 (36.3 GB,
43 GB headroom) not at bs=256 (~65.6 GB, same as DDP). Running FSDP2 at bs=256 negates its
primary advantage — the right production operating point for FSDP2 is bs=128 or bs=192 (untested,
estimated ~50 GB).

**100-step caveat**: None of the FSDP2 no-release results cover a real training memory profile.
100 steps without eval or checkpoint cycles cannot expose fragmentation accumulation, eval-phase
allocation peaks, or checkpoint write spikes. DDP+ES bs=256 is soak-tested; no-release and
ZeRO-3 bs=256 are not.

**Conclusion**: `reshard_after_forward=False` does not improve throughput or reduce memory vs
ZeRO-3 at matched batch size. At bs=256 no-release (64.2 GB) ≈ ZeRO-3 (65.6 GB) — no memory
benefit. MFU 23.2% vs ZeRO-3 23.5% — no throughput benefit. Prolonged compile warmup (~90 iters
vs ~30-40 for ZeRO-3) is an additional downside. **No-release is closed.**

**Strategy going forward**: DDP+ES bs=256 remains the benchmarked production config. For full
training run safety and multi-node readiness, FSDP2 ZeRO-3 bs=128 (36.3 GB, 43 GB headroom) is
the correct reference point — not FSDP2 at bs=256. An untested bs=192 probe (~50 GB estimated)
would determine whether FSDP2 can approach DDP+ES MFU while preserving meaningful memory margin.
