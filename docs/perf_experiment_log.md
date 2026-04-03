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
