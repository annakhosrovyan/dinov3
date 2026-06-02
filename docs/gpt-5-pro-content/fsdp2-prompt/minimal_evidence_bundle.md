# Minimal Evidence Bundle For GPT-5 Pro: bs=128 vs bs=96 DINOv3 Phase 5 Puzzle

Paste this file into GPT-5 Pro after `docs/gpt5-pro-bs128-context-pack.md`.

This is intentionally compressed. It is not a full repository dump. It is the smallest self-contained evidence pack I would give a strong reasoning model so it can reason about why `bs=128` is not materially faster than `bs=96` in the current 8-GPU FSDP2 setup.

----- BEGIN TASK -----

We are training a satellite-specialized fork of Meta DINOv3 on one DGX H100-class node with 8x H100 GPUs.

Question:

Why does per-GPU batch size `bs=128` fail to significantly outperform per-GPU batch size `bs=96` under the current Phase 5 FSDP2 + selective activation checkpointing setup, even though a single-GPU disambiguator shows `bs=128` is locally faster?

Important: produce hypotheses that are consistent with both:

1. `bs=128` is faster than `bs=96` on 1 GPU.
2. `bs=128` is not materially faster, and can be slower, on 8 GPUs.

Separate observed evidence from causal inference. Use `HYPOTHESIS:` for anything not directly proven.

----- END TASK -----

----- BEGIN HARDWARE AND ENVIRONMENT -----

Hardware target:

- One DGX H100-class node.
- 8x H100 GPUs requested via Slurm: `#SBATCH --gres=gpu:h100:8`.
- Runs are single-node.
- Node heterogeneity matters: gpu05 vs gpu07 comparisons are treated as weaker than same-node paired comparisons.

Working environment:

- Conda env: `/home/adovlatyan/.conda/envs/test-conda-slurm`.
- PyTorch 2.6.0+cu124.
- Required env vars in run scripts:
  - `PYTHONPATH=.`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`
  - `OMP_NUM_THREADS=8`
- Shared Weka paths are used for logs/output.

Current production-safe target:

- FSDP2.
- `train.batch_size_per_gpu=96`.
- selective activation checkpointing.
- `train.fsdp_reshard_after_forward=true`.
- `train.compile=true`.
- no `expandable_segments` for FSDP2.

`bs=128` status:

- Investigational only.
- A researcher-reported long-training FSDP2 OOM remains unresolved.
- Short screening/profiling runs do not prove long-run memory safety.

----- END HARDWARE AND ENVIRONMENT -----

----- BEGIN KEY RUN LEDGER -----

Most important rows:

| Job | Node | GPUs | Batch/GPU | FSDP reshard | AC | Iters/window | Throughput | MFU | Peak alloc | Interpretation |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 45280 | gpu07 | 8 | 96 | true | selective | steady 400-999 | 1351 img/s | 7.73% | 14.3 GB | Current best/default cell |
| 45366 | gpu07 | 8 | 128 | true | selective | 300-ish rerun | 1276 img/s | 7.30% | 18.9 GB | Clean bs=128 rerun; no advantage over bs=96 |
| 45364 | gpu03 | 8 | 128 | true | full | 300-ish | 1297 img/s | 7.42% | 11.7 GB | Full AC, cross-node |
| 45282 | gpu07 | 8 | 128 | true | none | 1000 mem profile | 1267 img/s | 7.25% | 33.0 GB | Survived 1000 iters, but OOM risk unresolved |
| 47554 | gpu05 | 8 | 128 | false | selective | 1000 | mean 1341, median 1312 img/s | 7.67% | about 18.6 GB rank0 | Interesting, but cross-node vs best bs=96 anchor |
| 51069 | gpu05 | 8 | 128 | true | selective | pending/running | TBD | TBD | TBD | Paired same-node A/B vs 47554 |
| 45476 | gpu07 | 1 | 96 | degenerate/no-shard | selective | 300 | 448 img/s/GPU | 20.49% | 16.1 GB | Single-GPU ref |
| 45477 | gpu07 | 1 | 128 | degenerate/no-shard | selective | 300 | 478 img/s/GPU | 21.88% | 20.7 GB | bs=128 is +6.7% locally |

Core tension:

- On 1 GPU: `bs=128` improves local CUDA-event throughput by about 6.7%.
- On 8 GPUs: `bs=128 reshT selective AC` is worse than `bs=96 reshT selective AC` in the clean gpu07 comparison.
- `bs=128 reshF selective AC` on gpu05 is close to the best `bs=96` number, but the comparison crosses node/date and does not prove a win.

Same-node decision rule for job 51069:

- If job 51069 (`bs=128 reshT sel`, gpu05) is within about 2% of 47554 (`bs=128 reshF sel`, gpu05), then reshF probably buys little at bs=128 and the 47554 lift was mostly node/date/noise.
- If job 51069 is at least 3% slower than 47554, then `reshard_after_forward=false` may be a real bs=128 lever.
- If job 51069 reaches or exceeds the bs=96 best anchor of 1351 img/s, then bs=128 is competitive even without reshF, but memory safety is still not proven.

----- END KEY RUN LEDGER -----

----- BEGIN NSYS AND PROFILING FACTS -----

Existing nsys trace evidence:

- Jobs 39140 (`bs=128`) and 39141 (`bs=96`) produced usable nsys traces.
- Headline trace signal: NCCL is dominant, around 70-80% of kernel time.
- Reported overlap between NCCL and compute is low, around 11-16%.
- This is the strongest evidence that 8-GPU throughput is communication-exposed, not simply local compute-limited.

Interpretation constraint:

- Do not merely say "communication bottleneck."
- Explain why extra local compute from `bs=128` does not hide communication enough.
- Consider whether FSDP2 all-gather/reduce-scatter timing, per-block wrapping, selective activation checkpointing recompute, loss communication, bucketization, launch ordering, or stream placement keeps communication on the critical path.

----- END NSYS AND PROFILING FACTS -----

----- BEGIN MEASUREMENT BOUNDARIES -----

Throughput and MFU:

- Logged `images_per_sec` is computed from CUDA-event `step_time_ms`, not host wall-clock `iter_time`.
- `images_per_sec = global_batch_size / (step_time_ms / 1000.0)`.
- MFU uses `images_per_sec`, MACs per image, and dense H100 BF16 peak.
- MFU is useful comparatively but not absolute truth.

Data loader:

- `data`/`data_time` is host-side timing and can diverge from CUDA-event throughput.
- Relevant 8-GPU bs=128 logs show `data` near zero in steady regions, so the loader is probably not the main 8-GPU bottleneck.
- The single-GPU bs=128 disambiguator had a host-side data stall, so do not confuse wall-time data stalls with the CUDA-event compute-window win.

Eval/checkpoint:

- Some probes disable eval and checkpoint with large periods.
- Production behavior includes eval/checkpoint phase boundaries.
- A run surviving 1000 iterations without eval/checkpoint is not proof of long-training memory safety.

Cross-node comparisons:

- gpu05 vs gpu07 comparisons are weak evidence.
- Same-node paired runs are stronger.
- The `47554` vs `45280` comparison is not a clean bs=128-vs-bs=96 win because it crosses node and date.

----- END MEASUREMENT BOUNDARIES -----

----- BEGIN CURRENT CONFIG KNOBS -----

Common Phase 5 training shape:

- `train.distributed_strategy=fsdp2`
- `train.compile=true`
- `train.cudagraphs=false`
- `train.batch_size_per_gpu=96` or `128`
- `train.checkpointing=true` for selective AC
- `train.checkpointing_full=false` for selective AC
- `train.fsdp_reshard_after_forward=true` for reshT/ZeRO-3-like behavior
- `train.fsdp_reshard_after_forward=false` for reshF/no-release/DDP-like behavior
- `train.num_workers=20`
- `train.prefetch_factor=8`
- `train.persistent_workers=true`
- no NCCL knobs in the clean bs=128 reshT vs reshF A/B

Forbidden or parked knobs:

- Do not suggest `expandable_segments=True` as a default FSDP2 fix. It was a DDP/large-batch screening knob and is off-limits for current FSDP2 production experiments unless isolated.
- Do not suggest `compile_mode=max-autotune` or CUDA graphs as easy wins. Dynamic iBOT masked-token count and multi-crop behavior made those paths non-viable in prior tests.
- Do not bundle several knobs into the next A/B if the goal is causal isolation.

----- END CURRENT CONFIG KNOBS -----

----- BEGIN SOURCE DIGEST: train.py -----

File: `dinov3/train/train.py`

Training loop facts:

- `do_train()` is the core loop.
- Per iteration:
  1. schedules update
  2. optimizer zero grad
  3. `model.forward_backward(data, teacher_temp=...)`
  4. gradient clipping/all-reduce/optimizer step
  5. EMA teacher update
  6. CUDA-event step timing and MFU/images/sec update

Important timing lines:

```python
step_start_event.record()
...
loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
...
optimizer.step()
model.update_ema(mom)
step_end_event.record()
step_end_event.synchronize()
step_time_ms = step_start_event.elapsed_time(step_end_event)
images_per_sec = global_batch_size / (step_time_ms / 1000.0)
mfu = compute_mfu(images_per_sec, macs_per_image, num_gpus)
```

Interpretation:

- The reported `images_per_sec` is a CUDA-event compute-window metric.
- It includes H2D copies inside `forward_backward()` if they occur between the CUDA events.
- It does not directly include all host-side waiting represented by `MetricLogger` `data_time`.

----- END SOURCE DIGEST: train.py -----

----- BEGIN SOURCE DIGEST: ssl_meta_arch.py -----

File: `dinov3/train/ssl_meta_arch.py`

Forward/backward facts:

- `forward_backward()` moves collated tensors to GPU using `cuda(non_blocking=True)`.
- Teacher forward uses global crops under no-grad.
- Student forward processes both global and local crops.
- Losses include DINO CLS, iBOT masked patch, KoLeo, and optional Gram.
- Backprop is `loss.backward()`.
- EMA update uses foreach ops.

H2D transfer shape:

```python
global_crops = data["collated_global_crops"].cuda(non_blocking=True)
local_crops = data["collated_local_crops"].cuda(non_blocking=True)
masks = data["collated_masks"].cuda(non_blocking=True)
mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
masks_weight = data["masks_weight"].cuda(non_blocking=True)
n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)
```

Student forward bottleneck:

```python
student_backbone_output = self.student.backbone.forward_features_list(
    [global_crops, local_crops],
    [masks, None],
)
```

Interpretation:

- The two crop resolutions cannot be trivially batched together because sequence lengths differ.
- Global crops are 224px, local crops are 96px.
- This creates a multi-resolution pass through the transformer stack.
- A hypothesis about launch overhead, graph capture difficulty, or overlap failure should account for this structure.

Loss communication:

- DINO and iBOT center updates use async all-reduce in the loss files:
  - `dinov3/loss/dino_clstoken_loss.py`: `dist.all_reduce(..., async_op=True, ...)`
  - `dinov3/loss/ibot_patch_loss.py`: `dist.all_reduce(..., async_op=True, ...)`

Interpretation:

- Loss center all-reduce is intended to overlap, but a deep analysis should check whether it can still create ordering/synchronization pressure around the end of the step.

----- END SOURCE DIGEST: ssl_meta_arch.py -----

----- BEGIN SOURCE DIGEST: vision_transformer.py -----

File: `dinov3/models/vision_transformer.py`

Key method:

```python
def forward_features_list(self, x_list, masks_list):
    x = []
    rope = []
    for t_x, t_masks in zip(x_list, masks_list):
        t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
        x.append(t2_x)
        rope.append(hw_tuple)
    for _, blk in enumerate(self.blocks):
        if self.chunked_blocks:
            outputs = blk(x, rope=rope)
        else:
            outputs = blk(x, rope=rope)
        x = outputs
    ...
```

Relevant architecture facts:

- ViT-B default:
  - hidden dim 768
  - depth 12
  - heads 12
  - patch size 16
  - input channels 5
  - no register/storage tokens in this fork
- Global crop tokens: 197 = 196 patches + CLS.
- Local crop tokens: 37 = 36 patches + CLS.
- Multi-crop setup: 2 global crops + 8 local crops.

Interpretation:

- Increasing per-GPU batch from 96 to 128 increases local work.
- But it does not change sequence lengths, model depth, number of FSDP-wrapped blocks, or the presence of multi-crop resolution splitting.
- If communication is exposed at block boundaries or loss/gradient boundaries, larger batch may not hide it as much as naive FLOP scaling suggests.

----- END SOURCE DIGEST: vision_transformer.py -----

----- BEGIN SOURCE DIGEST: fsdp/ac_compile_parallelize.py -----

File: `dinov3/fsdp/ac_compile_parallelize.py`

FSDP2 and activation checkpointing facts:

- Activation checkpointing wraps transformer blocks with PyTorch checkpoint wrappers.
- FSDP2 wrapping is applied around trained model components/blocks.
- `train.fsdp_reshard_after_forward` is wired into FSDP2 `fully_shard()` behavior in this fork.

Semantics:

- `train.fsdp_reshard_after_forward=true`:
  - reshT.
  - ZeRO-3-like behavior.
  - parameters are reshared after forward.
  - lower memory, more repeated communication.
- `train.fsdp_reshard_after_forward=false`:
  - reshF/no-release behavior.
  - DDP-like in the sense that parameters stay materialized through backward.
  - higher memory, potentially less communication exposure.

Important reasoning prompt:

- If `bs=128` is not faster under reshT, check whether per-block all-gather/release patterns dominate the critical path.
- If `bs=128 reshF` is faster only on same-node paired runs, that supports a communication-exposure explanation.
- If `bs=128 reshF` is not faster on same-node paired runs, then the bottleneck is likely elsewhere or noise dominates.

----- END SOURCE DIGEST: fsdp/ac_compile_parallelize.py -----

----- BEGIN SOURCE DIGEST: data pipeline -----

Files:

- `dinov3/data/loaders.py`
- `dinov3/data/collate.py`
- `dinov3/data/datasets/mixed_satlas_dataset.py`
- `dinov3/data/datasets/channel_utils.py`

Facts:

- `pin_memory=True` is always enabled in the data loader.
- `num_workers=20`, `prefetch_factor=8`, and `persistent_workers=true` are used in run scripts.
- H2D copies in `ssl_meta_arch.py` use `non_blocking=True`.
- All satellite sources are normalized to 5 channels via `to_five_channels()`.
- Multi-crop collation creates 2 global crops and 8 local crops plus iBOT masks.

Interpretation:

- The obvious data-loader knobs are already on.
- In the relevant 8-GPU bs=128 logs, `data` is near zero in steady regions, so a data-loader-first explanation is weak.
- However, H2D stream placement and actual overlap with compute/NCCL remains a possible instrumentation question.

----- END SOURCE DIGEST: data pipeline -----

----- BEGIN SOURCE DIGEST: mfu.py -----

File: `dinov3/utils/mfu.py`

Facts:

- `compute_dino_flops_per_image()` returns MACs per image, not hardware FLOPs.
- `compute_mfu()` converts MACs to FLOPs with a 2x MAC-to-FLOP factor.
- Dense H100 BF16 peak is treated as 989 TFLOPS, not the 1979 TFLOPS sparse peak.

Formula:

```python
actual_tflops = (images_per_sec * 2 * macs_per_image) / 1e12
mfu = actual_tflops / (num_gpus * H100_BF16_TFLOPS) * 100
```

Interpretation:

- MFU tracks CUDA-event throughput, not full training wall-clock including all host stalls and periodic eval/checkpoint.
- Activation checkpointing may change hardware work/HFU without changing model-FLOP MFU in the intuitive way.

----- END SOURCE DIGEST: mfu.py -----

----- BEGIN CLOSED OR WEAK EXPLANATIONS -----

Weak as primary explanation:

1. Dataloader starvation on 8 GPUs.
   - Against: steady `data` in bs=128 8-GPU logs is near zero.
   - Still worth instrumenting H2D overlap separately if needed.

2. Learning-rate or optimizer hyperparameter scaling.
   - Against: throughput puzzle is in short performance probes, not quality convergence. LR is auto-scaled.

3. `torch.compile` mode.
   - Against: viable path is `compile_mode: null`; max-autotune/CUDA graphs are already known problematic.

4. Batch size tensor-core alignment.
   - Against: hidden dim 768 is already tensor-core friendly; sequence length and distributed exposure are more plausible.

5. `expandable_segments=True`.
   - Against: DDP/large-batch knob, not a current FSDP2 production default.

----- END CLOSED OR WEAK EXPLANATIONS -----

----- BEGIN HYPOTHESES TO RANK -----

Ask GPT-5 Pro to rank these and add any missing hypotheses.

HYPOTHESIS 1: Communication exposure dominates at 8 GPUs.

- Single-GPU bs=128 improves because local matmul efficiency/work improves.
- 8-GPU bs=128 fails because FSDP2/NCCL communication remains exposed and is not sufficiently overlapped.
- Supported by nsys: NCCL is 70-80% of kernel time with low overlap.
- Predicts 1-GPU vs 8-GPU sign flip.

HYPOTHESIS 2: reshT per-block all-gather/release pattern limits bs=128.

- With `reshard_after_forward=true`, FSDP2 repeatedly materializes and reshards params around block forward/backward.
- Increasing batch increases activation/work but not the number of block-boundary communication events.
- If communication lies on the critical path, extra compute may not hide it.
- Job 51069 vs 47554 is the clean test.

HYPOTHESIS 3: selective AC changes the overlap schedule.

- Selective activation checkpointing reduces memory, but recompute in backward can alter when all-gathers/reduce-scatters happen.
- It may improve memory while failing to improve, or even hurting, communication overlap.
- Compare no AC / selective AC / full AC within same node and reshard setting.

HYPOTHESIS 4: multi-crop/dynamic-shape structure causes launch and overlap inefficiency.

- `forward_features_list([global_crops, local_crops])` processes different sequence lengths through each block.
- Dynamic iBOT masked-token count and multi-crop shape make CUDA graphs/max-autotune non-viable.
- The resulting launch pattern may limit overlap and prevent larger batch from reaching expected local efficiency at 8 GPUs.

HYPOTHESIS 5: node/date variance explains apparent bs=128 reshF improvement.

- `47554` on gpu05 looks close to bs=96 best on gpu07, but this is not a clean comparison.
- If 51069 on gpu05 matches 47554, reshF is not the lever.
- If 51069 is much slower, reshF likely matters.

HYPOTHESIS 6: memory pressure or allocator behavior causes hidden inefficiency.

- bs=128 uses more memory, especially no-AC and reshF variants.
- Short runs may avoid OOM while long runs hit phase-boundary or fragmentation issues.
- `[MEMPROFILE]` and `[MEMFRAG]` evidence is needed before promotion.

----- END HYPOTHESES TO RANK -----

----- BEGIN JOB 51069 INTERPRETATION MATRIX -----

Job 51069:

- Script: `scripts/fsdp2/fsdp2_bs128_reshT_sel_resoak.sh`
- Node: gpu05
- Matched against job 47554 on gpu05.
- Only variable vs 47554: `train.fsdp_reshard_after_forward=false -> true`.
- Same bs=128, selective AC, compile=true, no NCCL knobs, eval/checkpoint disabled, memprofile period 50.

Interpretation:

1. 51069 within about 2% of 47554:
   - reshF probably buys little at bs=128.
   - 47554's apparent lift may be node/date/noise.
   - bs=128 remains not clearly better than bs=96.

2. 51069 at least 3% slower than 47554:
   - reshF is likely a real lever at bs=128.
   - Next test should isolate reshF at bs=96 or same-node bs=96 vs bs=128 under reshF, not stack new knobs.

3. 51069 reaches/exceeds 1351 img/s:
   - bs=128 may be competitive even with reshT.
   - Still do not promote until long-run memory/OOM question is resolved.

4. 51069 OOMs or shows allocator instability:
   - bs=128 safety concern strengthens.
   - Need memory-phase diagnosis, not throughput tuning.

----- END JOB 51069 INTERPRETATION MATRIX -----

----- BEGIN REQUESTED OUTPUT FROM GPT-5 PRO -----

Please produce:

1. Executive conclusion, max 10 bullets.
2. Ranked hypothesis table.
3. Mechanistic explanation for the 1-GPU vs 8-GPU sign flip.
4. Explanation of whether `bs=128` should have hidden NCCL better, and why it apparently did not.
5. Interpretation matrix for job 51069.
6. Minimal next three experiments, with exactly one changed variable each.
7. Minimal instrumentation changes that would reduce uncertainty most.

Rules:

- Do not give generic advice.
- Do not recommend changing multiple knobs at once.
- Do not treat cross-node comparisons as decisive.
- Do not treat `bs=128` throughput and `bs=128` long-run safety as the same question.
- Label causal claims as `HYPOTHESIS:` unless directly proven by the evidence above.

----- END REQUESTED OUTPUT FROM GPT-5 PRO -----

