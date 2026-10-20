# torch.compile Mode Screening - DINOv3

Current status of compile modes in this repo, with Phase 4/6 stale conclusions
cleaned up so this file can be safely used as agent context.

Last corrected: 2026-06-11.

---

## Current operating rule

1. The production DDP path in `run_ddp.sh` uses `train.compile=true`,
   `train.cudagraphs=true`, and leaves `train.compile_mode=null`.
2. With `cudagraphs=true`, ViT backbone blocks take the explicit
   `wrap_compile_block()` branch:

   ```python
   module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
   ```

3. `train.compile_mode` is not consulted for backbone blocks when
   `cudagraphs=true`, because that branch wins first.
4. Heads (`dino_head`, `ibot_head`) are compiled separately with
   `use_cuda_graphs=False`. In the current production path, because
   `compile_mode=null`, they use plain `module.compile()`.
5. DDP wrapping is applied after compile. DDP does not replace or bypass the
   compile dispatcher.

See:

- `run_ddp.sh:111-138`
- `dinov3/configs/config.py:71-90`
- `dinov3/fsdp/ac_compile_parallelize.py:65-77`
- `dinov3/fsdp/ac_compile_parallelize.py:217-242`
- `dinov3/fsdp/ac_compile_parallelize.py:316-357`
- `docs/phase7_compile_dispatch_agent_handoff.md`

---

## Mode terminology in this repo

| Setting | What it means here | Current use |
|---|---|---|
| `train.compile=true` | Enables this repo's wrapper-level compile loop over student plus inference-only model dictionaries. | On by default and on in `run_ddp.sh`. |
| `train.compile_mode=null` | `_get_compile_mode()` returns `None`; eligible modules use default `module.compile()`. | Current production value for heads; ignored by backbone branch when `cudagraphs=true`. |
| `train.cudagraphs=true` | Sends backbone blocks through the fullgraph/static-shape/Inductor CUDA graph branch. | Current production DDP champion path. |
| `train.compile_mode=max-autotune-no-cudagraphs` with `cudagraphs=false` | Config-only isolation test for Triton matmul autotuning without graph replay. | Retestable as a Phase 7 go/no-go experiment. |
| `train.compile_mode=max-autotune-no-cudagraphs` with `cudagraphs=true` | Does not affect backbone blocks; reaches non-backbone heads only. | Not a valid backbone autotune test. |
| Branch-1 code edit to `options={"max_autotune": True, "triton.cudagraphs": True}` | Combines autotuned matmul choices with the current graph-replay backbone path. | Gated Phase 7 experiment, not a default. |

Important distinction:

- CUDA graph replay reduces CPU launch overhead.
- `max_autotune` searches faster matmul kernel configurations.
- They are different optimization axes. The champion path already has graph
  replay for backbone blocks, but has not yet tested backbone max-autotune in
  the same branch.

---

## Current recommended Phase 7 ladder

1. Run nsys residual attribution first. Measure head/loss wall share instead of
   inferring it only from MACs.
2. If autotune is worth testing, run the config-only isolation:
   `train.cudagraphs=false train.compile_mode=max-autotune-no-cudagraphs`.
   Use matched per-GPU batch size and a single-rank warmup so Triton/Inductor
   cache entries are populated before the 8-rank run.
3. If the isolation test shows a real step-time gain, edit branch 1 in
   `wrap_compile_block()` to:

   ```python
   module.compile(
       fullgraph=True,
       dynamic=False,
       options={"max_autotune": True, "triton.cudagraphs": True},
   )
   ```

4. Treat head CUDA-graph/fullgraph work as conditional and likely low ROI unless
   nsys shows dispatch overhead in that path.

---

## Confirmed historical evidence

### Default mode result (2026-04-07, job 16719)

Config: DDP bs=128, 8xH100, 200 iters.

- Steady-state MFU, iters 20-199: about 23.1%, range 22.9-23.5%.
- Throughput: about 4040 img/s.
- This was a pre-Phase-6 screening result, not the final Phase 6.A champion
  measurement.

### Multi-rank autotune OOM hazard (2026-04-07, job 16720)

Observed:

- Job started at 13:31 and crashed at 13:49, about 18 minutes into iter-0
  compile.
- Zero training log lines were produced.
- SIGABRT occurred on rank 2.
- Stderr contained large `triton_mm_XX` benchmarking output from iter-0 compile.

Still-valid interpretation:

- `max-autotune-no-cudagraphs` benchmarks candidate Triton kernels by running
  test tensors on GPU.
- When 8 ranks autotune concurrently, every GPU carries model weights plus
  temporary benchmark allocations.
- This can exceed the 80 GB H100 limit even though steady-state DDP bs=128 uses
  much less VRAM.

Operational implication:

- If retesting autotune, use single-rank warmup at the same per-GPU batch size
  before the 8-rank run.

### Triton/Inductor cache warmup assumption

Autotune cache validity depends on:

- operation shape,
- hardware type,
- PyTorch and Triton version,
- relevant compiler options.

For backbone blocks, nproc=1 and nproc=8 should share shapes when
`train.batch_size_per_gpu` is identical. DDP gradient all-reduce happens outside
the compiled backbone regions and should not change backbone matmul cache keys.

---

## Retracted Phase 4/6 interpretation

Do not use the old Phase 4/6 shape-instability explanation as current context.
It was an unverified hypothesis and is wrong for the default fixed per-GPU
batch-size path.

Retracted interpretation:

- Autotune failures should not be attributed to per-iteration changes in iBOT K
  for the current fixed-bs path.
- A matched-bs single-rank warmup should not be rejected on the basis of world
  size alone; the relevant cache shape is local-batch driven for backbone blocks.
- `max-autotune-no-cudagraphs` is not ruled out by the old shape-instability
  explanation. It remains a retestable isolation experiment with known autotune
  memory-spike risk.
- Earlier verdicts against CUDA-graph autotune modes predate the Phase 6
  cudagraph fixes and should be treated as stale until retested.

Current evidence:

- `dinov3/data/collate.py:42-56` deterministically chooses how many samples are
  masked and each masked sample's target patch count.
- `dinov3/data/masking.py:87-93` calls `complete_mask_randomly()`, which
  force-tops-up every mask to exactly the requested count.
- Therefore, at fixed per-GPU batch size, the number of selected masked patches
  is constant. Mask positions vary; count does not.
- Phase 7 verification reported K = 7503 for 5/5 trials at bs=128.

Caveats:

- Changing per-GPU batch size changes K and can require new compile/cache
  entries.
- Multidistillation and some Gram token-selection modes can change shape
  behavior. They are not part of the current DDP champion config.

---

## Short version

The safe default is still `train.compile_mode=null` with `train.cudagraphs=true`
for the DDP champion path. The old shape-instability explanation for rejecting
`max-autotune-no-cudagraphs` is retracted. If autotune is pursued in Phase 7,
first test it with `cudagraphs=false` as an isolation run; combining autotune
with the champion backbone path requires a code edit in `wrap_compile_block()`,
not just a config override.
