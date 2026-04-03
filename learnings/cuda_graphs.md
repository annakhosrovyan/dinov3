# CUDA Graphs Learnings — DINOv3

What CUDA graphs means in this repo, why upstream left it off by default, and why the current satellite ViT-B recipe is a plausible candidate for it.

---

## What "compiled backbone blocks use cudagraphs" means here (2026-04-02)

In this repo, CUDA graphs are not manually capturing the entire training step. The implementation is narrower:

- `train.compile=true` enables `module.compile()` during model setup.
- If `train.cudagraphs=true`, only backbone blocks take the stricter path:
  - `fullgraph=True`
  - `dynamic=False`
  - `options={"triton.cudagraphs": True}`
- Everything else is still just compiled normally, without the explicit cudagraphs option.

Code path:
- `dinov3/fsdp/ac_compile_parallelize.py:65-69` defines `wrap_compile_block(...)`.
- `dinov3/fsdp/ac_compile_parallelize.py:84-86` applies that only to transformer `model.blocks`.
- Non-backbone modules are compiled through the generic path in `dinov3/fsdp/ac_compile_parallelize.py:179-181`.

**Why non-trivial**: "CUDA graphs enabled" can sound like "the whole step is graphed." That is not what this code does. The graphed region is the compiled backbone block forward, not the entire Python training loop, optimizer step, logging, checkpointing, etc.

**Decision implication**: Expect a narrower but safer win. This knob is about reducing launch overhead and stabilizing execution inside repeated transformer blocks, not about converting the full trainer into one giant captured graph.

---

## What a backbone block is here, and in general (2026-04-02)

In general:
- The **backbone** is the main feature extractor before task-specific heads.
- A **block** is one repeated layer unit inside that backbone.

In this repo specifically:
- `SSLMetaArch` builds `student["backbone"]`, `teacher["backbone"]`, and separate heads such as `dino_head` and `ibot_head`.
- For ViT models, the backbone is `DinoVisionTransformer`.
- A ViT backbone block is one `SelfAttentionBlock` inside `DinoVisionTransformer.blocks`.

Code path:
- `dinov3/train/ssl_meta_arch.py:47-56` builds the backbones and records `embed_dim`.
- `dinov3/models/vision_transformer.py:131-148` constructs a `blocks_list` of `SelfAttentionBlock`.
- `dinov3/models/vision_transformer.py:151` stores them as `self.blocks = nn.ModuleList(blocks_list)`.
- `dinov3/fsdp/ac_compile_parallelize.py:84-86` compiles each of those blocks individually.

For the active recipe:
- `student.arch=vit_base`
- `vit_base` maps to `embed_dim=768`, `depth=12`
- so the backbone is a 12-block ViT-B/16, and those 12 transformer blocks are the "backbone blocks" relevant to CUDA graphs.

Code path:
- `dinov3/configs/ssl_default_config.yaml:87` sets `student.arch: vit_base`
- `run.sh` and `latest_run.sh` both override `student.arch=vit_base`
- `dinov3/models/vision_transformer.py:344-349` defines `vit_base(...)` with `embed_dim=768`, `depth=12`

**Why non-trivial**: It is easy to confuse the backbone width/depth with head settings like `head_hidden_dim=2048`. CUDA graphs in this code are targeting the repeated ViT trunk blocks, not the DINO/iBOT projection heads.

**Decision implication**: If CUDA graphs help here, they help the repeated ViT trunk first. Any benefit on the heads is secondary because they do not take the explicit cudagraphs path.

---

## What upstream DINOv3 actually did (2026-04-02)

The CUDA-graphs path is not a local satellite fork invention. It was already present in the imported DINOv3 code and still matches the public upstream project structure:

- upstream wires `train.cudagraphs`
- upstream defaults it to `false`
- upstream only applies the stricter cudagraphs compile mode to backbone blocks

Local git evidence:
- `git blame` shows `dinov3/fsdp/ac_compile_parallelize.py:65-69` came from the initial import commit
- `git blame` shows `dinov3/configs/ssl_default_config.yaml:80-81` had `compile: true` and `cudagraphs: false` in the initial import commit

Interpretation:
- upstream considered CUDA graphs useful enough to wire in
- upstream did not consider them robust enough to make the global default

**Why non-trivial**: The important upstream signal is not only that the feature exists; it is that they paired it with `dynamic=False` and left the config default at `false`. That is a design statement: this is an opt-in steady-state optimization, not a universally safe default.

**Decision implication**: Treat `train.cudagraphs=true` as a recipe-level benchmark flag, not as a config that should be enabled blindly across all training jobs.

---

## Why CUDA graphs are off by default (best-supported reading) (2026-04-02)

There is no explicit comment saying "off because X", but the implementation and the training code strongly suggest the reason:

- cudagraph mode requires a more static execution shape than plain `module.compile()`
- the repo supports crop configurations that are not universally single-shape across all recipes
- the repo also has explicit `reset_cudagraph_trees()` hooks for feature changes like FP8 and 2:4 sparsity

Evidence:
- `dinov3/fsdp/ac_compile_parallelize.py:67` uses `dynamic=False`
- `dinov3/train/train.py:368-410` supports multi-resolution loader configuration
- `dinov3/layers/fp8_linear.py:138-140` resets cudagraph trees after FP8 conversion
- `dinov3/layers/sparse_linear.py:87-89` resets cudagraph trees after sparsity updates

This means the intended usage is:
- plain compile as the robust baseline
- cudagraphs only when the recipe settles into a stable repeated shape

**Why non-trivial**: "Dynamic values" and "dynamic shapes" are different. CUDA graphs can tolerate changing tensor values; the main risk is changing tensor shapes or graph structure often enough that capture stability disappears.

**Decision implication**: Re-capturing after occasional phase changes can still be useful, but that is not the main optimization model. The main optimization model is long steady-state reuse of the same captured compiled region.

---

## Current satellite recipe: why it is a plausible cudagraph candidate (2026-04-02)

The current training recipe is materially more graph-friendly than the general codebase:

- `student.arch=vit_base`
- fixed global crop size: `224`
- fixed local crop size: `96`
- fixed crop counts: `2` global, `8` local
- fixed patch size: `16`
- fixed per-run batch size
- `train.compile=true`
- `train.cudagraphs=false`

Code path:
- `dinov3/configs/ssl_default_config.yaml:80-81` sets `compile: true`, `cudagraphs: false`
- `dinov3/configs/ssl_default_config.yaml:87-88` sets `student.arch: vit_base`, `patch_size: 16`
- `dinov3/configs/ssl_default_config.yaml:141-147` sets `global_crops_size: 224`, `local_crops_number: 8`, `local_crops_size: 96`
- `dinov3/train/ssl_meta_arch.py:369-403` uses exactly `n_global_crops = 2` and `n_local_crops = self.n_local_crops`
- `dinov3/train/ssl_meta_arch.py:537-547` forwards global and local crops through the student backbone jointly as two fixed-shape tensors
- `run.sh` keeps the default cudagraphs setting and explicitly requests `gpu:h100:8`
- `latest_run.sh` also keeps the default cudagraphs setting and does not override crop or cudagraph settings

What this means in practice:
- every iteration still has changing data values and changing mask contents
- but the main backbone tensor shapes look stable for this recipe
- stable shapes are the key precondition for the current cudagraph path

**Why non-trivial**: The codebase as a whole is more dynamic than the active recipe. The right question is not "can all DINOv3 recipes use cudagraphs?" but "is this specific recipe stable enough to try?"

**Decision implication**: For this project, `train.cudagraphs=true` is worth benchmarking on the current ViT-B satellite recipe.

---

## Important caveats: why "plausible candidate" is not "guaranteed win" (2026-04-02)

Several things can still reduce or erase the benefit:

- The code only applies cudagraphs to backbone blocks, not the full step.
- Training still includes Python-side orchestration, FSDP, EMA update, reductions, optimizer step, and logging outside the graphed region.
- `drop_path_rate=0.3` means the transformer block training path uses per-iteration randomness, although tensor shapes remain fixed.
- If a future recipe switches to multi-resolution crop lists or changes batch shape across phases, the stricter static path may stop paying off.

Evidence:
- `dinov3/configs/ssl_default_config.yaml:90` sets `drop_path_rate: 0.3`
- `dinov3/layers/block.py:134-176` uses `torch.randperm(...)` and subset indexing in the training path when sample drop is active

Interpretation:
- dynamic values and RNG do not automatically kill cudagraph viability
- but they narrow the confidence margin compared with a purely deterministic static block

**Decision implication**: Benchmark first. Do not assume a gain just because the recipe is stable-shaped.

---

## Environment and hardware evidence behind this conclusion (2026-04-02)

Repo-level evidence:
- `dinov3/train/train.py:39` requires PyTorch `>= 2.1`
- local environment reports `torch 2.5.1+cu121`
- local environment reports `torch.version.cuda == 12.1`
- the current local shell environment has no visible GPU, so GPU capability was inferred from training scripts rather than direct runtime inspection
- `run.sh` explicitly requests `--gres=gpu:h100:8`
- `dinov3/utils/mfu.py` uses H100 BF16 dense peak as the default MFU denominator

Practical reading:
- the software stack is new enough to support the current `torch.compile` + Triton cudagraph path
- the intended target hardware for the main training script is 8x H100
- `latest_run.sh` does not pin H100 explicitly, so the strongest hardware claim is attached to `run.sh` and to the MFU code assumptions

**Why non-trivial**: "Repo supports cudagraphs" and "our actual training environment is set up to benefit" are different claims. The first is about code paths; the second also needs recipe stability, GPU class, and a modern PyTorch stack.

**Decision implication**: The current evidence supports "viable to benchmark on our H100-targeted ViT-B recipe with PyTorch 2.5.1+cu121," not "universally beneficial on every machine or launch script."

---

## Experimental result: CUDA graphs FAIL on multi-crop architecture (2026-04-02)

**Tested** `train.cudagraphs=true` on the current ViT-B satellite recipe (jobs 7564, 7565).

**Result**: Runtime crash with:
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run.
```

**Root cause**: `forward_features_list` in `dinov3/layers/block.py:194` processes global
crops (197 tokens) and local crops (37 tokens) **sequentially** through the same compiled
blocks. The CUDAGraph tree captures the first forward (global), but the second forward
(local) reuses the same tensors with different shapes/strides, triggering the overwrite error.

**Error location**: `block.py:210` → `_forward_list` → `x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))`

**Fix would require**: Either `torch.compiler.cudagraph_mark_step_begin()` between the
global and local crop forwards, or restructuring `forward_features_list` to batch both
resolutions (which requires padding or separate block forward calls).

**Decision**: CUDA graphs are **not viable** on this multi-crop DINOv3 architecture without
non-trivial code changes. Deprioritized in favor of DDP switch and batch-size scaling.

---

## Bottom line (2026-04-02, updated with experimental result)

Conclusion:
- upstream DINOv3 wired CUDA-graph-enabled compile for backbone blocks and left it off
- the satellite recipe LOOKED like a plausible candidate due to fixed shapes
- **but the multi-crop sequential forward (global then local through same blocks) breaks CUDAGraph tree**
- this is a fundamental architectural incompatibility, not a configuration issue

Recommended operating rule:
- keep `compile=true` as baseline
- do NOT enable `cudagraphs=true` on the current multi-crop recipe
- if the architecture is changed to batch or separate the multi-crop forwards, re-test
