# GPT-5 Pro Content Organization

This folder is organized by prompt goal. Prefer sending GPT-5 Pro one prompt-specific folder plus
only the shared files needed for that question.

## `ddp-prompt/`

Use for current Phase 6 ideation around DDP, CUDA graphs, static-shape iBOT/head/loss work, and
delivered wall-clock throughput.

Files:

- `ideation_prompt.md` — the actual GPT-5 Pro prompt for DDP + CUDA graph brainstorming.
- `compressed_phase6_perf_plan.md` — compact Phase 6 context. Paste this instead of the full
  `docs/phase6_perf_plan.md`.

Add here later:

- DDP-specific nsys or profiler summaries.
- DDP-specific source excerpts if they differ from the shared excerpts.
- Multi-node DDP/NCCL evidence once available.

## `fsdp2-prompt/`

Use for the older Phase 5 FSDP2 question: why bs=128 did not beat bs=96 under FSDP2/selective AC,
and whether communication/resharding/AC explains the result.

Files:

- `context_pack.md` — FSDP2 bs=128 vs bs=96 prompt scaffold and attachment list.
- `minimal_evidence_bundle.md` — compressed FSDP2 evidence bundle for the Phase 5 puzzle.

Add here later:

- FSDP2-specific nsys summaries.
- FSDP2 reshard / AC sweep summaries.
- Any refreshed conservative-baseline context for `run.sh`.

## `shared/`

Use for reusable DINOv3 architecture/source context that can support either DDP or FSDP2 prompts.
Do not duplicate these into prompt-specific folders unless the excerpt has been edited for that
specific question.

Files:

- `lateral_systems_context_pack.md` — broad lateral-systems framing. Useful for both prompts, but
  older sections mention FSDP2; treat it as shared background, not current Phase 6 truth.
- `lateral_source_excerpts.md` — selected source excerpts for multi-crop ViT execution, losses,
  block scheduling, wrapping, and MFU measurement. Useful for both prompts.

Good shared candidates if created later:

- compact architecture summary;
- source excerpts for `train.py`, `ssl_meta_arch.py`, `collate.py`, DINO/iBOT losses;
- measurement semantics: GPU-event throughput vs wall-clock throughput;
- MFU formula/context.

## Quick Choice

- Current DDP + cudagraph performance brainstorming: paste `ddp-prompt/ideation_prompt.md`,
  `ddp-prompt/compressed_phase6_perf_plan.md`, then selected `shared/` excerpts.
- Old FSDP2 bs=128 vs bs=96 analysis: paste `fsdp2-prompt/context_pack.md`,
  `fsdp2-prompt/minimal_evidence_bundle.md`, then selected `shared/` excerpts.
- Broad algorithmic/systems ideation: start with `shared/lateral_systems_context_pack.md`, but
  pair it with either `ddp-prompt/compressed_phase6_perf_plan.md` or
  `fsdp2-prompt/minimal_evidence_bundle.md` so the model knows which era/config is current.
