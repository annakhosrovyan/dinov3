#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-docs/gpt-5-pro-content/ddp-prompt/minimum_useful_bundle_source_excerpts.md}"

mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<'HEADER'
# GPT-5 Pro DDP Prompt Minimum Useful Bundle Source Excerpts

Purpose: paste this after `ddp-prompt/compressed_phase6_perf_plan.md` when using
`ddp-prompt/ideation_prompt.md`. This file consolidates the Python source excerpts
called out by the DDP prompt's "Minimum useful bundle" into one markdown artifact.

Notes:

- These are cropped excerpts, not full files.
- The cuts are centered on the DDP + CUDA-graph execution path, dynamic iBOT masking,
  loss/collective behavior, and the train-loop timing/logging boundary.
- `compressed_phase6_perf_plan.md` remains a separate companion file.

HEADER

emit_range() {
  local title="$1"
  local file="$2"
  local start="$3"
  local end="$4"
  local language="${5:-python}"

  {
    printf '\n## %s\n\n' "$title"
    printf 'Source: `%s:%s-%s`\n\n' "$file" "$start" "$end"
    printf '%s\n\n' "----- BEGIN ${file}:${start}-${end} -----"
    printf '```%s\n' "$language"
    sed -n "${start},${end}p" "$file"
    printf '```\n\n'
    printf '%s\n' "----- END ${file}:${start}-${end} -----"
  } >> "$OUT"
}

emit_range \
  "1. Compile and CUDA-Graph Gating for Backbone Blocks" \
  "dinov3/fsdp/ac_compile_parallelize.py" \
  24 \
  126

emit_range \
  "2. DDP Wrapping Path Used by the Current Champion Configuration" \
  "dinov3/fsdp/ac_compile_parallelize.py" \
  315 \
  353

emit_range \
  "3. Forward/Backward Step Skeleton in SSLMetaArch" \
  "dinov3/train/ssl_meta_arch.py" \
  379 \
  465

emit_range \
  "4. Teacher Path: Masked-Patch Gather and Centering" \
  "dinov3/train/ssl_meta_arch.py" \
  467 \
  510

emit_range \
  "5. Student Path: Joint Global/Local Backbone Pass and Head Dispatch" \
  "dinov3/train/ssl_meta_arch.py" \
  566 \
  618

emit_range \
  "6. Loss Assembly: DINO, iBOT, and Weighting Logic" \
  "dinov3/train/ssl_meta_arch.py" \
  620 \
  720

emit_range \
  "7. Collate Path: Dynamic iBOT Mask Construction and n_masked Derivation" \
  "dinov3/data/collate.py" \
  10 \
  77

emit_range \
  "8. iBOT Patch Loss and Center Update Path" \
  "dinov3/loss/ibot_patch_loss.py" \
  60 \
  141

emit_range \
  "9. DINO CLS-Token Loss and Async Center All-Reduce Path" \
  "dinov3/loss/dino_clstoken_loss.py" \
  15 \
  123

emit_range \
  "10. Train Loop: MFU Setup, CUDA-Event Timing, Logging, and Wall-Clock Boundary" \
  "dinov3/train/train.py" \
  427 \
  747

emit_range \
  "11. Train Loop Tail: Eval and Checkpoint Synchronization Points" \
  "dinov3/train/train.py" \
  758 \
  789

printf 'Wrote %s\n' "$OUT"
