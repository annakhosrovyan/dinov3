#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-docs/gpt5-pro-dinov3-lateral-source-excerpts.md}"

cat > "$OUT" <<'HEADER'
# GPT-5 Pro DINOv3 Lateral Systems Source Excerpts

Purpose: paste this after the DINOv3 lateral-systems context pack and minimal evidence bundle. These are selected source excerpts for reasoning about multi-crop ViT execution, attention/block scheduling, iBOT/DINO losses, FSDP2 wrapping, and MFU measurement.

Note: `dinov3/train/ssl_meta_arch.py` is intentionally omitted because it was extracted separately.

HEADER

emit_range() {
  local file="$1"
  local start="$2"
  local end="$3"

  {
    printf '\n----- BEGIN %s:%s-%s -----\n\n' "$file" "$start" "$end"
    printf '```python\n'
    sed -n "${start},${end}p" "$file"
    printf '```\n\n'
    printf '%s\n' "----- END ${file}:${start}-${end} -----"
  } >> "$OUT"
}

emit_range "dinov3/models/vision_transformer.py" 180 265
emit_range "dinov3/layers/block.py" 21 210
emit_range "dinov3/layers/attention.py" 43 118
emit_range "dinov3/loss/ibot_patch_loss.py" 61 142
emit_range "dinov3/loss/dino_clstoken_loss.py" 16 124
emit_range "dinov3/data/collate.py" 11 120
emit_range "dinov3/fsdp/ac_compile_parallelize.py" 25 124
emit_range "dinov3/utils/mfu.py" 20 139

printf 'Wrote %s\n' "$OUT"
