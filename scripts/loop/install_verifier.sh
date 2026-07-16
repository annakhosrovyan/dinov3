#!/bin/bash
# Install the hermetic verifier OUTSIDE the agent-editable worktree.
#
# The loop's writer agent works in a repo worktree and must never be able to
# edit the code that scores it (METR: 30.4% hack rate with a visible/editable
# scorer vs 0.7% opaque). The verifier is now three files — the thin CLI
# (score.py), the reusable engine (score_core.py), and the project adapter
# (adapters/dinov3.py) — so all three are installed read-only and a SINGLE
# COMBINED digest is recorded over them. The loop driver calls the INSTALLED
# score.py and compares the combined digest at each iteration to detect drift.
#
# Re-run after any reviewed change to score_core.py or the adapter (bump: review
# the diff first — the verifier is the one component where a bug corrupts every
# future verdict; see docs/phase7_harness_audit.md for what un-audited
# measurement code costs). ADAPTERS.md documents the engine/adapter contract.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DST_DIR="${HOME}/scripts/loop-verifier"
FILES=(score_core.py score.py adapters/__init__.py adapters/dinov3.py)

# Build the EXACT manifest in a staging dir on the same filesystem, then swap it
# in atomically (Codex GPT-5.6 review, finding 4). Replacing named files in place
# left any pre-existing stray module (sitecustomize.py, a shadow json.py/pathlib.py)
# behind — it would then be importable AND folded into the freshly recorded digest,
# blessing code that never came from this repo. A fresh dir guarantees the
# installed tree is nothing but the manifest.
PARENT="$(dirname "${DST_DIR}")"
mkdir -p "${PARENT}"
STAGE="$(mktemp -d "${PARENT}/.loop-verifier.XXXXXX")"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/adapters"
for f in "${FILES[@]}"; do
  cp "${SRC_DIR}/${f}" "${STAGE}/${f}"
done

# No bytecode in the installed tree (score.py also sets sys.dont_write_bytecode).
find "${STAGE}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Atomic-ish replace (Codex 2026-07-16, finding 6). Move any existing tree ASIDE
# first (an atomic rename), verify the target path is clear, THEN rename staging
# into it. A bare `rm -rf "${DST_DIR}"; mv "${STAGE}" "${DST_DIR}"` has a window
# where, if a concurrent same-user install recreated the dir, `mv` would nest the
# stage INSIDE it and the digest below would attest the wrong tree. The driver's
# per-iteration digest check fails closed if the verifier is briefly absent.
OLD=""
if [ -e "${DST_DIR}" ]; then
  OLD="$(mktemp -d "${PARENT}/.loop-verifier-old.XXXXXX")"
  mv "${DST_DIR}" "${OLD}/tree"
fi
if [ -e "${DST_DIR}" ]; then
  echo "FATAL: ${DST_DIR} reappeared during install (concurrent install?) — aborting to avoid nesting"; exit 70
fi
mv "${STAGE}" "${DST_DIR}"
trap - EXIT
[ -n "${OLD}" ] && rm -rf "${OLD}"
find "${DST_DIR}" -type f -exec chmod 555 {} +

# One combined digest over ALL installed files except the digest file itself
# (sorted, path-relative). Hashing every file — not just *.py — closes the
# bytecode-substitution gap: a pre-planted __pycache__/*.pyc would be imported
# without changing any *.py, but it DOES change this digest, so the driver's
# per-iteration check catches it. Preserves the "single digest in state.md"
# doctrine across the multi-file layout.
COMBINED="$(cd "${DST_DIR}" && find . -type f ! -name 'verifier.sha256' | sort | xargs sha256sum | sha256sum | awk '{print $1}')"
echo "${COMBINED}  (combined over all files in ${DST_DIR} except verifier.sha256)" > "${DST_DIR}/verifier.sha256"
echo "${COMBINED}"
echo ""
echo "installed (read-only): ${DST_DIR}/{score.py, score_core.py, adapters/dinov3.py}"
# Always invoke via `python3 -I -B` (isolated + no-bytecode): -I ignores
# PYTHONPATH/user-site/sitecustomize so a planted import cannot hijack the scorer
# before it protects itself; -B writes no bytecode. (Codex 2026-07-16, finding 5.)
echo "run: python3 -I -B ${DST_DIR}/score.py <run_dir> --baseline <dir> --expect-iters N --world-size 8 --slurm-log <log> ..."
echo ""
echo "TRUST CHAIN (Codex note: a hash file NEXT TO the target only detects"
echo "accidents, not replacement — anyone who can swap the files can swap the"
echo "hash). Record the COMBINED digest above in the loop's state.md at loop"
echo "start; the driver re-derives it from the live files and compares to THAT:"
echo "  cd ${DST_DIR} && find . -type f ! -name 'verifier.sha256' | sort | xargs sha256sum | sha256sum | awk '{print \$1}'"
echo "chmod 555 guards against accidental edits only, not the owning user."
