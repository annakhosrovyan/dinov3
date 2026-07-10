#!/usr/bin/env python3
"""Hermetic T2 verifier for dinov3 screening runs — thin CLI.

All logic is in the reusable engine (score_core.py); everything dinov3-specific
is in adapters/dinov3.py. This file only wires the two together, so the CLI,
flags, and behavior are unchanged from the original single-file score.py.

To verify a different PyTorch training project: copy score_core.py verbatim,
write adapters/<project>.py (see ADAPTERS.md), and make a 4-line sibling of this
file that imports that adapter.

Deployment: run the INSTALLED copy (~/scripts/loop-verifier/, see
install_verifier.sh). Exit code: 0 = scored (gates may still FAIL — read the
JSON), 2 = could not score.
"""

import sys

# Do not read/write __pycache__ for the verifier: the hermetic trust chain hashes
# the on-disk tree, and a pre-planted .pyc would otherwise be imported without
# changing any source hash. install_verifier.sh hashes ALL files (so a planted
# .pyc is caught) and this keeps the installed tree from accreting .pyc at runtime.
sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_core import run_cli  # noqa: E402
from adapters.dinov3 import ADAPTER  # noqa: E402

if __name__ == "__main__":
    run_cli(ADAPTER)
