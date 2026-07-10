"""dinov3 adapter for the T2 verifier engine (score_core.py).

Everything project-specific lives here. To verify a DIFFERENT PyTorch training
project, copy score_core.py verbatim and write a sibling of this file — see
ADAPTERS.md for the field-by-field contract. Keep this file small: it is only
names, a column map, two parsers, and two key-sets. All gate/verdict LOGIC is in
the engine and must not be duplicated here.
"""

import re
from collections import defaultdict
from pathlib import Path

from score_core import Adapter, Columns

# --- per-rank straggler parser -------------------------------------------------
# dinov3 emits these to stderr when DINOV3_PERRANK_DIAG=1 (run_candidate.sh sets
# it). data_time_max is optional: logs from before its addition to helpers.py
# lack it. The engine's straggler_gate consumes the parsed dicts.
RANKDATA_RE = re.compile(r"\[RANKDATA\] rank=(\d+) iter=(\d+) data_time=([\d.eE+-]+)(?: data_time_max=([\d.eE+-]+))?")


def parse_per_rank(slurm_log, skip_iters):
    per_rank_avg = defaultdict(list)
    per_rank_max = defaultdict(list)
    with open(slurm_log, errors="replace") as f:
        for line in f:
            m = RANKDATA_RE.search(line)
            if m and int(m.group(2)) >= skip_iters:
                per_rank_avg[int(m.group(1))].append(float(m.group(3)))
                if m.group(4) is not None:
                    per_rank_max[int(m.group(1))].append(float(m.group(4)))
    return per_rank_avg, per_rank_max


def extra_notes(run_dir):
    """dinov3 writes offending tensors to nan_logs/ when a NaN is caught. A
    non-empty dir means the run hit (and logged) numerical trouble worth a look."""
    notes = []
    nan_logs = Path(run_dir) / "nan_logs"
    if nan_logs.is_dir() and any(nan_logs.iterdir()):
        notes.append("nan_logs/ is non-empty — inspect before trusting this run")
    return notes


def read_expected_iters(run_dir):
    """Best-effort: OFFICIAL_EPOCH_LENGTH * epochs from config.yaml (regex — no yaml dep)."""
    cfg = Path(run_dir) / "config.yaml"
    if not cfg.is_file():
        return None
    text = cfg.read_text()
    epoch_len = re.search(r"OFFICIAL_EPOCH_LENGTH:\s*(\d+)", text)
    epochs = re.search(r"^\s*epochs:\s*(\d+)", text, re.MULTILINE)
    if epoch_len:
        return int(epoch_len.group(1)) * (int(epochs.group(1)) if epochs else 1)
    return None


ADAPTER = Adapter(
    name="dinov3",
    metrics_file="training_metrics.json",
    config_file="config.yaml",
    columns=Columns(
        iteration="iteration",
        iter_time="iter_time",
        batch_size="global_batch_size",
        loss="total_loss",
        data_time="data_time",
        diag_metric_src="mfu",
        diag_metric_out="mfu_pct",
        diag_metric_label="MFU",
        extra_diag=[
            ("step_time_ms", "step_time_ms_cuda_event"),
            ("images_per_sec", "images_per_sec_cuda_event"),
        ],
    ),
    # config.yaml keys allowed to differ from baseline without failing integrity.
    # These are the named EXTRA_OPTS knobs of the menu (candidates.example.yaml).
    # Env-channel knobs (MAX_CONN, OMP, DINOV3_RANK_CPU_SLICE) never appear in
    # config.yaml and are invisible to this diff — the driver owns those.
    allowed_diff_keys={
        "train.num_workers",
        "train.prefetch_factor",
    },
    # Run-identity keys that always differ and carry no objective content.
    ignored_diff_keys={
        "train.output_dir",
        "wandb.group",
        "wandb.run_name",
    },
    read_expected_iters=read_expected_iters,
    parse_per_rank=parse_per_rank,
    extra_notes=extra_notes,
    # 8×H100 single-node operating point. The straggler gate FAILS if the [RANKDATA]
    # log is missing a rank (a candidate can't hide a starving rank by dropping it).
    # Cross-node runs (bs192, 16 ranks) override with --world-size.
    expected_world_size=8,
)
