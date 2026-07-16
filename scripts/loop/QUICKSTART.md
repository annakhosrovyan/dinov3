# scripts/loop/ — Quickstart

The five-minute version of how to run one screening candidate and score it. Full
rationale is in `DESIGN.md`; the loop contract is in `README.md`; porting the
verifier to another PyTorch project is in `ADAPTERS.md`.

## What this is

A toolkit for **T2 config/systems screening**: pick a training knob, run a short
(700-iter) job at the fixed operating point, and get a trustworthy verdict on
whether it beats the current champion — improvement / regression / no-evidence /
incomplete (mandatory gate unevaluated — supply the missing input) / invalid
(mandatory gate failed).
It is the standing infrastructure a `/loop` will drive later; today you can drive
it by hand.

## One-time setup

Install the verifier outside the repo so a writer agent can never edit its own
scorer, and record the digest:

```bash
scripts/loop/install_verifier.sh
# copy the printed sha256 into your loop's state.md (see DESIGN.md "trust chain")
```

## Run one candidate

```bash
CANDIDATE_TAG=nw16 \
HYPOTHESIS="fewer workers → less CPU oversubscription (job 80711 lever)" \
EXTRA_OPTS="train.num_workers=16" \
sbatch scripts/loop/run_candidate.sh
```

Channels you can set (see `candidates.example.yaml` for the menu):
- `EXTRA_OPTS` — OmegaConf `key=value` overrides, appended last (config knobs).
- `EXTRA_ENV` — `K=V` env vars, validated (e.g. `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `DINOV3_RANK_CPU_SLICE=1`).
- `CANDIDATE_TAG`, `HYPOTHESIS` — recorded for lineage.

Everything else (recipe, bs=128, 700 iters, env, Slurm resources) is pinned.

## Poll and score

```bash
~/scripts/jobcheck <jobid>          # wait for completion, find the outdir + log

# Always invoke via `python3 -I -B` (isolated + no-bytecode) so no PYTHONPATH /
# user-site / sitecustomize import can hijack the scorer before it protects itself.
python3 -I -B ~/scripts/loop-verifier/score.py <outdir> \
    --baseline /mnt/weka/adovlatyan/runs/phase7-compile-nsys/maxconn/output_maxconn1_ddp_bs128_24w8pf_77672 \
    --expect-iters 700 --world-size 8 \
    --slurm-log /mnt/weka/adovlatyan/logs/<candidate-log>.out \
    --lineage <loop_dir>/lineage.jsonl --tag nw16 --hypothesis "..."
```

Omitting `--slurm-log` leaves the mandatory `per_rank_straggler` gate
`not_evaluated`, so the verdict is `incomplete` (not a certified pass).

Read the verdict line. `candidate-improvement` is **not** a promotion — schedule a
matched-node A/B replicate before believing it (see README rules).

## The three things you must not do

1. **Do not optimize MFU.** Wall img/s is the score; MFU is diagnostic only
   (§7.2: MAX_CONN=8 raised MFU 9.5% while wall img/s got 7.7% worse).
2. **Do not omit `--expect-iters`.** It is driver-owned; omitting it lets a
   candidate that shortened its own schedule self-attest completion.
3. **Do not promote on one run.** Single replicate never promotes; deltas under
   the 5% noise floor stay `no-evidence`.

## Exit codes

`0` = the run was scored — **including when a gate FAILED** (verdict `invalid`) or a
mandatory gate was unevaluated (verdict `incomplete`). Always read the verdict line,
not just the exit code. `2` = could NOT score at all (missing/empty metrics file,
no steady-state points, unparseable JSON) — read the stderr note.
