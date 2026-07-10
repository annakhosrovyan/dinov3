# scripts/loop/ — Quickstart

The five-minute version of how to run one screening candidate and score it. Full
rationale is in `DESIGN.md`; the loop contract is in `README.md`; porting the
verifier to another PyTorch project is in `ADAPTERS.md`.

## What this is

A toolkit for **T2 config/systems screening**: pick a training knob, run a short
(700-iter) job at the fixed operating point, and get a trustworthy verdict on
whether it beats the current champion — better / worse / no-evidence / invalid.
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

~/scripts/loop-verifier/score.py <outdir> \
    --baseline /mnt/weka/adovlatyan/runs/phase7-compile-nsys/maxconn/output_maxconn1_ddp_bs128_24w8pf_77672 \
    --expect-iters 700 \
    --slurm-log /mnt/weka/adovlatyan/logs/<candidate-log>.out \
    --lineage <loop_dir>/lineage.jsonl --tag nw16 --hypothesis "..."
```

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

`0` scored (read verdict), `2` cannot score (gate/parse failure — read the note).
