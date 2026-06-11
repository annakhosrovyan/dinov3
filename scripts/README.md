# Script Layout

Most scripts are Slurm recipes. Submit them from the repository root so their
cd "${SLURM_SUBMIT_DIR}" and PYTHONPATH=. assumptions keep pointing at this
repo checkout.

## Groups

- smoke/: quick validation and MFU baseline jobs.
- screening/: short-horizon comparison runs, compile-mode probes, and DDP/FSDP screening recipes.
- fsdp2/: FSDP2-specific experiment recipes, including activation-checkpointing, reshard, single-GPU, and NCCL sweep runs.
- profiling/: memory profiling, fragmentation probes, nsys capture, and nsys summary tooling.
- soak/: longer stability or memory-creep runs.
- docs/: helper scripts for generating or extracting documentation artifacts.

## Common Commands

```bash
sbatch scripts/smoke/mfu_validation_run.sh
sbatch scripts/screening/ddp_bs96_calibration.sh
python scripts/profiling/nsys_dinov3_summary.py /path/to/trace.sqlite
```
