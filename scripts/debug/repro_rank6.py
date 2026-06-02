#!/usr/bin/env python3
"""Reproduce the deterministic rank-6 crash (iter ~1310) on CPU, no GPU / no DDP.

Why this works: the training loader uses ShardedInfiniteSampler (cache_dataset=true)
which is fully seeded/deterministic. Rank 6 of 8 = start=6, step=8, seed=train.seed+1.
DataLoader re-raises a worker decode exception in BATCH ORDER, so the offending sample
sits in rank-6's batch ~1310 regardless of num_workers/prefetch. We replay exactly that
index stream single-process and call dataset[idx], so any bad sample throws here with a
full synchronous traceback — and we print the resolved (source, base_idx) for it.
"""
import argparse
import sys
import traceback
from itertools import islice

from omegaconf import OmegaConf

from dinov3.data import make_dataset
from dinov3.data.datasets.augmentation import MAIDAugmentation
from dinov3.data.datasets.hdf5_augmentation import HDF5Augmentation
from dinov3.data.samplers import ShardedInfiniteSampler

# Exact dataset spec from the bs=128 soak scripts.
DATASET_STR = (
    "MixedSatelliteDataset:"
    "intelinair_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/intelinair/intelinair.h5:"
    "maid_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/maid:"
    "sen1_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/sentinel1:"
    "sen1_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/sentinel1_stats:"
    "naip_data_path=/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip:"
    "naip_stats_dir=/mnt/weka/akhosrovyan/re-id/pretraining/satlas_dataset/stats/naip_stats:"
    "naip_weight=1.0"
)


def build_augmentation(cfg):
    """Replicates SSLMetaArch.build_data_augmentation_dino without the model."""
    kw = dict(
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        global_crops_number=2,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )
    maid_aug = MAIDAugmentation(**kw)
    hdf5_aug = HDF5Augmentation(**kw)

    class DatasetAwareAugmentation:
        def __call__(self, image, dataset_name=None):
            return maid_aug(image) if dataset_name == "MAID" else hdf5_aug(image)

    return DatasetAwareAugmentation()


def resolve_source(ds, index):
    """Map a global MixedSatelliteDataset index -> (source_name, base_idx)."""
    for i, (start, end) in enumerate(zip(ds.cumulative_sizes[:-1], ds.cumulative_sizes[1:])):
        if start <= index < end:
            base_idx = (index - start) % ds.dataset_raw_sizes[i]
            return ds.dataset_names[i], base_idx
    return "OUT_OF_RANGE", -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="dinov3/configs/ssl_default_config.yaml")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--start-batch", type=int, default=1280)
    ap.add_argument("--end-batch", type=int, default=1340)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_file)
    seed = int(cfg.train.seed) + 0 + 1  # start_iter=0 -> seed = train.seed + 1

    print(f"Building dataset (transform=multi-crop aug)...", flush=True)
    augmentation = build_augmentation(cfg)
    ds = make_dataset(dataset_str=DATASET_STR, transform=augmentation, target_transform=lambda _: ())
    n = len(ds)
    print(f"Dataset len={n:,}  cumulative_sizes={ds.cumulative_sizes}  names={ds.dataset_names}", flush=True)

    sampler = ShardedInfiniteSampler(
        sample_count=n, shuffle=True, seed=seed,
        start=args.rank, step=args.world_size, use_new_shuffle_tensor_slice=False,
    )

    lo = args.start_batch * args.batch_size
    hi = args.end_batch * args.batch_size
    print(f"Replaying rank={args.rank}/{args.world_size} seed={seed} stream positions "
          f"[{lo:,}:{hi:,}] (batches {args.start_batch}..{args.end_batch})", flush=True)

    indices = list(islice(iter(sampler), lo, hi))
    bad = []
    for pos, idx in enumerate(indices, start=lo):
        src, base = resolve_source(ds, idx)
        batch_no = pos // args.batch_size
        try:
            ds[idx]
        except BaseException as e:  # noqa: BLE001 - we want EVERYTHING, incl. SystemExit
            print(f"\n!!! CRASH at stream_pos={pos} (batch {batch_no}) "
                  f"global_idx={idx} source={src} base_idx={base}", flush=True)
            print(f"    exception: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            bad.append((batch_no, idx, src, base, type(e).__name__, str(e)))

    print("\n===== SUMMARY =====", flush=True)
    if bad:
        for b in bad:
            print(f"  batch={b[0]} idx={b[1]} source={b[2]} base_idx={b[3]} {b[4]}: {b[5]}", flush=True)
        sys.exit(2)
    print(f"  No crash in batches {args.start_batch}..{args.end_batch} for rank {args.rank}. "
          f"Widen the window or check a different rank.", flush=True)


if __name__ == "__main__":
    main()
