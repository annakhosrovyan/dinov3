# Rank-6 / iter-~1310 crash — ROOT CAUSE (job 59847)

## Verdict
**NOT** an iBOT-mask, CUDA-graph, dynamic-shape, NaN, or Sinkhorn-collective issue (all prior
hypotheses — mine and Codex's — were wrong). The crash is a **corrupt NAIP PNG tile** whose
**error-recovery path tries to write a log file into akhosrovyan's read-only directory**, raising
an uncaught `PermissionError` that kills the DataLoader worker → kills rank 6 → kills the job.

## The captured traceback (job 59847, rank 6, gpu-node; per-rank log via @record + --redirects=3)
`/mnt/weka/adovlatyan/logs/rank6-capture-59847-perrank/none_okz52y0h/attempt_0/6/{stderr.log,error.json}`

```
libpng error: Read Error
PermissionError: Caught PermissionError in DataLoader worker process 11.
  ...
  mixed_satlas_dataset.py:169   images, band_names, data_path = self.datasets[i][base_idx]
  satlas_datasets.py:342        self.save_error_path(tci_path)
  satlas_datasets.py:58         with open(save_path, "a") as f:
PermissionError: [Errno 13] Permission denied:
  '/mnt/weka/akhosrovyan/re-id/pretraining/satlas-dataset-v1-naip-2020/naip_error_paths.txt'
```

## Exact chain (source-grounded)
1. `satlas_datasets.py:338` NAIP `__getitem__` does `cv2.imread(tci_path)` on a PNG tile.
   The tile is corrupt/truncated → libpng prints `Read Error`, `tci_img` is `None`.
2. `satlas_datasets.py:341` detects `tci_img is None` and enters the **self-heal recovery**:
   `save_error_path()` → `_invalidate_index()` → `_sample_ok_index()` → `continue`. This design
   is correct: skip the bad tile, log it, resample a good one.
3. `satlas_datasets.py:342` → `save_error_path(tci_path)` (`:56-59`) computes
   `save_path = dirname(self.data_path)/<basename>_error_paths.txt` — i.e. **inside the dataset's
   own directory, owned by akhosrovyan**. `open(save_path, "a")` raises `PermissionError` because
   `adovlatyan` has read-but-not-write access to `/mnt/weka/akhosrovyan/re-id/...`.
4. The `PermissionError` is uncaught → propagates out of the worker → DataLoader re-raises it in
   the main process (`fetch.py:52` → `reraise`) → rank 6's `do_train` loop dies.

## Why it is deterministic (always rank 6, always iter ~1310, 4 jobs / 2 nodes)
The determinism is an **observed fact** (4 jobs, 2 nodes, same rank, same iter). Mechanism,
corrected after Codex review + source check:
- `_sample_ok_index()` (`satlas_datasets.py:111-114`) uses the **global numpy RNG**
  (`np.random.randint`).
- The SSL training loader supplies **no `worker_init_fn`** (`loaders.py:210,250` default `None`;
  only the eval/depth + eval/seg paths set one). So there is **no per-worker numpy re-seed** — my
  earlier "seeded per worker" wording was wrong.
- Instead: `fix_random_seeds(seed + rank)` runs in each rank's **main process**
  (`config.py:165,206` → `utils.py:87 np.random.seed(seed+rank)`). On Linux (fork start method)
  DataLoader workers **inherit the parent's numpy RNG state at fork**. So rank 6's workers all
  start from a numpy state seeded `seed+6`, identical across runs → the `_sample_ok_index` walk is
  reproducible run-to-run and reaches the same corrupt tile at the same global step.
- Caveat (Codex): this fork-inheritance is more fragile than explicit seeding — a switch to
  `spawn`, a different torch version, or adding a `worker_init_fn` could shift *which* tile/iter
  triggers it. The crash class (corrupt tile → fatal log write) is robust; the exact iter is not.

## Why the CPU repro (job 59814) FALSELY cleared __getitem__
The repro replayed only the **outer ShardedInfiniteSampler index stream** and called `ds[idx]`
single-process. It did **not** model (a) the NAIP dataset's internal **ok-index pool** remapping
(`__getitem__:332-333` rewrites the incoming index through `ok_index_pos`), nor (b) the
**per-worker global numpy RNG state** that drives `_sample_ok_index`. With a different `np.random`
state, the repro sampled different tiles and never hit the corrupt PNG. Lesson: reproducing the
sampler stream ≠ reproducing the dataset's internal stochastic index selection.

## Fix surface (NOT yet implemented — for review). Codex independent review: PARTIALLY AGREE.
Codex agreed with the root-cause chain from source (it could not read the Weka traceback —
sandbox has no /mnt/weka), corrected the determinism wording (above), and added two refinements.
Recommended fix = **A + B + C, applied to all three datasets** (Sen1/Sen2/NAIP):
- **A. Make `save_error_path` non-fatal**: wrap `open(...,"a")` in `try/except OSError` (warn +
  return). Preserves skip-bad-tile self-heal; never fatal when the dir is read-only. Catching
  `OSError` is safe — the real recovery is `_invalidate_index` shrinking the pool, not the log.
- **B. Redirect the error log to a writable path** (`~/.cache/dinov3_manifests/` or
  `/mnt/weka/adovlatyan/logs/`) so the bad-tile audit trail survives on read-only datasets.
- **C (Codex add): bound the `while True` recovery loops** at `satlas_datasets.py:179` (Sen1),
  `:257` (Sen2), `:334` (NAIP). `_sample_ok_index` only raises when `ok_indices` is fully
  exhausted; if a whole shard is unreadable, one rank spins resampling while the others reach the
  DDP collective → **deadlock/hang risk**. Cap retries at `len(self.ok_indices)` before raising.
- **Same unguarded `save_error_path` pattern exists in Sen1 (`:189,:196,:201`) and Sen2
  (`:265,:274,:280`)** — fix all three together.
Open design question: make bad-tile logging opt-in via config?
NOTE: libpng "Read Error" could be a *transient* Weka I/O failure rather than a permanently
corrupt file. Codex: this does not change the fix — a read failure must never become a fatal
write to a read-only dir, transient or not. (If transient, A+B+C also makes the run self-recover.)
