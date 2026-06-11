#!/usr/bin/env python3
"""Regression test for the rank-6 / iter~1310 soak crash root cause.

The bug: SatlasDataset.save_error_path() wrote a bad-tile log *into the dataset
directory*. When that directory is read-only (datasets owned by another user),
the write raised PermissionError inside a DataLoader worker and killed the rank.
Root-cause write-up: scripts/debug/rank6_root_cause.md.

This test asserts the fix:
  1. PRE-FIX BEHAVIOR (reproduction): writing into a read-only dir raises OSError.
  2. POST-FIX: save_error_path() redirects to DINOV3_ERROR_LOG_DIR and never raises,
     even when the dataset directory is read-only.

Run: python scripts/debug/test_save_error_path_fix.py   (no GPU, no cluster needed)
"""
import os
import stat
import sys
import tempfile

from dinov3.data.datasets import satlas_datasets
from dinov3.data.datasets.satlas_datasets import SatlasDataset


def _make_ds(data_path):
    # Build a SatlasDataset shell without touching disk-backed init: we only exercise
    # save_error_path, which needs nothing but self.data_path.
    ds = SatlasDataset.__new__(SatlasDataset)
    ds.data_path = data_path
    return ds


def test_part1_reproduces_original_crash():
    """A raw write into a read-only dir raises — this is the original fatal path."""
    with tempfile.TemporaryDirectory() as tmp:
        ro_dataset_dir = os.path.join(tmp, "someone_elses_dataset")
        os.makedirs(ro_dataset_dir)
        os.chmod(ro_dataset_dir, stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
        try:
            with open(os.path.join(ro_dataset_dir, "x_error_paths.txt"), "a") as f:
                f.write("bad_tile\n")
            print("PART 1: FAIL — expected PermissionError writing to read-only dir")
            return False
        except OSError:
            print("PART 1: OK — confirmed a raw write into a read-only dataset dir raises (the original crash)")
            return True
        finally:
            os.chmod(ro_dataset_dir, stat.S_IRWXU)


def test_part2_fix_redirects_and_never_raises(monkeypatched_dir):
    """save_error_path must redirect to the writable dir and not raise, even when the
    dataset dir is read-only."""
    with tempfile.TemporaryDirectory() as tmp:
        ro_dataset_dir = os.path.join(tmp, "akhosrovyan_dataset")
        os.makedirs(ro_dataset_dir)
        os.chmod(ro_dataset_dir, stat.S_IRUSR | stat.S_IXUSR)  # simulate read-only Weka
        try:
            ds = _make_ds(os.path.join(ro_dataset_dir, "naip"))
            # Should NOT raise — redirected to monkeypatched_dir, guarded against OSError.
            ds.save_error_path("/some/corrupt/tile.png")
            expected = os.path.join(monkeypatched_dir, "naip_error_paths.txt")
            if not os.path.exists(expected):
                print(f"PART 2: FAIL — redirected log not found at {expected}")
                return False
            with open(expected) as f:
                contents = f.read()
            if "/some/corrupt/tile.png" not in contents:
                print("PART 2: FAIL — bad-tile path not recorded in redirected log")
                return False
            print(f"PART 2: OK — save_error_path redirected to {expected} and did not raise")
            return True
        finally:
            os.chmod(ro_dataset_dir, stat.S_IRWXU)


def test_part3_guard_survives_unwritable_target():
    """Even if the *redirect* target is itself unwritable, the guard must swallow it."""
    with tempfile.TemporaryDirectory() as tmp:
        ro_target = os.path.join(tmp, "ro_target")
        os.makedirs(ro_target)
        os.chmod(ro_target, stat.S_IRUSR | stat.S_IXUSR)
        satlas_datasets._ERROR_LOG_DIR = ro_target
        try:
            ds = _make_ds("/anything/naip")
            ds.save_error_path("/corrupt/tile.png")  # must NOT raise (guard warns + continues)
            print("PART 3: OK — guard swallowed OSError when the redirect target is read-only")
            return True
        except OSError:
            print("PART 3: FAIL — guard did not swallow OSError")
            return False
        finally:
            os.chmod(ro_target, stat.S_IRWXU)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as writable:
        # Point the redirect at a writable dir for part 2.
        satlas_datasets._ERROR_LOG_DIR = writable
        results = [
            test_part1_reproduces_original_crash(),
            test_part2_fix_redirects_and_never_raises(writable),
            test_part3_guard_survives_unwritable_target(),
        ]
    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
    sys.exit(0 if all(results) else 1)
