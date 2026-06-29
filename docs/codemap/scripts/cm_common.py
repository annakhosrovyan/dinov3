"""Shared helpers for the codemap toolchain. Stdlib + PyYAML only."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import yaml

SCHEMA_VERSION = 1

def load_config(repo_root: Path) -> dict:
    p = Path(repo_root) / "docs/codemap/codemap.config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"missing codemap config: {p}")
    return yaml.safe_load(p.read_text())

def git_sha(repo_root: Path) -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                         check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                           check=True, capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha

def snapshot_dir(repo_root: Path, cfg: dict, sha: str) -> Path:
    d = Path(repo_root) / cfg["snapshots"]["dir"] / sha
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def prune_snapshots(repo_root: Path, cfg: dict) -> list[str]:
    snap_root = Path(repo_root) / cfg["snapshots"]["dir"]
    keep = cfg["snapshots"].get("keep_last", 10)
    dirs = [p for p in snap_root.iterdir() if p.is_dir() and not p.is_symlink()]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    removed = []
    for p in dirs[:-keep] if keep < len(dirs) else []:
        for child in sorted(p.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        p.rmdir(); removed.append(p.name)
    return removed

def update_latest_symlink(snap_root: Path, sha: str) -> None:
    link = Path(snap_root) / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(sha)
