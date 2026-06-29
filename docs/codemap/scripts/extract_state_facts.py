"""Project-state FACTS only (no narrative). Stdlib + git."""
from __future__ import annotations
import subprocess, glob, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_common as cm

def _run(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout

def _branch_exists(repo, b):
    return subprocess.run(["git", "rev-parse", "--verify", "-q", b], cwd=repo,
                          capture_output=True).returncode == 0

def vs_compare(repo_root, branch) -> dict:
    if not branch or not _branch_exists(repo_root, branch):
        return {"branch": branch, "ahead": None, "behind": None, "changed_files": []}
    counts = _run(repo_root, "rev-list", "--left-right", "--count", f"{branch}...HEAD").split()
    behind, ahead = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (None, None)
    changed = []
    for line in _run(repo_root, "diff", "--numstat", f"{branch}...HEAD").splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            add, dele, path = parts
            changed.append({"path": path, "added": None if add == "-" else int(add),
                            "deleted": None if dele == "-" else int(dele)})
    return {"branch": branch, "ahead": ahead, "behind": behind, "changed_files": changed}

def recent_commits(repo_root, n=10) -> list:
    fmt = "%h\x1f%s\x1f%cI"
    out = _run(repo_root, "log", f"-{n}", f"--pretty={fmt}")
    res = []
    for line in out.splitlines():
        sha, subj, date = line.split("\x1f")
        res.append({"sha": sha, "subject": subj, "date": date})
    return res

def phase_doc_headings(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.rstrip() for ln in p.read_text().splitlines() if ln.lstrip().startswith("#")]

def _memory_index(memory_dir) -> list:
    d = Path(memory_dir) if memory_dir else None
    if not d or not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        desc = ""
        for ln in f.read_text().splitlines():
            if ln.startswith("description:"):
                desc = ln.split(":", 1)[1].strip(); break
        out.append({"name": f.stem, "desc": desc})
    return out

def _recent_jobs(job_glob) -> list:
    if not job_glob:
        return []
    files = sorted(glob.glob(job_glob, recursive=True), key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0)[-10:]
    return [{"log": f} for f in files]  # jobid/state enrichment is the skill's job

def build_state_facts(repo_root, cfg) -> dict:
    st = cfg.get("state", {})
    branch = _run(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    phase = []
    for pd in st.get("phase_docs", []):
        p = Path(repo_root) / pd
        if p.exists():
            phase.append({"path": pd, "headings": phase_doc_headings(p)})
    return {"schema_version": cm.SCHEMA_VERSION,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_sha": cm.git_sha(repo_root), "branch": branch,
            "vs_compare_branch": vs_compare(repo_root, st.get("compare_branch")),
            "recent_commits": recent_commits(repo_root, 10),
            "phase_docs": phase, "memory_index": _memory_index(st.get("memory_dir")),
            "recent_jobs": _recent_jobs(st.get("job_log_glob"))}

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); args = ap.parse_args(argv)
    repo = Path(args.repo).resolve(); cfg = cm.load_config(repo)
    out = build_state_facts(repo, cfg)
    d = cm.snapshot_dir(repo, cfg, cm.git_sha(repo))
    cm.write_json(d / "state.facts.json", out); print(d / "state.facts.json")

if __name__ == "__main__":
    main()
