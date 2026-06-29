import json, subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import cm_common as cm

def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

def make_repo(tmp_path):
    repo = tmp_path / "r"; repo.mkdir()
    _git(repo, "init", "-q"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    return repo

def test_schema_version_is_one():
    assert cm.SCHEMA_VERSION == 1

def test_git_sha_clean_and_dirty(tmp_path):
    repo = make_repo(tmp_path)
    sha = cm.git_sha(repo)
    assert sha and "-dirty" not in sha
    (repo / "f.txt").write_text("y")
    assert cm.git_sha(repo).endswith("-dirty")

def test_write_json_roundtrip(tmp_path):
    p = tmp_path / "a.json"
    cm.write_json(p, {"b": 1, "a": 2})
    text = p.read_text()
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b"]  # sorted keys

def test_prune_keeps_newest(tmp_path):
    repo = make_repo(tmp_path)
    snap = repo / "docs/codemap/snapshots"; snap.mkdir(parents=True)
    import os, time
    for i, name in enumerate(["aaa", "bbb", "ccc"]):
        d = snap / name; d.mkdir()
        os.utime(d, (1000 + i, 1000 + i))
    cfg = {"snapshots": {"dir": "docs/codemap/snapshots", "keep_last": 2}}
    removed = cm.prune_snapshots(repo, cfg)
    assert removed == ["aaa"]
    assert {p.name for p in snap.iterdir()} == {"bbb", "ccc"}
