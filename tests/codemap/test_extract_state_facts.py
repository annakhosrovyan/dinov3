import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/codemap/scripts"))
import extract_state_facts as sf

def _git(repo, *a): subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)

def make_repo(tmp_path):
    repo = tmp_path / "r"; repo.mkdir()
    _git(repo, "init", "-q", "-b", "master"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x=1\n"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.py").write_text("x=1\ny=2\n"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "feat: add y")
    return repo

def test_vs_compare_counts_ahead(tmp_path):
    repo = make_repo(tmp_path)
    out = sf.vs_compare(repo, "master")
    assert out["ahead"] == 1 and out["behind"] == 0
    assert any(c["path"] == "a.py" for c in out["changed_files"])

def test_recent_commits(tmp_path):
    repo = make_repo(tmp_path)
    cs = sf.recent_commits(repo, n=5)
    assert cs[0]["subject"].startswith("feat: add y")

def test_phase_headings(tmp_path):
    p = tmp_path / "phase.md"; p.write_text("# A\nblah\n## B.1\n### deep\n")
    assert sf.phase_doc_headings(p) == ["# A", "## B.1", "### deep"]

def test_build_state_facts_tolerates_missing(tmp_path):
    repo = make_repo(tmp_path)
    cfg = {"state": {"compare_branch": "master", "phase_docs": ["nope.md"],
                     "memory_dir": "/does/not/exist"}, "project": {}}
    out = sf.build_state_facts(repo, cfg)
    assert out["schema_version"] == 1 and out["branch"] == "feature"
    assert out["memory_index"] == [] and out["phase_docs"] == []
