#!/usr/bin/env python3
"""Break things on purpose and check that ``verify_claims.py`` notices.

A check nobody has seen fail is not evidence. Three checks in this project were
written, looked right, and could not fail:

* one compared with ``c[1] is False`` while the comparison produced a *numpy*
  bool, so a failing check printed FAIL and was then not counted, and the
  script exited 0;
* one matched ``\\d+`` where the document said "five of six", so it passed on
  the exact string it existed to catch;
* one keyed gradient-fit counts by arm into a dict, which kept only the last
  row per arm, so a wrong value in any earlier row was invisible.

None of those were found by reading. All three were found by mutating the
thing the check is supposed to protect and watching what happened.

    python scripts/mutation_sweep.py            # every case
    python scripts/mutation_sweep.py --list     # names only

This mutates files and restores them from an in-memory snapshot taken
immediately before each case. It refuses to start with a dirty tree, so an
interrupted run can be recovered with ``git checkout``; three cases also touch
untracked files under ``private/``, which git cannot restore, so those are
copied to a temporary directory first and the path is printed. It is not part
of CI: it is slow, it writes to the working tree, and it is a thing you run
deliberately after adding or editing a check.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.split()


def snapshot(extra: set[str] = frozenset()) -> dict[str, bytes]:
    """Tracked files, plus any path a case is going to touch.

    Three cases mutate files under ``private/``, which is gitignored, so
    ``git ls-files`` does not list them. Snapshotting only tracked files would
    mutate those and never put them back.
    """
    files = set(tracked()) | set(extra)
    return {f: (REPO / f).read_bytes() for f in sorted(files) if (REPO / f).exists()}


def restore(snap: dict[str, bytes]) -> None:
    for f, blob in snap.items():
        p = REPO / f
        if not p.exists() or p.read_bytes() != blob:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
    for hidden in REPO.rglob("*.hidden"):
        hidden.rename(hidden.with_suffix(""))


def backup_untracked(paths: set[str]) -> tuple[pathlib.Path, list[str]] | None:
    """Copy the paths git does not track to a temp directory.

    ``restore`` puts every mutated file back from memory, but a run killed
    part-way leaves whatever the last mutation wrote, and ``git checkout``
    cannot recover a file git has never seen.
    """
    known = set(tracked())
    at_risk = sorted(p for p in paths if p not in known and (REPO / p).exists())
    if not at_risk:
        return None
    into = pathlib.Path(tempfile.mkdtemp(prefix="mutation_sweep_"))
    for rel in at_risk:
        dest = into / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    return into, at_risk


def verify() -> tuple[int, str]:
    r = subprocess.run([sys.executable, "scripts/verify_claims.py"],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout


# ---- mutations: each breaks exactly what one check is meant to protect -----

def _sub(path: str, old: str, new: str):
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        s = p.read_text()
        if old not in s:
            return False
        p.write_text(s.replace(old, new, 1))
        return True
    go.path = path  # type: ignore[attr-defined]
    return go


def _hide(path: str):
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        p.rename(p.with_suffix(p.suffix + ".hidden"))
        return True
    go.path = path  # type: ignore[attr-defined]
    return go


def _jsonl(path: str, pick, mutate):
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            d = json.loads(line)
            if pick(d):
                mutate(d)
                lines[i] = json.dumps(d)
                p.write_text("\n".join(lines) + "\n")
                return True
        return False
    go.path = path  # type: ignore[attr-defined]
    return go


def _json(path: str, mutate):
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        d = json.loads(p.read_text())
        mutate(d)
        p.write_text(json.dumps(d, indent=1))
        return True
    go.path = path  # type: ignore[attr-defined]
    return go


def _e6_wider() -> bool:
    p = REPO / "results/e6.jsonl"
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        d = json.loads(line)
        if d.get("strategy") == "cross":
            for a in d.get("alphas", {}).values():
                a["set_size"] *= 2.5
            lines[i] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n")
    return True


def _drop_row(path: str):
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        lines = p.read_text().splitlines()
        if len(lines) < 2:
            return False
        p.write_text("\n".join(lines[:-1]) + "\n")
        return True
    go.path = path  # type: ignore[attr-defined]
    return go


def _drop_download() -> bool:
    p = REPO / "README.md"
    p.write_text("\n".join(l for l in p.read_text().splitlines()
                           if "scripts/download_data.py" not in l) + "\n")
    return True


CASES: list[tuple[str, object]] = [
    ("E6 significantly wider is zero", _e6_wider),
    ("the E6 table has all nine rows",
     _sub("README.md", "| Variant III | 100 |", "| Variant III x | 100 |")),
    ("cost ratio equals K in every quote",
     _json("results/cost_kfold.json",
           lambda d: d["rows"][0].__setitem__("ratio", d["rows"][0]["k"] + 3.0))),
    ("documented layout exists",
     _sub("private/CAHIER-DES-CHARGES.md", "├── results/", "├── nonexistent_dir/")),
    ("E4 wall-clock confound flagged",
     _jsonl("results/e4.jsonl", lambda d: True,
            lambda d: d.__setitem__("wallclock_comparable", True))),
    # Keyed by arm, this check once kept only the last row per arm. Mutate the
    # FIRST tabpfn row so a collapsing implementation cannot hide it again.
    ("TabPFN arms do zero gradient fits",
     _jsonl("results/e4.jsonl", lambda d: d.get("family") == "tabpfn",
            lambda d: d.__setitem__("n_grad_fits", 4))),
    ("package version matches pyproject",
     _sub("src/tabpfn_conformal/__init__.py", '__version__ = "0.1.0"',
          '__version__ = "9.9.9"')),
    ("no TODO left in package metadata",
     _sub("pyproject.toml", 'description = "', 'description = "TODO ')),
    ("every script the README runs exists",
     _sub("README.md", "python scripts/build_demo.py", "python scripts/nope.py")),
    ("README documents the data download", _drop_download),
    ("every experiment supports --dry-run",
     _sub("experiments/api/e7_second_domain.py", '"--dry-run"', '"--dryrun"')),
    ("every figure the README shows exists",
     _sub("README.md", "figures/e5_scale_alpha005.png", "figures/ghost.png")),
    ("extensions payload has a changelog fragment",
     _hide("contrib/tabpfn-extensions/changelog/PRNUMBER.added.md")),
    ("the quoted HTTP 422 is the recorded one",
     _sub("README.md", "FIT_WITH_CACHE fit mode is not compatible with thinking mode",
          "SOME OTHER ERROR TEXT")),
    ("py.typed exists", _hide("src/tabpfn_conformal/py.typed")),
    ("py.typed is declared to the build backend",
     _sub("pyproject.toml", 'artifacts = ["src/tabpfn_conformal/py.typed"]',
          "artifacts = []")),
    ("SUBMISSION does not claim identical cache sets",
     _sub("private/SUBMISSION.md", "- **The KV cache makes the evaluation pass",
          "- The KV cache gives identical prediction sets. "
          "**The KV cache makes the evaluation pass")),
    ("STATUS names the changelog rename step",
     _sub("private/STATUS.md", "PRNUMBER.added.md", "SOMEFILE.md")),
    ("E7 ran the full grid", _drop_row("results/e7.jsonl")),
    ("E7 is a different dataset from BAF",
     _jsonl("results/e7.jsonl", lambda d: True, lambda d: d.__setitem__("dataset", "baf"))),
    ("the fair-hardware run is 24 rows on a GPU",
     _json("results/kaggle_wallclock.json",
           lambda d: d["device"].__setitem__("device", "cpu"))),
    # The check reads the GENERATED PR.md, so mutating the generator would not
    # reach it without a rebuild.
    ("the PR description states the measured cost",
     _sub("contrib/tabpfn-extensions/PR.md", "3 of 6", "several")),
    # Compares the git commit dates of a figure and its generator, which no
    # edit to the working tree can change. Proven separately against real
    # history: run over the tree at aa98d33 it named exactly the three figures
    # that were genuinely stale and cleared the rest.
    ("no figure predates its generator", None),
    ("the upstream module README reports the paired test",
     _sub("contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/README.md",
          "Across four datasets",
          "with narrower prediction sets in five of six comparisons. Across four datasets")),
    ("vendored cross-conformal keeps the validity caveat",
     _sub("contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/crossconformal.py",
          "Vovk 2015", "an unnamed source")),
    ("E5 scale set size @100,200",
     _sub("README.md", "| 100,200 | 0.20% | 1.504 |", "| 100,200 | 0.20% | 9.999 |")),
    ("E5 scale table lists every context run",
     _sub("README.md", "| 100,200 | 0.20% | 1.504 |\n", "")),
    ("demo data regenerates from results",
     _json("figures/demo_data.json", lambda d: d.__setitem__("n_fraud", 99999))),
    # Their pre-commit rejects this and ours never used to look for it.
    ("payload declares float but returns a bare numpy scalar",
     _sub("contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/calibration.py",
          "return math.inf", "return np.inf")),
    ("extensions payload matches src",
     _sub("contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/metrics.py",
          "import numpy", "import numpy  # drifted")),
]


TOUCHED = {q for _, m in CASES if (q := getattr(m, "path", None)) is not None}


def main() -> int:
    if "--list" in sys.argv:
        for label, _ in CASES:
            print(label)
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("refusing to run with uncommitted changes; commit or stash first.",
              file=sys.stderr)
        return 2

    code, _ = verify()
    if code != 0:
        print("verify_claims already fails; fix that before sweeping.", file=sys.stderr)
        return 2

    spare = backup_untracked(TOUCHED)
    if spare is not None:
        into, at_risk = spare
        print(f"{len(at_risk)} untracked file(s) get mutated and git cannot put "
              f"them back; copies are in {into}")

    blind, skipped, not_mutable = [], [], []
    for label, mutate in CASES:
        if mutate is None:
            not_mutable.append(label)
            continue
        snap = snapshot(TOUCHED)
        applied = mutate()
        if not applied:
            skipped.append(label)
            restore(snap)
            continue
        code, out = verify()
        caught = any(l.startswith("FAIL") and label[:34] in l for l in out.splitlines())
        if not caught:
            blind.append((label, code))
        restore(snap)
        print(f"  {'caught ' if caught else 'BLIND  '} {label}")

    code, _ = verify()
    proven = len(CASES) - len(blind) - len(skipped) - len(not_mutable)
    print(f"\n{proven} of {len(CASES)} checks proven able to fail; "
          f"{len(blind)} blind; {len(not_mutable)} verified by other means; "
          f"{len(skipped)} unreachable")
    for label, c in blind:
        print(f"  BLIND: {label} (exit {c} -- something else caught it, or nothing did)")
    for label in not_mutable:
        print(f"  by other means: {label} (see the comment beside its case)")
    for label in skipped:
        print(f"  unreachable: {label} (mutation target not found; the code moved)")
    if code != 0:
        print("\nthe tree did not restore cleanly; run: git checkout .", file=sys.stderr)
        if spare is not None:
            print(f"untracked files are not covered by that; copy them back "
                  f"from {spare[0]}", file=sys.stderr)
        return 2
    return 1 if blind else 0


if __name__ == "__main__":
    raise SystemExit(main())
