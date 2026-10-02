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
interrupted run can be recovered with ``git checkout``; sixteen cases also touch
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
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.split()


def snapshot(extra: set[str] = frozenset()) -> dict[str, bytes]:
    """Tracked files, plus any path a case is going to touch.

    Sixteen cases mutate files under ``private/``, which is gitignored, so
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


def _docx(path: str, old: str, new: str):
    """Edit the text inside a .docx, which is a zip of XML parts.

    The recording script is read on camera, so a stale number in it is spoken
    aloud. It is the one copy no check opened, and it sat at "Three hundred and
    sixty" while the script said four hundred and five.
    """
    def go() -> bool:
        p = REPO / path
        if not p.exists():
            return False
        with zipfile.ZipFile(p) as zin:
            if old not in zin.read("word/document.xml").decode("utf8"):
                return False
            items = [(i, zin.read(i.filename)) for i in zin.infolist()]
        tmp = p.with_suffix(".docx.mutating")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item, data in items:
                if item.filename == "word/document.xml":
                    data = data.decode("utf8").replace(old, new, 1).encode("utf8")
                zout.writestr(item, data)
        shutil.move(tmp, p)
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
    # The drift that actually happened: BRIEFING is gitignored, CI never reads
    # it, and it sat at 360 while the README said 405.
    ("BRIEFING claim count (prose) agrees with the README",
     _sub("private/BRIEFING.md", "recomputes 405 published numbers",
          "recomputes 360 published numbers")),
    # The exact artefact that was wrong: the docx read on camera.
    ("the recording docx speaks the README's claim count",
     _docx("private/Demo video script.docx", "Four hundred and five of them",
           "Three hundred and sixty of them")),
    ("the recording docx speaks the collected test count",
     _docx("private/Demo video script.docx", "A hundred and thirty-two tests",
           "A hundred and thirty tests")),
    # The script had quoted the best of E4's four comparisons as if typical.
    ("the script speaks the README's narrower range",
     _sub("private/VIDEO.md", "**seven and twelve percent**",
          "**seven and fifteen percent**")),
    ("the recording docx speaks the script's narrower range",
     _docx("private/Demo video script.docx", "seven and twelve percent",
           "six and twelve percent")),
    ("the script states its own spoken word count",
     _sub("private/VIDEO.md", "424 spoken words", "419 spoken words")),
    # A sweep of the spoken script found that only its digit-form numbers were
    # checked; every number spelled out for reading aloud had nothing on it.
    # Each anchor below is the one in section 3, not a planning-table copy: the
    # first attempt at the slowdown case mutated a notes table on line 31 and
    # read as an unguarded claim.
    ("the script speaks the certifiable ceiling at 100 frauds",
     _sub("private/VIDEO.md", "**ninety-eight percent**", "**ninety-six percent**")),
    ("the script restates what cross certifies at 100 frauds",
     _sub("private/VIDEO.md", "Same hundred frauds, **ninety-nine**",
          "Same hundred frauds, **ninety-seven**")),
    ("the script speaks the gradient-fit counts",
     _sub("private/VIDEO.md", "Zero against six.", "Zero against eight.")),
    ("the script speaks the routed transaction count",
     _sub("private/VIDEO.md", "Four hundred real held-out transactions",
          "Five hundred real held-out transactions")),
    ("the script speaks the prediction scoreboard",
     _sub("private/VIDEO.md", "**Four turned out wrong**",
          "**Three turned out wrong**")),
    ("the script speaks a slowdown inside the measured range",
     _sub("private/VIDEO.md", "TabPFN is about **sixty times slower**",
          "TabPFN is about **forty times slower**")),
    ("BRIEFING has a table row for every experiment",
     _sub("private/BRIEFING.md", "| **E8** | Does the guarantee work",
          "| **E9** | Does the guarantee work")),
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
    ("E8 laya_zero_shot review",
     _sub("README.md", "| 0.900 | 1.833 | **83.3%** |", "| 0.900 | 1.833 | **55.5%** |")),
    ("E8 empty sets among fraud rows",
     _sub("README.md", "**1.48% of fraud rows receive an", "**9.99% of fraud rows receive an")),
    # A sweep of every number in the E8 section found these two were the only
    # ones a change could not fail: each sentence's figures are computed against
    # a hard-coded level, so a relabelled alpha read as correct.
    ("E8 shortfall sentence alpha",
     _sub("README.md", "\u03b1 = 0.20 both TabPFN arms",
          "\u03b1 = 0.23 both TabPFN arms")),
    ("E8 marginal sentence alpha",
     _sub("README.md", "abandons a class at \u03b1 = 0.10",
          "abandons a class at \u03b1 = 0.13")),
    # An external fact: nothing offline recomputes it, so what is checked is
    # that the README and the script that ran the model still agree.
    ("E8 model size agrees with the experiment that ran it",
     _sub("experiments/api/e8_zero_shot_guarantee.py", "a 421M-parameter",
          "a 415M-parameter")),
    ("E8 tabpfn_cross_200 shortfall @0.2",
     _sub("README.md", "\u22120.0102 \u00b1 0.0035", "\u22120.9999 \u00b1 0.0035")),
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
