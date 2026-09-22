#!/usr/bin/env python3
"""Fail if anything credential-shaped is about to be committed or published.

This repository is public and carries API-backed experiments, so the failure
mode is a token pasted into a script, a log captured verbatim from an API
response, or a notebook cell saved with its output. None of those look like
mistakes at the time.

Scans tracked files by default; `--history` scans every blob in every commit,
which is what to run before making a repository public.

    python scripts/check_secrets.py             # tracked files at HEAD
    python scripts/check_secrets.py --history   # all commits (slower)
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
GIT = "/usr/bin/git" if pathlib.Path("/usr/bin/git").exists() else "git"

# Each entry is (name, pattern). Kept narrow on purpose: a scanner that cries
# wolf gets switched off, and this one runs in CI.
PATTERNS = [
    ("GitHub token", r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"),
    ("OpenAI key", r"\bsk-[A-Za-z0-9_-]{20,}"),
    ("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("Google API key", r"\bAIza[0-9A-Za-z_-]{30,}"),
    ("Slack token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    ("private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("JWT", r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}"),
    # The one that matters here: the project's own variable with a value
    # attached, rather than the name alone, which appears legitimately in code.
    ("TabPFN token value", r"TABPFN_TOKEN\s*[=:]\s*['\"]?[A-Za-z0-9_.-]{12,}"),
    # A local absolute path exposes the machine's username for no benefit.
    ("local home path", r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
]

# Filenames that should never be tracked at all.
FORBIDDEN = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa|id_ed25519|kaggle\.json"
    r"|\.netrc|\.npmrc|\.pypirc|credentials(\.json)?|secrets?\.(json|ya?ml))$",
    re.I,
)

SKIP_SUFFIXES = {".png", ".svg", ".npz", ".jpg", ".jpeg", ".gif", ".pdf", ".ico"}
SELF = "scripts/check_secrets.py"


def scan_text(where: str, text: str) -> list[str]:
    hits = []
    for name, pat in PATTERNS:
        for m in re.finditer(pat, text):
            frag = m.group(0)
            hits.append(f"{where}: {name} -> {frag[:48]}")
    return hits


def tracked_files() -> list[str]:
    out = subprocess.run([GIT, "ls-files"], cwd=REPO, capture_output=True, text=True)
    return [f for f in out.stdout.split("\n") if f]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true",
                    help="scan every blob in every commit, not just HEAD")
    args = ap.parse_args()

    problems: list[str] = []

    for f in tracked_files():
        if FORBIDDEN.search(f):
            problems.append(f"{f}: a file of this name should never be tracked")

    if args.history:
        commits = subprocess.run([GIT, "rev-list", "--all"], cwd=REPO,
                                 capture_output=True, text=True).stdout.split()
        print(f"scanning {len(commits)} commits...")
        for name, pat in PATTERNS:
            if name == "local home path":
                continue            # history is not rewritten; HEAD is what ships
            out = subprocess.run([GIT, "grep", "-hIoE", pat, *commits],
                                 cwd=REPO, capture_output=True, text=True)
            for line in {l for l in out.stdout.split("\n") if l.strip()}:
                problems.append(f"HISTORY: {name} -> {line[:48]}")
        added = subprocess.run(
            [GIT, "log", "--all", "--diff-filter=A", "--name-only", "--format="],
            cwd=REPO, capture_output=True, text=True).stdout.split("\n")
        for f in {a for a in added if a.strip()}:
            if FORBIDDEN.search(f):
                problems.append(f"HISTORY: {f} was committed at some point")
    else:
        for f in tracked_files():
            p = REPO / f
            if p.suffix.lower() in SKIP_SUFFIXES or f == SELF or not p.exists():
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            problems.extend(scan_text(f, text))

    scope = "all history" if args.history else "tracked files at HEAD"
    if problems:
        print(f"\n{len(problems)} problem(s) in {scope}:\n")
        for p in sorted(set(problems))[:40]:
            print(f"  {p}")
        return 1
    print(f"clean: nothing credential-shaped in {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
