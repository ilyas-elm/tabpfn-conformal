"""Shared data protocol for every API experiment.

Lives in one place so E1, E2, E3 and E4 cannot silently disagree about what
"the pool" or "the evaluation set" means. If these functions change, every
experiment changes together.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import signal

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "Base.csv"

LABEL, TIME = "fraud_bool", "month"
POOL_MONTHS, EVAL_MONTHS = (0, 1, 2, 3, 4, 5), (6, 7)
EVAL_LEGIT = 3_000
BASE_RATE = 0.011


def load_token() -> bool:
    """Put TABPFN_TOKEN in the environment from .env if it is not already set."""
    if os.environ.get("TABPFN_TOKEN"):
        return True
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN=") and not line.lstrip().startswith("#"):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
                return True
    return False


def load_frames(keep_time: bool = False):
    """The BAF Base split into a labelled pool and an evaluation set, temporally.

    Never a random split: the fraud rate climbs from 0.875% in month 2 to 1.475%
    in month 7, so a random split would quietly leak the future into the past.
    """
    if not DATA.exists():
        raise SystemExit(
            f"Missing {DATA.relative_to(REPO)} -- run scripts/download_data.py first."
        )
    df = pd.read_csv(DATA)
    pool = df[df[TIME].isin(POOL_MONTHS)].reset_index(drop=True)
    ev = df[df[TIME].isin(EVAL_MONTHS)].reset_index(drop=True)
    if not keep_time:
        return pool, ev
    return pool, ev, df


def split_xy(frame: pd.DataFrame, keep_time: bool = False):
    drop = [LABEL] if keep_time else [LABEL, TIME]
    return frame.drop(columns=drop), frame[LABEL].to_numpy()


def make_eval(ev: pd.DataFrame, seed: int, keep_time: bool = False):
    """Every evaluation fraud, plus a fixed sample of legitimate rows.

    Class-conditional coverage is estimated *within* each class, so the class
    mix of the evaluation set cannot bias it. Keeping all the positives buys a
    tight fraud-coverage estimate for a fraction of the tokens a proportionally
    sampled set would cost.
    """
    rng = np.random.default_rng(seed)
    pos = ev[ev[LABEL] == 1]
    neg = ev[ev[LABEL] == 0]
    neg = neg.iloc[rng.choice(len(neg), min(EVAL_LEGIT, len(neg)), replace=False)]
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_xy(out, keep_time)


def make_pool(pool: pd.DataFrame, n_frauds: int, seed: int, keep_time: bool = False):
    """A labelled pool holding exactly ``n_frauds`` positives at the base rate."""
    rng = np.random.default_rng(1000 + seed)
    pos = pool[pool[LABEL] == 1]
    neg = pool[pool[LABEL] == 0]
    n_neg = int(round(n_frauds / BASE_RATE)) - n_frauds
    if n_frauds > len(pos) or n_neg > len(neg):
        return None, None
    take = pd.concat([
        pos.iloc[rng.choice(len(pos), n_frauds, replace=False)],
        neg.iloc[rng.choice(len(neg), n_neg, replace=False)],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_xy(take, keep_time)


def resume_keys(path: pathlib.Path, key_fn) -> set:
    """Keys already present in a JSONL results file, for resumable runs."""
    import json

    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(key_fn(json.loads(line)))
        except (ValueError, KeyError):
            continue
    return out


def save_proba(tag: str, key: str, proba, y_true) -> str:
    """Persist evaluation probabilities so alpha can be swept densely, offline.

    Without this, an experiment is locked to whatever alpha grid it happened to
    choose at runtime, and the honest comparison -- set width at MATCHED
    realised coverage -- cannot be computed after the fact without paying for
    the predictions again. Roughly 80 KB per configuration.
    """
    import re

    out = REPO / "results" / "proba" / tag
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    path = out / f"{safe}.npz"
    np.savez_compressed(path, proba=np.asarray(proba, dtype=np.float32),
                        y_true=np.asarray(y_true))
    return str(path.relative_to(REPO))


class ConfigTimeout(Exception):
    """A single configuration exceeded its wall-clock budget."""


@contextlib.contextmanager
def time_limit(seconds: int):
    """Abort a configuration that hangs instead of losing the whole run to it.

    Observed 19 Sept: a plain ``predict`` against the managed API blocked for
    1h50m with 18 seconds of CPU time -- the client sets no timeout on ordinary
    calls, so a dropped response stalls forever and every later configuration
    waits behind it. Every experiment here is resumable, so the right response
    to a stall is to abandon that configuration and move on.

    Uses SIGALRM, so it only interrupts the main thread on Unix. Zero or None
    disables it.
    """
    if not seconds:
        yield
        return

    def _fire(signum, frame):
        raise ConfigTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
