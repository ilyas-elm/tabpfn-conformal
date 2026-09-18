"""Spike S1 -- does the KV cache work together with Thinking on the managed API?

The documentation contradicts itself. https://docs.priorlabs.ai/capabilities/kv-cache
states that caching is incompatible with Thinking mode on the managed API, while
other material says the client sets ``use_kv_cache=True`` automatically when
thinking is enabled. The answer decides whether the cache story and the Thinking
story can share a figure, so it is worth ten minutes and ~50k tokens to settle
empirically.

Also prints estimate_cost() for BAF-shaped workloads. Those calls send dimensions
only -- no data upload, no quota consumed -- so the cost table below is free.

Run:
    export PRIORLABS_API_TOKEN=...        # or let the client prompt you once
    python experiments/api/spike_s1_cache_thinking.py

Writes results/spike_s1.json.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "spike_s1.json"

# Deliberately tiny: each billable operation costs a 10,000-token minimum
# anyway, so there is nothing to gain from larger probes.
N_TRAIN, N_TEST, N_FEATURES = 200, 50, 8

# Shapes we actually care about later, for the free cost quotes.
BAF_SHAPES = [
    ("E1 split conformal, 10k pool", 5_000, 5_000),
    ("E1 cross-conformal, 10k pool, per fold", 8_000, 2_000),
    ("E2 budget sweep, one cell", 4_000, 4_000),
    ("E3 one month of drift", 20_000, 5_000),
]
BAF_FEATURES = 30


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES))
    y = (X[:, 0] + 0.5 * rng.normal(size=len(X)) > 1.8).astype(int)
    if y[:N_TRAIN].sum() < 5:           # make sure both classes are present
        y[:5] = 1
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]


def _try(label: str, build, X_tr, y_tr, X_te) -> dict:
    """Run one configuration and record what happened, success or failure."""
    record: dict = {"config": label}
    try:
        clf = build()
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        record["fit_seconds"] = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        proba = clf.predict_proba(X_te)
        record["predict_seconds"] = round(time.perf_counter() - t1, 2)

        # A second predict against the same fit: this is where a cache pays off,
        # and it is exactly the conformal workload (one context, many batches).
        t2 = time.perf_counter()
        clf.predict_proba(X_te)
        record["second_predict_seconds"] = round(time.perf_counter() - t2, 2)

        record["ok"] = True
        record["proba_shape"] = list(np.shape(proba))
    except Exception as exc:  # noqa: BLE001 -- recording the failure IS the result
        record["ok"] = False
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:500]
    print(f"  {label:<46} -> {'ok' if record.get('ok') else record.get('error_type')}")
    return record


def main() -> int:
    try:
        from tabpfn_client import TabPFNClassifier, estimate_cost
    except ImportError:
        print(
            "tabpfn-client is not installed.\n"
            '    pip install -e ".[experiments]"\n'
            "Then set PRIORLABS_API_TOKEN, or run once interactively to log in.",
            file=sys.stderr,
        )
        return 1

    results: dict = {"shapes": {"n_train": N_TRAIN, "n_test": N_TEST, "n_features": N_FEATURES}}

    # ---- free: cost quotes for the real experiments -------------------------
    print("\nCost quotes (no upload, no quota consumed):")
    quotes = []
    for label, n_tr, n_te in BAF_SHAPES:
        entry = {"label": label, "n_train": n_tr, "n_test": n_te}
        try:
            q = estimate_cost(
                np.zeros((n_tr, BAF_FEATURES)), np.zeros((n_te, BAF_FEATURES))
            )
            entry["estimated_cost"] = getattr(q, "estimated_cost", str(q))
            print(f"  {label:<46} {entry['estimated_cost']}")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"  {label:<46} ! {entry['error']}")
        quotes.append(entry)
    results["cost_quotes"] = quotes

    # ---- billed, but trivially: the actual S1 question ----------------------
    X_tr, y_tr, X_te = _toy()
    print("\nConfigurations (each costs the 10,000-token minimum):")
    results["configs"] = [
        _try("baseline", lambda: TabPFNClassifier(), X_tr, y_tr, X_te),
        _try(
            "fit_with_cache",
            lambda: TabPFNClassifier(fit_mode="fit_with_cache"),
            X_tr, y_tr, X_te,
        ),
        _try(
            "thinking_effort=medium",
            lambda: TabPFNClassifier(thinking_effort="medium"),
            X_tr, y_tr, X_te,
        ),
        _try(
            "fit_with_cache + thinking_effort=medium   <-- S1",
            lambda: TabPFNClassifier(
                fit_mode="fit_with_cache", thinking_effort="medium"
            ),
            X_tr, y_tr, X_te,
        ),
    ]

    combined = results["configs"][-1]
    results["s1_answer"] = (
        "compatible" if combined.get("ok") else f"incompatible: {combined.get('error_type')}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nS1: KV cache + Thinking is {results['s1_answer']}")
    print(f"Written to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
