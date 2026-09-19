"""Spike S1 -- settle the KV-cache/Thinking contradiction, and price the plan.

Two jobs, in order of cost:

1. FREE. ``estimate_cost`` accepts ``operation`` in {predict, cache_predict,
   thinking_fit, thinking_predict} and sends dimensions only -- no upload, no
   quota. That prices every planned experiment AND quantifies the KV-cache
   discount before a single token is spent. Run this part alone with --quotes-only.

2. CHEAP. The docs contradict themselves on whether the KV cache works with
   Thinking on the managed API (https://docs.priorlabs.ai/capabilities/kv-cache
   says incompatible; other material says the client forces the cache on when
   thinking is enabled). Four tiny fits settle it empirically. Each billable
   operation costs the 10,000-token minimum, so the whole probe is ~40k of a
   5,000,000/day budget.

Auth -- the token is never passed on the command line and never printed:
    python -c "import tabpfn_client; tabpfn_client.init()"    # interactive, caches it
or  export TABPFN_TOKEN=...                                   # in your own shell
or  a TABPFN_TOKEN=... line in .env at the repo root          # gitignored

Run:
    python experiments/api/spike_s1_cache_thinking.py [--quotes-only]

Writes results/spike_s1.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "spike_s1.json"

N_TRAIN, N_TEST, N_FEATURES = 200, 50, 8
BAF_FEATURES = 30

# (label, n_context, n_scored, operation) for the experiments in the cahier.
PLANNED = [
    ("E1 split conformal, 10k pool", 5_000, 5_000, "predict"),
    ("E1 cross-conformal, 10k pool, per fold", 8_000, 2_000, "predict"),
    ("E1 cross-conformal, same fold, cached", 8_000, 2_000, "cache_predict"),
    ("E2 budget sweep, one cell", 4_000, 4_000, "predict"),
    ("E2 budget sweep, one cell, cached", 4_000, 4_000, "cache_predict"),
    ("E3 one month of drift", 20_000, 5_000, "predict"),
    ("E3 one month, Thinking fit", 20_000, None, "thinking_fit"),
    ("E3 one month, Thinking predict", 20_000, 5_000, "thinking_predict"),
]


def _load_dotenv() -> None:
    """Read TABPFN_TOKEN from .env if it is not already in the environment."""
    if os.environ.get("TABPFN_TOKEN"):
        return
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("TABPFN_TOKEN=") and not line.startswith("#"):
            os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
            print("Loaded TABPFN_TOKEN from .env")
            return


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES))
    y = (X[:, 0] + 0.5 * rng.normal(size=len(X)) > 1.8).astype(int)
    y[:5] = 1  # guarantee both classes are in the context
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]


def quote_plan(estimate_cost) -> list[dict]:
    """Price every planned experiment. Costs nothing."""
    print("\nCost quotes -- dimensions only, no upload, no quota consumed:")
    print(f"  {'workload':<44} {'operation':<18} {'tokens':>14}")
    rows = []
    for label, n_ctx, n_scored, op in PLANNED:
        row = {"label": label, "n_context": n_ctx, "n_scored": n_scored, "operation": op}
        try:
            kwargs = {"operation": op}
            if op.startswith("thinking"):
                kwargs["thinking_effort"] = "medium"
            q = estimate_cost(
                np.zeros((n_ctx, BAF_FEATURES)),
                None if n_scored is None else np.zeros((n_scored, BAF_FEATURES)),
                **kwargs,
            )
            row["tokens"] = getattr(q, "estimated_cost", None)
            print(f"  {label:<44} {op:<18} {row['tokens']:>14,}")
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"  {label:<44} {op:<18} {'!':>14}  {row['error']}")
        rows.append(row)

    # The headline number this spike exists to produce, for free.
    paid = {r["label"]: r.get("tokens") for r in rows}
    a = paid.get("E2 budget sweep, one cell")
    b = paid.get("E2 budget sweep, one cell, cached")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
        print(f"\n  KV-cache saving on a repeat prediction: {100 * (1 - b / a):.0f}%")
    return rows


def probe(label: str, build, X_tr, y_tr, X_te) -> dict:
    """Run one configuration; record success or the exact failure."""
    rec: dict = {"config": label}
    try:
        clf = build()
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        rec["fit_seconds"] = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        proba = clf.predict_proba(X_te)
        rec["predict_seconds"] = round(time.perf_counter() - t1, 2)

        # The conformal workload: a second batch against the same context.
        t2 = time.perf_counter()
        clf.predict_proba(X_te)
        rec["second_predict_seconds"] = round(time.perf_counter() - t2, 2)

        rec["ok"] = True
        rec["proba_shape"] = list(np.shape(proba))
        print(
            f"  {label:<46} ok   fit {rec['fit_seconds']}s  "
            f"predict {rec['predict_seconds']}s  repeat {rec['second_predict_seconds']}s"
        )
    except Exception as exc:  # noqa: BLE001 -- recording the failure IS the result
        rec["ok"] = False
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)[:500]
        print(f"  {label:<46} FAILED  {rec['error_type']}: {rec['error'][:110]}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quotes-only",
        action="store_true",
        help="print the free cost quotes and stop; spend nothing.",
    )
    args = ap.parse_args()

    _load_dotenv()

    try:
        from tabpfn_client import TabPFNClassifier, estimate_cost, get_api_usage
    except ImportError:
        print(
            'tabpfn-client is not installed.\n    pip install -e ".[experiments]"',
            file=sys.stderr,
        )
        return 1

    if not os.environ.get("TABPFN_TOKEN"):
        print(
            "No TABPFN_TOKEN found. Either log in once interactively --\n"
            '    python -c "import tabpfn_client; tabpfn_client.init()"\n'
            "-- or put a TABPFN_TOKEN= line in .env (already gitignored), or export it.",
            file=sys.stderr,
        )
        return 2

    results: dict = {
        "probe_shape": {"n_train": N_TRAIN, "n_test": N_TEST, "n_features": N_FEATURES},
        "cost_quotes": quote_plan(estimate_cost),
    }

    try:
        results["api_usage_before"] = get_api_usage()
        print(f"\nAPI usage before: {results['api_usage_before']}")
    except Exception as exc:  # noqa: BLE001
        results["api_usage_before"] = f"{type(exc).__name__}: {exc}"[:200]

    if args.quotes_only:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"\nQuotes only; nothing spent. Written to {OUT.relative_to(REPO)}")
        return 0

    X_tr, y_tr, X_te = _toy()
    print("\nProbes (each billable op costs the 10,000-token minimum):")
    results["configs"] = [
        probe("baseline", lambda: TabPFNClassifier(), X_tr, y_tr, X_te),
        probe(
            "fit_with_cache",
            lambda: TabPFNClassifier(fit_mode="fit_with_cache"),
            X_tr, y_tr, X_te,
        ),
        probe(
            "thinking_mode",
            lambda: TabPFNClassifier(thinking_mode=True, thinking_effort="medium"),
            X_tr, y_tr, X_te,
        ),
        probe(
            "fit_with_cache + thinking_mode   <-- S1",
            lambda: TabPFNClassifier(
                fit_mode="fit_with_cache", thinking_mode=True, thinking_effort="medium"
            ),
            X_tr, y_tr, X_te,
        ),
    ]

    combined = results["configs"][-1]
    results["s1_answer"] = (
        "compatible"
        if combined.get("ok")
        else f"incompatible ({combined.get('error_type')})"
    )

    try:
        results["api_usage_after"] = get_api_usage()
    except Exception as exc:  # noqa: BLE001
        results["api_usage_after"] = f"{type(exc).__name__}: {exc}"[:200]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nS1: KV cache + Thinking is {results['s1_answer']}")
    print(f"API usage after: {results.get('api_usage_after')}")
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
