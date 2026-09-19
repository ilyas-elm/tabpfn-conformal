"""Fetch the Bank Account Fraud (BAF) suite and verify what arrived.

BAF (Jesus et al., NeurIPS 2022, Feedzai) is distributed through Kaggle only.
This script downloads it, then *checks* the shape, the fraud rate and the month
column rather than assuming them, and prints what it actually found. If the
numbers drift from what the cahier records, you want to know here and not
halfway through E1.

Usage:
    python scripts/download_data.py            # download + verify
    python scripts/download_data.py --verify   # verify an existing copy only

Kaggle needs credentials. Either accept the dataset's terms and log in via
`kagglehub` when prompted, or place an API token at ~/.kaggle/kaggle.json
(Kaggle -> Settings -> API -> Create New Token). If neither works, the manual
fallback is printed.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SLUG = "sgpjesus/bank-account-fraud-dataset-neurips-2022"

# What the NeurIPS paper and datasheet describe. Deviations are reported, not
# enforced -- the dataset is the source of truth, this is a tripwire.
EXPECTED = {
    "rows": 1_000_000,
    "columns": 32,
    "label": "fraud_bool",
    "time": "month",
    "fraud_rate": 0.011,
}


def manual_instructions() -> None:
    print(
        f"""
Could not download automatically. To do it by hand:

  1. Open https://www.kaggle.com/datasets/{SLUG}
  2. Sign in, accept the dataset terms, click Download
  3. Unzip it into:  {DATA}
     so that {DATA / 'Base.csv'} exists

Then re-run:  python scripts/download_data.py --verify
""",
        file=sys.stderr,
    )


def download() -> pathlib.Path | None:
    try:
        import kagglehub
    except ImportError:
        print('kagglehub is not installed.  pip install -e ".[experiments]"', file=sys.stderr)
        return None

    print(f"Downloading {SLUG} (about 500 MB, this takes a few minutes)...")
    try:
        cached = pathlib.Path(kagglehub.dataset_download(SLUG))
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    DATA.mkdir(exist_ok=True)
    copied = 0
    for src in sorted(cached.rglob("*.csv")):
        dst = DATA / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} new CSV file(s) into {DATA.relative_to(REPO)}/")
    return DATA


def verify() -> int:
    if not DATA.exists():
        print(f"No {DATA.relative_to(REPO)}/ directory.", file=sys.stderr)
        manual_instructions()
        return 1

    csvs = sorted(DATA.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {DATA.relative_to(REPO)}/.", file=sys.stderr)
        manual_instructions()
        return 1

    print(f"\nFound {len(csvs)} file(s):")
    for f in csvs:
        print(f"  {f.name:<24} {f.stat().st_size / 1e6:>8.1f} MB")

    base = next((f for f in csvs if f.stem.lower() == "base"), csvs[0])
    print(f"\nInspecting {base.name} ...")
    df = pd.read_csv(base)

    print(f"  rows                {len(df):>12,}   (expected {EXPECTED['rows']:,})")
    print(f"  columns             {df.shape[1]:>12}   (expected {EXPECTED['columns']})")

    ok = True
    label, time_col = EXPECTED["label"], EXPECTED["time"]

    if label in df.columns:
        rate = df[label].mean()
        print(f"  {label:<18}  {rate:>11.4%}   (expected ~{EXPECTED['fraud_rate']:.1%})")
        print(f"  positives           {int(df[label].sum()):>12,}")
    else:
        print(f"  !! no '{label}' column; found {list(df.columns)[:8]}...")
        ok = False

    if time_col in df.columns:
        months = sorted(df[time_col].unique())
        print(f"  {time_col:<18}  {str(months):>12}")
        if label in df.columns:
            print(f"\n  fraud rate by {time_col} (the drift E3 depends on):")
            by = df.groupby(time_col)[label].agg(["mean", "size"])
            for m, row in by.iterrows():
                bar = "#" * int(row["mean"] * 2000)
                print(f"    {m}  {row['mean']:>7.3%}  n={int(row['size']):>7,}  {bar}")
    else:
        print(f"  !! no '{time_col}' column -- the drift experiment needs it")
        ok = False

    dtypes = df.dtypes.value_counts()
    print(f"\n  dtypes: {dict(zip(dtypes.index.astype(str), dtypes.values))}")
    cats = [c for c in df.columns if df[c].dtype == object]
    if cats:
        print(f"  categorical columns ({len(cats)}): {cats}")

    print("\nOK" if ok else "\nVerification found problems -- see the !! lines above.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="skip the download")
    args = ap.parse_args()

    if not args.verify and download() is None:
        manual_instructions()
        return 1
    return verify()


if __name__ == "__main__":
    sys.exit(main())
