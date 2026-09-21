# Tier 2 — the wall-clock comparison, on hardware where it is fair

Every wall-clock number in E4 carries `wallclock_comparable: false`. TabPFN ran
remotely on Prior Labs' GPUs; LightGBM ran on a laptop CPU. The TabPFN figure is
dominated by upload and round-trip, not inference, so the two were never a race
— see [`../../docs/limitations.md`](../../docs/limitations.md). It is the one
item from the plan that was never run.

`wallclock.py` settles it by putting **both models on one machine with one
accelerator**, using local TabPFN weights so there is no network in the
measurement. The protocol is E4's, unchanged: same matched budgets, same pool
and evaluation construction.

## Running it

One Kaggle session, about fifteen minutes.

1. New notebook → **Accelerator: GPU T4 x2** → **Add Input** → search
   *Bank Account Fraud Dataset NeurIPS 2022* and add it.
2. One cell:

```python
!pip install -q tabpfn lightgbm
!git clone -q https://github.com/ilyas-elm/tabpfn-conformal.git
%cd tabpfn-conformal
!pip install -q -e .
!python experiments/kaggle/wallclock.py \
    --data /kaggle/input/bank-account-fraud-dataset-neurips-2022
```

3. Download `results/kaggle_wallclock.json` from the notebook output, drop it
   into `results/` here, and run:

```bash
python experiments/analyze_kaggle.py
```

The script refuses to pretend: run it without a GPU and it says so, both on
stderr while running and in the analysis output, because on CPU it settles
nothing.

## What it does and does not measure

It measures **local** TabPFN, not the managed API. That is deliberate — removing
the network is the whole point — but it means these numbers do not reproduce the
API timings in E4 and are not meant to. Gradient fits are unchanged either way:
0 for TabPFN, 1 for LightGBM split, 6 for LightGBM cross at K=5. That count is
the hardware-independent one, and it is the number the README leads with.
