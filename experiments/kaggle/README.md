# The fair-hardware wall-clock run (optional)

**This is optional.** It settles P5, the one prediction still reported as
confounded. The submission does not depend on it, and `docs/limitations.md` says
plainly that it has not been run. Do the four things in
[`../../docs/STATUS.md`](../../docs/STATUS.md) first.

## Why it exists

Every wall-clock number in E4 is tagged `wallclock_comparable: false`. TabPFN ran
remotely on Prior Labs' GPUs; LightGBM ran on this laptop's CPU. That is not a
race — the TabPFN figure is mostly network round-trip, not inference.

`wallclock.py` puts **both models on one machine with one GPU**, using TabPFN's
downloadable weights so nothing goes over the network mid-measurement. The
protocol is E4's, unchanged.

## What you need

- **A Kaggle account.** Free, at <https://www.kaggle.com>. If you downloaded the
  dataset with `scripts/download_data.py` you may already have one.
- **Phone verification.** This is the part that catches people out: Kaggle will
  not give you a GPU until you verify a phone number. Settings → Phone
  Verification. Do this *before* you start, or you will build the notebook and
  find the GPU option greyed out.
- About **fifteen minutes**, and GPU quota (Kaggle gives you a number of GPU
  hours per week and shows what you have left in the session sidebar).

## What a Kaggle notebook is

A free Jupyter notebook that runs on Kaggle's machines instead of yours. You
type into cells in the browser, press run, and it executes on their hardware —
including a GPU, which this laptop does not have. Nothing installs locally.

## Step by step

1. Go to <https://www.kaggle.com/code> and click **+ New Notebook**.

2. Right-hand sidebar → **Session options** → **Accelerator** → pick **GPU T4 x2**.
   If it is greyed out, you have not done the phone verification above.

3. Same sidebar → **+ Add Input** → search `Bank Account Fraud Dataset NeurIPS 2022`
   → **Add**. It appears under `/kaggle/input/`. You may have to accept the
   dataset's terms once.

4. Click the first cell, paste this in, and press the ▶ button:

   ```python
   !pip install -q tabpfn lightgbm
   !git clone -q https://github.com/ilyas-elm/tabpfn-conformal.git
   %cd tabpfn-conformal
   !pip install -q -e .
   !python experiments/kaggle/wallclock.py \
       --data /kaggle/input/bank-account-fraud-dataset-neurips-2022
   ```

   The clone only works once the repository is public. It prints one line per
   configuration as it goes, so you can see it working.

5. When it finishes, the sidebar's **Output** (or Data → output) tab has
   `results/kaggle_wallclock.json`. Download it.

6. Back here, put that file in `results/` and run:

   ```bash
   python experiments/analyze_kaggle.py
   ```

   That prints the comparison and a verdict per configuration.

## If something goes wrong

- **No GPU option** → phone verification, step above.
- **`FileNotFoundError` on `Base.csv`** → the dataset input was not added, or its
  folder is named differently. Run `!ls /kaggle/input/` in a cell and pass the
  real path to `--data`.
- **`Repository not found` on the clone** → the repo is still private.
- **It says it is running on CPU** → the accelerator was not set. The script
  warns on stderr and the analysis refuses to draw a conclusion, because on CPU
  it settles nothing.

## What it measures, and what it does not

It measures **local** TabPFN weights, not the managed API. That is the point —
removing the network is the whole reason for the exercise — but it means these
timings do not reproduce E4's API numbers and are not meant to. The gradient-fit
count does not change either way: 0 for TabPFN, 1 for LightGBM split, 6 for
LightGBM cross at K=5. That count is the hardware-independent one, and it is
what the README leads with.
