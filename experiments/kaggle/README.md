# The fair-hardware wall-clock run (optional)

**This is optional.** It settles P5, the one prediction still reported as
confounded. The submission does not depend on it, and `docs/limitations.md` says
plainly that it has not been run. Do the four things in
[`../../docs/STATUS.md`](../../docs/STATUS.md) first.

## Why it exists

Every wall-clock number in E4 is tagged `wallclock_comparable: false`. TabPFN ran
remotely on Prior Labs' GPUs; LightGBM ran on this laptop's CPU. That is not a
race, the TabPFN figure is mostly network round-trip, not inference.

`wallclock.py` puts **both models on one machine with one GPU**, using TabPFN's
downloadable weights so nothing goes over the network mid-measurement. The
protocol is E4's, unchanged.

## What you need

- **A Kaggle account**, free, at <https://www.kaggle.com>.
- **Phone verification.** Settings, then Phone Verification. This is what
  unlocks both the GPU *and* internet access in a notebook. Do it before you
  start, or both options are greyed out.
- **The TabPFN licence accepted, and your API key.** Local weights are not just
  a download: TabPFN asks for a one-time licence acceptance, and a notebook is
  not an interactive terminal, so it reads `TABPFN_TOKEN` instead. Log in at
  <https://ux.priorlabs.ai>, accept on the **Licenses** tab, then copy the key
  from <https://ux.priorlabs.ai/account>. If you already have a key for
  `tabpfn-client`, it is the same key; you may still need the licence tick.

  The token only authorises the download. Inference is local, which is the
  whole point of this run, so nothing goes over the network while the clock is
  running. `wallclock.py` downloads and loads the weights in a warm-up phase
  before it times anything.

- About **half an hour** of GPU quota. Kaggle shows what you have left in the
  session sidebar. The run is 24 configurations and writes its JSON after every
  one, so a session that dies part-way still leaves usable rows.

## Step by step

1. Go to <https://www.kaggle.com/code> and click **+ New Notebook**.

2. Right-hand sidebar, **Session options**:
   - **Accelerator**, pick **GPU T4 x2**.
   - **Internet**, switch it **on**. It is off by default, and without it the
     `pip install` and the `git clone` below both fail.
   If either is greyed out, you have not done the phone verification.

3. **Add-ons**, then **Secrets**, then **Add a new secret**. Label it
   `TABPFN_TOKEN`, paste your key as the value, and make sure it is attached to
   this notebook. Put it here rather than in a cell: a cell is saved with the
   notebook, and a notebook can be shared.

4. Same sidebar, **+ Add Input**, search `Bank Account Fraud Dataset NeurIPS 2022`,
   then **Add**. It appears under `/kaggle/input/`. You may have to accept the
   dataset's terms once.

5. Click the first cell, paste this in, and press the play button:

   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")

   !pip install -q tabpfn lightgbm
   !git clone -q https://github.com/ilyas-elm/tabpfn-conformal.git
   %cd tabpfn-conformal
   !pip install -q -e .
   !python experiments/kaggle/wallclock.py \
       --data /kaggle/input/bank-account-fraud-dataset-neurips-2022
   ```

   It prints `device: cuda (Tesla T4)`, then two warm-up lines, then one line per
   configuration, so you can see it working.

6. When it finishes, the sidebar's **Output** tab has
   `results/kaggle_wallclock.json`. Download it.

7. Back here, put that file in `results/` and run:

   ```bash
   python experiments/analyze_kaggle.py
   ```

   That prints the comparison and a verdict per configuration.

## If something goes wrong

- **No GPU option, or Internet cannot be switched on** → phone verification,
  step above. Both are gated on it.
- **`TabPFNLicenseError`, or the script exits saying TabPFN could not load its
  weights** → the licence is not accepted, or the secret is not attached to
  this notebook. The script checks this in its first seconds, on purpose, so
  you lose nothing by rerunning the cell once it is fixed.
- **`No such file: kaggle_secrets`** → you are not on Kaggle. Set
  `TABPFN_TOKEN` in the environment instead.
- **`FileNotFoundError` on `Base.csv`** → the dataset input was not added, or its
  folder is named differently. Run `!ls /kaggle/input/` in a cell and pass the
  real path to `--data`.
- **`Repository not found` on the clone** → the repo is still private.
- **Segfault on a Mac** → this is a Linux script. torch and lightgbm each load
  their own libomp on macOS and the process dies when the second one fits. It
  does not happen on Kaggle, and it is why the local dry run below only covers
  one family at a time.
- **It says it is running on CPU** → the accelerator was not set. The script
  warns on stderr and the analysis refuses to draw a conclusion, because on CPU
  it settles nothing.

## What it measures, and what it does not

It measures **local** TabPFN weights, not the managed API. That is the point,
removing the network is the whole reason for the exercise, but it means these
timings do not reproduce E4's API numbers and are not meant to. The gradient-fit
count does not change either way: 0 for TabPFN, 1 for LightGBM split, 6 for
LightGBM cross at K=5. That count is the hardware-independent one, and it is
what the README leads with.
