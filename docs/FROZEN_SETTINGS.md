# Frozen settings and test protocol (T14, T15)

Frozen on 16 Nov 2026, at the commit that adds this file (`git log -1 -- docs/FROZEN_SETTINGS.md`).
Everything below was chosen using the training and validation sets only, and none of it changes
after the test predictions are made.

## Data and evaluation
- Data: the instructor's preprocessed recordings (20 bipolar channels, 64 Hz, 0.5–30 Hz) and splits
  (825 / 103 / 104 children). All ages (0.16–16 years) are kept.
- Each recording is used up to its last non-zero sample. 0–300 s is pre-N2, 300–900 s is N2.
- Scores are per child: MAE in years, and balanced MAE, the mean of the MAEs of five age groups
  (<1, 1–3, 3–6, 6–12, 12+, each including its lower edge). r and R² are reported but not used to choose.
- Reference: predicting the training median age (2.98 years) for every child.

## Baseline: ridge regression (notebooks/01_baseline.ipynb)
- 13 features per channel from 30 s windows: log variance, Hjorth mobility and complexity, and log
  and relative power in delta (0.5–4 Hz), theta (4–8), alpha (8–12), sigma (12–16) and beta (16–30),
  from Welch spectra with 4 s segments.
- The median over windows, separately for pre-N2 and N2: 520 features per child.
- Standardised features; ridge regression on log(age); predictions converted back to years. The
  regularisation strength is chosen by leave-one-out on train from 25 values between 0.01 and 10,000
  (chosen: 56). Fitted on train only.

## CNN (scripts/train_cnn.py)
- Architecture: figure 3 of the assignment, with ReLU and batch norm. Three blocks of two (1×3)
  convolutions (16, 32 and 64 filters) and (1×2) max pooling; a fully connected layer per channel with
  64 outputs and batch norm; the maximum over the 20 channels; dropout 0.3 and a linear output for
  log(age). About 549,000 parameters.
- Input: 16 s windows divided by one global scale, the RMS amplitude of the training set (42.4 µV).
- Training on train only: random 16 s crops inside the real signal, 8 per child per epoch, batch 64,
  Huber loss on log(age), AdamW (learning rate 1e-3, weight decay 1e-2), cosine schedule over 60 epochs.
- The model after the last epoch is kept; validation never picks an epoch. Batch-norm statistics are
  recomputed over one fixed epoch of training crops before saving.
- Prediction: every non-overlapping 16 s window of the real signal; the median predicted log(age) per
  child, converted to years.
- Three runs, seeds 0, 1 and 2 (outputs/cnn16s_seed0, _seed1, _seed2). The development run
  outputs/cnn16s is not used in the report.

## Test protocol
1. Train the three CNN runs. Check the comparison notebook on validation before any test
   prediction exists.
2. Predict test once: `--predict-test` for each CNN run and the last cell of 01 for ridge. Both first
   check that the saved model reproduces its validation predictions.
3. CNN numbers are the mean over the three runs; each run's numbers and the range are also reported.
   The runs are not averaged into one model, and no run is picked.
4. Comparison with ridge: the paired bootstrap used on validation (10,000 resamples of children within
   each age group, seed 0). The CNN's error for a child is its mean absolute error over the three runs.
   For MAE and balanced MAE we report the observed difference (CNN − ridge), its 95% interval and the
   share of resamples in which the CNN is worse.
5. A difference is called clear only if its 95% interval excludes 0. Otherwise it is reported as not
   distinguishable at this test size (104 children, 9 aged 12+).
6. The same table, bootstrap and figures are made for validation.
7. If a bug is found after the test predictions are made, we fix it, rerun steps 1–6, and report both
   results and the reason.

Expected before seeing test: test has more children aged 6 or over than validation (30 vs. 23), where
errors are largest, so plain MAE will likely be higher on test for both models.
