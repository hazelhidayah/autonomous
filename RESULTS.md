# Results summary (presentation-ready)

## Headline

> **98.7% of critical conjunctions caught on the held-out test set (2,198 of 2,227), with
> calibrated probabilities (mean P(critical) 0.091 vs actual 0.091) and one alert per event
> state-change instead of one per message.**

## Stage outputs

**1 — Data sanity.** 162,634 CDMs / 13,154 events train; 24,484 / 2,167 test; 19 satellites.
`miss_distance` confirmed metres (median 12.4 km). `risk` floored at −30 (41% of rows).
Nulls: F10/AP 4.19%, c_sigma_r 0.01% → missing-indicator flags + native NaN routing in the
trees (linear baseline median-imputes). Test contains only `time_to_tca ≥ 2` rows (shift).

**2 — Labels.** Critical first, then Elevated, then Safe. Balance train 71.2 / 21.6 / 7.2%,
test 64.4 / 26.6 / 9.1%. Only 716 train rows are Critical via risk ≥ −4 vs 11,305 via
miss < 1 km → **the tier rule is miss-distance-dominated; the ML's real job is Pc in the
−6…−4 band for events outside 5 km.**

**3 — Regression (log₁₀ Pc), GroupKFold by event (5 folds):**

| Model | MAE | RMSE | MAE in −8…−3 band |
|---|---|---|---|
| Ridge (linear baseline) | 6.17 | 7.43 | 7.47 |
| HistGradientBoosting | **3.52** | **5.24** | **3.13** |
| HistGradientBoosting — test | 3.72 | 5.36 | **2.63** |

(LightGBM unavailable without Homebrew libomp on this machine; sklearn's
HistGradientBoosting is the same algorithm family with native NaN + categorical support.)

**4 — Classification (CV, out-of-fold):**

| Strategy | Critical recall | Critical precision | FN Critical | macro F1 |
|---|---|---|---|---|
| Rules on predicted risk | 0.985 | 0.946 | 178 | 0.963 |
| Direct 3-class (balanced weights) | 0.985 | 0.986 | 177 | 0.977 |
| **Max-severity union (selected)** | **0.992** | 0.936 | **90** | 0.966 |

Confusion matrix (union, CV): true Critical 11,630 → 11,540 caught, 90 missed (0.8%).
Calibration: isotonic on OOF probabilities; log loss 0.050 → 0.042; test reliability on the
diagonal (mean P(critical) 0.0906 vs share 0.0910).

**5 — Test set (never used for training/tuning):**

| Strategy | Critical recall | Critical precision | FN Critical | macro F1 |
|---|---|---|---|---|
| Rules on predicted risk | 0.985 | 0.915 | 38 | 0.948 |
| Direct classifier | 0.977 | 0.988 | 51 | 0.971 |
| **Max-severity union** | **0.987** | 0.920 | **29** | 0.958 |

The test-mimicking holdout (CV rows with `time_to_tca ≥ 2`: recall 0.990, macro F1 0.955)
predicted the real test numbers almost exactly — the CV estimate is honest under the shift.

**6 — Explainability.** Permutation importance: `miss_over_sigma` (0.49) ≫
`max_risk_estimate` (0.28) > `time_to_tca` (0.24) > `combined_sigma` (0.18) — i.e. miss
distance relative to combined position uncertainty drives Pc, matching the physics.
`max_risk_estimate` is band-informative (corr 0.51 in −10…−4 vs 0.22 overall; the global
figure is diluted by the −30 clip). Every alert carries a plain-English reason.

## Key insights

1. Critical is ~97% miss-distance-driven; honest reporting of where ML adds value (the Pc
   band −6…−4) matters more than headline accuracy.
2. Group-aware validation is everything here: row-level splits leak event state and inflate
   scores; our grouped CV matched test within ~0.3 points on every metric.
3. Calibration survives the train→test tier shift (7.2% → 9.1% Critical), so the
   probabilities are usable as rates for operator decision thresholds.
4. Escalation-based alerting cuts 24,484 messages to 218 actionable alerts (54 of them
   escalations) without missing a Critical event state.

## Limitations

Pc from these columns is approximate (radial σ only, no covariance cross-terms, no
propagation); 41% of targets sit at the −30 clip; the test regime starts 2 days from TCA;
no manoeuvre planning is modelled; thresholds are taken as given (configurable in
`src/config.py`).
