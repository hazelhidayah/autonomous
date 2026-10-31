# Aegis LEO — AI-Driven Satellite Collision Avoidance

End-to-end solution: a collision risk evaluator (Pc regression), a calibrated
Safe/Elevated/Critical classifier, a multi-page mission-control console with live feed,
alerts, archive and data ingestion, plus a static shareable dashboard.

Built on the ESA Kelvins Collision Avoidance Challenge dataset (162,634 CDMs / 13,154
events train; 24,484 CDMs / 2,167 events test; 19 satellites).

**Metrics at a glance** (full tables in [RESULTS.md](RESULTS.md); product doc in [PMD.md](PMD.md)):

| Metric | CV (GroupKFold by event) | Test (held out) |
|---|---|---|
| Critical recall | **0.992** | **0.987** |
| False-negative Critical | 90 / 11,630 | 29 / 2,227 |
| Critical precision | 0.936 | 0.920 |
| Macro F1 | 0.966 | 0.958 |
| Risk MAE (log₁₀ Pc), −8…−3 band | 3.13 | 2.63 |
| Mean P(critical) vs actual share | — | 0.0906 vs 0.0910 |

## Project structure

```
src/
  config.py        thresholds, paths, seed, feature lists (single source of truth)
  common.py        loading, tier labelling, feature engineering, alert reasons
  eda.py           stage-1 sanity checks (units, nulls, distributions, tier balance)
  train.py         baselines + GBM, GroupKFold CV, calibration, metrics, predictions
  save_models.py   fits final models + isotonic calibrators → outputs/models.joblib
  followup_checks.py  tca≥2 holdout, max_risk_estimate band analysis, calibration shift
  plot_importance.py  permutation-importance PNG
  make_dashboard.py   static dashboard.html generator
app/
  core.py          SQLite store, scoring-on-ingest, alert engine, chart helpers
  Overview.py      KPI row, filters, risk table, scatter, 24-h panel, insights
  pages/1_Live_Watch.py      stream replay with counters + per-event escalation sim
  pages/2_Satellites.py      satellite selector, 3D nearby objects, drill-down
  pages/3_Archive.py         historical store, searchable, event history
  pages/4_Add_Data.py        CSV upload + manual CDM entry, scored on ingest
  pages/5_Alerts.py          escalation alerts, urgency-sorted, CSV report
  pages/6_Model_Performance.py  confusion matrix, reliability plot, importance
outputs/           metrics.json, predictions.csv, test_predictions.csv,
                   oof_predictions.csv, models.joblib, feature_importance.png
dashboard.html     self-contained static demo (no server needed)
```

## Setup & run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# dataset: place the Kelvins files so that
# data/Collision Avoidance Challenge - Dataset/kelvins_competition_data/{train,test}_data.csv
# exist (download: https://zenodo.org/record/4463683)

.venv/bin/python src/eda.py              # stage 1: sanity checks
.venv/bin/python src/train.py            # stages 2-5: labels, CV, calibration, test preds
.venv/bin/python src/save_models.py      # persist model + calibrators for the app
.venv/bin/python src/followup_checks.py  # shift/holdout + feature checks
.venv/bin/python src/plot_importance.py
.venv/bin/python src/make_dashboard.py   # static dashboard.html

.venv/bin/streamlit run app/Overview.py  # the console (first run builds data/ops.db)
```

## Design decisions

- **Labels** (`src/config.py`): Critical = risk ≥ −4 **or** miss < 1 km; Elevated = risk ≥ −6
  or miss ≤ 5 km; Safe otherwise; Critical applied last so nothing is downgraded.
  `miss_distance` verified to be metres. ~97% of Critical train rows come from the
  miss-distance rule — the ML's real value is Pc in the −6…−4 band for events beyond 5 km,
  and we report band-specific MAE for exactly that reason.
- **Validation**: GroupKFold by `event_id` (rows of one event never straddle folds), plus a
  test-mimicking holdout on `time_to_tca ≥ 2` rows, which predicted real test metrics within
  ~0.3 points. Test labels were used only for final scoring.
- **Models**: Ridge baseline → `HistGradientBoostingRegressor/Classifier`. LightGBM/XGBoost
  need Homebrew `libomp` (absent here); sklearn's HGB is the same histogram-GBM family with
  native NaN routing and categorical support, so no imputation bias for the main models.
  Missing-indicator flags added for F10/AP (4.2% missing, co-occurring) and c_sigma_r.
- **Classification**: three strategies compared (rules on predicted risk / direct 3-class
  with balanced weights / max-severity union). Union selected for Critical recall (0.992),
  accepting ~5 pts of precision — missed events cost more than false alarms.
- **Calibration**: isotonic per class on out-of-fold probabilities (log loss 0.050 → 0.042);
  reliability verified on test despite the 7.2% → 9.1% Critical share shift.
- **Alerts**: fire on Critical events and tier escalations only, deduplicated per event,
  urgency-sorted, each with a plain-English reason; webhook/email stub behind
  `config.ALERT_WEBHOOK_URL`, off by default.
- **Operational impact**: automatic scoring and triage reduce manual labour and human
  error by turning every incoming CDM into a consistent, explainable, prioritised review.
- **max_risk_estimate**: not assumed to be the top feature — measured. Corr with risk is 0.22
  overall but 0.51 inside −10…−4 (the clip at −30 dilutes the global number); permutation
  importance ranks it 2nd behind the engineered miss/σ ratio.

## Deliverables map

| Deliverable | Location |
|---|---|
| Test predictions (spec format: event_id, row_index, predicted_risk, predicted_tier, p_*) | `outputs/predictions.csv` |
| Metrics (CV + holdout + test, confusion matrices, FN counts) | `outputs/metrics.json`, printed by `src/train.py` |
| Trained pipeline | `outputs/models.joblib` |
| Dashboard + alerts app | `app/` (Streamlit, dark mission-control theme) |
| Static demo | `dashboard.html` |
| Results summary for slides | `RESULTS.md` |
| Product management document | `PMD.md` |

## Limitations

Pc estimated from this column subset is an approximation (radial sigmas only, no 6×6
covariance or cross-terms; relative geometry, not orbit propagation). 41% of training
targets sit at the −30 floor, so error there is partly an artefact of the clip. The test
regime is time-shifted (≥ 2 days to TCA); behaviour on late messages is validated only
out-of-fold on train. No manoeuvre planning — the output is a screening decision backed by
calibrated probabilities, with the human making the manoeuvre call.
