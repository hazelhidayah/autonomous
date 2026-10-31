# Aegis LEO

## AI-assisted satellite collision-avoidance console

Aegis LEO is a **working prototype** that helps satellite operators screen
Conjunction Data Messages (CDMs), identify the events most likely to need
attention, and investigate them in one mission-control-style application.

The app combines:

- Collision-risk regression for predicted log10 probability of collision (Pc)
- Calibrated Safe / Elevated / Critical classification
- Conservative domain thresholds for miss distance and risk
- Live CDM replay and escalation detection
- Fleet and satellite-level visual analysis
- Historical event investigation
- CSV and manual data ingestion with immediate scoring
- Deduplicated alerts with plain-English reasons
- Model validation, calibration, and feature-importance diagnostics

It addresses the problem of high-volume, manual conjunction screening by
automating first-pass triage while keeping the operator responsible for
investigation and manoeuvre decisions.

> **Scope:** This is a screening and decision-support prototype, not a
> certified flight-safety system or an autonomous manoeuvre planner.

## App pages

The app is launched from `app/Overview.py` and uses Streamlit multipage
navigation.

| Page | Purpose |
|---|---|
| **Overview** | Fleet KPIs, prioritized conjunction table, filters, sorting, 24-hour workload, and urgent work queue. |
| **Live Watch** | Replays incoming CDMs, tracks counters, shows current tier mix, and demonstrates escalation alerts. |
| **Fleet Analysis** | Satellite-level risk, fleet workload, symbolic 3D view, risk landscape, and conjunction matrix. |
| **Historical Archive** | Searches archived events, displays full CDM risk history, and exports event data. |
| **Add Data** | Uploads a CSV or accepts a manual CDM, scores it immediately, previews the result, and stores it. |
| **Alerts** | Shows deduplicated Critical events and tier escalations, sorted by urgency, with reports and explanations. |
| **Model Performance** | Displays regression/classification metrics, confusion matrix, reliability curves, error slices, and feature importance. |

### Typical operator workflow

1. Start on **Overview** to identify the highest-priority events.
2. Use **Alerts** to review Critical conditions and tier escalations.
3. Open **Fleet Analysis** to understand the affected satellite and fleet context.
4. Use **Historical Archive** to inspect how an event changed across CDMs.
5. Add new observations through **Add Data** for immediate scoring.
6. Use **Model Performance** to review model quality and calibration.

## How it addresses the problem

Satellite operators may receive many CDMs for the same event, while only a
small proportion requires urgent review. Aegis LEO:

1. **Scores each message** using telemetry, geometry, uncertainty, object type,
   and time-to-closest-approach features.
2. **Assigns an operational tier**:
   - Critical: predicted risk >= -4 or miss distance < 1 km
   - Elevated: predicted risk >= -6 or miss distance <= 5 km
   - Safe: otherwise
3. **Prioritizes review** by severity, time to TCA, miss distance, and
   calibrated Critical probability.
4. **Reduces alert fatigue** by alerting on Critical states and tier
   escalations rather than every message.
5. **Explains alerts** with reasons such as a keep-out threshold or predicted
   risk threshold.
6. **Keeps humans in the loop** with event history, fleet context, downloadable
   reports, and no automatic manoeuvre execution.

## Model and data

The prototype uses the ESA Kelvins Collision Avoidance Challenge dataset:

- 162,634 training CDMs across 13,154 events
- 24,484 held-out test CDMs across 2,167 events
- 19 satellites

The feature pipeline uses available CDM values such as miss distance, relative
speed, relative position, radial uncertainty, object type, F10/AP values, and
time to TCA. It also derives geometry ratios, logarithmic features, and
missing-value indicators.

The model stack includes:

- Ridge regression baseline
- HistGradientBoosting regression for predicted log10 Pc
- HistGradientBoosting classification
- Isotonic calibration for tier probabilities
- A max-severity operational union so a serious event is not downgraded

Validation uses GroupKFold by `event_id`, ensuring rows from one conjunction
event do not appear in both training and validation folds. A time-to-TCA
holdout is also used to approximate the shifted test regime.

## Reported performance

| Metric | Cross-validation | Held-out test |
|---|---:|---:|
| Critical recall | **0.992** | **0.987** |
| Critical precision | **0.936** | **0.920** |
| Macro F1 | **0.966** | **0.958** |
| Critical false negatives | 90 / 11,630 | 29 / 2,227 |
| Risk MAE in log10 Pc -8 to -3 band | 3.13 | 2.63 |
| Mean P(Critical) vs actual share | - | 0.0906 vs 0.0910 |

The system prioritizes Critical recall because a missed high-risk event is
more costly than an additional event being sent for analyst review.

## Repository structure

```text
app/
  Overview.py                 Main operations dashboard
  core.py                     Shared scoring, SQLite, alerts, and chart helpers
  pages/1_Live_Watch.py       Live CDM replay and escalation simulator
  pages/2_Satellites.py       Fleet and satellite analysis
  pages/3_Archive.py          Historical event archive
  pages/4_Add_Data.py         CSV and manual CDM ingestion
  pages/5_Alerts.py           Alert review and reports
  pages/6_Model_Performance.py Model diagnostics

src/
  config.py                   Paths, thresholds, features, and random seed
  common.py                   Data loading and feature engineering
  eda.py                      Dataset sanity checks
  train.py                    Training, cross-validation, and predictions
  save_models.py              Model and calibrator persistence
  followup_checks.py          Shift and feature checks
  plot_importance.py          Feature-importance plot generation
  make_dashboard.py           Static dashboard generation

outputs/
  models.joblib               Saved scoring models and calibrators
  metrics.json                Validation and test metrics
  predictions.csv             Submission-format predictions
  oof_predictions.csv         Out-of-fold predictions
  test_predictions.csv        Held-out test predictions
  feature_importance.png      Permutation-importance visualization

data/ops.db                   Checked-in demo SQLite store
dashboard.html                Self-contained static dashboard
```

## Local setup

### Requirements

- Python 3.10+
- Git
- Dependencies listed in `requirements.txt`

### Install and run the app

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/Overview.py
```

The checked-in `data/ops.db` and `outputs/models.joblib` provide the demo
runtime data and model. On a fresh local build, the database can be rebuilt
from the challenge dataset when the required source files are available.

### Rebuild the model pipeline

Place the dataset files at:

```text
data/Collision Avoidance Challenge - Dataset/kelvins_competition_data/
  train_data.csv
  test_data.csv
```

Then run:

```bash
.venv/bin/python src/eda.py
.venv/bin/python src/train.py
.venv/bin/python src/save_models.py
.venv/bin/python src/followup_checks.py
.venv/bin/python src/plot_importance.py
.venv/bin/python src/make_dashboard.py
```

The dataset is available from the
[ESA Kelvins challenge archive](https://zenodo.org/record/4463683).

## Public deployment

The app is configured for
[Streamlit Community Cloud](https://share.streamlit.io/).

1. Sign in with GitHub.
2. Select `hazelhidayah/autonomous`.
3. Select branch `main`.
4. Set the main file to `app/Overview.py`.
5. Click **Deploy**.

Streamlit Cloud will provide a public `streamlit.app` URL. The checked-in
`data/ops.db` means the demo does not need the original challenge dataset at
runtime.

The repository also includes `dashboard.html`, a static dashboard that can be
opened or hosted independently without a Streamlit server.

## Design and operational decisions

- **Conservative tiers:** Critical is evaluated before Elevated so a severe
  event cannot be downgraded.
- **Max-severity union:** Rule-based and model-based signals are combined in
  favour of Critical recall.
- **Alert deduplication:** Repeated CDMs for one event do not create repeated
  alerts unless the event state changes.
- **Calibrated probabilities:** Isotonic calibration makes displayed tier
  probabilities more useful as operational estimates.
- **Explainability:** Alerts include threshold-based, plain-English reasons.
- **Human oversight:** The app supports screening and review; it does not
  recommend or execute manoeuvres.

## Limitations

- Pc is estimated from the available feature subset, not a full operational
  6x6 covariance calculation with all cross-terms.
- The 3D fleet view is symbolic and does not propagate physical orbits.
- There is no manoeuvre planning, delta-v optimization, authentication, or
  production-grade message queue.
- External notifications are disabled by default; the webhook configuration is
  a stub for future integration.
- SQLite and Streamlit are suitable for this prototype but not a production
  multi-user architecture.
- A portion of the training risk targets is clipped at -30, which affects
  regression error interpretation.

## Project documentation

- [RESULTS.md](RESULTS.md) - detailed experiment results
- [PMD.md](PMD.md) - product requirements, decisions, risks, and roadmap
- [dashboard.html](dashboard.html) - static shareable demo
- [requirements.txt](requirements.txt) - Python dependencies

## License and dataset notice

This repository is a technical prototype built for demonstration and
evaluation. Review the source dataset's terms and attribution requirements
before redistributing the dataset or using the system operationally.
