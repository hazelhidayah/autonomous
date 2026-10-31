# Aegis LEO — AI-Driven Satellite Collision Avoidance

Working prototype: an AI-assisted console that scores conjunctions, classifies
them as Safe/Elevated/Critical, and helps operators prioritize collision-risk reviews.

Built on the ESA Kelvins Collision Avoidance Challenge dataset (162,634 CDMs / 13,154
events train; 24,484 CDMs / 2,167 events test; 19 satellites).

**Metrics at a glance:**

| Metric | CV (GroupKFold by event) | Test (held out) |
|---|---|---|
| Critical recall | **0.992** | **0.987** |
| False-negative Critical | 90 / 11,630 | 29 / 2,227 |
| Critical precision | 0.936 | 0.920 |
| Macro F1 | 0.966 | 0.958 |
| Risk MAE (log₁₀ Pc), −8…−3 band | 3.13 | 2.63 |
| Mean P(critical) vs actual share | — | 0.0906 vs 0.0910 |

## Pages

- **Overview** — fleet KPIs, prioritized risk table, filters, and urgent work.
- **Live Watch** — simulated CDM stream, counters, and escalation replay.
- **Fleet Analysis** — satellite selector, 3D view, risk landscape, and matrix.
- **Archive** — searchable event history and CSV export.
- **Add Data** — CSV/manual ingestion with immediate model scoring.
- **Alerts** — deduplicated Critical/escalation alerts and reports.
- **Model Performance** — metrics, calibration, confusion matrix, and feature importance.

## How it addresses the problem

Operators receive many conjunction messages but need to find the few requiring
urgent review. Aegis LEO scores each CDM, applies conservative risk tiers,
deduplicates alerts, explains why an event was flagged, and keeps the operator
in control of any manoeuvre decision. It reduces manual triage and missed-critical
risk while preserving audit and review tools.

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

## Public Streamlit deployment

The app can be deployed from this repository with
[Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with GitHub and select `hazelhidayah/autonomous`.
2. Set the branch to `main`.
3. Set the main file to `app/Overview.py`.
4. Click **Deploy**.

The deployed app will receive a public `streamlit.app` URL. The checked-in
`data/ops.db` supplies the demo runtime data, so the cloud deployment does not
need the original challenge dataset.

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

This is a screening prototype, not a certified flight-safety system: Pc is
approximate, the 3D view is symbolic, and manoeuvre planning, authentication,
and production-grade messaging are out of scope.

More detail: [RESULTS.md](RESULTS.md) · [PMD.md](PMD.md)
