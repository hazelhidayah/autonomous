# Product Management Document — Conjunction Operations Console

**Product:** AI-driven satellite collision avoidance console
**Challenge:** "How could we leverage AI-driven technologies for satellite collision avoidance?"
**Reference standards:** NASA Spacecraft Conjunction Assessment and Collision Avoidance Best
Practices Handbook (NASA/SP-20205011318); ESA Space Debris Office practice.

## 1. Problem & vision

Satellite operators receive thousands of conjunction data messages (CDMs) daily. Manual
evaluation is slow and error-prone, and the cost of a missed critical conjunction (collision,
debris cascade) vastly exceeds the cost of a false alarm (an unnecessary manoeuvre review).

**Vision:** a single console where every incoming CDM is automatically scored for collision
probability, triaged into Safe / Elevated / Critical, and surfaced to operators through
dashboards and escalation alerts — with calibrated probabilities and explainable reasons, so
humans stay in the loop for the manoeuvre decision.

## 2. Users

| Persona | Need |
|---|---|
| **On-console operator** | "What needs my attention right now?" — alerts, 24-h panel, live feed |
| **Flight dynamics analyst** | Drill into one event: geometry, Pc trend across messages, chaser details |
| **Mission manager** | Fleet view per satellite; tier mix; downloadable reports |
| **ML / safety reviewer** | Model metrics, calibration, importance; audit of thresholds and labels |

## 3. Requirements → where they live

| # | Requirement | Delivered in |
|---|---|---|
| R1 | Collision risk evaluator (Pc from conjunction telemetry) | `src/train.py` regression head; `outputs/models.joblib`; scoring on ingest in `app/core.py` |
| R2 | Safe/Elevated/Critical classifier backed by probabilities | tier rules + direct classifier + max-severity union; calibrated P(tier) per row |
| R3 | Dashboard: KPIs, risk table with filters (tier/type/time), scatter with thresholds, trajectory view, 24-h panel, insights | `app/Overview.py`, `app/pages/2_Satellites.py` |
| R4 | Notification/alert layer for manoeuvre planning | `app/pages/5_Alerts.py` + `build_alerts()` in `app/core.py` |
| R5 | Live update feature, counter, live watch page | `app/pages/1_Live_Watch.py` (stream replay + per-event escalation simulator) |
| R6 | Storage for old data, viewable | SQLite store (`data/ops.db`), `app/pages/3_Archive.py` |
| R7 | Place to add new data (scored on ingest) | `app/pages/4_Add_Data.py` (CSV upload + manual form) |
| R8 | Satellite selector with nearby-object visualisation and drill-down | `app/pages/2_Satellites.py` |
| R9 | Model performance transparency | `app/pages/6_Model_Performance.py` (confusion matrix, reliability plot, importance) |
| R10 | Static shareable demo | `dashboard.html` (no server needed) |

## 4. Product decisions (and why)

- **Alert on escalation, not on every CDM.** A 12-message event would otherwise fire 12
  alerts. Elevated is a watch list; alerts fire only for Critical events and tier escalations,
  deduplicated by event and sorted by urgency. This mirrors operator practice and prevents
  alert fatigue.
- **Max-severity union for the operational tier.** Rules-on-predicted-risk and the direct
  classifier are combined by never downgrading. Trade: Critical recall 99.2% at 93.6%
  precision (CV). In this domain a false alarm costs an analyst-minute; a false negative can
  cost the mission.
- **Calibrated probabilities, verified under shift.** Isotonic calibration fit on out-of-fold
  predictions; on the time-shifted test set mean P(critical) 0.0906 vs actual share 0.0910.
  Operators can treat the probability as a real rate, which is what "backed by probability"
  must mean in a safety context.
- **Explainable by construction.** Every alert carries a plain-English reason ("miss distance
  640 m < 1 km keep-out"; "predicted log₁₀ Pc −3.2 ≥ −4"), and the model page shows which
  physical quantities drive predictions (miss/σ ratio dominates — matching Pc physics).
- **Honest visualisation.** The trajectory view is the chaser's relative position in the RTN
  frame from CDM columns — labelled as such, not presented as orbit propagation.
- **Thresholds in one config** (`src/config.py`) so mission-specific tuning (−4/−6, 1 km/5 km)
  is a one-line change.

## 5. Data & model summary

- ESA Kelvins collision-avoidance dataset: 162,634 CDMs / 13,154 events (train),
  24,484 / 2,167 (test); 19 satellites; `risk` = log₁₀ Pc, floored at −30.
- Features: the 13 allowed columns + engineered (miss/σ Mahalanobis proxy, logs, geometry
  ratios, missing-indicator flags). NaNs routed natively by the trees; linear baseline
  median-imputes.
- Validation: GroupKFold by event (no leakage), plus a test-mimicking holdout
  (`time_to_tca ≥ 2` only) that predicted the real test score within ~0.3 points.
- Headline: **Critical recall 0.992 CV / 0.987 test; 29 missed Critical events of 2,227 on
  test; macro F1 0.96; band (−8…−3) MAE 3.1 CV / 2.6 test** vs 7.5 for the linear baseline.

## 6. Success metrics (KPIs)

| KPI | Target | Achieved |
|---|---|---|
| Critical recall (missed-event rate) | ≥ 0.98 | 0.992 CV / 0.987 test |
| Critical precision (alert quality) | ≥ 0.85 | 0.936 CV / 0.920 test |
| Probability calibration (mean P vs actual share) | ±0.01 | 0.0906 vs 0.0910 |
| Alert dedup | 1 alert per event state-change | yes (218 alerts from 24,484 CDMs) |
| Scoring latency on ingest | interactive (< 1 s/batch) | yes |

## 7. Demo script (5 min)

1. **Overview** — read the KPI row, filter to ▲ Critical, point at the scatter quadrants.
2. **Live Watch** — press ▶; counters climb, alerts appear; then replay one event with the
   slider to show an Elevated → Critical escalation firing exactly once.
3. **Satellites** — pick SAT-02; 3D nearby objects inside the 1/5 km spheres; click a row,
   show the Pc-trend chart and chaser details.
4. **Add Data** — type a 450 m miss CDM in the form → instant Critical with P(critical)=1.0;
   commit; show it in Overview and Alerts.
5. **Model Performance** — confusion matrix (88 FN of 11,630), reliability plot on the
   diagonal, importance = physics.

## 8. Risks & limitations

- Pc from these columns is approximate: radial sigmas only, no full 6×6 covariance, no
  cross-terms; geometry is summarised, not propagated.
- 41% of training targets sit at the −30 clip; regression error there is partly an artefact.
- Test regime is time-shifted (only CDMs ≥ 2 days from TCA); late-message behaviour is
  validated out-of-fold on train only.
- No manoeuvre planning: output is a screening decision (review / monitor / ignore).
- Single-node SQLite + Streamlit is a prototype architecture; production would need a message
  queue, model service, and auth.

## 9. Roadmap (post-hackathon)

1. Real CDM ingestion (CCSDS CDM format parser; Space-Track/EUSST feeds).
2. Full-covariance Pc (Foster/Chan) alongside the ML estimate; disagreement flags.
3. Webhook/email/SMS notifications (flag exists in config, off by default).
4. Manoeuvre-suggestion module with Δv estimates and post-manoeuvre Pc.
5. Model monitoring: drift of input distributions and calibration over time.
