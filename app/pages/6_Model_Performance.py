"""Model performance: metrics tables, confusion matrix, reliability plot,
feature importance. Everything from grouped CV / the held-out test set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (GRID, ROOT, SERIES_1, SERIES_2, MUTED, TIER_META, base_layout,
                  inject_css, page_header, query, sidebar_brand)

st.set_page_config(page_title="Aegis LEO — Model performance", layout="wide")
inject_css()
sidebar_brand()
page_header(
    "Model performance",
    "Validation, calibration and feature diagnostics for the collision-risk models.",
    "model",
)

metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
TIERS = ["Safe", "Elevated", "Critical"]

sel = metrics["selected_strategy"]
st.caption(
    f"Validation: 5-fold GroupKFold by event_id (no event straddles folds). Selected "
    f"classification strategy: **{sel.replace('_', ' ')}** — chosen to maximise Critical "
    f"recall, since a missed critical event costs far more than a false alarm."
)

sanity = metrics.get("shuffled_label_sanity")
if sanity:
    st.info(
        f"Leakage sanity check: a GroupKFold fold trained on shuffled tier labels reached "
        f"macro F1 {sanity['macro_f1']:.3f} (chance baseline "
        f"{sanity['expected_macro_f1_baseline']:.3f})."
    )

# ---- regression comparison ----
st.subheader("Risk regression (log₁₀ Pc)")
reg = metrics["regression"]
rows = []
for name, label in [("ridge_baseline", "Ridge (linear baseline)"),
                    ("hgb", "HistGradientBoosting")]:
    r = reg[name]
    rows.append({
        "model": label, "MAE": round(r["mae"], 2), "RMSE": round(r["rmse"], 2),
        "MAE in −8…−3 band": round(r["band_-8_-3"]["mae"], 2),
        "band n": r["band_-8_-3"]["n"],
    })
t = metrics["test_regression"]
rows.append({"model": "HistGradientBoosting — test set", "MAE": round(t["mae"], 2),
             "RMSE": round(t["rmse"], 2),
             "MAE in −8…−3 band": round(t["band_-8_-3"]["mae"], 2),
             "band n": t["band_-8_-3"]["n"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "The −8…−3 band is where Pc decides the tier for events the miss-distance rule "
    "doesn't settle — only ~0.4% of rows are Critical via Pc, so this band is the "
    "model's real job."
)

# ---- classification ----
st.subheader("Tier classification — strategy comparison (CV, out-of-fold)")
rows = []
for name, block in metrics["cv_classification"].items():
    rep = block["report"]
    rows.append({
        "strategy": name.replace("_", " ") + (" (selected)" if name == sel else ""),
        "Critical recall": round(rep["Critical"]["recall"], 3),
        "Critical precision": round(rep["Critical"]["precision"], 3),
        "FN Critical": block["fn_critical"],
        "macro F1": round(rep["macro avg"]["f1-score"], 3),
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Confusion matrix** — selected strategy, CV")
    cm = np.array(metrics["cv_classification"][sel]["confusion_matrix"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"pred {t}" for t in TIERS], y=[f"true {t}" for t in TIERS],
        colorscale=[[0, "#E0F2FE"], [1, "#3B82F6"]], showscale=False,
        text=cm, texttemplate="%{text:,}", hoverinfo="skip",
    ))
    fig.update_layout(**base_layout(height=340))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    fn = metrics["cv_classification"][sel]["fn_critical"]
    st.caption(f"False-negative Critical events (the metric that matters most): "
               f"**{fn} of {cm[2].sum():,}** — {fn / cm[2].sum() * 100:.1f}%.")

with c2:
    st.markdown("**Per-class metrics** — selected strategy")
    rep = metrics["cv_classification"][sel]["report"]
    per = pd.DataFrame({
        "tier": TIERS,
        "precision": [round(rep[t]["precision"], 3) for t in TIERS],
        "recall": [round(rep[t]["recall"], 3) for t in TIERS],
        "F1": [round(rep[t]["f1-score"], 3) for t in TIERS],
        "support": [int(rep[t]["support"]) for t in TIERS],
    })
    st.dataframe(per, hide_index=True, width="stretch")
    hold = metrics.get("cv_tca_ge_2", {})
    if hold:
        hr = hold["report"]
        st.markdown(
            f"**Test-mimicking holdout** (CV rows with `time_to_tca ≥ 2`): Critical recall "
            f"{hr['Critical']['recall']:.3f}, precision {hr['Critical']['precision']:.3f}, "
            f"macro F1 {hr['macro avg']['f1-score']:.3f}, FN Critical {hold['fn_critical']} — "
            f"predicts the actual test score almost exactly."
        )
    tr = metrics["test_classification"][sel]["report"]
    st.markdown(
        f"**Actual test set**: Critical recall {tr['Critical']['recall']:.3f}, precision "
        f"{tr['Critical']['precision']:.3f}, macro F1 {tr['macro avg']['f1-score']:.3f}, "
        f"FN Critical {metrics['test_classification'][sel]['fn_critical']}."
    )

# ---- reliability plot ----
st.subheader("Calibration (isotonic, fit on out-of-fold probabilities)")
st.caption(
    f"Log loss {metrics['calibration']['log_loss_raw']:.4f} → "
    f"{metrics['calibration']['log_loss_calibrated']:.4f} after calibration. "
    "Reliability below is computed on the held-out test set."
)
oof = query("SELECT predicted_tier, p_safe, p_elevated, p_critical, risk, miss_distance "
            "FROM cdms WHERE source='current'")
from common import assign_tier  # noqa: E402

true_tier = assign_tier(oof["risk"], oof["miss_distance"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                         line=dict(color=MUTED, width=1, dash="dash")))
for tier, col in [("Safe", "p_safe"), ("Elevated", "p_elevated"), ("Critical", "p_critical")]:
    p = oof[col].to_numpy()
    y = (true_tier == tier).astype(float)
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    mp = [p[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).any()]
    fy = [y[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).any()]
    m = TIER_META[tier]
    fig.add_trace(go.Scatter(
        x=mp, y=fy, mode="lines+markers", name=f"{m['glyph']} P({tier})",
        line=dict(color=m["color"], width=2),
        marker=dict(size=9, symbol=m["symbol"]),
        hovertemplate="mean p %{x:.3f} · observed %{y:.3f}<extra>" + tier + "</extra>",
    ))
fig.update_layout(**base_layout(height=380))
fig.update_xaxes(title_text="mean predicted probability (decile bins)", gridcolor=GRID,
                 range=[-0.02, 1.02])
fig.update_yaxes(title_text="observed frequency", gridcolor=GRID, range=[-0.02, 1.02])
st.plotly_chart(fig, width="stretch")

# ---- feature importance ----
st.subheader("What drives the predicted risk")
imp = pd.DataFrame(metrics["permutation_importance"]).head(12)[::-1]
fig = go.Figure(go.Bar(
    x=imp["mean"], y=imp["feature"], orientation="h",
    error_x=dict(array=imp["std"], color=MUTED),
    marker_color=SERIES_1,
))
fig.update_layout(**base_layout(height=420))
fig.update_xaxes(title_text="permutation importance (increase in MSE of log₁₀ Pc)",
                 gridcolor=GRID)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Physics check: chaser radial σ and the miss/σ ratio dominate — Pc is driven by miss "
    "distance relative to combined position uncertainty. `max_risk_estimate` is "
    "band-informative (corr 0.51 inside −10…−4 vs 0.22 overall, diluted by the −30 clip)."
)

with st.expander("Error slices and failure cases"):
    breakdown = metrics.get("error_breakdown", {})
    for label, key in [("Object type", "object_type"), ("TCA bucket", "time_to_tca"),
                       ("Risk band", "risk_band")]:
        st.markdown(f"**{label}**")
        rows = [{"slice": name, **values} for name, values in breakdown.get(key, {}).items()]
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    failures = metrics.get("failure_cases", {})
    st.markdown(
        f"**Critical false negatives in OOF:** {failures.get('false_negative_critical', 0)}. "
        "Worst misses and false-negative examples are retained in metrics.json for review."
    )
