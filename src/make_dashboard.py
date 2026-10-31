"""Build the static HTML dashboard from model outputs.

Reads outputs/oof_predictions.csv + outputs/test_predictions.csv, joins
back to the raw CDMs, reduces to the latest CDM per event, and writes
dashboard.html — a self-contained page (Plotly via CDN) with:
  stat tiles / 24-h warning panel, auto-generated insights, risk table,
  miss-distance-vs-Pc scatter with threshold lines, and a 3D relative
  trajectory view.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    MISS_CRITICAL_M,
    MISS_ELEVATED_M,
    RISK_CRITICAL,
    RISK_ELEVATED,
    assign_tier,
    load,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

KEEP = [
    "event_id",
    "time_to_tca",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "c_object_type",
    "risk",
]


def latest_per_event(raw: pd.DataFrame, preds: pd.DataFrame, split: str) -> pd.DataFrame:
    df = raw[KEEP].copy()
    df["predicted_risk"] = preds["predicted_risk"].to_numpy()
    df["predicted_tier"] = preds["predicted_tier"].to_numpy()
    df["p_critical"] = preds["p_critical"].to_numpy()
    df = df.sort_values("time_to_tca").groupby("event_id", as_index=False).first()
    df["split"] = split
    return df


def main():
    train = load("train")
    test = load("test")
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    tpr = pd.read_csv(OUT / "test_predictions.csv")

    ev_train = latest_per_event(train, oof, "train (out-of-fold)")
    ev_test = latest_per_event(test, tpr, "test")
    # train and test event_ids overlap numerically; namespace them
    ev_test["event_id"] = ev_test["event_id"] + 100_000
    ev = pd.concat([ev_train, ev_test], ignore_index=True)
    ev["true_tier"] = assign_tier(ev["risk"], ev["miss_distance"])

    tier_rank = {"Safe": 0, "Elevated": 1, "Critical": 2}
    ev["tier_rank"] = ev["predicted_tier"].map(tier_rank)

    # ---- 24-hour warning panel ----
    # some train CDMs are issued after TCA (negative time_to_tca); exclude past events
    soon = ev[(ev["time_to_tca"] >= 0.0) & (ev["time_to_tca"] <= 1.0)]
    panel = {
        "total_events": int(len(ev)),
        "soon_total": int(len(soon)),
        "soon_by_tier": soon["predicted_tier"].value_counts().to_dict(),
        "soon_inside_5km": int((soon["miss_distance"] <= MISS_ELEVATED_M).sum()),
        "soon_inside_1km": int((soon["miss_distance"] < MISS_CRITICAL_M).sum()),
    }

    # ---- auto-generated insights ----
    crit = ev[ev["predicted_tier"] == "Critical"]
    crit_soon = soon[soon["predicted_tier"] == "Critical"]
    upcoming = ev[ev["time_to_tca"] >= 0.0]
    closest = upcoming.loc[upcoming["miss_distance"].idxmin()]
    insights = []
    insights.append(
        f"{len(crit_soon)} critical event{'s' if len(crit_soon) != 1 else ''} within the next 24 hours "
        f"({panel['soon_total']} conjunctions total in that window, "
        f"{panel['soon_inside_1km']} inside the 1 km keep-out zone)."
    )
    insights.append(
        f"Closest predicted approach: {closest['miss_distance']/1000:.2f} km in "
        f"{closest['time_to_tca']*24:.1f} hours (event {int(closest['event_id'])}, "
        f"predicted log₁₀ Pc {closest['predicted_risk']:.1f})."
    )
    if len(crit):
        top_type = crit["c_object_type"].mode().iat[0]
        share = (crit["c_object_type"] == top_type).mean() * 100
        insights.append(
            f"Most common chaser type among the {len(crit)} critical events: "
            f"{top_type} ({share:.0f}%)."
        )
        med_speed = crit["relative_speed"].median()
        insights.append(
            f"Median relative speed in critical events is {med_speed/1000:.1f} km/s, so "
            f"collision geometry, not closing speed, is what separates tiers here."
        )

    # ---- risk table: top 30 by predicted risk ----
    table = (
        ev.sort_values("predicted_risk", ascending=False)
        .head(30)[
            [
                "event_id",
                "split",
                "time_to_tca",
                "miss_distance",
                "predicted_risk",
                "p_critical",
                "predicted_tier",
                "c_object_type",
            ]
        ]
    )
    table_rows = table.round(
        {"time_to_tca": 2, "predicted_risk": 2, "p_critical": 3}
    ).to_dict("records")

    # ---- scatter data: all elevated/critical + a sample of safe ----
    rng = np.random.default_rng(42)
    safe = ev[ev["predicted_tier"] == "Safe"]
    keep_safe = safe.sample(n=min(3000, len(safe)), random_state=42)
    sc = pd.concat([ev[ev["predicted_tier"] != "Safe"], keep_safe])
    scatter = {
        tier: {
            "x": grp["miss_distance"].round(1).tolist(),
            "y": np.power(10.0, grp["predicted_risk"]).tolist(),
            "event": grp["event_id"].astype(int).tolist(),
            "tca": grp["time_to_tca"].round(2).tolist(),
        }
        for tier, grp in sc.groupby("predicted_tier")
    }

    # ---- 3D view: riskiest events whose geometry fits the 8 km frame ----
    rel_mag = np.sqrt(
        ev["relative_position_r"] ** 2
        + ev["relative_position_t"] ** 2
        + ev["relative_position_n"] ** 2
    )
    top3d = (
        ev[rel_mag <= 8000]
        .sort_values("predicted_risk", ascending=False)
        .head(40)
    )
    three_d = {
        tier: {
            "r": grp["relative_position_r"].tolist(),
            "t": grp["relative_position_t"].tolist(),
            "n": grp["relative_position_n"].tolist(),
            "event": grp["event_id"].astype(int).tolist(),
            "risk": grp["predicted_risk"].round(2).tolist(),
        }
        for tier, grp in top3d.groupby("predicted_tier")
    }

    metrics = json.loads((OUT / "metrics.json").read_text())
    sel = metrics["selected_strategy"]
    cv_rep = metrics["cv_classification"][sel]["report"]
    headline = {
        "strategy": sel.replace("_", " "),
        "critical_recall": cv_rep["Critical"]["recall"],
        "critical_precision": cv_rep["Critical"]["precision"],
        "macro_f1": cv_rep["macro avg"]["f1-score"],
        "reg_mae": metrics["regression"]["hgb"]["mae"],
        "test_critical_recall": metrics["test_classification"][sel]["report"]["Critical"]["recall"],
    }

    payload = {
        "panel": panel,
        "insights": insights,
        "table": table_rows,
        "scatter": scatter,
        "three_d": three_d,
        "headline": headline,
        "thresholds": {
            "miss_crit": MISS_CRITICAL_M,
            "miss_elev": MISS_ELEVATED_M,
            "pc_crit": 10 ** RISK_CRITICAL,
            "pc_elev": 10 ** RISK_ELEVATED,
        },
    }

    template = (ROOT / "src" / "dashboard_template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(payload))
    (ROOT / "dashboard.html").write_text(html)
    print(f"Wrote dashboard.html ({len(html)/1e6:.1f} MB), {len(ev)} events")


if __name__ == "__main__":
    main()
