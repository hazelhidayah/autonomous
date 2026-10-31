"""Add data: ingest new CDMs by CSV upload or manual entry; every row is
scored by the saved model and stored with source='upload'."""

import io

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    inject_css,
    CDM_COLS,
    REQUIRED_UPLOAD,
    insert_cdms,
    page_header,
    query,
    sat_label,
    score,
    sidebar_brand,
    style_tiers,
    tier_chip,
)

st.set_page_config(page_title="Aegis LEO — Add data", layout="wide")
inject_css()
sidebar_brand()
page_header(
    "Add conjunction data",
    "Score and ingest new collision data from a CSV or a manual observation.",
    "upload",
)

n_up = int(query("SELECT count(*) n FROM cdms WHERE source='upload'")["n"].iat[0])
st.caption(f"{n_up:,} uploaded CDMs currently in the store. New rows are scored on ingest "
           "and appear immediately in Overview, Satellites and Alerts.")

tab_csv, tab_form = st.tabs(["CSV upload", "Manual entry"])

with tab_csv:
    tmpl = pd.DataFrame(columns=CDM_COLS).to_csv(index=False)
    st.download_button("Download CSV template", tmpl, "cdm_template.csv", "text/csv")
    up = st.file_uploader("CDM file (same schema as the challenge CSVs)", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:  # malformed file
            st.error(f"Could not parse the CSV: {e}")
            raw = None
        if raw is not None:
            missing = [c for c in REQUIRED_UPLOAD if c not in raw.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                scored = score(raw)
                st.success(f"Parsed and scored {len(scored)} CDMs.")
                prev = scored[
                    ["event_id", "mission_id", "time_to_tca", "miss_distance",
                     "predicted_risk", "predicted_tier", "p_critical"]
                ].copy()
                prev["tier"] = prev["predicted_tier"].map(tier_chip)
                st.dataframe(
                    style_tiers(prev.drop(columns=["predicted_tier"]), col="tier")
                    .format({"time_to_tca": "{:.2f}", "miss_distance": "{:,.0f}",
                             "predicted_risk": "{:.2f}", "p_critical": "{:.3f}"}),
                    width="stretch", height=300, hide_index=True,
                )
                counts = scored["predicted_tier"].value_counts().to_dict()
                st.markdown("Tier result: " + " · ".join(
                    f"**{tier_chip(t)}: {n}**" for t, n in counts.items()))
                if st.button("Commit to data store", type="primary"):
                    insert_cdms(scored, source="upload")
                    st.success("Stored. Check Overview / Satellites / Alerts.")

with tab_form:
    st.caption("Score a single CDM by hand — useful for what-if checks.")
    with st.form("manual_cdm"):
        c1, c2, c3 = st.columns(3)
        with c1:
            event_id = st.number_input("Event id", 300000, 999999, 300001,
                                       help="Use 3xxxxx to avoid clashing with stored ids")
            mission_id = st.selectbox(
                "Satellite", sorted(query(
                    "SELECT DISTINCT mission_id FROM cdms")["mission_id"]),
                format_func=sat_label)
            c_object_type = st.selectbox(
                "Chaser type", ["DEBRIS", "UNKNOWN", "PAYLOAD", "ROCKET BODY", "TBA"])
            time_to_tca = st.number_input("Days to TCA", 0.0, 7.0, 2.0, 0.1)
        with c2:
            miss_distance = st.number_input("Miss distance (m)", 1.0, 100000.0, 3000.0, 100.0)
            relative_speed = st.number_input("Relative speed (m/s)", 1.0, 20000.0, 10000.0, 100.0)
            rel_r = st.number_input("Relative position R (m)", -50000.0, 50000.0, 500.0)
            rel_t = st.number_input("Relative position T (m)", -50000.0, 50000.0, 2000.0)
        with c3:
            rel_n = st.number_input("Relative position N (m)", -50000.0, 50000.0, 2000.0)
            t_sigma_r = st.number_input("Target radial σ (m)", 0.01, 5000.0, 50.0)
            c_sigma_r = st.number_input("Chaser radial σ (m)", 0.01, 50000.0, 500.0)
            max_risk_estimate = st.number_input("max_risk_estimate (log₁₀, optional)",
                                                -30.0, 0.0, -6.0)
        submitted = st.form_submit_button("Score CDM", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "event_id": event_id, "mission_id": mission_id,
            "time_to_tca": time_to_tca, "miss_distance": miss_distance,
            "relative_speed": relative_speed,
            "relative_position_r": rel_r, "relative_position_t": rel_t,
            "relative_position_n": rel_n,
            "t_sigma_r": t_sigma_r, "c_sigma_r": c_sigma_r,
            "c_object_type": c_object_type,
            "F10": np.nan, "AP": np.nan,
            "max_risk_estimate": max_risk_estimate, "risk": np.nan,
        }])
        scored = score(row)
        r = scored.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Tier", tier_chip(r["predicted_tier"]))
        b.metric("Predicted log₁₀ Pc", f"{r['predicted_risk']:.2f}")
        c.metric("P(critical)", f"{r['p_critical']:.3f}")
        d.metric("P(elevated)", f"{r['p_elevated']:.3f}")
        st.session_state["pending_manual"] = scored
    if "pending_manual" in st.session_state:
        if st.button("Commit this CDM to the store"):
            insert_cdms(st.session_state.pop("pending_manual"), source="upload")
            st.success("Stored — it now shows up across the console.")
