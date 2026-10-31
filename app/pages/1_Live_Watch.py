"""Live watch: replay of the incoming CDM stream with counters, plus a
single-event replay to demonstrate escalation alerts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    inject_css,
    GRID,
    TIER_META,
    build_alerts,
    base_layout,
    page_header,
    query,
    risk_history_fig,
    sat_label,
    sidebar_brand,
    style_tiers,
    tier_chip,
)

st.set_page_config(page_title="Aegis LEO — Live Watch", layout="wide")
inject_css()
sidebar_brand()
page_header(
    "Live Watch",
    "Incoming CDMs replayed in arrival order and scored as a live operations simulation.",
    "live",
)

TOTAL = int(query("SELECT count(*) n FROM cdms WHERE stream_seq IS NOT NULL")["n"].iat[0])

ss = st.session_state
ss.setdefault("live_pos", 0)
ss.setdefault("live_running", False)

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
if c1.button("Start" if not ss.live_running else "Pause", width="stretch"):
    ss.live_running = not ss.live_running
    st.rerun()
if c2.button("Reset", width="stretch"):
    ss.live_pos, ss.live_running = 0, False
    st.rerun()
speed = c3.select_slider("CDMs / tick", [10, 25, 50, 100, 250], value=50)


@st.fragment(run_every="1.5s" if st.session_state.live_running else None)
def live_panel():
    ss = st.session_state
    if ss.live_running and ss.live_pos < TOTAL:
        ss.live_pos = min(ss.live_pos + speed, TOTAL)

    recv = query(
        "SELECT * FROM cdms WHERE stream_seq IS NOT NULL AND stream_seq < ?",
        (ss.live_pos,),
    )
    k = st.columns(5)
    k[0].metric("CDMs received", f"{ss.live_pos:,}")
    k[1].metric("Events tracked", f"{recv['event_id'].nunique():,}")
    latest = (recv.sort_values("time_to_tca").groupby("event_id").first()
              if len(recv) else pd.DataFrame())
    n_crit = int((latest["predicted_tier"] == "Critical").sum()) if len(latest) else 0
    k[2].metric("Critical now", f"{n_crit:,}")
    alerts = build_alerts(sources=("current",), max_seq=ss.live_pos) if ss.live_pos else pd.DataFrame()
    k[3].metric("Alerts", 0 if alerts.empty else len(alerts))
    k[4].metric("Feed status", "LIVE" if ss.live_running else "PAUSED")
    st.progress(min(ss.live_pos / TOTAL, 1.0),
                text=f"{ss.live_pos:,} of {TOTAL:,} CDMs received")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Incoming messages")
        if len(recv):
            feed = recv.sort_values("stream_seq", ascending=False).head(12)[
                ["stream_seq", "event_id", "mission_id", "time_to_tca",
                 "miss_distance", "predicted_risk", "predicted_tier"]
            ].copy()
            feed["satellite"] = feed["mission_id"].map(sat_label)
            feed["tier"] = feed["predicted_tier"].map(tier_chip)
            feed = feed[["stream_seq", "event_id", "satellite", "time_to_tca",
                         "miss_distance", "predicted_risk", "tier"]]
            st.dataframe(
                style_tiers(feed, col="tier").format({
                    "time_to_tca": "{:.2f}", "miss_distance": "{:,.0f}",
                    "predicted_risk": "{:.2f}",
                }),
                width="stretch", height=400, hide_index=True,
                column_config={"stream_seq": "#", "time_to_tca": "TCA (d)",
                               "miss_distance": "miss (m)",
                               "predicted_risk": "log₁₀ Pc"},
            )
        else:
            st.info("Press Start to begin the replay.")
    with right:
        st.subheader("Events tracked, by tier")
        if len(latest):
            counts = latest["predicted_tier"].value_counts()
            fig = go.Figure()
            for tier in ["Safe", "Elevated", "Critical"]:
                fig.add_trace(go.Bar(
                    x=[int(counts.get(tier, 0))], y=[f"{TIER_META[tier]['glyph']} {tier}"],
                    orientation="h", marker_color=TIER_META[tier]["color"],
                    name=tier, showlegend=False,
                    text=[int(counts.get(tier, 0))], textposition="outside",
                ))
            fig.update_layout(**base_layout(height=220), bargap=0.35)
            fig.update_xaxes(gridcolor=GRID, title_text="events")
            st.plotly_chart(fig, width="stretch")
        if not alerts.empty:
            st.subheader("Newest alerts")
            st.dataframe(
                alerts.head(6)[["event_id", "satellite", "tier", "alert_type",
                                "time_to_tca_d", "reason"]],
                width="stretch", hide_index=True, height=220,
            )


live_panel()

st.divider()

# ---- single-event escalation replay ----
st.subheader("Simulate new CDM arrivals for one event")
st.caption(
    "Replays an event's messages in time order to demonstrate escalation alerts "
    "(alerts fire on tier escalation, not on every CDM)."
)
esc = query(
    """SELECT event_id, count(*) n,
              min(predicted_tier='Critical') always_crit
       FROM cdms WHERE source IN ('current','upload')
       GROUP BY event_id HAVING n >= 5"""
)
# prefer events that actually escalate during replay
cand = query(
    """SELECT event_id FROM cdms WHERE source IN ('current','upload')
       GROUP BY event_id
       HAVING count(DISTINCT predicted_tier) > 1 AND count(*) >= 5
       LIMIT 400"""
)["event_id"].tolist()
pick = st.selectbox(
    "Event to replay", cand,
    format_func=lambda e: f"event {int(e)}",
) if cand else None

if pick is not None:
    hist = query("SELECT * FROM cdms WHERE event_id = ?", (int(pick),)).sort_values(
        "time_to_tca", ascending=False
    ).reset_index(drop=True)
    step = st.slider("CDMs received", 1, len(hist), len(hist))
    seen = hist.iloc[:step]
    tiers = seen["predicted_tier"].tolist()
    current = tiers[-1]
    m = TIER_META[current]
    cA, cB = st.columns([1, 2])
    with cA:
        st.metric("Current tier", f"{m['glyph']} {current}")
        st.metric("Latest log₁₀ Pc", f"{seen['predicted_risk'].iat[-1]:.2f}")
        st.metric("Miss distance", f"{seen['miss_distance'].iat[-1]:,.0f} m")
        escalations = [
            f"CDM {i+1}: {tiers[i-1]} → {tiers[i]}"
            for i in range(1, len(tiers))
            if ["Safe", "Elevated", "Critical"].index(tiers[i])
            > ["Safe", "Elevated", "Critical"].index(tiers[i - 1])
        ]
        if escalations and current != "Safe":
            st.error("ESCALATION ALERT\n\n" + "\n\n".join(escalations[-2:]))
        elif current == "Critical":
            st.error("CRITICAL — manoeuvre review required")
        else:
            st.success("No escalation so far")
    with cB:
        st.plotly_chart(risk_history_fig(seen), width="stretch")
