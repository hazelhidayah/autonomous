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
st.markdown(
    """<style>
    .live-tier-zero {min-height:156px;padding:14px 16px;background:#F4F6F9;
        border:1px solid #CBD5E1;border-radius:14px;display:grid;gap:8px;align-content:center}
    .live-tier-zero-row {display:flex;align-items:center;justify-content:space-between;
        padding:7px 10px;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:9px;
        color:#475569;font-size:12px}
    .live-tier-zero-row span {display:flex;align-items:center;gap:8px}
    .live-tier-dot {width:9px;height:9px;border-radius:999px;display:inline-block}
    .live-tier-zero-row strong {color:#0F172A;font-size:13px}
    .live-tier-zero-note {margin:2px 0 0;color:#475569;font-size:11px;text-align:center}
    .st-key-live_current_tier_safe [data-testid="stMetric"] {
        background:#D1FAE5!important;border-color:#10B981!important}
    .st-key-live_current_tier_safe [data-testid="stMetric"] * {color:#047857!important}
    .st-key-live_current_tier_elevated [data-testid="stMetric"] {
        background:#FEF3C7!important;border-color:#F59E0B!important}
    .st-key-live_current_tier_elevated [data-testid="stMetric"] * {color:#92400E!important}
    .st-key-live_current_tier_critical [data-testid="stMetric"] {
        background:#FEE2E2!important;border-color:#EF4444!important}
    .st-key-live_current_tier_critical [data-testid="stMetric"] * {color:#B91C1C!important}
    </style>""",
    unsafe_allow_html=True,
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

    left, right = st.columns([1, 1], gap="large")
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
                width="stretch", height=500, hide_index=True,
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
        else:
            st.markdown(
                '<div class="live-tier-zero" role="status" aria-live="polite">'
                '<div class="live-tier-zero-row"><span><i class="live-tier-dot" '
                'style="background:#EF4444"></i>Critical</span><strong>0</strong></div>'
                '<div class="live-tier-zero-row"><span><i class="live-tier-dot" '
                'style="background:#F59E0B"></i>Elevated</span><strong>0</strong></div>'
                '<div class="live-tier-zero-row"><span><i class="live-tier-dot" '
                'style="background:#10B981"></i>Safe</span><strong>0</strong></div>'
                '<p class="live-tier-zero-note">Waiting for the first incoming CDM</p></div>',
                unsafe_allow_html=True,
            )
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
        with st.container(key=f"live_current_tier_{current.lower()}"):
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
