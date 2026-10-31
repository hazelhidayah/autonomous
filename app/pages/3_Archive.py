"""Archive: the historical CDM store (resolved events), searchable."""

import pandas as pd
import streamlit as st

from core import (
    inject_css,
    TIER_ORDER,
    latest_per_event,
    page_header,
    query,
    risk_history_fig,
    sat_label,
    sidebar_brand,
    style_tiers,
    tier_chip,
)

st.set_page_config(page_title="Aegis LEO — Archive", layout="wide")
inject_css()
sidebar_brand()
page_header(
    "Historical archive",
    "Resolved conjunction history with out-of-fold model predictions for every event.",
    "archive",
)
st.markdown(
    """<style>
    .st-key-archive_tier_filter [data-tag][aria-label="[C] Critical"] {
        background:#FEE2E2!important;border:1px solid #EF4444!important;
        color:#B91C1C!important;
    }
    .st-key-archive_tier_filter [data-tag][aria-label="[E] Elevated"] {
        background:#FEF3C7!important;border:1px solid #F59E0B!important;
        color:#92400E!important;
    }
    .st-key-archive_tier_filter [data-tag][aria-label="[S] Safe"] {
        background:#D1FAE5!important;border:1px solid #10B981!important;
        color:#047857!important;
    }
    .st-key-archive_tier_filter [data-tag] > span,
    .st-key-archive_tier_filter [data-tag] > button {
        color:inherit!important;
    }
    </style>""",
    unsafe_allow_html=True,
)

stats = query(
    """SELECT count(*) rows, count(DISTINCT event_id) events
       FROM cdms WHERE source='archive'"""
)
k = st.columns(4)
k[0].metric("Archived CDMs", f"{int(stats['rows'].iat[0]):,}")
k[1].metric("Archived events", f"{int(stats['events'].iat[0]):,}")

ev = latest_per_event(sources=("archive",))
k[2].metric("Critical outcomes", f"{(ev['predicted_tier']=='Critical').sum():,}")
k[3].metric("Satellites covered", ev["mission_id"].nunique())

f1, f2, f3 = st.columns(3)
with f1:
    tiers = st.multiselect(
        "Tier", TIER_ORDER, default=TIER_ORDER, format_func=tier_chip,
        key="archive_tier_filter",
    )
with f2:
    sats = st.multiselect("Satellite", sorted(ev["mission_id"].unique()),
                          format_func=sat_label, placeholder="All satellites")
with f3:
    eid_search = st.text_input("Find event id", placeholder="e.g. 8767")

view = ev[ev["predicted_tier"].isin(tiers)]
if sats:
    view = view[view["mission_id"].isin(sats)]
if eid_search.strip().isdigit():
    view = view[view["event_id"] == int(eid_search)]

st.caption(f"{len(view):,} events match.")
tbl = view.sort_values("predicted_risk", ascending=False)[
    ["event_id", "mission_id", "c_object_type", "predicted_tier", "time_to_tca",
     "miss_distance", "risk", "predicted_risk"]
].copy()
tbl["satellite"] = tbl["mission_id"].map(sat_label)
tbl["tier"] = tbl["predicted_tier"].map(tier_chip)
show = tbl[["event_id", "satellite", "c_object_type", "tier",
            "time_to_tca", "miss_distance", "risk", "predicted_risk"]]
sel = st.dataframe(
    style_tiers(show, col="tier").format({
        "time_to_tca": "{:.2f}", "miss_distance": "{:,.0f}",
        "risk": "{:.2f}", "predicted_risk": "{:.2f}",
    }),
    width="stretch", height=380, hide_index=True,
    on_select="rerun", selection_mode="single-row",
    column_config={"event_id": "event", "c_object_type": "chaser",
                   "time_to_tca": "final TCA (d)", "miss_distance": "miss (m)",
                   "risk": "actual log₁₀ Pc", "predicted_risk": "predicted"},
)

rows = sel.selection.rows if sel and sel.selection else []
if rows:
    eid = int(show.iloc[rows[0]]["event_id"])
    hist = query("SELECT * FROM cdms WHERE event_id = ? AND source='archive'", (eid,))
    st.divider()
    st.subheader(f"Event {eid} — full CDM history ({len(hist)} messages)")
    a, b = st.columns([1.3, 1])
    with a:
        st.plotly_chart(risk_history_fig(hist), width="stretch")
    with b:
        h = hist.sort_values("time_to_tca", ascending=False)[
            ["time_to_tca", "miss_distance", "risk", "predicted_risk", "predicted_tier"]
        ].copy()
        h["tier"] = h["predicted_tier"].map(tier_chip)
        st.dataframe(
            h.drop(columns=["predicted_tier"]).style.format({
                "time_to_tca": "{:.2f}", "miss_distance": "{:,.0f}",
                "risk": "{:.2f}", "predicted_risk": "{:.2f}",
            }),
            width="stretch", height=360, hide_index=True,
        )
    st.download_button(
        "Download this event's CDMs (CSV)",
        hist.to_csv(index=False).encode(), f"event_{eid}_history.csv", "text/csv",
    )
else:
    st.info("Select a row to open the event's full message history.")
