"""Operational landing page: status, urgent work, and direct next actions."""

from datetime import datetime, timezone

import numpy as np
import streamlit as st

from core import (
    TIER_META,
    build_alerts,
    inject_css,
    latest_per_event,
    page_header,
    sat_label,
    sidebar_brand,
    style_criticality,
)

st.set_page_config(page_title="Aegis LEO — Overview", layout="wide")
inject_css()
sidebar_brand()

ev = latest_per_event()
alerts = build_alerts()
now = datetime.now(timezone.utc)

head, status = st.columns([4, 1], vertical_alignment="center")
with head:
    page_header(
        "Operations overview",
        "A concise view of fleet health, urgent conjunctions and the next operator actions.",
        None,
    )
status.markdown(
    f'<div style="text-align:right"><span class="pill" style="background:#D1FAE5;'
    f'border:1px solid #10B981;color:#047857">LIVE · NOMINAL</span>'
    f'<div class="aegis-sub" style="margin-top:6px">updated {now:%H:%M:%S} UTC</div></div>',
    unsafe_allow_html=True,
)

n_critical = int((ev["predicted_tier"] == "Critical").sum())
n_elevated = int((ev["predicted_tier"] == "Elevated").sum())
within_24h = ev[ev["time_to_tca"].between(0, 1)]
inside_5km = int((ev["miss_distance"] <= 5_000).sum())
inside_1km = int((ev["miss_distance"] < 1_000).sum())
nearest = ev.loc[ev["miss_distance"].idxmin()]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Events screened", f"{len(ev):,}", "latest CDM per event")
k2.metric("Critical / Elevated", f"{n_critical:,} / {n_elevated:,}",
          "review / watchlist")
k3.metric("Next 24 hours", f"{len(within_24h):,}", "0 ≤ TCA ≤ 1 day")
k4.metric("Inside 5 km / 1 km", f"{inside_5km:,} / {inside_1km:,}",
          "miss distance")
k5.metric("Active alerts", f"{len(alerts):,}", "deduplicated events")

st.markdown(
    '<div class="fleet-sectionbar"><strong>Prioritize the conjunction view</strong>'
    '<span>Filters and sorting apply to the event table below</span></div>',
    unsafe_allow_html=True,
)
filter_col, sort_col, rows_col = st.columns([1.45, 1.35, .7])
with filter_col:
    tier_filter = st.multiselect(
        "Risk tier", ["Critical", "Elevated", "Safe"],
        default=["Critical", "Elevated", "Safe"],
        help="Limit the table to one or more operational risk tiers.",
    )
with sort_col:
    sort_by = st.selectbox(
        "Prioritize by",
        [
            "Operational priority",
            "Time to closest approach",
            "Miss distance",
            "Predicted collision probability",
            "Calibrated critical probability",
            "Satellite",
        ],
        help="Each option uses the operationally meaningful sort direction.",
    )
with rows_col:
    rows_to_show = st.selectbox("Rows", [10, 25, 50, 100], index=0)

rank = {"Critical": 0, "Elevated": 1, "Safe": 2}
view = ev[ev["predicted_tier"].isin(tier_filter)].copy()
view["_tier_rank"] = view["predicted_tier"].map(rank)
view["collision_probability"] = np.power(10.0, view["predicted_risk"])
sort_rules = {
    "Operational priority": (["_tier_rank", "time_to_tca", "miss_distance"], [True, True, True]),
    "Time to closest approach": (["time_to_tca", "_tier_rank"], [True, True]),
    "Miss distance": (["miss_distance", "_tier_rank"], [True, True]),
    "Predicted collision probability": (["predicted_risk", "_tier_rank"], [False, True]),
    "Calibrated critical probability": (["p_critical", "_tier_rank"], [False, True]),
    "Satellite": (["mission_id", "_tier_rank", "time_to_tca"], [True, True, True]),
}
sort_columns, ascending = sort_rules[sort_by]
view = view.sort_values(sort_columns, ascending=ascending)

page_signature = (tuple(tier_filter), sort_by, rows_to_show)
if st.session_state.get("overview_page_signature") != page_signature:
    st.session_state.overview_page = 0
    st.session_state.overview_page_signature = page_signature

total_events = len(view)
total_pages = max(1, int(np.ceil(total_events / rows_to_show)))
current_page = min(st.session_state.get("overview_page", 0), total_pages - 1)
st.session_state.overview_page = current_page
page_start = current_page * rows_to_show
page_view = view.iloc[page_start:page_start + rows_to_show].copy()

urgent_col, brief_col = st.columns([1.8, 1], gap="medium")

with urgent_col:
    with st.container(border=True):
        st.markdown(
            '<p class="card-title">Prioritized conjunctions</p>'
            f'<p class="card-sub">{page_start + 1 if total_events else 0:,}–'
            f'{min(page_start + len(page_view), total_events):,} of {total_events:,} events · '
            f'sorted by {sort_by.lower()}</p>',
            unsafe_allow_html=True,
        )
        page_view["satellite"] = page_view["mission_id"].map(sat_label)
        page_view["status"] = page_view["predicted_tier"].map(
            {"Critical": "[C] Critical", "Elevated": "[E] Elevated", "Safe": "[S] Safe"}
        )
        table = page_view[["status", "satellite", "c_object_type", "time_to_tca",
                           "miss_distance", "collision_probability", "p_critical"]]
        selected = st.dataframe(
            style_criticality(table, "status"),
            hide_index=True, width="stretch", height=390,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "status": "status", "satellite": "asset", "c_object_type": "chaser",
                "time_to_tca": st.column_config.NumberColumn("TCA", format="%.2f d"),
                "miss_distance": st.column_config.NumberColumn("miss", format="%,.0f m"),
                "collision_probability": st.column_config.NumberColumn("predicted Pc", format="%.2e"),
                "p_critical": st.column_config.ProgressColumn(
                    "P(critical)", min_value=0, max_value=1, format="%.3f"
                ),
            },
        )
        selected_rows = selected.selection.rows if selected and selected.selection else []
        if selected_rows:
            selected_sat = int(page_view.iloc[selected_rows[0]]["mission_id"])
            if st.button("Inspect selected satellite", type="primary", width="stretch"):
                st.session_state.selected_satellite = selected_sat
                st.switch_page("pages/2_Satellites.py")

        previous_col, page_col, next_col = st.columns([1, 1.1, 1])
        with previous_col:
            if st.button(
                "Previous", icon=":material/chevron_left:", width="stretch",
                disabled=current_page == 0, key="overview_previous_page",
            ):
                st.session_state.overview_page = current_page - 1
                st.rerun()
        page_col.markdown(
            f'<div class="table-page-status" aria-live="polite">Page '
            f'{current_page + 1} of {total_pages}</div>',
            unsafe_allow_html=True,
        )
        with next_col:
            if st.button(
                "Next", icon=":material/chevron_right:", width="stretch",
                disabled=current_page >= total_pages - 1, key="overview_next_page",
            ):
                st.session_state.overview_page = current_page + 1
                st.rerun()

with brief_col:
    with st.container(border=True):
        st.markdown(
            '<p class="card-title">Operational brief</p>'
            '<p class="card-sub">What the current screening picture means</p>',
            unsafe_allow_html=True,
        )
        closest_critical = ev[ev["predicted_tier"] == "Critical"].sort_values(
            "miss_distance"
        ).iloc[0]
        st.markdown(
            f'<div class="ops-brief-row"><span>Immediate window</span><strong>'
            f'{len(within_24h):,} events inside 24 h</strong></div>'
            f'<div class="ops-brief-row"><span>Closest critical pass</span><strong>'
            f'{closest_critical["miss_distance"]:,.0f} m · '
            f'{sat_label(closest_critical["mission_id"])}</strong></div>'
            f'<div class="ops-brief-row"><span>Fleet coverage</span><strong>'
            f'{ev["mission_id"].nunique()} satellites · {len(ev):,} events</strong></div>'
            f'<div class="ops-brief-row"><span>Model state</span><strong>'
            f'Loaded and calibrated</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glass-note" style="margin-top:24px">'
            '<b style="color:#0F172A">Recommended next action</b><br>'
            'Open Fleet Analysis to inspect the nearest critical conjunction in its '
            '3D orbital context, then review its CDM history before making a manoeuvre decision.'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown("#### Continue operations")
q1, q2, q3 = st.columns(3)
with q1:
    with st.container(border=True):
        st.markdown('<p class="card-title">Fleet analysis</p><p class="card-sub">'
                    'Live 3D tracking and satellite inspector</p>', unsafe_allow_html=True)
        st.page_link("pages/2_Satellites.py", label="Open fleet analysis")
with q2:
    with st.container(border=True):
        st.markdown('<p class="card-title">Alert queue</p><p class="card-sub">'
                    'Review escalations and coordination messages</p>', unsafe_allow_html=True)
        st.page_link("pages/5_Alerts.py", label="Open active alerts")
with q3:
    with st.container(border=True):
        st.markdown('<p class="card-title">Live watch</p><p class="card-sub">'
                    'Replay incoming CDMs and escalation changes</p>', unsafe_allow_html=True)
        st.page_link("pages/1_Live_Watch.py", label="Open live watch")

st.markdown(
    """<style>
    .ops-brief-row {display:flex;justify-content:space-between;gap:12px;padding:12px 0;
        border-bottom:1px solid #CBD5E1;font-size:12px}
    .ops-brief-row span {color:#475569}.ops-brief-row strong {color:#0F172A;text-align:right}
    .table-page-status {height:40px;display:flex;align-items:center;justify-content:center;
        color:#475569;font-size:12px;font-weight:700}
    </style>""",
    unsafe_allow_html=True,
)
