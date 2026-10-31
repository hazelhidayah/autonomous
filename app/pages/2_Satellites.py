"""Fleet-centric satellite explorer with an on-demand inspector panel."""

from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import (
    INK,
    TIER_META,
    base_layout,
    geometry_fig,
    inject_css,
    latest_per_event,
    page_header,
    query,
    risk_history_fig,
    sat_label,
    scatter_fig,
    sidebar_brand,
    tier_pill,
)

st.set_page_config(page_title="Aegis LEO — Fleet", layout="wide")
inject_css()
sidebar_brand()
st.markdown(
    """<style>
    .block-container { max-width:1680px !important; padding-top:2rem !important; }
    .fleet-topbar [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:0 !important; border-color:#CBD5E1 !important;
        background:#FFFFFF !important; box-shadow:none !important;
    }
    .fleet-brand {font-size:17px;font-weight:800;color:#0F172A;white-space:nowrap}
    .fleet-brand span {color:#3B82F6}
    .fleet-path {font-size:18px;font-weight:700;color:#0F172A;white-space:nowrap}
    .fleet-path small {font-size:11px;color:#475569;font-weight:500;margin-left:8px}
    [data-testid="stHorizontalBlock"]:has(.fleet-path) {
        align-items:center !important; row-gap:16px !important;
        container-type:inline-size;
    }
    [data-testid="stHorizontalBlock"]:has(.fleet-path) > [data-testid="stColumn"] {
        min-width:0 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.fleet-path)
    [data-testid="stSelectbox"] [role="combobox"] {
        min-width:0 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.fleet-path)
    [data-testid="stButtonGroup"] [role="radiogroup"] {
        flex-wrap:wrap !important; height:auto !important; overflow:visible !important;
    }
    [data-testid="stHorizontalBlock"]:has(.fleet-path)
    [data-testid="stButtonGroup"] {height:auto !important}
    @container (max-width:800px) {
        [data-testid="stHorizontalBlock"]:has(.fleet-path) > [data-testid="stColumn"] {
            flex:1 1 100% !important; max-width:100% !important;
        }
        [data-testid="stHorizontalBlock"]:has(.fleet-path)
        > [data-testid="stColumn"]:nth-child(3) {
            flex-basis:70% !important; max-width:70% !important;
        }
        [data-testid="stHorizontalBlock"]:has(.fleet-path)
        > [data-testid="stColumn"]:nth-child(4) {
            flex-basis:30% !important; max-width:30% !important;
        }
    }
    @container (max-width:500px) {
        .fleet-path {white-space:normal; line-height:1.25}
        .fleet-path small {display:block; margin:3px 0 0}
        [data-testid="stHorizontalBlock"]:has(.fleet-path)
        > [data-testid="stColumn"]:nth-child(3),
        [data-testid="stHorizontalBlock"]:has(.fleet-path)
        > [data-testid="stColumn"]:nth-child(4) {
            flex-basis:100% !important; max-width:100% !important;
        }
    }
    .fleet-sectionbar {display:flex;align-items:center;justify-content:space-between;
        padding:9px 2px 5px;border-bottom:1px solid #CBD5E1;margin-bottom:10px}
    .fleet-sectionbar strong {font-size:13px;color:#0F172A}
    .fleet-sectionbar span {font-size:11px;color:#475569}
    div[data-testid="stMetric"] {min-height:108px;padding:16px 18px}
    div[data-testid="stMetricValue"] {font-size:2rem}
    </style>""",
    unsafe_allow_html=True,
)


def fleet_figure(summary, phase=0.0):
    """Interactive 3D fleet overview; positions are operational, not ephemerides."""
    d = summary.sort_index().copy()
    n = len(d)
    groups = np.arange(n) % 3
    angular_speed = np.array([1.0, .82, .66])[groups]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.18 + phase * angular_speed
    radii = np.array([1.42, 1.68, 1.94])[groups]
    inclinations = np.deg2rad(np.array([18, 52, 76]))
    nodes = np.deg2rad(np.array([-30, 24, 68]))

    def orbit_xyz(radius, theta, inclination, node):
        """Circular orbit rotated by inclination and ascending-node angle."""
        x0, y0 = radius * np.cos(theta), radius * np.sin(theta)
        x1, y1, z1 = x0, y0 * np.cos(inclination), y0 * np.sin(inclination)
        return (x1 * np.cos(node) - y1 * np.sin(node),
                x1 * np.sin(node) + y1 * np.cos(node), z1)

    coords = [orbit_xyz(radii[i], angles[i], inclinations[groups[i]], nodes[groups[i]])
              for i in range(n)]
    d["x"], d["y"], d["z"] = zip(*coords)
    d["state"] = np.select(
        [d["critical"] > 0, d["elevated"] > 0], ["Critical", "Elevated"], default="Safe"
    )
    fig = go.Figure()
    # Earth with a restrained blue atmospheric gradient.
    u = np.linspace(0, 2 * np.pi, 64)
    v = np.linspace(0, np.pi, 40)
    earth_x = .62 * np.outer(np.cos(u), np.sin(v))
    earth_y = .62 * np.outer(np.sin(u), np.sin(v))
    earth_z = .62 * np.outer(np.ones_like(u), np.cos(v))
    light = (earth_z + .62) / 1.24
    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z, surfacecolor=light,
        colorscale=[[0, "#DBEAFE"], [.45, "#93C5FD"], [1, "#3B82F6"]],
        showscale=False, hoverinfo="skip", name="Earth",
        lighting=dict(ambient=.52, diffuse=.72, roughness=.8, specular=.25),
        lightposition=dict(x=2.5, y=-3, z=4),
    ))
    theta = np.linspace(0, 2 * np.pi, 260)
    for i, (radius, inclination, node) in enumerate(zip(
            (1.42, 1.68, 1.94), inclinations, nodes)):
        x, y, z = orbit_xyz(radius, theta, inclination, node)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=f"rgba(59,130,246,{.50 - i * .08})", width=2),
        ))
    for tier in ("Safe", "Elevated", "Critical"):
        g = d[d["state"] == tier]
        if g.empty:
            continue
        meta = TIER_META[tier]
        fig.add_trace(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"], mode="markers+text", name=tier,
            text=[sat_label(i) for i in g.index], textposition="top center",
            textfont=dict(color="#475569", size=9),
            customdata=np.column_stack([g.index, g["events"], g["critical"], g["closest"]]),
            marker=dict(size=np.clip(4 + np.sqrt(g["events"]) * .55, 5, 10),
                        color=meta["color"], symbol=meta["sym3d"], opacity=.98,
                        line=dict(color="#FFFFFF", width=.6)),
            hovertemplate=("<b>%{text}</b><br>%{customdata[1]} tracked conjunctions"
                           "<br>%{customdata[2]} critical<br>closest pass %{customdata[3]:,.0f} m"
                           "<extra>select satellite</extra>"),
        ))
    fig.update_layout(**base_layout(height=560))
    fig.update_layout(
        margin=dict(l=0, r=0, t=18, b=0), hovermode="closest",
        legend=dict(orientation="h", x=.02, y=1.04, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        scene=dict(
            xaxis=dict(visible=False, range=[-2.1, 2.1]),
            yaxis=dict(visible=False, range=[-2.1, 2.1]),
            zaxis=dict(visible=False, range=[-2.1, 2.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.45, z=.82)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


ev_all = latest_per_event()
counts = ev_all.groupby("mission_id").agg(
    events=("event_id", "count"),
    critical=("predicted_tier", lambda s: int((s == "Critical").sum())),
    elevated=("predicted_tier", lambda s: int((s == "Elevated").sum())),
    closest=("miss_distance", "min"),
    next_tca=("time_to_tca", "min"),
).sort_values(["critical", "events"], ascending=False)

if "selected_satellite" not in st.session_state:
    st.session_state.selected_satellite = None

sat = st.session_state.selected_satellite
st.markdown('<div class="fleet-topbar">', unsafe_allow_html=True)
with st.container(border=True):
    title_col, view_col, focus_col, live_col = st.columns(
        [2.3, 4.4, 2.5, 1.1], vertical_alignment="center")
    title_col.markdown('<div class="fleet-path">Fleet analysis <small>live operations</small></div>',
                       unsafe_allow_html=True)
    with view_col:
        dashboard_view = st.segmented_control(
            "Dashboard view", ["3D orbit", "Risk landscape", "Conjunction matrix"],
            default="3D orbit", label_visibility="collapsed", width="stretch",
        )
    with focus_col:
        focused_satellite = st.selectbox(
            "Focus satellite", counts.index.tolist(), index=None,
            format_func=lambda m: f"{sat_label(m)} · {counts.loc[m, 'events']} conjunctions",
            placeholder="Select a satellite",
            label_visibility="collapsed", key="top_satellite_selector",
        )
    with live_col:
        live_updates = st.toggle(
            "Live updates", value=True, key="fleet_live_updates",
            help="Pause automatic CDM and orbital updates",
        )
st.markdown('</div>', unsafe_allow_html=True)
if focused_satellite is not None and int(focused_satellite) != sat:
    sat = int(focused_satellite)
    st.session_state.selected_satellite = sat

stream_total = int(query(
    "SELECT count(*) AS n FROM cdms WHERE stream_seq IS NOT NULL"
)["n"].iat[0])


@st.fragment(run_every="2s" if live_updates else None)
def telemetry_tape():
    """Replay stored CDMs as a continuously updating operational telemetry tape."""
    now = datetime.now(timezone.utc)
    frame = int(now.timestamp() // 2)
    offset = (frame * 7) % max(1, stream_total)
    packet = query(
        """SELECT * FROM cdms WHERE stream_seq IS NOT NULL
           ORDER BY stream_seq LIMIT 1 OFFSET ?""",
        (offset,),
    ).iloc[0]
    tca_seconds = max(0, int(float(packet["time_to_tca"]) * 86400))
    days, remainder = divmod(tca_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    status = str(packet["predicted_tier"])
    color = TIER_META[status]["color"]
    st.markdown(
        '<div class="telemetry-shell">'
        '<div class="telemetry-top"><span><span class="live-pulse"></span>'
        f'<b>CDM REPLAY {"LIVE" if live_updates else "PAUSED"}</b></span>'
        f'<span>packet {int(packet["stream_seq"]) + 1:,} / {stream_total:,} · '
        f'{now:%H:%M:%S} UTC</span></div>'
        '<div class="telemetry-grid">'
        f'<div><span>Asset</span><strong>{sat_label(packet["mission_id"])}</strong></div>'
        f'<div><span>Separation</span><strong>{float(packet["miss_distance"]):,.0f} m</strong></div>'
        f'<div><span>TCA countdown</span><strong>{days:02d}d {hours:02d}:{minutes:02d}:{seconds:02d}</strong></div>'
        f'<div><span>Relative speed</span><strong>{float(packet["relative_speed"]):,.0f} m/s</strong></div>'
        f'<div><span>Risk state</span><strong style="color:{color}">{status}</strong></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


telemetry_tape()

total_critical = int(counts["critical"].sum())
total_elevated = int(counts["elevated"].sum())
within_day = int(ev_all["time_to_tca"].between(0, 1).sum())
k1, k2, k3 = st.columns(3)
k1.metric("Critical conjunctions", f"{total_critical:,}", "requires operator review")
k2.metric("Elevated conjunctions", f"{total_elevated:,}", f"{within_day} within 24 hours")
k3.metric("Nearest approach", f"{ev_all['miss_distance'].min() / 1000:.2f} km",
          f"across {len(counts)} satellites")

main_col, panel_col = st.columns([2.25, 1] if sat is not None else [1, .001], gap="medium")
with main_col:
    st.markdown('<div class="fleet-sectionbar"><strong>Overview</strong>'
                '<span>Live visualization and fleet analytics</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        title, hint = st.columns([3, 2])
        view_copy = {
            "3D orbit": ("Live orbital picture", "Interactive 3D fleet overview · symbolic orbit placement"),
            "Risk landscape": ("Risk landscape", "Miss distance against predicted collision probability"),
            "Conjunction matrix": ("Conjunction matrix", "Fleet workload by satellite and severity"),
        }
        view_title, view_sub = view_copy[dashboard_view]
        title.markdown(f'<p class="card-title">{view_title}</p><p class="card-sub">{view_sub}</p>',
                       unsafe_allow_html=True)
        hint.markdown('<p class="card-footer">Live replay · updates automatically</p>',
                      unsafe_allow_html=True)

        @st.fragment(run_every="3s" if live_updates else None)
        def live_orbits():
            now = datetime.now(timezone.utc)
            phase = (now.timestamp() % 600) / 600 * 2 * np.pi
            if dashboard_view == "3D orbit":
                fig = fleet_figure(counts, phase)
            elif dashboard_view == "Risk landscape":
                fig = scatter_fig(ev_all)
                fig.update_layout(height=560, margin=dict(l=50, r=15, t=25, b=45))
            else:
                matrix = (ev_all.groupby(["mission_id", "predicted_tier"]).size()
                          .unstack(fill_value=0).reindex(columns=["Critical", "Elevated", "Safe"]))
                fig = go.Figure(go.Heatmap(
                    z=matrix.to_numpy(), x=matrix.columns,
                    y=[sat_label(i) for i in matrix.index],
                    colorscale=[[0, "#F4F6F9"], [.35, "#BFDBFE"], [1, "#3B82F6"]],
                    text=matrix.to_numpy(), texttemplate="%{text}",
                    hovertemplate="%{y}<br>%{x}: %{z} events<extra></extra>",
                    colorbar=dict(title="events", thickness=10),
                ))
                fig.update_layout(**base_layout(height=560))
                fig.update_layout(margin=dict(l=65, r=20, t=25, b=40))
            st.plotly_chart(
                fig, width="stretch",
                config={"displayModeBar": False}, key="fleet_orbits_live",
            )
            st.caption(f"{view_title} updated {now:%H:%M:%S} UTC")

        live_orbits()

    info_left, info_right = st.columns(2)
    with info_left:
        with st.container(border=True):
            st.markdown('<p class="card-title">Fleet risk mix</p><p class="card-sub">Events by operational tier</p>',
                        unsafe_allow_html=True)
            tier_counts = ev_all["predicted_tier"].value_counts()
            fig = go.Figure(go.Bar(
                x=["Critical", "Elevated", "Safe"],
                y=[tier_counts.get(t, 0) for t in ("Critical", "Elevated", "Safe")],
                marker_color=[TIER_META[t]["color"] for t in ("Critical", "Elevated", "Safe")],
                text=[tier_counts.get(t, 0) for t in ("Critical", "Elevated", "Safe")],
                textposition="outside",
            ))
            fig.update_layout(**base_layout(height=250))
            fig.update_layout(margin=dict(l=35, r=10, t=25, b=30), showlegend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with info_right:
        with st.container(border=True):
            st.markdown('<p class="card-title">Approach windows</p><p class="card-sub">Conjunctions by days to TCA</p>',
                        unsafe_allow_html=True)
            fig = go.Figure(go.Histogram(
                x=ev_all["time_to_tca"], nbinsx=14, marker_color="#3B82F6",
                hovertemplate="%{x:.1f} days · %{y} events<extra></extra>",
            ))
            fig.update_layout(**base_layout(height=250))
            fig.update_layout(margin=dict(l=35, r=10, t=25, b=30), showlegend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Fleet information")
    fleet_table = counts.reset_index().copy()
    fleet_table["satellite"] = fleet_table["mission_id"].map(sat_label)
    st.dataframe(
        fleet_table[["satellite", "events", "critical", "elevated", "closest", "next_tca"]],
        hide_index=True, width="stretch",
        column_config={"satellite": "asset", "events": "conjunctions",
                       "critical": "critical", "elevated": "elevated",
                       "closest": st.column_config.NumberColumn("closest pass", format="%,.0f m"),
                       "next_tca": st.column_config.NumberColumn("next TCA", format="%.2f d")},
    )

if sat is not None:
    ev = ev_all[ev_all["mission_id"] == sat].copy()
    closest = ev.loc[ev["miss_distance"].idxmin()]
    hottest = ev.sort_values("predicted_risk", ascending=False).iloc[0]
    with panel_col:
        with st.container(border=True):
            close_col, _ = st.columns([1, 4])
            if close_col.button("×", help="Close inspector", key="close_sat"):
                st.session_state.selected_satellite = None
                st.session_state.top_satellite_selector = None
                st.rerun()
            state = "Critical" if (ev["predicted_tier"] == "Critical").any() else (
                "Elevated" if (ev["predicted_tier"] == "Elevated").any() else "Safe")
            st.markdown(
                f'<div class="aegis-eyebrow">Satellite inspector</div>'
                f'<div style="font-size:24px;font-weight:780;color:{INK};margin:4px 0">{sat_label(sat)}</div>'
                f'{tier_pill(state)}', unsafe_allow_html=True,
            )
            st.markdown("")
            a, b = st.columns(2)
            a.metric("Conjunctions", len(ev))
            b.metric("Critical", int((ev["predicted_tier"] == "Critical").sum()))
            st.markdown(
                f'<div class="kv"><span class="k">Closest pass</span><span class="v">'
                f'{closest["miss_distance"] / 1000:.2f} km</span></div>'
                f'<div class="kv"><span class="k">Time to TCA</span><span class="v">'
                f'{closest["time_to_tca"]:.2f} days</span></div>'
                f'<div class="kv"><span class="k">Highest log₁₀ Pc</span><span class="v">'
                f'{hottest["predicted_risk"]:.2f}</span></div>'
                f'<div class="kv"><span class="k">Primary chaser</span><span class="v">'
                f'{hottest["c_object_type"]}</span></div>', unsafe_allow_html=True,
            )
            st.markdown("##### Relative geometry")
            rel = np.sqrt(ev["relative_position_r"] ** 2 + ev["relative_position_t"] ** 2
                          + ev["relative_position_n"] ** 2)
            near = ev[rel <= 8000]
            geo = geometry_fig(near)
            geo.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(geo, width="stretch", config={"displayModeBar": False})
            st.markdown("##### Select conjunction")
            event_options = ev.sort_values("predicted_risk", ascending=False)["event_id"].astype(int).tolist()
            eid = st.selectbox(
                "Conjunction", event_options, label_visibility="collapsed",
                format_func=lambda x: f"Event {x} · {ev.loc[ev.event_id == x, 'predicted_tier'].iat[0]}",
            )
            hist = query("SELECT * FROM cdms WHERE event_id = ?", (eid,))
            detail = hist.sort_values("time_to_tca").iloc[0]
            st.markdown(
                f'<div class="glass-note"><b style="color:#0F172A">Event {eid}</b><br>'
                f'{detail["c_object_type"]} · {detail["miss_distance"]:,.0f} m miss · '
                f'{detail["relative_speed"]:,.0f} m/s relative</div>', unsafe_allow_html=True,
            )
            history = risk_history_fig(hist)
            history.update_layout(height=250, margin=dict(l=35, r=5, t=12, b=35), showlegend=False)
            st.plotly_chart(history, width="stretch", config={"displayModeBar": False})
