"""Alert panel: Critical events and tier escalations, deduplicated by event,
sorted by urgency, with a downloadable report. Selecting an event shows its
Pc trend and — for payload-vs-payload conjunctions — an operator-coordination
message (generated text only; this demo has no contact directory)."""

from datetime import datetime, timedelta, timezone

import streamlit as st

from core import (build_alerts, inject_css, page_header, query, risk_history_fig,
                  sidebar_brand, style_tiers, tier_chip)


def coordination_message(a, last_cdm) -> str:
    """Operator-to-operator coordination request for a payload-vs-payload event."""
    now = datetime.now(timezone.utc)
    tca = now + timedelta(days=float(a["time_to_tca_d"]))
    hours = float(a["time_to_tca_d"]) * 24
    pc = 10.0 ** float(a["predicted_log10_pc"])
    rel_speed = last_cdm.get("relative_speed")
    rel_line = (f"  Relative speed (latest CDM):      {rel_speed:,.0f} m/s\n"
                if rel_speed == rel_speed else "")  # NaN-safe
    return (
        f"Subject: Conjunction coordination request — event {a['event_id']}, "
        f"TCA ≈ {tca:%Y-%m-%d %H:%M} UTC\n"
        f"\n"
        f"To the operations team of the secondary payload in event {a['event_id']}:\n"
        f"\n"
        f"Our conjunction screening has flagged a close approach between our\n"
        f"satellite {a['satellite']} and your payload.\n"
        f"\n"
        f"  Event ID:                        {a['event_id']}\n"
        f"  Time of closest approach (est.): {tca:%Y-%m-%d %H:%M} UTC (~{hours:.0f} h from now)\n"
        f"  Miss distance (latest CDM):      {a['miss_distance_m']:,.0f} m\n"
        f"{rel_line}"
        f"  Predicted risk tier:             {a['tier']} ({a['reason']})\n"
        f"  Predicted collision probability: Pc ≈ {pc:.1e} (log10 Pc {a['predicted_log10_pc']:.2f})\n"
        f"  Calibrated P(critical):          {a['p_critical']:.3f} ({a['pc_trend']})\n"
        f"  CDMs received so far:            {a['n_cdms']}\n"
        f"\n"
        f"Requested actions:\n"
        f"  1. Confirm you are tracking this event, and share your latest\n"
        f"     ephemeris/covariance if it differs from the CDM solution.\n"
        f"  2. Share any manoeuvres planned between now and TCA.\n"
        f"  3. Agree which operator moves if the event is still "
        f"{a['tier']} at TCA−24 h\n"
        f"     — we are prepared to manoeuvre if you cannot.\n"
        f"\n"
        f"— Aegis LEO operations (demo message; replace with your ops contact)\n"
    )

st.set_page_config(page_title="Aegis LEO — Alerts", layout="wide")
inject_css()
sidebar_brand()

alerts = build_alerts()
n = 0 if alerts.empty else len(alerts)
page_header(
    f"Alerts ({n})",
    "Critical events and tier escalations, deduplicated and ordered by operational urgency.",
    "alert",
)

if alerts.empty:
    st.success("No active alerts.")
    st.stop()

crit = (alerts["tier"] == "Critical").sum()
esc = alerts["alert_type"].str.startswith("escalation").sum()
k = st.columns(4)
k[0].metric("Active alerts", n)
k[1].metric("Critical", int(crit))
k[2].metric("Escalations", int(esc))
k[3].metric("Within 24 h", int((alerts["time_to_tca_d"] <= 1.0).sum()))

view = alerts.copy()
view["tier"] = view["tier"].map(tier_chip)
sel = st.dataframe(
    style_tiers(view, col="tier").format({
        "time_to_tca_d": "{:.2f}", "miss_distance_m": "{:,.0f}",
        "predicted_log10_pc": "{:.2f}", "p_critical": "{:.3f}",
    }),
    width="stretch", height=480, hide_index=True,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "event_id": "event", "chaser": "chaser", "time_to_tca_d": "TCA (d)",
        "miss_distance_m": "miss (m)", "predicted_log10_pc": "log₁₀ Pc",
        "pc_trend": "Pc trend", "n_cdms": "CDMs",
    },
)

st.download_button(
    "Download alert report (CSV)",
    alerts.to_csv(index=False).encode(), "alert_report.csv", "text/csv",
    type="primary",
)

rows = sel.selection.rows if sel and sel.selection else []
if rows:
    a = alerts.iloc[rows[0]]
    eid = int(a["event_id"])
    st.divider()
    st.subheader(f"Event {eid} — Pc trend across messages")
    hist = query("SELECT * FROM cdms WHERE event_id = ?", (eid,))
    st.plotly_chart(risk_history_fig(hist), width="stretch")
    st.markdown(f"**Reason:** {a['reason']}")

    # ---- operator coordination (payload-vs-payload only) ----
    st.subheader("Coordinate with the other operator")
    if a["chaser"] == "PAYLOAD":
        st.markdown(
            "Both objects in this event are **active payloads**, so operator-to-operator "
            "coordination is possible — agree who manoeuvres rather than both moving blind. "
            "This demo has **no operator contact directory**, so the message below is "
            "generated for you to copy or download; nothing is actually sent."
        )
        last_cdm = hist.sort_values("time_to_tca").iloc[0].to_dict()
        msg = coordination_message(a, last_cdm)
        st.code(msg, language=None)
        c1, c2 = st.columns([1.2, 3])
        c1.download_button("Download message (.txt)", msg.encode(),
                           f"event_{eid}_coordination.txt", "text/plain")
        if c2.button("Send to operator (simulated)", key=f"send_{eid}"):
            st.success(
                "Simulated send only — no delivery happened. In production this would "
                "route via the operator's registered point of contact "
                "(e.g. a Space-Track / 18 SDS directory or a webhook per "
                "`config.ALERT_WEBHOOK_URL`)."
            )
    else:
        st.markdown(
            f"**Notify operator unavailable** — the chaser in this event is "
            f"**{a['chaser']}**. Debris, rocket bodies and unknown objects have no "
            f"operations team to coordinate with, so mitigation rests entirely on our "
            f"own manoeuvre planning. (Coordination is offered only for "
            f"payload-vs-payload events.)"
        )
