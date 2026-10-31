"""Shared backbone for the ops console: SQLite store, model scoring,
and chart helpers. Pages import everything from here."""

import sqlite3
import sys
import re
import os
from html import escape
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    BASE_FEATURES,
    TIERS,
    assign_tier,
    engineer,
    fit_imputation_values,
    load,
    tier_reason,
)
from config import ALERT_WEBHOOK_URL  # noqa: E402

DB_PATH = ROOT / "data" / "ops.db"
MODELS_PATH = ROOT / "outputs" / "models.joblib"

TIER_META = {
    "Critical": {"color": "#EF4444", "symbol": "triangle-up", "sym3d": "x", "glyph": "[C]"},
    "Elevated": {"color": "#F59E0B", "symbol": "diamond", "sym3d": "diamond", "glyph": "[E]"},
    "Safe": {"color": "#10B981", "symbol": "circle", "sym3d": "circle", "glyph": "[S]"},
}
TIER_ORDER = ["Critical", "Elevated", "Safe"]
# 60-30-10 palette: white canvas, cool-grey structural surfaces, blue accents.
# Risk tiers retain semantic red/amber/emerald independent of the brand palette.
SERIES_1 = "#3B82F6"
SERIES_2 = "#1D4ED8"
ACCENT = "#3B82F6"
MUTED = "#475569"
GRID = "#CBD5E1"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F4F6F9"
INK = "#0F172A"
INK_2 = "#475569"
BORDER = "#CBD5E1"

CDM_COLS = [
    "event_id", "mission_id", "time_to_tca", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "t_sigma_r", "c_sigma_r", "c_object_type", "F10", "AP",
    "max_risk_estimate", "risk",
]
PRED_COLS = ["predicted_risk", "predicted_tier", "p_safe", "p_elevated", "p_critical"]

# columns a CSV upload must provide (risk is optional — unknown for new data)
REQUIRED_UPLOAD = [c for c in CDM_COLS if c not in ("risk", "F10", "AP", "max_risk_estimate")]


# ---------- models ----------

@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return joblib.load(MODELS_PATH)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Score raw CDM rows: predicted risk, calibrated tier probabilities,
    and the max-severity-union tier used operationally."""
    models = get_models()
    d = df.copy()
    for c in BASE_FEATURES:
        if c not in d.columns:
            d[c] = np.nan
    d["c_object_type"] = (
        d.get("c_object_type", pd.Series(["UNKNOWN"] * len(d)))
        .fillna("UNKNOWN").astype(str)
    )
    X = engineer(d, models.get("imputation_values") or fit_imputation_values(d))
    X["c_object_type"] = pd.Categorical(
        X["c_object_type"], categories=models["category_levels"]
    )
    pred_risk = models["regressor"].predict(X)
    proba = models["classifier"].predict_proba(X)
    cal = np.column_stack(
        [models["isotonics"][k].predict(proba[:, k]) for k in range(3)]
    )
    cal /= cal.sum(axis=1, keepdims=True)
    rules = pd.Series(assign_tier(pred_risk, d["miss_distance"])).map(
        {t: i for i, t in enumerate(TIERS)}
    ).to_numpy()
    union = np.maximum(rules, cal.argmax(axis=1))
    out = df.copy()
    out["predicted_risk"] = pred_risk
    out["predicted_tier"] = [TIERS[i] for i in union]
    out["p_safe"], out["p_elevated"], out["p_critical"] = cal[:, 0], cal[:, 1], cal[:, 2]
    return out


# ---------- database ----------

def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _build_db(conn):
    train = load("train")
    test = load("test")
    oof = pd.read_csv(ROOT / "outputs" / "oof_predictions.csv")
    tpr = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")

    tr = train[CDM_COLS].copy()
    for c in PRED_COLS:
        tr[c] = oof[c].to_numpy()
    tr["source"] = "archive"

    te = test[CDM_COLS].copy()
    for c in PRED_COLS:
        te[c] = tpr[c].to_numpy()
    te["event_id"] = te["event_id"] + 100_000
    te["source"] = "current"
    # arrival order for the live-feed replay: furthest from TCA arrives first
    te = te.sort_values("time_to_tca", ascending=False).reset_index(drop=True)
    te["stream_seq"] = np.arange(len(te))

    tr["stream_seq"] = pd.NA
    allrows = pd.concat([tr, te], ignore_index=True)
    allrows["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    allrows.to_sql("cdms", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_event ON cdms(event_id)")
    conn.execute("CREATE INDEX idx_source ON cdms(source)")
    conn.execute("CREATE INDEX idx_mission ON cdms(mission_id)")
    conn.commit()


@st.cache_resource(show_spinner="Opening data store (first run builds it)…")
def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = _connect()
    n = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='cdms'"
    ).fetchone()[0]
    if n == 0:
        _build_db(conn)
    return conn


def query(sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_conn(), params=params)


def insert_cdms(df: pd.DataFrame, source: str):
    rows = df.copy()
    for c in CDM_COLS + PRED_COLS:
        if c not in rows.columns:
            rows[c] = np.nan
    rows = rows[CDM_COLS + PRED_COLS]
    rows["source"] = source
    rows["stream_seq"] = pd.NA
    rows["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows.to_sql("cdms", get_conn(), if_exists="append", index=False)
    get_conn().commit()


def latest_per_event(sources=("current", "upload"), missions=None, tiers=None) -> pd.DataFrame:
    """Latest CDM per event (smallest time_to_tca) for the given sources."""
    src = ",".join(f"'{s}'" for s in sources)
    df = query(
        f"""SELECT * FROM cdms WHERE source IN ({src})
            AND rowid IN (
              SELECT rowid FROM (
                SELECT rowid, ROW_NUMBER() OVER (
                  PARTITION BY event_id, source ORDER BY time_to_tca ASC) rn
                FROM cdms WHERE source IN ({src}))
              WHERE rn = 1)"""
    )
    if missions:
        df = df[df["mission_id"].isin(missions)]
    if tiers:
        df = df[df["predicted_tier"].isin(tiers)]
    return df


# ---------- alerts ----------

TIER_RANK = {t: i for i, t in enumerate(TIERS)}  # Safe=0 … Critical=2


def build_alerts(sources=("current", "upload"), max_seq=None) -> pd.DataFrame:
    """Alert on Critical events and on tier escalations per event — never on
    every CDM (alert fatigue). Deduplicated by event_id; latest state wins.

    max_seq limits the view to CDMs already 'received' in the live replay.
    """
    src = ",".join(f"'{s}'" for s in sources)
    sql = f"SELECT * FROM cdms WHERE source IN ({src})"
    if max_seq is not None:
        sql += f" AND (stream_seq IS NULL OR stream_seq < {int(max_seq)})"
    df = query(sql)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for eid, g in df.groupby("event_id"):
        g = g.sort_values("time_to_tca", ascending=False)  # arrival order
        tiers = g["predicted_tier"].map(TIER_RANK).to_numpy()
        last = g.iloc[-1]
        escalated = len(tiers) > 1 and tiers[-1] > tiers[-2]
        if not (last["predicted_tier"] == "Critical" or (escalated and tiers[-1] >= 1)):
            continue
        trend = (
            last["predicted_risk"] - g.iloc[-2]["predicted_risk"]
            if len(g) > 1 else 0.0
        )
        rows.append({
            "event_id": int(eid),
            "satellite": sat_label(last["mission_id"]),
            "chaser": str(last["c_object_type"]) if pd.notna(last["c_object_type"]) else "UNKNOWN",
            "tier": last["predicted_tier"],
            "alert_type": (
                f"escalation {TIERS[tiers[-2]]} → {TIERS[tiers[-1]]}"
                if escalated else "critical (steady)"
            ),
            "time_to_tca_d": round(float(last["time_to_tca"]), 2),
            "miss_distance_m": round(float(last["miss_distance"]), 0),
            "predicted_log10_pc": round(float(last["predicted_risk"]), 2),
            "p_critical": round(float(last["p_critical"]), 3),
            "pc_trend": "rising" if trend > 0.1 else ("falling" if trend < -0.1 else "steady"),
            "n_cdms": len(g),
            "reason": tier_reason(float(last["predicted_risk"]), float(last["miss_distance"])),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["urgency"] = out["tier"].map(TIER_RANK) * 100 - out["time_to_tca_d"]
    out = out.sort_values(["urgency"], ascending=False).drop(columns=["urgency"])
    if ALERT_WEBHOOK_URL:  # off by default (config.ALERT_WEBHOOK_URL = None)
        pass  # would POST new alerts to the webhook here
    return out.reset_index(drop=True)


# ---------- display helpers ----------

def sat_label(mission_id) -> str:
    return f"SAT-{int(mission_id):02d}"


def tier_chip(tier: str) -> str:
    return f"{TIER_META[tier]['glyph']} {tier}"


# cell backgrounds that signal criticality: red / orange / green, grey = neutral
CRITICALITY_CELL = {
    "Critical": "background-color: #FEE2E2; color: #B91C1C",
    "Elevated": "background-color: #FEF3C7; color: #92400E",
    "Safe": "background-color: #D1FAE5; color: #047857",
}
NEUTRAL_CELL = "background-color: #F4F6F9; color: #475569"


def style_criticality(df: pd.DataFrame, tier_col: str):
    """Tint every cell of a row by the row's criticality tier (red / orange /
    green), with a stronger accent on the tier cell itself; grey when neutral
    or unknown."""
    def _row(row):
        tier = next((t for t in TIER_META if t in str(row[tier_col])), None)
        base = CRITICALITY_CELL.get(tier, NEUTRAL_CELL)
        accent = (
            f"background-color: {TIER_META[tier]['color']}33; "
            f"color: {TIER_META[tier]['color']}; font-weight: 600"
            if tier else NEUTRAL_CELL
        )
        return [accent if c == tier_col else base for c in row.index]
    return df.style.apply(_row, axis=1)


def style_tiers(df: pd.DataFrame, col="tier"):
    def _style(v):
        for t, m in TIER_META.items():
            if t in str(v):
                return f"color: {m['color']}; font-weight: 600"
        return ""
    return df.style.map(_style, subset=[col])


def base_layout(height=420):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12,
                  color=INK_2),
        margin=dict(l=55, r=15, t=10, b=45), height=height,
        legend=dict(orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                        font=dict(color=INK, size=12,
                                  family='system-ui, -apple-system, "Segoe UI", sans-serif')),
    )


def inject_css():
    """Shared console chrome: soft rounded cards with a hairline ring, ghost
    buttons, tier pills, stat tiles, and the plain-language insight callout."""
    st.markdown(
        f"""<style>
        :root {{ color-scheme: light; }}
        .stApp {{
            background: #FFFFFF;
        }}
        .block-container {{ max-width: 1680px; padding-top: 2rem; padding-bottom: 4rem; }}
        h1, h2, h3 {{ color: #0F172A !important; letter-spacing: -.025em; }}
        h2 {{ font-size: 1.35rem !important; margin-top: 1.5rem !important; }}
        h3 {{ font-size: 1.05rem !important; color: #0F172A !important; }}
        p, label, [data-testid="stCaptionContainer"] {{ color: #475569; }}
        header[data-testid="stHeader"] {{ background: rgba(255,255,255,.92); }}
        section[data-testid="stSidebar"] {{
            width: 280px !important; min-width:280px !important; max-width:280px !important;
            background: #F4F6F9;
            border-right: 1px solid #CBD5E1;
            transition:width .2s ease,min-width .2s ease,max-width .2s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
            display:none !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
            display:none !important;
        }}
        [data-testid="stSidebarCollapsedControl"] {{
            display:none !important;
        }}
        .aegis-brand-block {{ position:relative;width:100%;padding:0 42px 8px 0;
                              min-height:66px;box-sizing:border-box; }}
        .aegis-mini-logo {{ position:relative;width:38px;height:38px;margin:0 auto 10px;
                            display:grid;place-items:center;border-radius:12px;color:#1D4ED8;
                            background:#E0F2FE;border:1px solid #BFDBFE; }}
        .aegis-mini-logo svg {{width:21px;height:21px;stroke:currentColor;fill:none;
                               stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            border-radius: 10px; min-height: 44px; margin: 4px 0; transition: .18s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
            background: #E0F2FE;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
            color: #1D4ED8; background: #E0F2FE;
            border: 1px solid #93C5FD;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 16px;
            box-shadow: 0 10px 28px rgba(15,23,42,.07);
        }}
        div.stButton > button {{
            background: #FFFFFF; color: #0F172A;
            border: 1px solid #CBD5E1; border-radius: 10px;
            padding: 0 16px; min-height: 40px; height: 40px;
            font-size: 12px; font-weight: 600; white-space: nowrap;
        }}
        div.stButton > button:hover {{
            border-color: #1D4ED8; color: #1D4ED8; background: #E0F2FE;
            transform: translateY(-1px);
        }}
        button:focus-visible, a:focus-visible, input:focus-visible,
        [role="button"]:focus-visible, [role="option"]:focus-visible {{
            outline: 3px solid #3B82F6 !important; outline-offset: 2px !important;
        }}
        [data-testid="stPageLink"] a {{ min-height:44px; display:flex; align-items:center; }}
        div.stButton > button[kind="primary"] {{
            color: #fff; border-color: transparent;
            background: #1D4ED8;
            box-shadow: 0 8px 20px rgba(29,78,216,.22);
        }}
        [data-testid="stMetric"] {{
            background: #F4F6F9; border: 1px solid #CBD5E1;
            border-radius: 15px; padding: 13px 15px;
            box-shadow: 0 6px 18px rgba(15,23,42,.05);
        }}
        [data-testid="stMetricValue"] {{ color: #0F172A; letter-spacing: -.03em; }}
        [data-testid="stDataFrame"] {{
            border: 1px solid #CBD5E1; border-radius: 14px; overflow: hidden;
        }}
        div[data-baseweb="select"] > div {{
            background: #FFFFFF; border-color: #CBD5E1;
            border-radius: 12px;
        }}
        [data-baseweb="tab-list"] {{
            gap: 6px; padding: 4px; background: #F4F6F9;
            border: 1px solid #CBD5E1; border-radius: 12px;
        }}
        [data-baseweb="tab"] {{ border-radius: 9px; padding: 8px 16px; }}
        [aria-selected="true"][data-baseweb="tab"] {{
            background: #E0F2FE; color: #1D4ED8;
        }}
        [data-testid="stFileUploaderDropzone"], [data-testid="stForm"] {{
            background: #F4F6F9; border: 1px solid #CBD5E1;
            border-radius: 15px;
        }}
        [data-testid="stHeaderActionElements"] {{ display: none !important; }}
        [data-testid="stAlert"] {{ border-radius: 13px; border-width: 1px; }}
        [data-testid="stProgress"] > div > div {{ background: linear-gradient(90deg,{ACCENT},{SERIES_1}); }}
        div.stDownloadButton > button {{
            background: #1D4ED8; color: #FFFFFF; border: none; border-radius: 10px;
            min-height: 30px; height: 30px; font-size: 12px; font-weight: 700;
            white-space: nowrap;
        }}
        div.stDownloadButton > button:hover {{ background: #1D4ED8; color: #FFFFFF; }}
        .card-title {{ font-size: 13px; font-weight: 700; color: #0F172A;
                       letter-spacing: 0.2px; margin: 0; white-space: nowrap;
                       overflow: hidden; text-overflow: ellipsis; }}
        .card-sub {{ font-size: 12px; color: {MUTED}; margin: 2px 0 0 0;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .card-footer {{ font-size: 12px; color: {MUTED}; text-align: right; }}
        .statrow {{ display: flex; gap: 10px; margin: 8px 0 2px 0; }}
        .statrow .stat {{ flex: 1; min-width: 0; }}
        .stat-value {{ font-size: clamp(20px, 1.9vw, 34px); font-weight: 700;
                       color: {INK}; line-height: 1.15; white-space: nowrap; }}
        .stat-value .dot {{ font-size: 0.55em; vertical-align: 0.18em; margin-right: 3px; }}
        .stat-label {{ font-size: 11.5px; color: {MUTED}; line-height: 1.35;
                       overflow-wrap: normal; word-break: normal; hyphens: none; }}
        .pill {{ display: inline-flex; align-items: center; gap: 6px;
                 padding: 2px 10px; border-radius: 999px;
                 font-size: 12px; font-weight: 600; color: {INK}; white-space: nowrap; }}
        .kv {{ display: flex; justify-content: space-between; gap: 10px;
               font-size: 12px; padding: 3px 0;
               border-top: 1px solid #E2E8F0; }}
        .kv .k {{ color: {MUTED}; white-space: nowrap; }}
        .kv .v {{ color: {INK_2}; font-weight: 600; text-align: right; white-space: nowrap; }}
        .strip {{ background: {SURFACE_2}; border-radius: 10px; padding: 8px 14px;
                  display: flex; justify-content: space-between; align-items: center;
                  gap: 8px; min-width: 150px; }}
        .strip .name {{ color: {INK_2}; font-size: 12.5px; white-space: nowrap; }}
        .strip .val {{ font-size: clamp(18px, 1.5vw, 26px); font-weight: 700;
                       color: {INK}; white-space: nowrap; }}
        .insight {{ background: #E0F2FE;
                    border: 1px solid #93C5FD;
                    border-radius: 12px; padding: 12px 16px; }}
        .insight .ttl {{ color: {ACCENT}; font-size: 12px; font-weight: 700;
                         letter-spacing: 0.6px; margin-bottom: 4px; }}
        .insight ul {{ margin: 4px 0 0 0; padding-left: 18px; }}
        .insight li {{ color: #0F172A; font-size: 13px; line-height: 1.55;
                       margin-bottom: 3px; }}
        .event-id {{ color: {ACCENT}; font-weight: 600; }}
        .aegis-eyebrow {{ color:{ACCENT}; font-size:11px; font-weight:800;
                          letter-spacing:1.7px; text-transform:uppercase; }}
        .aegis-title {{ color:#0F172A; font-size:28px; font-weight:760;
                       letter-spacing:-.7px; margin:2px 0 0; }}
        .aegis-sub {{ color:{MUTED}; font-size:12px; margin-top:3px; }}
        .page-head {{ display:flex; align-items:center; gap:14px; margin:0 0 22px; }}
        .page-icon {{ width:46px; height:46px; display:grid; place-items:center;
                      border-radius:14px; color:#1D4ED8;background:#E0F2FE;
                      border:1px solid #BFDBFE;box-shadow:0 8px 20px rgba(59,130,246,.12); }}
        .page-icon svg {{ width:22px; height:22px; stroke:currentColor; fill:none;
                          stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
        .page-copy {{ min-width:0; }}
        .page-title {{ color:#0F172A; font-size:27px; line-height:1.1; font-weight:760;
                       letter-spacing:-.7px; margin:0; }}
        .page-description {{ color:{MUTED}; font-size:12.5px; margin-top:5px; }}
        .glass-note {{ background:#E0F2FE;border:1px solid #BFDBFE;border-radius:12px;
                       padding:10px 13px;color:#0F172A;font-size:12px; }}
        .fleet-sectionbar {{ display:flex; align-items:center; justify-content:space-between;
                             gap:16px; padding:10px 2px 7px;
                             border-bottom:1px solid #CBD5E1; margin-bottom:10px; }}
        .fleet-sectionbar strong {{ font-size:13px; color:#0F172A; }}
        .fleet-sectionbar span {{ font-size:11px; color:#475569; text-align:right; }}
        .telemetry-shell {{ margin:2px 0 14px; padding:11px 14px 13px;
                            background:#F4F6F9;border:1px solid #CBD5E1;border-radius:14px;
                            box-shadow:0 8px 22px rgba(15,23,42,.06); }}
        .telemetry-top {{ display:flex; justify-content:space-between; align-items:center;
                          color:{MUTED}; font-size:10px; letter-spacing:.45px; margin-bottom:9px; }}
        .telemetry-top b {{ color:#0F172A; font-size:10px; letter-spacing:1px; }}
        .live-pulse {{ display:inline-block; width:7px; height:7px; margin-right:7px;
                       border-radius:50%; background:#10B981; box-shadow:0 0 0 4px #D1FAE5; }}
        .telemetry-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }}
        .telemetry-grid > div {{ padding:7px 10px; border-left:1px solid #CBD5E1; }}
        .telemetry-grid > div:first-child {{ border-left:0; }}
        .telemetry-grid span {{ display:block; color:{MUTED}; font-size:10px; margin-bottom:2px; }}
        .telemetry-grid strong {{ display:block; color:#0F172A; font-size:14px;
                                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        @media (max-width: 850px) {{
            .telemetry-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .telemetry-grid > div {{ border-left:0; border-top:1px solid #CBD5E1; }}
            .telemetry-top {{ align-items:flex-start; gap:8px; }}
            .page-head {{ align-items:flex-start; }}
            .page-title {{ font-size:22px; }}
            .card-sub, .card-footer {{ white-space:normal; }}
            .statrow {{ flex-wrap:wrap; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ animation-duration:.01ms !important;
                animation-iteration-count:1 !important; transition-duration:.01ms !important;
                scroll-behavior:auto !important; }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


PAGE_ICONS = {
    "overview": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    "live": '<path d="M3 12h3l2-5 4 10 2-5h7"/><circle cx="12" cy="12" r="9"/>',
    "satellite": '<path d="M8.5 15.5l7-7M5 19l3-3M16 8l3-3"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M4 13l-2-2 4-4 2 2M16 16l2-2 4 4-4 4-2-2"/>',
    "archive": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "upload": '<path d="M12 16V3M7 8l5-5 5 5"/><path d="M5 14v6h14v-6"/>',
    "alert": '<path d="M18 8a6 6 0 00-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/>',
    "model": '<path d="M4 19V9M10 19V5M16 19v-7M22 19V3"/><path d="M2 19h22"/>',
}


def page_header(title: str, description: str, icon: str | None = "overview"):
    """Consistent page heading using lightweight inline SVG, never emoji glyphs."""
    if icon is None:
        st.markdown(
            f'<div class="page-head page-head-no-icon"><div class="page-copy">'
            f'<div class="page-title">{escape(title)}</div>'
            f'<div class="page-description">{escape(description)}</div></div></div>',
            unsafe_allow_html=True,
        )
        return
    paths = PAGE_ICONS.get(icon, PAGE_ICONS["overview"])
    st.markdown(
        f'<div class="page-head"><div class="page-icon">'
        f'<svg viewBox="0 0 24 24" aria-hidden="true">{paths}</svg></div>'
        f'<div class="page-copy"><div class="page-title">{escape(title)}</div>'
        f'<div class="page-description">{escape(description)}</div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar_brand():
    """Shared accessible navigation with a persistent compact icon-rail state."""
    compact = st.session_state.get("aegis_menu_compact", False)
    width = 76 if compact else 280
    with st.sidebar:
        st.markdown(
            f"""<style>
            section[data-testid="stSidebar"] {{width:{width}px !important;min-width:{width}px !important;
                max-width:{width}px !important;overflow:hidden !important}}
            section[data-testid="stSidebar"] > div:first-child {{width:{width}px !important}}
            section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{padding-top:18px!important}}
            section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
                margin-top:-76px!important;padding-top:0!important}}
            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:6px}}
            section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
                min-height:44px;border-radius:10px;padding:8px 12px;white-space:nowrap;
                border:1px solid transparent}}
            section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
                background:#E0F2FE;border-color:#BFDBFE}}
            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.aegis-toggle-anchor) {{
                height:0!important;margin:0!important;padding:0!important}}
            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.aegis-toggle-anchor) + div {{
                position:absolute!important;top:18px;right:20px;width:40px!important;z-index:20}}
            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.aegis-toggle-anchor) + div button {{
                width:40px!important;min-width:40px!important;padding:0!important}}
            section[data-testid="stSidebar"] [data-testid="stTextInput"] {{
                margin:6px 0 16px!important}}
            section[data-testid="stSidebar"] [data-testid="InputInstructions"] {{
                display:none!important}}
            {'''section[data-testid="stSidebar"] [data-testid="stPageLink"] p {display:none!important}
                section[data-testid="stSidebar"] [data-testid="stPageLink"] a {justify-content:center!important;padding:8px!important}
                section[data-testid="stSidebar"] .stButton button {padding:0!important;width:44px!important}
                section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-left:16px!important;padding-right:16px!important}'''
                if compact else ''}
            @media (prefers-reduced-motion:reduce) {{section[data-testid="stSidebar"] {{transition:none!important}}}}
            </style>""",
            unsafe_allow_html=True,
        )
        if compact:
            st.markdown(
                '<div class="aegis-mini-logo" title="Aegis LEO">'
                '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l8 5v10l-8 5-8-5V7z"/>'
                '<path d="M8 12h8M12 8v8"/></svg></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="aegis-brand-block">'
                '<div class="aegis-eyebrow">Orbital intelligence</div>'
                '<div style="font-size:22px;font-weight:800;color:#0F172A;margin-top:4px">'
                f'Aegis <span style="color:{ACCENT}">LEO</span></div>'
                '<div class="aegis-sub">Conjunction operations console</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<span class="aegis-toggle-anchor" aria-hidden="true"></span>', unsafe_allow_html=True)
        toggle_label = ":material/left_panel_open:" if compact else ":material/left_panel_close:"
        if st.button(toggle_label, key="aegis_menu_toggle", width="stretch",
                     help="Expand navigation" if compact else "Minimize navigation"):
            st.session_state.aegis_menu_compact = not compact
            st.rerun()

        if not compact:
            def _queue_navigation_search():
                st.session_state.global_navigation_search_pending = True

            search_term = st.text_input(
                "Search",
                placeholder="Page or satellite, e.g. SAT-02",
                help="Search for a page name, SAT-02, satellite 2, or simply 2.",
                label_visibility="collapsed",
                key="global_navigation_search_term",
                on_change=_queue_navigation_search,
            )
            submitted = st.session_state.pop("global_navigation_search_pending", False)
            if submitted:
                destination = resolve_navigation_search(search_term)
                if destination is None:
                    st.error("No matching page or satellite. Try SAT-02, Alerts, Archive, or Live Watch.")
                elif destination[0] == "satellite":
                    mission_id = destination[1]
                    available = set(query("SELECT DISTINCT mission_id FROM cdms")["mission_id"].astype(int))
                    if mission_id not in available:
                        st.error(f"SAT-{mission_id:02d} is not present in the current fleet data.")
                    else:
                        st.session_state.selected_satellite = mission_id
                        st.session_state.top_satellite_selector = mission_id
                        st.switch_page(_entry_relative_page("pages/2_Satellites.py"))
                else:
                    st.switch_page(_entry_relative_page(destination[1]))

        alert_count = len(build_alerts())
        alert_badge = "99+" if alert_count > 99 else str(alert_count)
        nav_items = [
            ("Overview.py", "/", "Overview", ":material/dashboard:", "Operations overview"),
            ("pages/1_Live_Watch.py", "/Live_Watch", "Live Watch", ":material/monitoring:", "Live Watch"),
            ("pages/2_Satellites.py", "/Satellites", "Fleet Analysis", ":material/satellite_alt:", "Fleet Analysis"),
            ("pages/3_Archive.py", "/Archive", "Archive", ":material/database:", "Historical Archive"),
            ("pages/4_Add_Data.py", "/Add_Data", "Add Data", ":material/upload_file:", "Add Conjunction Data"),
            ("pages/5_Alerts.py", "/Alerts", f"Alerts ({alert_count})", ":material/notifications:",
             f"Alerts: {alert_count} active"),
            ("pages/6_Model_Performance.py", "/Model_Performance", "Model Performance", ":material/analytics:",
             "Model Performance"),
        ]
        primary_entrypoint = _using_primary_entrypoint()
        for page, fallback_url, label, icon, help_text in nav_items:
            if primary_entrypoint:
                st.page_link(page, label=label, icon=icon, help=help_text, width="stretch")
            else:
                st.link_button(label, fallback_url, icon=icon, help=help_text, width="stretch")
            if compact and page == "pages/5_Alerts.py":
                st.markdown(
                    f'<div title="{alert_count} active alerts" style="height:0;position:relative;'
                    f'top:-18px;left:27px;color:#fff;background:#B91C1C;border:2px solid #F4F6F9;'
                    f'width:20px;height:20px;border-radius:999px;display:grid;place-items:center;'
                    f'font-size:8px;font-weight:800;z-index:5">{alert_badge}</div>',
                    unsafe_allow_html=True,
                )


def _entry_relative_page(page: str) -> str:
    """Return a page path relative to the active Streamlit entrypoint."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx(suppress_warning=True)
    entry_dir = Path(ctx.main_script_path).resolve().parent if ctx else ROOT / "app"
    target = (ROOT / "app" / page).resolve()
    return os.path.relpath(target, entry_dir)


def _using_primary_entrypoint() -> bool:
    """True for the supported `streamlit run app/Overview.py` launch path."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx(suppress_warning=True)
    return bool(ctx and Path(ctx.main_script_path).resolve() == (ROOT / "app" / "Overview.py").resolve())


def resolve_navigation_search(search_term: str):
    """Resolve a global navigation query to a page or satellite without side effects."""
    term = " ".join(str(search_term).strip().lower().replace("_", " ").split())
    if not term:
        return None
    pages = {
        "overview": "Overview.py",
        "home": "Overview.py",
        "live": "pages/1_Live_Watch.py",
        "live watch": "pages/1_Live_Watch.py",
        "satellites": "pages/2_Satellites.py",
        "fleet": "pages/2_Satellites.py",
        "fleet analysis": "pages/2_Satellites.py",
        "archive": "pages/3_Archive.py",
        "history": "pages/3_Archive.py",
        "add data": "pages/4_Add_Data.py",
        "upload": "pages/4_Add_Data.py",
        "alerts": "pages/5_Alerts.py",
        "alert": "pages/5_Alerts.py",
        "model": "pages/6_Model_Performance.py",
        "model performance": "pages/6_Model_Performance.py",
    }
    if term in pages:
        return "page", pages[term]
    satellite_match = re.fullmatch(
        r"(?:(?:sat|satellite)(?:\s+number)?[\s#:\-]*)?(\d{1,3})", term
    )
    if satellite_match:
        return "satellite", int(satellite_match.group(1))
    return None


def tier_pill(tier: str) -> str:
    """Tier as an exon-style pill: tinted background + glyph + label."""
    m = TIER_META[tier]
    c = m["color"]
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return (f'<span class="pill" style="background: rgba({r},{g},{b},0.16); '
            f'border: 1px solid rgba({r},{g},{b},0.45)">'
            f'<span style="color:{c}">{m["glyph"]}</span>{tier}</span>')


def scatter_fig(ev: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for tier in TIER_ORDER:
        grp = ev[ev["predicted_tier"] == tier]
        if grp.empty:
            continue
        m = TIER_META[tier]
        fig.add_trace(go.Scattergl(
            x=grp["miss_distance"], y=np.power(10.0, grp["predicted_risk"]),
            mode="markers", name=f"{m['glyph']} {tier}",
            customdata=np.stack([grp["event_id"], grp["time_to_tca"]], axis=1),
            hovertemplate="event %{customdata[0]:.0f}<br>miss %{x:.0f} m · Pc %{y:.1e}"
                          "<br>TCA in %{customdata[1]:.2f} d<extra>" + tier + "</extra>",
            marker=dict(color=m["color"], symbol=m["symbol"], size=6, opacity=0.55,
                        line=dict(width=1, color=SURFACE)),
        ))
    for x in (1000, 5000):
        fig.add_vline(x=x, line=dict(color=MUTED, width=1, dash="dash"))
    for y in (1e-4, 1e-6):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(**base_layout())
    fig.update_xaxes(type="log", title_text="miss distance (m)", gridcolor=GRID)
    fig.update_yaxes(type="log", title_text="predicted collision probability",
                     gridcolor=GRID, exponentformat="power", range=[-12, 0])
    return fig


def _sphere_traces(radius):
    out, N = [], 24
    for lat in (-60, -30, 0, 30, 60):
        lon = np.linspace(0, 2 * np.pi, N + 1)
        la = np.deg2rad(lat)
        out.append((radius * np.cos(la) * np.cos(lon),
                    radius * np.cos(la) * np.sin(lon),
                    np.full(N + 1, radius * np.sin(la))))
    for mer in range(4):
        la = np.linspace(-np.pi / 2, np.pi / 2, N + 1)
        lon = mer / 4 * np.pi
        out.append((radius * np.cos(la) * np.cos(lon),
                    radius * np.cos(la) * np.sin(lon),
                    radius * np.sin(la)))
    traces = []
    for i, (x, y, z) in enumerate(out):
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(color=GRID, width=1.5),
            hoverinfo="skip", showlegend=(i == 0), name=f"{radius/1000:g} km zone",
            opacity=0.6,
        ))
    return traces


def geometry_fig(ev: pd.DataFrame, frame_km=8) -> go.Figure:
    """3D chaser positions relative to the satellite, with keep-out spheres."""
    fig = go.Figure()
    for tr in _sphere_traces(1000) + _sphere_traces(5000):
        fig.add_trace(tr)
    for tier in TIER_ORDER:
        grp = ev[ev["predicted_tier"] == tier]
        if grp.empty:
            continue
        m = TIER_META[tier]
        fig.add_trace(go.Scatter3d(
            x=grp["relative_position_r"], y=grp["relative_position_t"],
            z=grp["relative_position_n"], mode="markers",
            name=f"{m['glyph']} {tier}",
            customdata=np.stack([grp["event_id"], grp["predicted_risk"]], axis=1),
            hovertemplate="event %{customdata[0]:.0f}<br>R %{x:.0f} · T %{y:.0f} · N %{z:.0f} m"
                          "<br>log₁₀ Pc %{customdata[1]:.2f}<extra>" + tier + "</extra>",
            marker=dict(color=m["color"], symbol=m["sym3d"], size=3.5, opacity=0.9),
        ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", name="satellite",
        marker=dict(color=INK, size=7, symbol="diamond"),
        hovertemplate="satellite<extra></extra>",
    ))
    r = frame_km * 1000
    ax = dict(gridcolor=GRID, range=[-r, r])
    fig.update_layout(
        **base_layout(height=480),
        scene=dict(
            xaxis=dict(title="radial (m)", **ax),
            yaxis=dict(title="along-track (m)", **ax),
            zaxis=dict(title="cross-track (m)", **ax),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
    )
    return fig


def risk_history_fig(hist: pd.DataFrame) -> go.Figure:
    """Risk evolution for one event as CDMs arrive (x = days to TCA, reversed)."""
    hist = hist.sort_values("time_to_tca", ascending=False)
    fig = go.Figure()
    if hist["risk"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["time_to_tca"], y=hist["risk"], mode="lines+markers",
            name="reported risk", line=dict(color=SERIES_1, width=2),
            marker=dict(size=8),
            hovertemplate="TCA−%{x:.2f} d · log₁₀ Pc %{y:.2f}<extra>reported</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=hist["time_to_tca"], y=hist["predicted_risk"], mode="lines+markers",
        name="model prediction", line=dict(color=SERIES_2, width=2),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="TCA−%{x:.2f} d · log₁₀ Pc %{y:.2f}<extra>predicted</extra>",
    ))
    for y, lbl in ((-4, "Critical ≥ −4"), (-6, "Elevated ≥ −6")):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=lbl, annotation_font_color=MUTED)
    fig.update_layout(**base_layout(height=360))
    fig.update_xaxes(title_text="days to closest approach", autorange="reversed",
                     gridcolor=GRID)
    fig.update_yaxes(title_text="log₁₀ Pc", gridcolor=GRID)
    return fig
