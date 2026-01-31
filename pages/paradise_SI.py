# -*- coding: utf-8 -*-
import re
from itertools import cycle
from typing import Optional, Dict, Callable
from contextlib import nullcontext

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cable_factory import create_cable
from SI_Test import SI_Test
from Test import Test
from Cable import Cable
from _shared_ui import top_bar
from UploadSIData import process_SI_file

# ============================================================
#                     Page & Session Init
# ============================================================
st.set_page_config(page_title="SI Tools", page_icon="🐍", layout="wide")

def ensure_state():
    st.session_state.setdefault("cables", [])                   # list[ Cable ]
    st.session_state.setdefault("tests", [])                    # optional list[ Test ]
    st.session_state.setdefault("processed_files", [])          # processed DAT files
    st.session_state.setdefault("processed_trace_files", [])    # processed trace CSVs
    st.session_state.setdefault("unattached_traces", [])        # list of {"name", "df"}
    st.session_state.setdefault("continuity_figs", {})
    st.session_state.setdefault("continuity_defect_figs", {})
    st.session_state.setdefault("inv_continuity_figs", {})
    st.session_state.setdefault("inv_continuity_defect_figs", {})
    st.session_state.setdefault("DCR_figs", {})
    st.session_state.setdefault("DCR_defect_figs", {})
    st.session_state.setdefault("leakage_figs", {})
    st.session_state.setdefault("leakage_defects", {})
    st.session_state.setdefault("leakage_1s_figs", {})
    st.session_state.setdefault("leakage_1s_defects", {})
    st.session_state.setdefault("traces", [])

def migrate_si_tests_traces():
    """
    Normalize all SI_Test.traces to a dict for existing objects in session_state.
    Safe to run multiple times.
    """
    cables = st.session_state.get("cables", [])
    for c in cables:
        si_list = getattr(c, "TESTS", {}).get("si", []) or []
        for t in si_list:
            val = getattr(t, "traces", None)
            if not isinstance(val, dict):
                try:
                    t.traces = dict(val) if val is not None else {}
                except Exception:
                    t.traces = {}

ensure_state()
migrate_si_tests_traces()
top_bar(page_icon="🐍", title="🏝️SI Tools", home_page_path="Home.py")


# ============================================================
#                        Utilities
# ============================================================
def _ns_to_ps_series(s: pd.Series) -> pd.Series:
    """Convert a numeric pandas Series from nanoseconds to picoseconds."""
    return pd.to_numeric(s, errors="coerce") * 1000.0

def parse_trace_filename(name: str) -> Optional[Dict[str, str]]:
    """
    Parse 'SN-01ACA3A061_J2.A01_P1_M1(.csv)' → {'serial': '01ACA3A061', 'end': 'P1', 'channel': 'A01'}
    Channel is optional; serial and end required for auto-attach.
    """
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base_no_ext = re.sub(r"\.[A-Za-z0-9]+$", "", base)

    pat = re.compile(
        r"""(?ix)
        ^SN-? (?P<serial>[A-Z0-9]+)
        _
        [A-Z0-9]+ \. (?P<channel>[A-Z]\d{2})
        _
        (?P<end>P[12])
        (?:_.*)?$
        """
    )
    m = pat.match(base_no_ext)
    if not m:
        pat2 = re.compile(r"""(?ix)^SN-?(?P<serial>[A-Z0-9]+).*?(?P<end>P[12]).*$""")
        m = pat2.match(base_no_ext)
        if not m:
            return None

    info = {k: v for k, v in m.groupdict().items() if v}
    info["serial"] = info["serial"].upper()
    info["end"] = info["end"].upper()
    if "channel" in info and info["channel"]:
        info["channel"] = info["channel"].upper()
    return info

def get_or_create_cable(cables: list, serial: str) -> Cable:
    """Find a cable by serial_number or create and append one."""
    found = next((c for c in cables if getattr(c, "serial_number", None) == serial), None)
    if found is not None:
        return found
    cable = create_cable(serial)
    cables.append(cable)
    return cable

def get_or_create_si_test_for_end(cable: Cable, end: str) -> Test:
    """Find the latest SI test for end (P1/P2) or create one and append."""
    si_list = getattr(cable, "TESTS", {}).get("si", []) or []
    matches = [t for t in si_list if (getattr(t, "test_end", "") or "").upper() == end.upper()]
    if matches:
        return matches[-1]
    test = SI_Test(test_type="si", test_end=end)
    if "si" not in cable.TESTS or cable.TESTS["si"] is None:
        cable.TESTS["si"] = []
    cable.TESTS["si"].append(test)
    return test

def _iter_si_tests(cables):
    """Yield (cable, test) for each SI test."""
    for c in cables:
        for t in getattr(c, "TESTS", {}).get("si", []) or []:
            yield c, t

def _format_col_label(serial: str, date_str: str | None, time_str: str | None) -> str:
    """
    Build '<Serial> <M/D/YYYY H:MM>' if date/time present; else '<Serial> <no time>'.
    """
    serial = (serial or "").strip()
    if date_str and time_str:
        try:
            y, mth, d = map(int, date_str.split("-"))
            hh, mm, _ss = map(int, time_str.split(":"))
            label = f"{serial} {mth}/{d}/{y} {hh}:{mm:02d}"
        except Exception:
            label = f"{serial} {date_str} {time_str}"
    else:
        label = f"{serial} <no time>"
    return label

def _dedup_label(label: str, existing: set[str]) -> str:
    if label not in existing:
        return label
    k = 2
    while f"{label}_{k}" in existing:
        k += 1
    return f"{label}_{k}"

def _current_zo_tol_for_category(cat_key: str) -> float:
    """Return the current Zo tolerance (Ω) for a given category."""
    return 2.5 if str(cat_key).strip().lower() == "cable" else 5.0


# ============================================================
#                   Zo Data Builders / Matrices
# ============================================================
def build_zo_wide_matrix(cables, *, category: str, metric: str, end: str) -> pd.DataFrame:
    """
    Build a wide matrix for a given Zo category and metric:
      - category: 'paddleboard' | 'cable' | 'dib'
      - metric:   'max' | 'min' | 'avg'
      - end:      'P1' | 'P2'
    Returns DataFrame with columns: ['Channels', '<Serial> <M/D/YYYY H:MM>', ...]
    """
    cols = {}
    existing = set()

    for cable, test in _iter_si_tests(cables):
        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue
        z = getattr(test, "zo_data", None)
        if not isinstance(z, dict):
            continue
        df = z.get(category)
        if not isinstance(df, pd.DataFrame) or metric not in df.columns or "channel" not in df.columns:
            continue

        label = _format_col_label(cable.serial_number, getattr(test, "test_date", None), getattr(test, "test_time", None))
        label = _dedup_label(label, existing)
        existing.add(label)

        s = pd.to_numeric(df[metric], errors="coerce")
        s.index = df["channel"]
        cols[label] = s

    if not cols:
        return pd.DataFrame(columns=["Channels"])

    wide = pd.DataFrame(cols)
    wide.insert(0, "Channels", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide


# ============================================================
#                  Skew (Overall from Pair Cols)
# ============================================================
def _find_two_numeric_pre_skew_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Find two numeric columns immediately left of a 'skew' column if present.
    Fallback: first two numeric columns (ignoring first col if label-like).
    """
    cols = list(df.columns)

    skew_idx = None
    for i, c in enumerate(cols):
        if "skew" in str(c).strip().lower():
            skew_idx = i
            break

    if skew_idx is not None and skew_idx >= 2:
        c1 = cols[skew_idx - 2]
        c2 = cols[skew_idx - 1]
        s1 = pd.to_numeric(df[c1], errors="coerce")
        s2 = pd.to_numeric(df[c2], errors="coerce")
        if s1.notna().sum() > 0 and s2.notna().sum() > 0:
            return c1, c2

    numeric_cols = []
    for c in cols[1:] if len(cols) > 1 else cols:
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().sum() > 0:
            numeric_cols.append(c)
        if len(numeric_cols) >= 2:
            break

    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    return None, None

def _collect_skew_overall_from_paircols_per_test(cables, end: str) -> pd.DataFrame:
    """
    For each SI_Test.skew_data on the given end, compute:
      Overall Max (pS) = 1000 * max of the first numeric col left of 'skew'
      Overall Min (pS) = 1000 * min of the second numeric col left of 'skew'
      Span (pS)        = Overall Max - Overall Min
    """
    rows = []
    for cable, test in _iter_si_tests(cables):
        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        v1_col, v2_col = _find_two_numeric_pre_skew_cols(df)
        if not v1_col or not v2_col:
            continue

        v1_ns = pd.to_numeric(df[v1_col], errors="coerce")
        v2_ns = pd.to_numeric(df[v2_col], errors="coerce")
        v1_ns = v1_ns[np.isfinite(v1_ns)]
        v2_ns = v2_ns[np.isfinite(v2_ns)]
        if v1_ns.empty or v2_ns.empty:
            continue

        v1_ps = v1_ns * 1000.0
        v2_ps = v2_ns * 1000.0

        overall_max_ps = float(np.nanmax(v1_ps))
        overall_min_ps = float(np.nanmin(v2_ps))
        span_ps = overall_max_ps - overall_min_ps

        label = _format_col_label(
            getattr(cable, "serial_number", "?"),
            getattr(test, "test_date", None),
            getattr(test, "test_time", None),
        )

        rows.append({
            "Serial": getattr(cable, "serial_number", "?"),
            "End": end.upper(),
            "Label": label,
            "Overall Max (pS)": overall_max_ps,
            "Overall Min (pS)": overall_min_ps,
            "Span (pS)": span_ps,
        })

    return pd.DataFrame(rows)

def build_skew_overall_from_paircols_wide(cables, end: str) -> pd.DataFrame:
    df = _collect_skew_overall_from_paircols_per_test(cables, end=end)
    if df.empty:
        return pd.DataFrame(columns=["Metric"])

    cols = {}
    existing = set()
    for _, r in df.iterrows():
        label = _dedup_label(r["Label"], existing)
        existing.add(label)
        cols[label] = pd.Series({
            "Overall Max": r["Overall Max (pS)"],
            "Overall Min": r["Overall Min (pS)"],
            "Span":        r["Span (pS)"],
        })

    wide = pd.DataFrame(cols)
    wide.insert(0, "Metric", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide


# ============================================================
#                   Per-Cable Zo (Max/Min)
# ============================================================
def collect_zo_min_values_df(
    cables, category: str, metric: str, end: str, key_fn=None,
) -> pd.DataFrame:
    def default_key_fn(cable):
        for attr in ("serial", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)

    key_fn = key_fn or default_key_fn
    per_cable_min = {}
    order = []

    for cable, test in _iter_si_tests(cables):
        key = key_fn(cable)
        if key not in per_cable_min:
            per_cable_min[key] = np.nan
            order.append(key)

        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        z = getattr(test, "zo_data", None)
        if not isinstance(z, dict):
            continue

        df = z.get(category)
        if isinstance(df, pd.DataFrame) and metric in df.columns:
            col = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not col.empty:
                cur = col.min()
                prev = per_cable_min[key]
                per_cable_min[key] = cur if np.isnan(prev) else max(prev, cur)

    values = [per_cable_min[k] for k in order]
    return pd.DataFrame({"value": values})

def collect_zo_max_values_df(
    cables, category: str, metric: str, end: str, key_fn=None,
) -> pd.DataFrame:
    def default_key_fn(cable):
        for attr in ("serial", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)

    key_fn = key_fn or default_key_fn
    per_cable_max = {}
    order = []

    for cable, test in _iter_si_tests(cables):
        key = key_fn(cable)
        if key not in per_cable_max:
            per_cable_max[key] = np.nan
            order.append(key)

        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        z = getattr(test, "zo_data", None)
        if not isinstance(z, dict):
            continue

        df = z.get(category)
        if isinstance(df, pd.DataFrame) and metric in df.columns:
            col = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not col.empty:
                cur = col.max()
                prev = per_cable_max[key]
                per_cable_max[key] = cur if np.isnan(prev) else max(prev, cur)

    values = [per_cable_max[k] for k in order]
    return pd.DataFrame({"value": values})


# ============================================================
#              Zo Collect (serial-aware long form)
# ============================================================
def _collect_zo_values_with_serial(cables, category: str, metric: str, end: str) -> pd.DataFrame:
    rows = []
    for cable, test in _iter_si_tests(cables):
        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        z = getattr(test, "zo_data", None)
        if not isinstance(z, dict):
            continue
        df = z.get(category)
        if isinstance(df, pd.DataFrame) and metric in df.columns:
            col = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not col.empty:
                sn = getattr(cable, "serial_number", "?")
                rows.extend({"serial": sn, "value": float(v)} for v in col.tolist())
    return pd.DataFrame(rows, columns=["serial", "value"])


# ============================================================
#                     Skew (Category-Aware)
# ============================================================
def _collect_skew_delta(
    cables,
    end: str,
    category: str,
    *,
    selector: Optional[Callable[[pd.Series, str], bool]] = None,
    channel_col: str = "channel site",
    delta_col: str = "delta",
    category_col_candidates: tuple[str, ...] = ("category", "cat", "group"),
) -> pd.DataFrame:
    """
    Category-aware: returns one-column DataFrame {'value_ps'} with
    the per-cable max Δskew in pS for the specified end.
    """
    def _key_for_cable(cable):
        for attr in ("serial", "serial_number", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)

    def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
        for c in category_col_candidates:
            if c in df.columns:
                return c
        lower_map = {str(c).lower(): c for c in df.columns}
        for c in category_col_candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    per_cable_max = {}
    order = []

    for cable, test in _iter_si_tests(cables):
        key = _key_for_cable(cable)
        if key not in per_cable_max:
            per_cable_max[key] = np.nan
            order.append(key)

        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue

        skew_obj = getattr(test, "skew_data", None)
        if skew_obj is None:
            continue

        if isinstance(skew_obj, dict):
            df = skew_obj.get(category)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
        else:
            df = skew_obj if isinstance(skew_obj, pd.DataFrame) else None
            if df is None or df.empty:
                continue

            if delta_col not in df.columns:
                alt = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
                if alt:
                    delta_use = alt
                else:
                    continue
            else:
                delta_use = delta_col

            cat_col = _pick_category_col(df)
            if cat_col is not None:
                cat_mask = df[cat_col].astype(str).str.strip().str.lower() == category.strip().lower()
                df = df.loc[cat_mask].copy()
            else:
                if callable(selector):
                    try:
                        row_mask = df.apply(lambda r: bool(selector(r, category)), axis=1)
                        df = df.loc[row_mask].copy()
                    except Exception:
                        pass

            delta_col = delta_use

        if df.empty:
            continue

        if delta_col not in df.columns:
            alt = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
            if not alt:
                continue
            delta_col = alt

        series_ps = _ns_to_ps_series(df[delta_col]).dropna()
        if series_ps.empty:
            continue

        cur = float(series_ps.max())
        prev = per_cable_max[key]
        per_cable_max[key] = cur if np.isnan(prev) else max(prev, cur)

    values = [per_cable_max[k] for k in order]
    return pd.DataFrame({"value_ps": values})

def _collect_skew_delta_with_serial(
    cables,
    end: str,
    category: str,
    *,
    selector: Optional[Callable[[pd.Series, str], bool]] = None,
    channel_col: str = "channel site",
    delta_col: str = "delta",
    category_col_candidates: tuple[str, ...] = ("category", "cat", "group"),
) -> pd.DataFrame:
    """
    Category-aware: returns rows ['serial','value_ps'] for Δskew in pS.
    """
    rows: list[dict] = []

    def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
        for c in category_col_candidates:
            if c in df.columns:
                return c
        lower_map = {str(c).lower(): c for c in df.columns}
        for c in category_col_candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    for cable, test in _iter_si_tests(cables):
        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue

        skew_obj = getattr(test, "skew_data", None)
        if skew_obj is None:
            continue

        if isinstance(skew_obj, dict):
            df = skew_obj.get(category)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
        else:
            df = skew_obj if isinstance(skew_obj, pd.DataFrame) else None
            if df is None or df.empty:
                continue

            delta_use = delta_col if delta_col in df.columns else next(
                (c for c in df.columns if str(c).strip().lower() == "delta"), None
            )
            if not delta_use:
                continue

            cat_col = _pick_category_col(df)
            if cat_col is not None:
                cat_mask = df[cat_col].astype(str).str.strip().str.lower() == category.strip().lower()
                df = df.loc[cat_mask].copy()
            else:
                if callable(selector):
                    try:
                        row_mask = df.apply(lambda r: bool(selector(r, category)), axis=1)
                        df = df.loc[row_mask].copy()
                    except Exception:
                        pass

            delta_col = delta_use

        if df is None or df.empty:
            continue

        dcol = delta_col if delta_col in df.columns else next(
            (c for c in df.columns if str(c).strip().lower() == "delta"), None
        )
        if not dcol:
            continue

        series_ps = _ns_to_ps_series(df[dcol]).dropna()
        if series_ps.empty:
            continue

        sn = getattr(cable, "serial_number", None) or getattr(cable, "serial", None) or "?"
        rows.extend({"serial": sn, "value_ps": float(v)} for v in series_ps.tolist())

    return pd.DataFrame(rows, columns=["serial", "value_ps"])


# ============================================================
#               Counts & Small Reductions (fast)
# ============================================================
def _zo_counts_by_cable(
    df: pd.DataFrame,
    target: float,
    tol_curr: float,
    tol_prop: float,
    value_col: str = "value",
    serial_col: str = "serial",
    per_cable_rule: str = "all_channels",
):
    if df is None or df.empty or value_col not in df.columns or serial_col not in df.columns:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    vals = pd.to_numeric(df[value_col], errors="coerce")
    ser = df[serial_col]
    mask = vals.notna() & ser.notna()
    vals = vals[mask]
    ser = ser[mask]

    if vals.empty:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    dev = (vals - float(target)).abs()
    ok_curr = dev <= float(tol_curr)
    ok_prop = dev <= float(tol_prop)

    if per_cable_rule == "all_channels":
        pass_curr_by_cable = ok_curr.groupby(ser).all()
        pass_prop_by_cable = ok_prop.groupby(ser).all()
    elif per_cable_rule == "any_channel":
        pass_curr_by_cable = ok_curr.groupby(ser).any()
        pass_prop_by_cable = ok_prop.groupby(ser).any()
    else:
        raise ValueError("per_cable_rule must be 'all_channels' or 'any_channel'")

    total_cables = int(pass_curr_by_cable.size)
    pass_current = int(pass_curr_by_cable.sum())
    pass_prop    = int(pass_prop_by_cable.sum())
    fail_current = total_cables - pass_current
    fail_prop    = total_cables - pass_prop

    return dict(
        total=total_cables,
        pass_current=pass_current,
        fail_current=fail_current,
        pass_prop=pass_prop,
        fail_prop=fail_prop,
    )

def _skew_counts_by_cable(
    df: pd.DataFrame,
    curr_ps: float,
    prop_ps: float,
    value_col: str = "value_ps",
    serial_col: str = "serial",
    per_cable_rule: str = "all_channels",
) -> dict:
    if df is None or df.empty or value_col not in df.columns or serial_col not in df.columns:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    vals = pd.to_numeric(df[value_col], errors="coerce")
    ser = df[serial_col]
    mask = vals.notna() & ser.notna()
    vals = vals[mask].abs()
    ser = ser[mask]

    if vals.empty:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    ok_curr = vals <= float(curr_ps)
    ok_prop = vals <= float(prop_ps)

    if per_cable_rule == "all_channels":
        pass_curr_by_cable = ok_curr.groupby(ser).all()
        pass_prop_by_cable = ok_prop.groupby(ser).all()
    elif per_cable_rule == "any_channel":
        pass_curr_by_cable = ok_curr.groupby(ser).any()
        pass_prop_by_cable = ok_prop.groupby(ser).any()
    else:
        raise ValueError("per_cable_rule must be 'all_channels' or 'any_channel'")

    total_cables = int(pass_curr_by_cable.size)
    pass_current = int(pass_curr_by_cable.sum())
    pass_prop    = int(pass_prop_by_cable.sum())
    fail_current = total_cables - pass_current
    fail_prop    = total_cables - pass_prop

    return dict(
        total=total_cables,
        pass_current=pass_current,
        fail_current=fail_current,
        pass_prop=pass_prop,
        fail_prop=fail_prop,
    )

def _reduce_and(series_list: list[pd.Series]) -> pd.Series:
    """Per-cable AND across slices; if a cable appears, it must pass in every slice it appears."""
    if not series_list:
        return pd.Series(dtype=bool)
    return pd.concat(series_list, axis=1).all(axis=1)


# ============================================================
#                  Uploaders (DAT & Traces)
# ============================================================
uploaded_files = st.file_uploader("Upload your SI files", type="DAT", accept_multiple_files=True)
uploaded_trace_csvs = st.file_uploader("Upload your Trace CSV files", type=["csv"], accept_multiple_files=True, key="trace_uploader")

cables = st.session_state["cables"]
processed = st.session_state["processed_files"]

if uploaded_files:
    for uf in uploaded_files:
        if uf.name in processed:
            continue
        cable, test = process_SI_file(uf, cables)
        if cable is not None:
            exists = next((c for c in st.session_state["cables"]
                           if getattr(c, "serial_number", None) == cable.serial_number), None)
            if exists is None:
                st.session_state["cables"].append(cable)
            processed.append(uf.name)
        st.session_state["processed_files"] = list(processed)


# ============================================================
#                        Specs Defaults
# ============================================================
if "zo_target_ohm" not in st.session_state:
    st.session_state["zo_target_ohm"] = 50.0
if "zo_tol_current_ohm" not in st.session_state:
    st.session_state["zo_tol_current_ohm"] = 5.0
if "zo_tol_proposed_ohm" not in st.session_state:
    st.session_state["zo_tol_proposed_ohm"] = st.session_state["zo_tol_current_ohm"]

if "zo_min_current_ohm" not in st.session_state:
    st.session_state["zo_min_current_ohm"] = 45.0
if "zo_max_current_ohm" not in st.session_state:
    st.session_state["zo_max_current_ohm"] = 55.0
if "zo_min_proposed_ohm" not in st.session_state:
    st.session_state["zo_min_proposed_ohm"] = st.session_state["zo_min_current_ohm"]
if "zo_max_proposed_ohm" not in st.session_state:
    st.session_state["zo_max_proposed_ohm"] = st.session_state["zo_max_current_ohm"]

if "skew_current_spec_ps" not in st.session_state:
    st.session_state["skew_current_spec_ps"] = 23.0
if "skew_tol_proposed" not in st.session_state:
    st.session_state["skew_tol_proposed"] = 23.0

st.session_state.setdefault("skew_proposed_ps", float(st.session_state.get("skew_current_spec_ps", 50.0)))

zo_target_ohm   = float(st.session_state.get("zo_target_ohm", 50.0))
zo_tol_curr_ohm = float(st.session_state.get("zo_tol_current_ohm", 5.0))
curr_skew_ps    = float(st.session_state.get("skew_current_spec_ps", 50.0))


# ============================================================
#                 Performance / Caching Helpers
# ============================================================
def _data_cache_key():
    pf = tuple(sorted(st.session_state.get("processed_files", [])))
    pt = tuple(sorted(st.session_state.get("processed_trace_files", [])))
    num_cables = len(st.session_state.get("cables", []))
    num_tests = sum(len(getattr(c, "TESTS", {}).get("si", []) or []) 
                    for c in st.session_state.get("cables", []))
    return (pf, pt, num_cables, num_tests)

from contextlib import contextmanager
@contextmanager
def fragment_ctx():
    """
    A guaranteed context manager that isolates rendering.
    Uses st.container() (a context manager) under the hood.
    This avoids relying on st.fragment behaviors across versions.
    """
    with st.container():
        yield


# Cached wrappers (they reference st.session_state["cables"] inside)
@st.cache_data(show_spinner=False)
def collect_zo_max_values_df_cached(cache_key, category, metric, end):
    return collect_zo_max_values_df(st.session_state["cables"], category, metric, end)

@st.cache_data(show_spinner=False)
def collect_zo_min_values_df_cached(cache_key, category, metric, end):
    return collect_zo_min_values_df(st.session_state["cables"], category, metric, end)

@st.cache_data(show_spinner=False)
def collect_zo_values_with_serial_cached(cache_key, category, metric, end):
    return _collect_zo_values_with_serial(st.session_state["cables"], category, metric, end)

@st.cache_data(show_spinner=False)
def collect_skew_delta_cached(cache_key, end, category):
    return _collect_skew_delta(st.session_state["cables"], end=end, category=category)

@st.cache_data(show_spinner=False)
def collect_skew_delta_with_serial_cached(cache_key, end, category):
    return _collect_skew_delta_with_serial(st.session_state["cables"], end=end, category=category)

@st.cache_data(show_spinner=False)
def build_zo_wide_matrix_cached(cache_key, category, metric, end):
    return build_zo_wide_matrix(st.session_state["cables"], category=category, metric=metric, end=end)

@st.cache_data(show_spinner=False)
def build_skew_wide_matrix_cached(cache_key, end):
    return build_skew_wide_matrix(st.session_state["cables"], end=end)

@st.cache_data(show_spinner=False)
def collect_skew_overall_from_paircols_per_test_cached(cache_key, end):
    return _collect_skew_overall_from_paircols_per_test(st.session_state["cables"], end=end)

@st.cache_data(show_spinner=False)
def build_skew_overall_from_paircols_wide_cached(cache_key, end):
    return build_skew_overall_from_paircols_wide(st.session_state["cables"], end=end)


# ============================================================
#                         Main Tabs
# ============================================================
st.divider()

categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
ends = ["P1", "P2"]

tabs = st.tabs([label for _, label in categories])
for (cat_key, cat_label), tab in zip(categories, tabs):
    with tab:
        st.write(f"**Category:** {cat_label}")
        st.subheader("Zo Histograms")

        cache_key = _data_cache_key()

        # ====== Precompute (cached) ======
        # Max
        df_max_p1 = collect_zo_max_values_df_cached(cache_key, cat_key, "max", "P1"); df_max_p1["end"] = "P1"
        df_max_p2 = collect_zo_max_values_df_cached(cache_key, cat_key, "max", "P2"); df_max_p2["end"] = "P2"
        df_max_both = pd.concat([df_max_p1, df_max_p2], ignore_index=True).rename(columns={"value": "max_value_ohm"})

        df_max_serial_long = pd.concat([
            collect_zo_values_with_serial_cached(cache_key, cat_key, "max", "P1").assign(end="P1"),
            collect_zo_values_with_serial_cached(cache_key, cat_key, "max", "P2").assign(end="P2"),
        ], ignore_index=True)

        st.divider()

        # --------- MAX (hist + counts) ---------
        with fragment_ctx():
            st.markdown(f"##### {cat_label} — Max Impedance (Ω)")
            if df_max_both.empty or df_max_both["max_value_ohm"].dropna().empty:
                st.info("No Max data across P1/P2")
            else:
                fig = px.histogram(
                    df_max_both, x="max_value_ohm", color="end",
                    nbins=30, barmode="overlay", opacity=0.60,
                    title=f"{cat_label} — Max Impedance",
                )
                fig.update_layout(
                    xaxis_title="Max Impedance (Ω)", yaxis_title="Count",
                    margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                    legend_title_text="End",
                )
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

                if df_max_serial_long is None or df_max_serial_long.empty:
                    st.info("No cable-level Max impedance data available for this category.")
                else:
                    current_tol_max = _current_zo_tol_for_category(cat_key)
                    target_ohm = float(st.session_state.get("zo_target_ohm", 50.0))

                    prop_key_max = f"zo_tol_proposed_max_ohm__{cat_key}"
                    if prop_key_max not in st.session_state:
                        st.session_state[prop_key_max] = current_tol_max

                    proposed_tol_max_local = st.number_input(
                        f"Proposed tolerance (Ω) — Max — {cat_label}",
                        key=f"zo_tol_prop_max_{cat_key}",
                        min_value=0.0, max_value=1000.0,
                        value=float(st.session_state[prop_key_max]),
                        step=0.5,
                        format="%.2f",  # ← prevents reruns on every keystroke
                    )
                    st.session_state[prop_key_max] = float(proposed_tol_max_local)

                    counts_max = _zo_counts_by_cable(
                        df=df_max_serial_long,
                        target=target_ohm,
                        tol_curr=current_tol_max,
                        tol_prop=float(proposed_tol_max_local),
                        value_col="value",
                        serial_col="serial",
                        per_cable_rule="all_channels",
                    )

                    def _pct(n: int, d: int) -> float:
                        return (100.0 * float(n) / float(d)) if d and d > 0 else 0.0

                    total = counts_max["total"]
                    row_max = {
                        "Cables — Total": total,
                        "Pass (Current)": counts_max["pass_current"],
                        "% Pass (Current)": f"{_pct(counts_max['pass_current'], total):.1f}%",
                        "Fail (Current)": counts_max["fail_current"],
                        "% Fail (Current)": f"{_pct(counts_max['fail_current'], total):.1f}%",
                        "Pass (Proposed)": counts_max["pass_prop"],
                        "% Pass (Proposed)": f"{_pct(counts_max['pass_prop'], total):.1f}%",
                        "Fail (Proposed)": counts_max["fail_prop"],
                        "% Fail (Proposed)": f"{_pct(counts_max['fail_prop'], total):.1f}%",
                    }
                    st.dataframe(pd.DataFrame([row_max]), hide_index=True, use_container_width=True)

        st.markdown("---")

        # ====== Precompute (cached) ======
        # Min
        df_min_p1 = collect_zo_min_values_df_cached(cache_key, cat_key, "min", "P1"); df_min_p1["end"] = "P1"
        df_min_p2 = collect_zo_min_values_df_cached(cache_key, cat_key, "min", "P2"); df_min_p2["end"] = "P2"
        df_min_both = pd.concat([df_min_p1, df_min_p2], ignore_index=True).rename(columns={"value": "min_value_ohm"})

        df_min_serial_long = pd.concat([
            collect_zo_values_with_serial_cached(cache_key, cat_key, "min", "P1").assign(end="P1"),
            collect_zo_values_with_serial_cached(cache_key, cat_key, "min", "P2").assign(end="P2"),
        ], ignore_index=True)

        # --------- MIN (hist + counts) ---------
        with fragment_ctx():
            st.markdown(f"##### {cat_label} — Min Impedance (Ω)")
            if df_min_both.empty or df_min_both["min_value_ohm"].dropna().empty:
                st.info("No Min data across P1/P2")
            else:
                fig = px.histogram(
                    df_min_both, x="min_value_ohm", color="end",
                    nbins=30, barmode="overlay", opacity=0.60,
                    title=f"{cat_label} — Min Impedance",
                )
                fig.update_layout(
                    xaxis_title="Min Impedance (Ω)", yaxis_title="Count",
                    margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                    legend_title_text="End",
                )
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

                if df_min_serial_long is None or df_min_serial_long.empty:
                    st.info("No cable-level Min impedance data available for this category.")
                else:
                    current_tol_min = _current_zo_tol_for_category(cat_key)
                    target_ohm = float(st.session_state.get("zo_target_ohm", 50.0))

                    prop_key_min = f"zo_tol_proposed_min_ohm__{cat_key}"
                    if prop_key_min not in st.session_state:
                        st.session_state[prop_key_min] = current_tol_min

                    proposed_tol_min_local = st.number_input(
                        f"Proposed tolerance (Ω) — Min — {cat_label}",
                        key=f"zo_tol_prop_min_{cat_key}",
                        min_value=0.0, max_value=1000.0,
                        value=float(st.session_state[prop_key_min]),
                        step=0.5,
                        format="%.2f",  # ← prevents reruns on every keystroke
                    )
                    st.session_state[prop_key_min] = float(proposed_tol_min_local)

                    counts_min = _zo_counts_by_cable(
                        df=df_min_serial_long,
                        target=target_ohm,
                        tol_curr=current_tol_min,
                        tol_prop=float(proposed_tol_min_local),
                        value_col="value",
                        serial_col="serial",
                        per_cable_rule="all_channels",
                    )

                    def _pct(n: int, d: int) -> float:
                        return (100.0 * float(n) / float(d)) if d and d > 0 else 0.0

                    total = counts_min["total"]
                    row_min = {
                        "Cables — Total": total,
                        "Pass (Current)": counts_min["pass_current"],
                        "% Pass (Current)": f"{_pct(counts_min['pass_current'], total):.1f}%",
                        "Fail (Current)": counts_min["fail_current"],
                        "% Fail (Current)": f"{_pct(counts_min['fail_current'], total):.1f}%",
                        "Pass (Proposed)": counts_min["pass_prop"],
                        "% Pass (Proposed)": f"{_pct(counts_min['pass_prop'], total):.1f}%",
                        "Fail (Proposed)": counts_min["fail_prop"],
                        "% Fail (Proposed)": f"{_pct(counts_min['fail_prop'], total):.1f}%",
                    }
                    st.dataframe(pd.DataFrame([row_min]), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Skew Histograms (Δ skew [pS])")

        # ====== Skew Δ both ends (cached) ======
        df_delta_p1 = collect_skew_delta_cached(cache_key, end="P1", category=cat_key).rename(columns={"value_ps": "delta_ps"})
        df_delta_p1["end"] = "P1"
        df_delta_p2 = collect_skew_delta_cached(cache_key, end="P2", category=cat_key).rename(columns={"value_ps": "delta_ps"})
        df_delta_p2["end"] = "P2"
        df_delta_both = pd.concat([df_delta_p1, df_delta_p2], ignore_index=True)

        with fragment_ctx():
            if df_delta_both.empty or df_delta_both["delta_ps"].dropna().empty:
                st.info("No Δ skew data across P1/P2")
            else:
                fig = px.histogram(
                    df_delta_both, x="delta_ps", color="end",
                    nbins=30, barmode="overlay", opacity=0.6,
                    title=f"Δ Skew (pS) — {cat_label} ",
                )
                fig.update_layout(
                    xaxis_title="Δ skew [pS]", yaxis_title="Count",
                    margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                    legend_title_text="End",
                )
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

                df_delta_s = pd.concat([
                    collect_skew_delta_with_serial_cached(cache_key, end="P1", category=cat_key),
                    collect_skew_delta_with_serial_cached(cache_key, end="P2", category=cat_key),
                ], ignore_index=True)

                if df_delta_s is None or df_delta_s.empty:
                    st.info("No cable-level skew data available for this category.")
                else:
                    current_spec_ps  = float(st.session_state.get("skew_current_spec_ps", 50.0))
                    proposed_spec_ps = float(st.session_state.get("skew_proposed_ps", current_spec_ps))

                    proposed_skew_ps_local = st.number_input(
                        f"Proposed tolerance [pS] — {cat_label}",
                        key=f"skew_tol_prop_combined_ps_{cat_key}",
                        min_value=0.0, max_value=100000.0,
                        value=proposed_spec_ps, step=1.0,
                        format="%.1f",  # ← prevents reruns on every keystroke
                    )
                    st.session_state["skew_proposed_ps"] = float(proposed_skew_ps_local)

                    counts_skew = _skew_counts_by_cable(
                        df_delta_s,
                        curr_ps=current_spec_ps,
                        prop_ps=float(proposed_skew_ps_local),
                        value_col="value_ps",
                        serial_col="serial",
                        per_cable_rule="all_channels",
                    )

                    def _pct(n: int, d: int) -> float:
                        return (100.0 * float(n) / float(d)) if d and d > 0 else 0.0

                    total = counts_skew["total"]
                    row_skew = {
                        "Cables — Total": total,
                        "Pass (Current)": counts_skew["pass_current"],
                        "% Pass (Current)": f"{_pct(counts_skew['pass_current'], total):.1f}%",
                        "Fail (Current)": counts_skew["fail_current"],
                        "% Fail (Current)": f"{_pct(counts_skew['fail_current'], total):.1f}%",
                        "Pass (Proposed)": counts_skew["pass_prop"],
                        "% Pass (Proposed)": f"{_pct(counts_skew['pass_prop'], total):.1f}%",
                        "Fail (Proposed)": counts_skew["fail_prop"],
                        "% Fail (Proposed)": f"{_pct(counts_skew['fail_prop'], total):.1f}%",
                    }
                    st.dataframe(pd.DataFrame([row_skew]), hide_index=True, use_container_width=True)


# ============================================================
#                      Combined Downloads
# ============================================================
st.divider()
st.subheader("Download Combined CSVs")

cache_key = _data_cache_key()

# Zo Combined Downloads
zo_categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
zo_metrics = [("max", "Max"), ("min", "Min")]

for cat_key, cat_label in zo_categories:
    with st.expander(f"Zo — {cat_label}", expanded=False):
        for metric_key, metric_label in zo_metrics:
            wide_p1 = build_zo_wide_matrix_cached(cache_key, category=cat_key, metric=metric_key, end="P1"); wide_p1["End"] = "P1"
            wide_p2 = build_zo_wide_matrix_cached(cache_key, category=cat_key, metric=metric_key, end="P2"); wide_p2["End"] = "P2"
            wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

            st.markdown(f"**{cat_label} — {metric_label} Impedance (Ω)**")
            if wide_both.empty:
                st.caption("No data available.")
            else:
                st.dataframe(wide_both.head(8), use_container_width=True)
                st.download_button(
                    label=f"Download {cat_label} {metric_label}",
                    data=wide_both.to_csv(index=False),
                    file_name=f"Zo_{cat_label}_{metric_label}_.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# Skew Δ Combined Download
def build_skew_wide_matrix(cables, *, end: str) -> pd.DataFrame:
    """
    Build a wide matrix for Skew where rows are 'channel site' and
    columns are '<Serial> <M/D/YYYY H:MM>', values are Δ in pS.
    """
    cols = {}
    existing = set()

    for cable, test in _iter_si_tests(cables):
        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        ch_col = next((c for c in df.columns if str(c).strip().lower() == "channel site"), None)
        d_col  = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
        if ch_col is None or d_col is None:
            continue

        label = _format_col_label(cable.serial_number, getattr(test, "test_date", None), getattr(test, "test_time", None))
        label = _dedup_label(label, existing)
        existing.add(label)

        s_ps = _ns_to_ps_series(df[d_col])
        s_ps.index = df[ch_col]
        cols[label] = s_ps

    if not cols:
        return pd.DataFrame(columns=["Channels"])

    wide = pd.DataFrame(cols)
    wide.insert(0, "Channels", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide

with st.expander("Skew Δ (pS)", expanded=False):
    wide_p1 = build_skew_wide_matrix_cached(cache_key, end="P1"); wide_p1["End"] = "P1"
    wide_p2 = build_skew_wide_matrix_cached(cache_key, end="P2"); wide_p2["End"] = "P2"
    wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

    if wide_both.empty:
        st.caption("No skew Δ data.")
    else:
        st.dataframe(wide_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Δ",
            data=wide_both.to_csv(index=False),
            file_name="Skew_Delta_Combined.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Skew Overall From Pair Cols Combined
with st.expander("Skew Overall from Pair Columns ", expanded=False):
    long_p1 = collect_skew_overall_from_paircols_per_test_cached(cache_key, end="P1"); long_p1["End"] = "P1"
    long_p2 = collect_skew_overall_from_paircols_per_test_cached(cache_key, end="P2"); long_p2["End"] = "P2"
    long_both = pd.concat([long_p1, long_p2], ignore_index=True)

    wide_p1 = build_skew_overall_from_paircols_wide_cached(cache_key, end="P1"); wide_p1["End"] = "P1"
    wide_p2 = build_skew_overall_from_paircols_wide_cached(cache_key, end="P2"); wide_p2["End"] = "P2"
    wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

    if long_both.empty:
        st.caption("No overall skew data.")
    else:
        st.markdown("**Preview (LONG Format)**")
        st.dataframe(long_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Overall — LONG",
            data=long_both.to_csv(index=False),
            file_name="Skew_Overall_Combined_LONG.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Preview (WIDE Format)**")
        st.dataframe(wide_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Overall WIDE",
            data=wide_both.to_csv(index=False),
            file_name="Skew_Overall_Combined_WIDE.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
#                     Traces Upload & Overlay
# ============================================================
processed_trace = st.session_state.setdefault("processed_trace_files", [])

if uploaded_trace_csvs:
    for tf in uploaded_trace_csvs:
        if tf.name in processed_trace:
            continue

        try:
            df = pd.read_csv(tf, header=None, names=["time_s", "value"])
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)

            meta = parse_trace_filename(tf.name)
            if not meta:
                st.session_state["unattached_traces"].append({"name": tf.name, "df": df})
                processed_trace.append(tf.name)
                st.warning(f"Could not parse '{tf.name}' for serial/end; kept in unattached pool.")
                continue

            serial = meta["serial"]
            end = meta["end"]
            trace_key = meta.get("channel", tf.name)

            cable = get_or_create_cable(st.session_state["cables"], serial=serial)
            si_test = get_or_create_si_test_for_end(cable, end=end)

            if not isinstance(si_test.traces, dict):
                si_test.traces = {}

            si_test.traces[trace_key] = df
            processed_trace.append(tf.name)

        except Exception as e:
            st.warning(f"Failed to read/attach '{tf.name}': {e}")

    st.session_state["processed_trace_files"] = list(processed_trace)

st.divider()
st.subheader("Traces Overlays")

def pick_xy_columns(df: pd.DataFrame):
    """
    Always choose a time-like column as X.
    Preference order: time_s, time, t, x, sample_idx.
    Then choose any other numeric column as Y.
    Fallbacks: first two numeric columns; then first two columns.
    """
    cols = list(df.columns)
    time_candidates = ["time_s", "time", "t", "x", "sample_idx"]

    x_col = next((c for c in time_candidates if c in df.columns), None)

    if x_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        x_col = numeric_cols[0] if numeric_cols else None

    y_col = None
    if x_col is not None:
        numeric_others = [c for c in cols if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        y_col = numeric_others[0] if numeric_others else None

    if x_col is None or y_col is None:
        if len(cols) >= 2:
            x_col, y_col = cols[0], cols[1]
        else:
            return None, None

    return x_col, y_col

# Optional: Only build overlays when requested (speeds up spec edits)
show_traces = st.checkbox("Show trace overlays", value=False, key="show_traces_overlay")

if show_traces:
    with fragment_ctx():
        unique_palette = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.D3
            + px.colors.qualitative.Set2
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Pastel
        )

        for cable in st.session_state.get("cables", []):
            sn = getattr(cable, "serial_number", "?")
            si_list = getattr(cable, "TESTS", {}).get("si", []) or []

            traces_all = []  # list of (end, name, df)
            for t in si_list:
                test_end = (getattr(t, "test_end", "") or "").upper()
                traces_obj = getattr(t, "traces", None)
                if not isinstance(traces_obj, dict) or not traces_obj:
                    continue
                for name, df in traces_obj.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        traces_all.append((test_end, name, df))

            if not traces_all:
                continue

            st.markdown(f"### Cable **{sn}**")
            fig = go.Figure()

            # Reset a color cycle per cable so each cable starts the same palette
            per_cable_cycle = cycle(unique_palette)

            for end, name, df in traces_all:
                x_col, y_col = pick_xy_columns(df)
                if x_col is None or y_col is None:
                    st.warning(f"Skipping trace '{name}' (could not infer x/y columns).")
                    continue

                x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                y_vals = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                x_vals, y_vals = x_vals[mask], y_vals[mask]

                if len(x_vals) < 2:
                    st.warning(f"Skipping trace '{name}' (too few points after cleaning).")
                    continue

                use_gl = len(x_vals) > 5000
                trace_cls = go.Scattergl if use_gl else go.Scatter

                color_this_trace = next(per_cable_cycle)

                fig.add_trace(
                    trace_cls(
                        x=x_vals, y=y_vals, mode="lines",
                        name=f"{end} — {name}",
                        line=dict(width=1.4, color=color_this_trace),
                        hovertemplate=f"<b>{end} — {name}</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                    )
                )

            fig.update_layout(
                title=f"Overlay — Cable {sn} — P1 & P2",
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",  y=1.0,
                    xanchor="left", x=1.02,
                    traceorder="normal",
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
                margin=dict(l=10, r=180, t=48, b=10),
                hovermode="x unified",
            )
            fig.update_xaxes(title_text="Time [s]")
            fig.update_yaxes(title_text="Resistance (Ω)")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Trace overlays are disabled to keep the page snappy while you tune specs. Check the box above to render them.")