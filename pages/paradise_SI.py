import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Optional, Dict, Callable
from dataclasses import Field as DataclassField

from cable_factory import create_cable
from SI_Test import SI_Test
from Test import Test
from Cable import Cable
from _shared_ui import top_bar
from UploadSIData import process_SI_file

# ----------------------- Page & Session Init -----------------------
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
    Must be safe to run multiple times.
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

# ------------------------- Utilities -------------------------
def _ns_to_ps_series(s: pd.Series) -> pd.Series:
    """Convert a numeric pandas Series from nanoseconds to picoseconds."""
    return pd.to_numeric(s, errors="coerce") * 1000.0

def _ns_to_ps_value(x: float | int | None):
    if x is None:
        return None
    return float(x) * 1000.0

def parse_trace_filename(name: str) -> Optional[Dict[str, str]]:
    """
    Parse 'SN-01ACA3A061_J2.A01_P1_M1(.csv)' → {'serial': '01ACA3A061', 'end': 'P1', 'channel': 'A01'}
    Channel is optional to use; serial and end are required for auto-attach.
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

# ---------------------- Zo Data Builders ----------------------
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

# ---------------------- Skew Overall (Pair Cols) ----------------------
def _find_two_numeric_pre_skew_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Find two numeric columns immediately left of a 'skew' column if present.
    Fallback: first two numeric columns (ignoring col0 label).
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

# ---------------------- Per-Cable Zo (Max/Min) ----------------------
def collect_zo_min_values_df(
    cables,
    category: str,
    metric: str,
    end: str,
    key_fn=None,
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
    cables,
    category: str,
    metric: str,
    end: str,
    key_fn=None,
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

# ---------------------- Zo Collect (with serials) ----------------------
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

# ---------------------- Skew (Category-Aware) ----------------------
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

# ---------------------- Counts & Reductions ----------------------
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

# ---------------------- Uploaders ----------------------
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

# ---------------------- Specs Defaults ----------------------
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
st.session_state.setdefault("zo_tol_proposed_max_ohm", float(st.session_state.get("zo_tol_current_ohm", 5.0)))
st.session_state.setdefault("zo_tol_proposed_min_ohm", float(st.session_state.get("zo_tol_current_ohm", 5.0)))
st.session_state.setdefault("skew_proposed_ps", float(st.session_state.get("skew_current_spec_ps", 50.0)))

zo_target_ohm   = float(st.session_state.get("zo_target_ohm", 50.0))
zo_tol_curr_ohm = float(st.session_state.get("zo_tol_current_ohm", 5.0))
curr_skew_ps    = float(st.session_state.get("skew_current_spec_ps", 50.0))



categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
ends = ["P1", "P2"]

st.divider()

# -------- Category-aware Executive Summaries (per tab) --------
def _zo_exec_summary_for_category(cat_key: str) -> pd.DataFrame:
    """
    Category-aware Zo summary (AND across ends) for the current category tab.
    Returns a DataFrame with two rows: Zo Max, Zo Min.
    """
    pass_curr_list_max, pass_prop_list_max = [], []
    pass_curr_list_min, pass_prop_list_min = [], []

    target_ohm = float(st.session_state.get("zo_target_ohm", 50.0))
    tol_curr   = float(st.session_state.get("zo_tol_current_ohm", 5.0))
    tol_prop_max = float(st.session_state.get("zo_tol_proposed_max_ohm", tol_curr))
    tol_prop_min = float(st.session_state.get("zo_tol_proposed_min_ohm", tol_curr))

    # P1 & P2
    for end in ("P1", "P2"):
        # ---- MAX ----
        df_max_s = _collect_zo_values_with_serial(
            st.session_state["cables"], category=cat_key, metric="max", end=end
        )
        if df_max_s is not None and not df_max_s.empty:
            vals = pd.to_numeric(df_max_s["value"], errors="coerce")
            ser  = df_max_s["serial"]
            m    = vals.notna() & ser.notna()
            vals, ser = vals[m], ser[m]
            dev = (vals - target_ohm).abs()
            ok_curr = (dev <= tol_curr).groupby(ser).all()
            ok_prop = (dev <= tol_prop_max).groupby(ser).all()
            pass_curr_list_max.append(ok_curr)
            pass_prop_list_max.append(ok_prop)

        # ---- MIN ----
        df_min_s = _collect_zo_values_with_serial(
            st.session_state["cables"], category=cat_key, metric="min", end=end
        )
        if df_min_s is not None and not df_min_s.empty:
            vals = pd.to_numeric(df_min_s["value"], errors="coerce")
            ser  = df_min_s["serial"]
            m    = vals.notna() & ser.notna()
            vals, ser = vals[m], ser[m]
            dev = (vals - target_ohm).abs()
            ok_curr = (dev <= tol_curr).groupby(ser).all()
            ok_prop = (dev <= tol_prop_min).groupby(ser).all()
            pass_curr_list_min.append(ok_curr)
            pass_prop_list_min.append(ok_prop)

    # AND across ends for the current category
    max_curr_all = _reduce_and(pass_curr_list_max)
    max_prop_all = _reduce_and(pass_prop_list_max)
    min_curr_all = _reduce_and(pass_curr_list_min)
    min_prop_all = _reduce_and(pass_prop_list_min)

    def _counts(sr: pd.Series):
        total = int(sr.size)
        passed = int(sr.sum())
        return total, passed, total - passed

    total_max, pass_max_curr, fail_max_curr = _counts(max_curr_all)
    _,        pass_max_prop, fail_max_prop  = _counts(max_prop_all)
    total_min, pass_min_curr, fail_min_curr = _counts(min_curr_all)
    _,        pass_min_prop, fail_min_prop  = _counts(min_prop_all)

    return pd.DataFrame([
        {
            "Metric": "Zo Max",
            "Current Spec":  f"Target {target_ohm} Ω, ±{tol_curr} Ω",
            "Proposed Spec": f"Target {target_ohm} Ω, ±{tol_prop_max} Ω",
            "Cables — Pass (Current)": pass_max_curr,
            "Cables — Fail (Current)": fail_max_curr,
            "Cables — Pass (Proposed)": pass_max_prop,
            "Cables — Fail (Proposed)": fail_max_prop,
            "Cables — Total": total_max,
        },
        {
            "Metric": "Zo Min",
            "Current Spec":  f"Target {target_ohm} Ω, ±{tol_curr} Ω",
            "Proposed Spec": f"Target {target_ohm} Ω, ±{tol_prop_min} Ω",
            "Cables — Pass (Current)": pass_min_curr,
            "Cables — Fail (Current)": fail_min_curr,
            "Cables — Pass (Proposed)": pass_min_prop,
            "Cables — Fail (Proposed)": fail_min_prop,
            "Cables — Total": total_min,
        },
    ])


def _skew_exec_summary_for_category(cat_key: str) -> pd.DataFrame:
    """
    Category-aware skew summary (AND across ends) for the current category tab.
    Returns a single-row DataFrame for Δ Skew.
    """
    curr_ps = float(st.session_state.get("skew_current_spec_ps", 50.0))
    prop_ps = float(st.session_state.get("skew_proposed_ps", curr_ps))

    per_end_curr, per_end_prop = [], []

    for end in ("P1", "P2"):
        df_s = _collect_skew_delta_with_serial(
            st.session_state["cables"], end=end, category=cat_key
        )
        if df_s is None or df_s.empty:
            continue

        vals = pd.to_numeric(df_s["value_ps"], errors="coerce").abs()
        ser  = df_s["serial"]
        m    = vals.notna() & ser.notna()
        vals, ser = vals[m], ser[m]

        ok_curr = (vals <= curr_ps).groupby(ser).all()
        ok_prop = (vals <= prop_ps).groupby(ser).all()
        per_end_curr.append(ok_curr)
        per_end_prop.append(ok_prop)

    curr_all = _reduce_and(per_end_curr)
    prop_all = _reduce_and(per_end_prop)

    total     = int(curr_all.size)
    pass_curr = int(curr_all.sum())
    fail_curr = total - pass_curr
    pass_prop = int(prop_all.sum())
    fail_prop = total - pass_prop

    return pd.DataFrame([{
        "Metric": "Δ Skew",
        "Current Spec":  f"±{curr_ps} pS",
        "Proposed Spec": f"±{prop_ps} pS",
        "Cables — Pass (Current)": pass_curr,
        "Cables — Fail (Current)": fail_curr,
        "Cables — Pass (Proposed)": pass_prop,
        "Cables — Fail (Proposed)": fail_prop,
        "Cables — Total": total,
    }])

# ---------------------- Zo Histograms (Combined P1+P2) ----------------------

tabs = st.tabs([label for _, label in categories])
for (cat_key, cat_label), tab in zip(categories, tabs):
    with tab:

        st.write(f"**Category:** {cat_label}")
        # ---------- Category-aware Executive Summary (inside tab) ----------
        st.markdown("### Executive Summary")
        zo_prop_df = pd.DataFrame([
    {
        "Metric": "Zo Max",
        "Target (Ω)": zo_target_ohm,
        "Current Tol (Ω)": zo_tol_curr_ohm,
        "Proposed Tol (Ω)": float(st.session_state["zo_tol_proposed_max_ohm"]),
    },
    {
        "Metric": "Zo Min",
        "Target (Ω)": zo_target_ohm,
        "Current Tol (Ω)": zo_tol_curr_ohm,
        "Proposed Tol (Ω)": float(st.session_state["zo_tol_proposed_min_ohm"]),
    },
])
        zo_prop_edited = st.data_editor(
            zo_prop_df,
            key=f"zo_prop_editor_{cat_key}",
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(disabled=True),
                "Target (Ω)": st.column_config.NumberColumn(format="%.3f", disabled=True),
                "Current Tol (Ω)": st.column_config.NumberColumn(format="%.3f", disabled=True),
                "Proposed Tol (Ω)": st.column_config.NumberColumn(format="%.3f", min_value=0.0, max_value=50.0, step=0.5),
            },
        )
        try:
            st.session_state["zo_tol_proposed_max_ohm"] = float(
                zo_prop_edited.loc[zo_prop_edited["Metric"] == "Zo Max", "Proposed Tol (Ω)"].iloc[0]
            )
            st.session_state["zo_tol_proposed_min_ohm"] = float(
                zo_prop_edited.loc[zo_prop_edited["Metric"] == "Zo Min", "Proposed Tol (Ω)"].iloc[0]
            )
        except Exception:
            pass

        # Skew Proposed editor
        skew_prop_df = pd.DataFrame([{
            "Metric": "Δ Skew",
            "Current Spec (pS)": curr_skew_ps,
            "Proposed Spec (pS)": float(st.session_state["skew_proposed_ps"]),
        }])
        skew_prop_edited = st.data_editor(
            skew_prop_df,
            key=f"skew_prop_editor_{cat_key}",
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(disabled=True),
                "Current Spec (pS)": st.column_config.NumberColumn(format="%.0f", disabled=True),
                "Proposed Spec (pS)": st.column_config.NumberColumn(format="%.0f", min_value=0.0, max_value=100000.0, step=1.0),
            },
        )
        try:
            st.session_state["skew_proposed_ps"] = float(skew_prop_edited.at[0, "Proposed Spec (pS)"])
        except Exception:
            pass
        zo_cat_df   = _zo_exec_summary_for_category(cat_key)
        if zo_cat_df is None or zo_cat_df.empty:
            st.info("No Zo data for this category.")
        else:
            st.dataframe(zo_cat_df, use_container_width=True)
        skew_cat_df = _skew_exec_summary_for_category(cat_key)

        if skew_cat_df is None or skew_cat_df.empty:
            st.info("No Δ Skew data for this category.")
        else:
            st.dataframe(skew_cat_df, use_container_width=True)

        st.divider()
        st.subheader("Zo Histograms (P1 + P2 Combined)")


        # Per-cable arrays for BOTH ends (Max)
        df_max_p1 = collect_zo_max_values_df(st.session_state["cables"], category=cat_key, metric="max", end="P1")
        df_max_p1["end"] = "P1"
        df_max_p2 = collect_zo_max_values_df(st.session_state["cables"], category=cat_key, metric="max", end="P2")
        df_max_p2["end"] = "P2"
        df_max_both = pd.concat([df_max_p1, df_max_p2], ignore_index=True).rename(columns={"value": "max_value_ohm"})

        # Per-cable arrays for BOTH ends (Min)
        df_min_p1 = collect_zo_min_values_df(st.session_state["cables"], category=cat_key, metric="min", end="P1")
        df_min_p1["end"] = "P1"
        df_min_p2 = collect_zo_min_values_df(st.session_state["cables"], category=cat_key, metric="min", end="P2")
        df_min_p2["end"] = "P2"
        df_min_both = pd.concat([df_min_p1, df_min_p2], ignore_index=True).rename(columns={"value": "min_value_ohm"})

        cmax, cmin = st.columns(2)
        with cmax:
            st.markdown(f"##### {cat_label} — Max Impedance (Ω)")
            if df_max_both.empty or df_max_both["max_value_ohm"].dropna().empty:
                st.info("No Max data across P1/P2")
            else:
                fig = px.histogram(
                    df_max_both, x="max_value_ohm", color="end",
                    nbins=30, barmode="overlay", opacity=0.60,
                    title=f"{cat_label} — Max Impedance (P1 + P2)",
                )
                fig.update_layout(
                    xaxis_title="Max Impedance (Ω)", yaxis_title="Count",
                    margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                    legend_title_text="End",
                )
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

        with cmin:
            st.markdown(f"##### {cat_label} — Min Impedance (Ω)")
            if df_min_both.empty or df_min_both["min_value_ohm"].dropna().empty:
                st.info("No Min data across P1/P2")
            else:
                fig = px.histogram(
                    df_min_both, x="min_value_ohm", color="end",
                    nbins=30, barmode="overlay", opacity=0.60,
                    title=f"{cat_label} — Min Impedance (P1 + P2)",
                )
                fig.update_layout(
                    xaxis_title="Min Impedance (Ω)", yaxis_title="Count",
                    margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                    legend_title_text="End",
                )
                fig.update_xaxes(range=[0, None])
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Skew Histograms (Δ skew [pS], P1 + P2 Combined)")

        df_delta_p1 = _collect_skew_delta(st.session_state["cables"], end="P1", category=cat_key).rename(columns={"value_ps": "delta_ps"})
        df_delta_p1["end"] = "P1"
        df_delta_p2 = _collect_skew_delta(st.session_state["cables"], end="P2", category=cat_key).rename(columns={"value_ps": "delta_ps"})
        df_delta_p2["end"] = "P2"
        df_delta_both = pd.concat([df_delta_p1, df_delta_p2], ignore_index=True)

        if df_delta_both.empty or df_delta_both["delta_ps"].dropna().empty:
            st.info("No Δ skew data across P1/P2")
        else:
            fig = px.histogram(
                df_delta_both, x="delta_ps", color="end",
                nbins=30, barmode="overlay", opacity=0.6,
                title=f"Δ Skew (pS) — {cat_label} — P1 + P2",
            )
            fig.update_layout(
                xaxis_title="Δ skew [pS]", yaxis_title="Count",
                margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                legend_title_text="End",
            )
            fig.update_xaxes(range=[0, None])
            st.plotly_chart(fig, use_container_width=True)

            # Cable-level counts across both ends
            df_delta_s_p1 = _collect_skew_delta_with_serial(st.session_state["cables"], end="P1", category=cat_key)
            df_delta_s_p2 = _collect_skew_delta_with_serial(st.session_state["cables"], end="P2", category=cat_key)
            df_delta_s = pd.concat([df_delta_s_p1, df_delta_s_p2], ignore_index=True)

            if df_delta_s is None or df_delta_s.empty:
                st.info("No cable-level skew data available for this category.")
            else:
                current_spec_ps  = float(st.session_state.get("skew_current_spec_ps", 50.0))
                proposed_spec_ps = float(st.session_state.get("skew_proposed_ps", current_spec_ps))

                # Optional per-tab override input (unique key per category)
                proposed_skew_ps_local = st.number_input(
                    f"Proposed tolerance [pS] — {cat_label}",
                    key=f"skew_tol_prop_combined_ps_{cat_key}",
                    min_value=0.0,
                    max_value=100000.0,
                    value=proposed_spec_ps,
                    step=1.0,
                )

                c_counts = _skew_counts_by_cable(
                    df_delta_s,
                    curr_ps=current_spec_ps,
                    prop_ps=float(proposed_skew_ps_local),
                    value_col="value_ps",
                    serial_col="serial",
                    per_cable_rule="all_channels",
                )
                mcol1, mcol2, mcol3, mcol4 = st.columns([1,1,1,1])
                mcol1.metric("Cables — Total", c_counts["total"])
                mcol2.metric("Pass (Current)", c_counts["pass_current"])
                mcol3.metric("Fail (Current)", c_counts["fail_current"])
                mcol4.metric("Fail (Proposed)", c_counts["fail_prop"])

# ---------------------- Combined Downloads ----------------------
st.divider()
st.subheader("Download Combined CSVs (P1 + P2)")

# Zo Combined Downloads
zo_categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
zo_metrics = [("max", "Max"), ("min", "Min")]

for cat_key, cat_label in zo_categories:
    with st.expander(f"Zo — {cat_label} (Combined P1 + P2)", expanded=False):
        for metric_key, metric_label in zo_metrics:
            wide_p1 = build_zo_wide_matrix(st.session_state["cables"], category=cat_key, metric=metric_key, end="P1")
            wide_p1["End"] = "P1"
            wide_p2 = build_zo_wide_matrix(st.session_state["cables"], category=cat_key, metric=metric_key, end="P2")
            wide_p2["End"] = "P2"
            wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

            st.markdown(f"**{cat_label} — {metric_label} Impedance (Ω)**")
            if wide_both.empty:
                st.caption("No data available.")
            else:
                st.dataframe(wide_both.head(8), use_container_width=True)
                st.download_button(
                    label=f"Download {cat_label} {metric_label} (Combined P1+P2)",
                    data=wide_both.to_csv(index=False),
                    file_name=f"Zo_{cat_label}_{metric_label}_Combined_P1P2.csv",
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

with st.expander("Skew Δ (pS) — Combined P1 + P2", expanded=False):
    wide_p1 = build_skew_wide_matrix(st.session_state["cables"], end="P1")
    wide_p1["End"] = "P1"
    wide_p2 = build_skew_wide_matrix(st.session_state["cables"], end="P2")
    wide_p2["End"] = "P2"
    wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

    if wide_both.empty:
        st.caption("No skew Δ data.")
    else:
        st.dataframe(wide_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Δ (Combined P1+P2)",
            data=wide_both.to_csv(index=False),
            file_name="Skew_Delta_Combined_P1P2.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Skew Overall From Pair Cols Combined
with st.expander("Skew Overall from Pair Columns — Combined P1 + P2", expanded=False):
    long_p1 = _collect_skew_overall_from_paircols_per_test(st.session_state["cables"], end="P1")
    long_p1["End"] = "P1"
    long_p2 = _collect_skew_overall_from_paircols_per_test(st.session_state["cables"], end="P2")
    long_p2["End"] = "P2"
    long_both = pd.concat([long_p1, long_p2], ignore_index=True)

    wide_p1 = build_skew_overall_from_paircols_wide(st.session_state["cables"], end="P1")
    wide_p1["End"] = "P1"
    wide_p2 = build_skew_overall_from_paircols_wide(st.session_state["cables"], end="P2")
    wide_p2["End"] = "P2"
    wide_both = pd.concat([wide_p1, wide_p2], ignore_index=True)

    if long_both.empty:
        st.caption("No overall skew data.")
    else:
        st.markdown("**Preview (LONG Format)**")
        st.dataframe(long_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Overall (Combined P1+P2) — LONG",
            data=long_both.to_csv(index=False),
            file_name="Skew_Overall_Combined_P1P2_LONG.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Preview (WIDE Format)**")
        st.dataframe(wide_both.head(8), use_container_width=True)
        st.download_button(
            label="Download Skew Overall (Combined P1+P2) — WIDE",
            data=wide_both.to_csv(index=False),
            file_name="Skew_Overall_Combined_P1P2_WIDE.csv",
            mime="text/csv",
            use_container_width=True,
        )
# ---------------------- Trace Upload & Overlay (Combined P1+P2) ----------------------
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

from itertools import cycle
unique_palette = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Pastel
)
# Prevent extremely long legends from reusing the same first few colors too soon
color_cycle = cycle(unique_palette)

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

    # Reset a color cycle **per cable** so each cable starts at the same palette
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