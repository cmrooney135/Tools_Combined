import streamlit as st 

import plotly.express as px
from cable_factory import create_cable
from SI_Test import SI_Test
from dataclasses import dataclass, field
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from Test import Test
from Cable import Cable
import streamlit as st
from _shared_ui import top_bar
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import re
from typing import Optional, Dict


import re
from typing import Optional, Dict
import re
from typing import Optional, Dict
from UploadSIData import process_SI_file
def ensure_state():
    # Lists to hold your domain objects
    st.session_state.setdefault("cables", [])       # list[ Cable ]
    st.session_state.setdefault("tests", [])        # optional list[ Test ]

    # Files you’ve already processed (use a list for Streamlit friendliness)
    st.session_state.setdefault("processed_files", [])

    # Any caches for plots/maps you use later in the file
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
    
    st.session_state.setdefault("unattached_traces", [])
     
    st.session_state.setdefault("processed_trace_files", []) 
# At top of file (near other imports):
from dataclasses import Field as DataclassField


st.set_page_config(page_title="SI Tools", page_icon = "🐍", layout="wide")

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
            # If legacy: Field, None, or anything non-dict → reset to {}
            if not isinstance(val, dict):
                try:
                    # If it's a Mapping-like object, try to coerce
                    t.traces = dict(val) if val is not None else {}
                except Exception:
                    t.traces = {}

migrate_si_tests_traces()
ensure_state()
top_bar(page_icon="🐍", title="🏝️SI Tools", home_page_path="Home.py")



def _ns_to_ps_series(s):
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
    # Create a new cable shell
    cable = create_cable(serial)
    cables.append(cable)
    return cable

def get_or_create_si_test_for_end(cable: Cable, end: str) -> Test:
    """Find the latest SI test for end (P1/P2) or create one and append."""
    si_list = getattr(cable, "TESTS", {}).get("si", []) or []
    matches = [t for t in si_list if (getattr(t, "test_end", "") or "").upper() == end.upper()]
    if matches:
        return matches[-1]
    # Create a new SI test shell
    test = SI_Test(
                test_type="si",
                test_end=end,
            )
    # Ensure list exists in TESTS dict
    if "si" not in cable.TESTS or cable.TESTS["si"] is None:
        cable.TESTS["si"] = []
    cable.TESTS["si"].append(test)
    return test
def validate_parsed(info: Dict[str, str]) -> None:
    """
    Raise ValueError if any piece is malformed.
    - serial: alphanumeric (10–12 typical; we won't hard fail on length)
    - channel: Letter + 2 digits (e.g., A01, B12)
    - end: P1 or P2
    """
    if not info:
        raise ValueError("Could not parse filename.")
    if not re.fullmatch(r"[A-Z0-9]+", info["serial"]):
        raise ValueError(f"Invalid serial: {info['serial']}")
    if not re.fullmatch(r"[A-Z]\d{2}", info["channel"]):
        raise ValueError(f"Invalid channel: {info['channel']}")
    if info["end"] not in ("P1", "P2"):
        raise ValueError(f"Invalid end: {info['end']}")
    

def _find_two_numeric_pre_skew_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Try to find the two numeric columns that precede the 'skew' marker column.
    Expected layout (example):
      [pair_label, v1, v2, 'skew [nS]', delta, result]
    Strategy:
      1) Find column whose name contains 'skew' (case-insensitive).
      2) Take the two columns immediately to its left (if present).
      3) Validate they are numeric (or coercible).
      4) Fallback: choose the first two numeric-looking columns after the first column.
    Returns (col_v1, col_v2) or (None, None) if not found.
    """
    cols = list(df.columns)

    # Heuristic 1: locate the 'skew' column by name
    skew_idx = None
    for i, c in enumerate(cols):
        cname = str(c).strip().lower()
        if "skew" in cname:
            skew_idx = i
            break

    # If found, pick two columns to the left of skew column
    if skew_idx is not None and skew_idx >= 2:
        c1 = cols[skew_idx - 2]
        c2 = cols[skew_idx - 1]

        # Check numeric-ness (allow coercion)
        s1 = pd.to_numeric(df[c1], errors="coerce")
        s2 = pd.to_numeric(df[c2], errors="coerce")
        if s1.notna().sum() > 0 and s2.notna().sum() > 0:
            return c1, c2

    # Heuristic 2: fallback—pick the first two numeric columns ignoring the leftmost label-like column
    # Often col0 is the "J2.A01 / J2.A02" label—so start from col1
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
    Returns long DataFrame:
      ['Serial','End','Label','Overall Max (pS)','Overall Min (pS)','Span (pS)']
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

        # Convert to pS
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
    """
    Wide matrix with rows = ['Overall Max','Overall Min','Span'] (pS)
    and columns = '<Serial> <M/D/YYYY H:MM>' per test.
    """
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

def attach_trace_by_serial_end(cables, df: pd.DataFrame, serial: str, end: str, trace_name: str) -> bool:
    """
    Find cable by serial_number, then attach the trace to the latest SI test for the given end (P1/P2).
    Returns True if successful, False otherwise.
    """
    # 1) Find cable
    cable = next((c for c in cables if getattr(c, "serial_number", None) == serial), None)
    if cable is None:
        return False

    # 2) Find latest SI test with matching end
    si_tests = getattr(cable, "TESTS", {}).get("si", []) or []
    candidates = [t for t in si_tests if (getattr(t, "test_end", "") or "").upper() == end.upper()]
    if not candidates:
        return False

    latest = candidates[-1]
    latest.traces[trace_name] = df
    return True
def attach_trace_to_latest_test(cable, df: pd.DataFrame, trace_name: str, end: str):
    si_tests = getattr(cable, "TESTS", {}).get("si", []) or []
    candidates = [t for t in si_tests if (getattr(t, "test_end", "") or "").upper() == end.upper()]
    if not candidates:
        return False
    latest = candidates[-1]
    if not isinstance(latest.traces, dict):
        latest.traces = {}
    latest.traces[trace_name] = df
    return True

def _iter_si_tests(cables):
    """Yield (cable, test) for each SI test."""
    for c in cables:
        for t in getattr(c, "TESTS", {}).get("si", []) or []:
            yield c, t

# ---- timestamp + column labeling ----
def _format_col_label(serial: str, date_str: str | None, time_str: str | None) -> str:
    """
    Build '<Serial> <M/D/YYYY H:MM>' if date/time present;
    else '<Serial> <no time>'.
    """
    serial = (serial or "").strip()
    if date_str and time_str:
        # date_str: 'YYYY-MM-DD'; time_str: 'HH:MM:SS'
        try:
            y, mth, d = map(int, date_str.split("-"))
            hh, mm, _ss = map(int, time_str.split(":"))
            label = f"{serial} {mth}/{d}/{y} {hh}:{mm:02d}"  # e.g., 1/27/2026 5:43
        except Exception:
            label = f"{serial} {date_str} {time_str}"
    else:
        label = f"{serial} <no time>"
    return label

def _dedup_label(label: str, existing: set[str]) -> str:
    """Ensure column names are unique by appending _N if needed."""
    if label not in existing:
        return label
    k = 2
    while f"{label}_{k}" in existing:
        k += 1
    return f"{label}_{k}"

# ---- build wide matrices for ZO ----
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

        # series of metric, indexed by channel (keep channels "as-is")
        s = pd.to_numeric(df[metric], errors="coerce")
        s.index = df["channel"]
        cols[label] = s

    if not cols:
        return pd.DataFrame(columns=["Channels"])

    wide = pd.DataFrame(cols)
    # Insert Channels as first column
    wide.insert(0, "Channels", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide
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

        s_ps = _ns_to_ps_series(df[d_col])  # nS -> pS
        s_ps.index = df[ch_col]
        cols[label] = s_ps

    if not cols:
        return pd.DataFrame(columns=["Channels"])

    wide = pd.DataFrame(cols)
    wide.insert(0, "Channels", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide
import numpy as np
import pandas as pd

def _collect_skew_delta(cables, end: str) -> pd.DataFrame:
    """
    For the given end ('P1' or 'P2'), compute the maximum skew 'delta' per cable
    from SI_Test.skew_data, returning a one-column DataFrame 'value_ps'.

    Behavior:
      - One row per cable (in the first-seen order from _iter_si_tests).
      - Combines multiple tests per cable via max.
      - Interprets 'delta' in nS and converts to pS (via _ns_to_ps_series).
      - If a cable has no valid 'delta' values, its entry is NaN.
    """
    def _key_for_cable(cable):
        # Create a stable grouping key for potentially unhashable cable objects
        for attr in ("serial", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)  # last resort, should still be stable within a run

    per_cable_max = {}  # key -> float (ps); initialize to NaN
    order = []          # preserve first-seen order of distinct cables

    for cable, test in _iter_si_tests(cables):
        key = _key_for_cable(cable)

        # Ensure every cable appears even if current test doesn't match
        if key not in per_cable_max:
            per_cable_max[key] = np.nan
            order.append(key)

        # Only process the requested end
        if getattr(test, "test_end", "").upper() != end.upper():
            continue

        # Pull skew_data DataFrame
        df = getattr(test, "skew_data", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Find a 'delta' column (case/space-insensitive)
        colname = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
        if not colname:
            continue

        # Convert nS -> pS and drop NaNs
        series_ps = _ns_to_ps_series(df[colname]).dropna()
        if series_ps.empty:
            continue

        # Max for this test, then aggregate with the cable's running max
        cur = series_ps.max()
        prev = per_cable_max[key]
        per_cable_max[key] = cur if np.isnan(prev) else max(prev, cur)

    # Build a one-column DataFrame in preserved order
    values = [per_cable_max[k] for k in order]
    return pd.DataFrame({"value_ps": values})


def _collect_zo_values(cables, category: str, metric: str, end: str) -> pd.DataFrame:
    """
    Collect values for a given Zo category ('paddleboard'|'cable'|'dib'),
    metric ('max'|'min'), and end ('P1'|'P2').
    Returns DataFrame with one column 'value'.
    """
    values = []
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
                values.extend(col.tolist())
    return pd.DataFrame({"value": values})
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from collections import OrderedDict
import numpy as np
import pandas as pd
def collect_zo_min_values_df(
    cables,
    category: str,
    metric: str,
    end: str,
    key_fn=None,
) -> pd.DataFrame:
    """
    Return a one-column DataFrame ('value') with the maximum Zo value per cable.
    - One row per cable (in first-seen order from _iter_si_tests).
    - Multiple matching tests per cable are reduced via max.
    - Cables with no valid values produce NaN.

    Parameters
    ----------
    cables : iterable
        Input consumed by _iter_si_tests(cables), which yields (cable, test).
    category : str
        e.g. 'paddleboard' | 'cable' | 'dib'
    metric : str
        e.g. 'max' | 'min' (column inside zo_data[category])
    end : str
        e.g. 'P1' | 'P2'
    key_fn : callable(cable) -> hashable, optional
        Provides a stable ID for grouping per cable. If None, a default tries
        cable.serial, cable.name, cable.id, else falls back to str(cable).

    Returns
    -------
    pd.DataFrame
        One column named 'value', each row = maximum per cable (float).
    """

    def default_key_fn(cable):
        for attr in ("serial", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)  # last resort, must be stable across occurrences

    key_fn = key_fn or default_key_fn

    per_cable_min = {}  # key -> float (nan initially)
    order = []          # preserve first-seen order of distinct cables

    for cable, test in _iter_si_tests(cables):
        key = key_fn(cable)

        # Ensure cable appears (even if this test won't match)
        if key not in per_cable_min:
            per_cable_min[key] = np.nan
            order.append(key)

        # Filter by end
        if getattr(test, "test_end", "").upper() != end.upper():
            continue

        # Pull Zo table
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

    # Build a one-column DataFrame in the preserved order
    values = [per_cable_min[k] for k in order]
    return pd.DataFrame({"value": values})

def collect_zo_max_values_df(
    cables,
    category: str,
    metric: str,
    end: str,
    key_fn=None,
) -> pd.DataFrame:
    """
    Return a one-column DataFrame ('value') with the maximum Zo value per cable.
    - One row per cable (in first-seen order from _iter_si_tests).
    - Multiple matching tests per cable are reduced via max.
    - Cables with no valid values produce NaN.

    Parameters
    ----------
    cables : iterable
        Input consumed by _iter_si_tests(cables), which yields (cable, test).
    category : str
        e.g. 'paddleboard' | 'cable' | 'dib'
    metric : str
        e.g. 'max' | 'min' (column inside zo_data[category])
    end : str
        e.g. 'P1' | 'P2'
    key_fn : callable(cable) -> hashable, optional
        Provides a stable ID for grouping per cable. If None, a default tries
        cable.serial, cable.name, cable.id, else falls back to str(cable).

    Returns
    -------
    pd.DataFrame
        One column named 'value', each row = maximum per cable (float).
    """

    def default_key_fn(cable):
        for attr in ("serial", "name", "id"):
            if hasattr(cable, attr):
                val = getattr(cable, attr)
                if val is not None:
                    return val
        return str(cable)  # last resort, must be stable across occurrences

    key_fn = key_fn or default_key_fn

    per_cable_max = {}  # key -> float (nan initially)
    order = []          # preserve first-seen order of distinct cables

    for cable, test in _iter_si_tests(cables):
        key = key_fn(cable)

        # Ensure cable appears (even if this test won't match)
        if key not in per_cable_max:
            per_cable_max[key] = np.nan
            order.append(key)

        # Filter by end
        if getattr(test, "test_end", "").upper() != end.upper():
            continue

        # Pull Zo table
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

    # Build a one-column DataFrame in the preserved order
    values = [per_cable_max[k] for k in order]
    return pd.DataFrame({"value": values})

def _histogram(values_df: pd.DataFrame, title: str, x_label: str = "Impedance"):
    if values_df.empty:
        st.info(f"No data for {title}")
        return
    fig = px.histogram(values_df, x="value", nbins=30, title=title)
    fig.update_layout(
        bargap=0.05,
        xaxis_title=x_label,
        yaxis_title="Count",
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

uploaded_files = st.file_uploader("Upload your SI files", type="DAT", accept_multiple_files=True)

cables = st.session_state["cables"]
processed = st.session_state["processed_files"]

uploaded_trace_csvs = st.file_uploader(
    "Upload your Trace CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="trace_uploader"
)

if uploaded_files:
    for uf in uploaded_files:
        if uf.name in processed:
            continue

        cable, test = process_SI_file(uf, cables)
    
        if cable is not None:
            # Optional: avoid duplicates by serial_number
            exists = next((c for c in st.session_state["cables"]
                           if getattr(c, "serial_number", None) == cable.serial_number), None)
            if exists is None:
                st.session_state["cables"].append(cable)

            processed.append(uf.name)

        st.session_state["processed_files"] = list(processed)

    #now there is a collection of SI test objects with dataframes inside 
    #i want to make a histogram with maximum impedance, minimum impedance , for each paddleboard, cable, dib
    # do the same thing for P1 and P2 if they are present 

      # --- Zo Spec Controls (place near the top or sidebar) ---

# Default/spec persistence
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
    st.session_state["skew_current_spec_ps"] = 23
if "skew_tol_proposed" not in st.session_state:
    st.session_state["skew_tol_proposed"] = 23

zo_target_ohm = 50.0

zo_tol_curr_ohm = 5
# --- Helpers to collect values WITH serials ---

def _collect_zo_values_with_serial(cables, category: str, metric: str, end: str) -> pd.DataFrame:
    """
    Return rows with columns ['serial', 'value'] for the requested Zo slice.
    """
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


def _collect_skew_delta_with_serial(cables, end: str) -> pd.DataFrame:
    """
    Return rows with columns ['serial', 'value_ps'] for Δ skew (converted to pS).
    """
    rows = []
    for cable, test in _iter_si_tests(cables):
        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            dcol = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
            if dcol:
                s = pd.to_numeric(df[dcol], errors="coerce").dropna() * 1000.0  # nS -> pS
                if not s.empty:
                    sn = getattr(cable, "serial_number", "?")
                    rows.extend({"serial": sn, "value_ps": float(v)} for v in s.tolist())
    return pd.DataFrame(rows, columns=["serial", "value_ps"])
def _zo_counts_by_cable(
    df: pd.DataFrame,
    target: float,
    tol_curr: float,
    tol_prop: float,
    value_col: str = "value",
    serial_col: str = "serial",
    per_cable_rule: str = "all_channels",  # or "any_channel"
):
    """
    Count pass/fail at the **cable** level for Zo using target ± tolerance.

    Returns: dict with counts of cables (not channels).
    """
    # Basic guards
    if df is None or df.empty or value_col not in df.columns or serial_col not in df.columns:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    # Clean values
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

    total_cables = int(pass_curr_by_cable.size)  # number of cables present
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
    per_cable_rule: str = "all_channels",  # or "any_channel"
) -> dict:
    """
    Count pass/fail at the **cable** level for skew, using absolute Δ skew thresholds.

    Cable pass rule:
      - 'all_channels': cable passes if ALL its channels meet |Δ skew| <= threshold
      - 'any_channel' : cable passes if ANY of its channels meets the threshold
    """
    if df is None or df.empty or value_col not in df.columns or serial_col not in df.columns:
        return dict(total=0, pass_current=0, fail_current=0, pass_prop=0, fail_prop=0)

    vals = pd.to_numeric(df[value_col], errors="coerce")
    ser = df[serial_col]
    mask = vals.notna() & ser.notna()
    vals = vals[mask].abs()  # absolute Δ skew in pS
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

#add two executive summary tables here : 
#impedance maximum and minimum 
# skew change 
# ===================== EXECUTIVE SUMMARY =====================
st.divider()
st.subheader("Executive Summary (Cable-Level)")

# --- Config ---
categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
ends = ["P1", "P2"]

# ---------------- Proposed Spec Editors (editable) ----------------
# Initialize proposed values in session_state once
st.session_state.setdefault(
    "zo_tol_proposed_max_ohm",
    float(st.session_state.get("zo_tol_current_ohm", 5.0))
)
st.session_state.setdefault(
    "zo_tol_proposed_min_ohm",
    float(st.session_state.get("zo_tol_current_ohm", 5.0))
)
st.session_state.setdefault(
    "skew_proposed_ps",
    float(st.session_state.get("skew_current_spec_ps", 50.0))
)

# Read current values (already defined elsewhere in your app)
zo_target_ohm   = float(st.session_state.get("zo_target_ohm", 50.0))
zo_tol_curr_ohm = float(st.session_state.get("zo_tol_current_ohm", 5.0))
curr_skew_ps    = float(st.session_state.get("skew_current_spec_ps", 50.0))

st.divider()
st.subheader("Proposed Specs (Editable)")

# ---- Table A: Impedance (Zo) Proposed Tolerances ----
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
    key="zo_prop_editor",
    num_rows="fixed",
    hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(disabled=True),
        "Target (Ω)": st.column_config.NumberColumn(format="%.3f", disabled=True),
        "Current Tol (Ω)": st.column_config.NumberColumn(format="%.3f", disabled=True),
        "Proposed Tol (Ω)": st.column_config.NumberColumn(
            format="%.3f", min_value=0.0, max_value=50.0, step=0.5
        ),
    },
)

# Persist edited Zo proposed tolerances (guard if user temporarily clears cells)
try:
    st.session_state["zo_tol_proposed_max_ohm"] = float(
        zo_prop_edited.loc[zo_prop_edited["Metric"] == "Zo Max", "Proposed Tol (Ω)"].iloc[0]
    )
    st.session_state["zo_tol_proposed_min_ohm"] = float(
        zo_prop_edited.loc[zo_prop_edited["Metric"] == "Zo Min", "Proposed Tol (Ω)"].iloc[0]
    )
except Exception:
    pass

# ---- Table B: Skew (Δ) Proposed Spec ----
skew_prop_df = pd.DataFrame([
    {
        "Metric": "Δ Skew",
        "Current Spec (pS)": curr_skew_ps,
        "Proposed Spec (pS)": float(st.session_state["skew_proposed_ps"]),
    }
])

skew_prop_edited = st.data_editor(
    skew_prop_df,
    key="skew_prop_editor",
    num_rows="fixed",
    hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn(disabled=True),
        "Current Spec (pS)": st.column_config.NumberColumn(format="%.0f", disabled=True),
        "Proposed Spec (pS)": st.column_config.NumberColumn(
            format="%.0f", min_value=0.0, max_value=100000.0, step=1.0
        ),
    },
)

# Persist edited Skew proposed spec
try:
    st.session_state["skew_proposed_ps"] = float(skew_prop_edited.at[0, "Proposed Spec (pS)"])
except Exception:
    pass

st.divider()

# ---------------- Build the two Executive Summary tables ----------------
# Helpers (with serials) — define if not already present in your file
def _collect_zo_values_with_serial(cables, category: str, metric: str, end: str) -> pd.DataFrame:
    """
    Return rows with columns ['serial', 'value'] for Zo.
    """
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

def _collect_skew_delta_with_serial(cables, end: str) -> pd.DataFrame:
    """
    Return rows with columns ['serial', 'value_ps'] for Δ skew (converted to pS).
    """
    rows = []
    for cable, test in _iter_si_tests(cables):
        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            dcol = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
            if dcol:
                s = pd.to_numeric(df[dcol], errors="coerce").dropna() * 1000.0  # nS -> pS
                if not s.empty:
                    sn = getattr(cable, "serial_number", "?")
                    rows.extend({"serial": sn, "value_ps": float(v)} for v in s.tolist())
    return pd.DataFrame(rows, columns=["serial", "value_ps"])

def _reduce_and(series_list: list[pd.Series]) -> pd.Series:
    """Per-cable AND across slices; if a cable appears, it must pass in every slice it appears."""
    if not series_list:
        return pd.Series(dtype=bool)
    return pd.concat(series_list, axis=1).all(axis=1)

# ---- Impedance (Zo) — Cable-level Executive Summary (Max & Min)
def _zo_exec_summary_tables_tol_only():
    pass_curr_list_max, pass_prop_list_max = [], []
    pass_curr_list_min, pass_prop_list_min = [], []

    for cat_key, _cat_label in categories:
        for end in ends:
            # MAX
            df_max_s = _collect_zo_values_with_serial(st.session_state["cables"], category=cat_key, metric="max", end=end)
            if df_max_s is not None and not df_max_s.empty:
                vals = pd.to_numeric(df_max_s["value"], errors="coerce")
                ser  = df_max_s["serial"]
                m    = vals.notna() & ser.notna()
                vals, ser = vals[m], ser[m]
                dev = (vals - zo_target_ohm).abs()
                ok_curr = (dev <= zo_tol_curr_ohm).groupby(ser).all()
                ok_prop = (dev <= float(st.session_state["zo_tol_proposed_max_ohm"])).groupby(ser).all()
                pass_curr_list_max.append(ok_curr)
                pass_prop_list_max.append(ok_prop)

            # MIN
            df_min_s = _collect_zo_values_with_serial(st.session_state["cables"], category=cat_key, metric="min", end=end)
            if df_min_s is not None and not df_min_s.empty:
                vals = pd.to_numeric(df_min_s["value"], errors="coerce")
                ser  = df_min_s["serial"]
                m    = vals.notna() & ser.notna()
                vals, ser = vals[m], ser[m]
                dev = (vals - zo_target_ohm).abs()
                ok_curr = (dev <= zo_tol_curr_ohm).groupby(ser).all()
                ok_prop = (dev <= float(st.session_state["zo_tol_proposed_min_ohm"])).groupby(ser).all()
                pass_curr_list_min.append(ok_curr)
                pass_prop_list_min.append(ok_prop)

    # Reduce across every Category×End slice per metric
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

    zo_overall_df = pd.DataFrame([
        {
            "Metric": "Zo Max",
            "Current Spec": f"Target {zo_target_ohm} Ω, ±{zo_tol_curr_ohm} Ω",
            "Proposed Spec": f"Target {zo_target_ohm} Ω, ±{st.session_state['zo_tol_proposed_max_ohm']} Ω",
            "Cables — Pass (Current)": pass_max_curr,
            "Cables — Fail (Current)": fail_max_curr,
            "Cables — Pass (Proposed)": pass_max_prop,
            "Cables — Fail (Proposed)": fail_max_prop,
            "Cables — Total": total_max,
        },
        {
            "Metric": "Zo Min",
            "Current Spec": f"Target {zo_target_ohm} Ω, ±{zo_tol_curr_ohm} Ω",
            "Proposed Spec": f"Target {zo_target_ohm} Ω, ±{st.session_state['zo_tol_proposed_min_ohm']} Ω",
            "Cables — Pass (Current)": pass_min_curr,
            "Cables — Fail (Current)": fail_min_curr,
            "Cables — Pass (Proposed)": pass_min_prop,
            "Cables — Fail (Proposed)": fail_min_prop,
            "Cables — Total": total_min,
        },
    ])
    return zo_overall_df

# ---- Skew (Δ) — Cable-level Executive Summary (aggregated across ends)
def _skew_exec_summary_table_tol_only():
    per_end_curr, per_end_prop = [], []

    for end in ends:
        df_s = _collect_skew_delta_with_serial(st.session_state["cables"], end=end)
        if df_s is None or df_s.empty:
            continue
        vals = pd.to_numeric(df_s["value_ps"], errors="coerce").abs()
        ser  = df_s["serial"]
        m    = vals.notna() & ser.notna()
        vals, ser = vals[m], ser[m]
        ok_curr = (vals <= curr_skew_ps).groupby(ser).all()
        ok_prop = (vals <= float(st.session_state["skew_proposed_ps"])).groupby(ser).all()
        per_end_curr.append(ok_curr)
        per_end_prop.append(ok_prop)

    curr_all = _reduce_and(per_end_curr)
    prop_all = _reduce_and(per_end_prop)

    total     = int(curr_all.size)
    pass_curr = int(curr_all.sum())
    fail_curr = total - pass_curr
    pass_prop = int(prop_all.sum())
    fail_prop = total - pass_prop

    skew_overall_df = pd.DataFrame([{
        "Metric": "Δ Skew",
        "Current Spec": f"±{curr_skew_ps} pS",
        "Proposed Spec": f"±{st.session_state['skew_proposed_ps']} pS",
        "Cables — Pass (Current)": pass_curr,
        "Cables — Fail (Current)": fail_curr,
        "Cables — Pass (Proposed)": pass_prop,
        "Cables — Fail (Proposed)": fail_prop,
        "Cables — Total": total,
    }])
    return skew_overall_df

# ---- Render both Executive Summary tables ----
zo_overall_df   = _zo_exec_summary_tables_tol_only()
skew_overall_df = _skew_exec_summary_table_tol_only()

st.markdown("#### Impedance (Zo) — Executive Summary")
st.dataframe(zo_overall_df, use_container_width=True)

st.markdown("#### Skew (Δ) — Executive Summary")
st.dataframe(skew_overall_df, use_container_width=True)


# =================================================================
st.subheader("Zo Histograms")

categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
ends = ["P1", "P2"]  # will quietly show "No data" when an end isn’t present



tabs = st.tabs([label for _, label in categories])
for (cat_key, cat_label), tab in zip(categories, tabs):
    with tab:
        st.write(f"**Category:** {cat_label}")
        cols = st.columns(2)
        for i, end in enumerate(ends):
            with cols[i]:
                st.markdown(f"##### End: {end}")
                
                # --- Max ---
                df_max = _collect_zo_values(st.session_state["cables"], category=cat_key, metric="max", end=end)
                #switch this to be the maximum from each cable not all the maximums 

                df_max_per_cable = collect_zo_max_values_df(
                    st.session_state["cables"],
                    category=cat_key,
                    metric="max",
                    end=end,
                    # If you have a known unique attribute, uncomment and set it explicitly:
                    # key_fn=lambda c: c.serial
                )

                _histogram(df_max_per_cable, title=f"{cat_label} — {end} — Max Impedance", x_label="Max Impedance")

                # New: collect with serial for cable-level counts
                df_max_s = _collect_zo_values_with_serial(
                    st.session_state["cables"], category=cat_key, metric="max", end=end
                )

                if df_max_s is None or df_max_s.empty:
                    st.info("No Max data")
                else:
                    zo_tol_prop_ohm = st.number_input(
                        "Proposed tolerance [Ω]",
                        key=f"zo_tol_prop_{cat_key}_{end}_max",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(st.session_state["zo_tol_proposed_ohm"]),
                        step=0.5,
                    )

                    # Choose rule: "all_channels" (default) or "any_channel"
                    # You can also expose this via a widget if you want
                    counts = _zo_counts_by_cable(
                        df_max_s,
                        target=zo_target_ohm,
                        tol_curr=zo_tol_curr_ohm,
                        tol_prop=zo_tol_prop_ohm,
                        value_col="value",
                        serial_col="serial",
                        per_cable_rule="all_channels",
                    )

                    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
                    c1.metric("Cables — Total", counts["total"])
                    c2.metric("Cables — Pass (Current)", counts["pass_current"])
                    c3.metric("Cables — Fail (Current)", counts["fail_current"])
                    c4.metric("Cables — Pass (Proposed)", counts["pass_prop"])
                    c5.metric("Cables — Fail (Proposed)", counts["fail_prop"])
                    st.caption(f"Spec — Current: Target {zo_target_ohm} Ω ±{zo_tol_curr_ohm} Ω | Proposed: ±{zo_tol_prop_ohm} Ω")

                # --- Min ---
                df_min = _collect_zo_values(st.session_state["cables"], category=cat_key, metric="min", end=end)
                df_min_per_cable = collect_zo_min_values_df(
                    st.session_state["cables"],
                    category=cat_key,
                    metric="min",
                    end=end,
                    # If you have a known unique attribute, uncomment and set it explicitly:
                    # key_fn=lambda c: c.serial
                )

                _histogram(df_min_per_cable, title=f"{cat_label} — {end} — Min Impedance", x_label="Min Impedance")


                # New: collect with serial for cable-level counts
                df_min_s = _collect_zo_values_with_serial(
                    st.session_state["cables"], category=cat_key, metric="min", end=end
                )

                if df_min_s is None or df_min_s.empty:
                    st.info("No Min data")
                else:
                    zo_tol_prop_ohm = st.number_input(
                        "Proposed tolerance [Ω]",
                        key=f"zo_tol_prop_{cat_key}_{end}_min",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(st.session_state["zo_tol_proposed_ohm"]),
                        step=0.5,
                    )

                    counts = _zo_counts_by_cable(
                        df_min_s,
                        target=zo_target_ohm,
                        tol_curr=zo_tol_curr_ohm,
                        tol_prop=zo_tol_prop_ohm,
                        value_col="value",
                        serial_col="serial",
                        per_cable_rule="all_channels",
                    )

                    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
                    c1.metric("Cables — Total", counts["total"])
                    c2.metric("Cables — Pass (Current)", counts["pass_current"])
                    c3.metric("Cables — Fail (Current)", counts["fail_current"])
                    c4.metric("Cables — Pass (Proposed)", counts["pass_prop"])
                    c5.metric("Cables — Fail (Proposed)", counts["fail_prop"])
                    st.caption(f"Spec — Current: Target {zo_target_ohm} Ω ±{zo_tol_curr_ohm} Ω | Proposed: ±{zo_tol_prop_ohm} Ω")

# Global summary across all slices


st.divider()


st.subheader("Skew Histograms (Δ skew [pS])")

ends = ["P1", "P2"]
cols = st.columns(2)

for i, end in enumerate(ends):
    with cols[i]:
        df_delta = _collect_skew_delta(st.session_state["cables"], end=end)  # for histogram (channels)
        df_delta_s = _collect_skew_delta_with_serial(st.session_state["cables"], end=end)
        if df_delta.empty:
            st.info(f"No data for Skew — {end} — Δ (pS)")
        else:
            # Histogram (channels)
            fig = px.histogram(df_delta, x="value_ps", nbins=30, title=f"Maximum Skew — {end} — Δ (pS)")
            fig.update_layout(
                bargap=0.05,
                xaxis_title="Δ skew [pS]",
                yaxis_title="Count",
                margin=dict(l=10, r=10, t=40, b=10),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
            proposed_skew_ps = st.number_input(
                "Proposed tolerance [pS]",
                key=f"skew_tol_prop_{cat_key}_{end}_ps",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state["skew_tol_proposed"]),
                step=0.5,
                )

            # Specs
            current_spec_ps = st.session_state["skew_current_spec_ps"]
            proposed_spec_ps = proposed_skew_ps

            # ----- Cable-level counts -----
            if df_delta_s is None or df_delta_s.empty:
                st.info("No cable-level skew data available.")
            else:
                # Choose rule: "all_channels" (default) or "any_channel"
                cable_rule = "all_channels"

                c_counts = _skew_counts_by_cable(
                    df_delta_s,
                    curr_ps=current_spec_ps,
                    prop_ps=proposed_spec_ps,
                    value_col="value_ps",
                    serial_col="serial",
                    per_cable_rule=cable_rule,
                )

                mcol1, mcol2, mcol3, mcol4 = st.columns([1,1,1,1])
                mcol1.metric(label=f"{end} Δ skew — Cables Total", value=c_counts["total"])
                mcol2.metric(label=f"{end} Δ skew — Cables Pass (Current ±{current_spec_ps} pS)", value=c_counts["pass_current"])
                mcol3.metric(label=f"{end} Δ skew — Cables Fail (Current)", value=c_counts["fail_current"])
                mcol4.metric(label=f"{end} Δ skew — Cables Fail (Proposed ±{proposed_spec_ps} pS)", value=c_counts["fail_prop"])

                st.caption(f"Cable rule: {'All channels must pass' if cable_rule=='all_channels' else 'Any channel may pass'}")
                st.dataframe(
                    {
                        "Spec": ["Current", "Proposed"],
                        "Cable Pass": [c_counts["pass_current"], c_counts["pass_prop"]],
                        "Cable Fail": [c_counts["fail_current"], c_counts["fail_prop"]],
                        "Cables Total": [c_counts["total"], c_counts["total"]],
                    },
                    use_container_width=True
                )


      
st.divider()
st.subheader("Download CSVs")

# ZO: per category (PB/Cable/DIB) × per end (P1/P2) × per metric (Max/Min)
zo_categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]
zo_metrics = [("max", "Max"), ("min", "Min")]
ends = ["P1", "P2"]

for cat_key, cat_label in zo_categories:
    with st.expander(f"Zo — {cat_label}", expanded=False):
        cols = st.columns(2)
        for i, end in enumerate(ends):
            with cols[i]:
                st.markdown(f"**End: {end}**")
                for metric_key, metric_label in zo_metrics:
                    wide = build_zo_wide_matrix(st.session_state["cables"], category=cat_key, metric=metric_key, end=end)
                    btn_label = f"Download {cat_label} {end} {metric_label} CSV"
                    fname = f"{cat_label}_{end}_{metric_label}_wide.csv".replace(" ", "")
                    if wide.empty:
                        st.caption(f"No data for {cat_label} {end} {metric_label}")
                    else:
                        st.dataframe(wide.head(8))  # optional preview
                        st.download_button(
                            label=btn_label,
                            data=wide.to_csv(index=False),
                            file_name=fname,
                            mime="text/csv",
                            use_container_width=True
                        )

with st.expander("Skew — Δ (pS)", expanded=False):
    cols = st.columns(2)
    for i, end in enumerate(ends):
        with cols[i]:
            st.markdown(f"**End: {end}**")
            wide = build_skew_wide_matrix(st.session_state["cables"], end=end)
            btn_label = f"Download Skew {end} Δ CSV"
            fname = f"Skew_{end}_Delta_wide.csv"
            if wide.empty:
                st.caption(f"No skew data for {end}")
            else:
                st.dataframe(wide.head(8))  # optional preview
                st.download_button(
                    label=btn_label,
                    data=wide.to_csv(index=False),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True
                )
import plotly.express as px

st.divider()
st.subheader("Skew Overall from Pair Columns (Max of Col2, Min of Col3)")

ends = ["P1", "P2"]
cols = st.columns(2)

for i, end in enumerate(ends):
    with cols[i]:
        df_overall = _collect_skew_overall_from_paircols_per_test(st.session_state["cables"], end=end)

        if df_overall.empty:
            st.info(f"No overall skew data (pair columns) for {end}")
            continue

        # ---- Global summary per end ----
        global_max = float(df_overall["Overall Max (pS)"].max())
        global_min = float(df_overall["Overall Min (pS)"].min())
        global_span = global_max - global_min

        st.metric(label="Global Overall Max (pS)", value=f"{global_max:.4f}")
        st.metric(label="Global Overall Min (pS)", value=f"{global_min:.4f}")
        st.metric(label="Global Span (pS)", value=f"{global_span:.4f}")

        st.markdown(f"**End: {end}**")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Global Overall Max (pS)", value=f"{global_max:.4f}")
        with m2:
            st.metric(label="Global Overall Min (pS)", value=f"{global_min:.4f}")
        with m3:
            st.metric(label="Global Span (pS)", value=f"{global_span:.4f}")

        # ---- Per-test table ----
        st.dataframe(
            df_overall.sort_values("Label"),
            use_container_width=True,
            height=260
        )

        # ---- Span bar chart ----
        fig = px.bar(
            df_overall.sort_values("Span (pS)", ascending=False),
            x="Label", y="Span (pS)",
            title=f"Skew Span from Pair Columns — {end}",
        )
        fig.update_layout(
            xaxis_title="Test",
            yaxis_title="Span (pS)",
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Downloads ----
        wide = build_skew_overall_from_paircols_wide(st.session_state["cables"], end=end)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label=f"Download Skew Overall (Pair Cols) — {end} — WIDE CSV",
                data=wide.to_csv(index=False),
                file_name=f"Skew_{end}_Overall_from_paircols_wide.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                label=f"Download Skew Overall (Pair Cols) — {end} — LONG CSV",
                data=df_overall.to_csv(index=False),
                file_name=f"Skew_{end}_Overall_from_paircols_long.csv",
                mime="text/csv",
                use_container_width=True,
            )
processed_trace = st.session_state["processed_trace_files"]

st.session_state.setdefault("traces", [])  



processed_trace = st.session_state.setdefault("processed_trace_files", [])
st.session_state.setdefault("traces", [])  

if uploaded_trace_csvs:
    for tf in uploaded_trace_csvs:
        if tf.name in processed_trace:
            continue

        try:
            # Read headerless x,y CSV
            df = pd.read_csv(tf, header=None, names=["time_s", "value"])          
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)

            meta = parse_trace_filename(tf.name)
            if not meta:
                # Could not parse filename → keep unattached for manual handling
                st.session_state["unattached_traces"].append({"name": tf.name, "df": df})
                processed_trace.append(tf.name)
                st.warning(f"Could not parse '{tf.name}' for serial/end; kept in unattached pool.")
                continue

            serial = meta["serial"]
            end = meta["end"]
            # optional: make trace key by channel if available
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
st.subheader("Trace Overlays (per Cable, grouped by End)")



def pick_xy_columns(df: pd.DataFrame):
    """
    Always choose a time-like column as X.
    Preference order: time_s, time, t, x, sample_idx.
    Then choose any other numeric column as Y.
    Fallbacks:
      - First two numeric columns
      - First two columns of the DataFrame
    Returns (x_col, y_col) or (None, None) if not possible.
    """
    cols = list(df.columns)
    time_candidates = ["time_s", "time", "t", "x", "sample_idx"]

    # 1) Prefer an explicit time-like column
    x_col = next((c for c in time_candidates if c in df.columns), None)

    # 2) If none of the above present, try the first numeric column as X
    if x_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        x_col = numeric_cols[0] if numeric_cols else None

    # 3) Pick Y as a different numeric column
    y_col = None
    if x_col is not None:
        numeric_others = [c for c in cols if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        y_col = numeric_others[0] if numeric_others else None

    # 4) Final fallback: if at least two columns exist, take them in order
    if x_col is None or y_col is None:
        if len(cols) >= 2:
            x_col, y_col = cols[0], cols[1]
        else:
            return None, None

    return x_col, y_col



# ----- Build overlay plots -----
# We’ll loop cables, then per end (P1/P2), gather traces from each SI_Test
ends = ["P1", "P2"]

for cable in st.session_state.get("cables", []):
    sn = getattr(cable, "serial_number", "?")
    si_list = getattr(cable, "TESTS", {}).get("si", []) or []

    # Map end -> list of (trace_name, df)
    end_to_traces = {e: [] for e in ends}
    for t in si_list:
        traces_obj = getattr(t, "traces", None)

        # Strict guard: skip if not a dict (covers Field/None/anything else)
        if not isinstance(traces_obj, dict) or not traces_obj:
            continue

        test_end = (getattr(t, "test_end", "") or "").upper()
        if test_end in end_to_traces:
            for name, df in traces_obj.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    end_to_traces[test_end].append((name, df))

    # Only render section for cable if there is at least one trace
    if not any(end_to_traces[e] for e in ends):
        continue

    st.markdown(f"### Cable **{sn}**")

    tab_p1, tab_p2 = st.tabs(["End P1", "End P2"])
    for e, tab in zip(ends, [tab_p1, tab_p2]):
        with tab:
            traces = end_to_traces[e]
            if not traces:
                st.caption(f"No traces for **{sn}** — **{e}**")
                continue

            fig = go.Figure()

            for name, df in traces:
                # Pick columns (x is always time-like by your rule)
                x_col, y_col = pick_xy_columns(df)
                if x_col is None or y_col is None:
                    st.warning(f"Skipping trace '{name}' (could not infer x/y columns).")
                    continue

                # Extract numeric arrays
                x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                y_vals = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)   # <- ensure this is &, not &amp;
                x_vals, y_vals = x_vals[mask], y_vals[mask]

                if len(x_vals) < 2:
                    st.warning(f"Skipping trace '{name}' (too few points after cleaning).")
                    continue

                # Add line to figure (use Scattergl for larger traces)
                use_gl = len(x_vals) > 5000
                trace_cls = go.Scattergl if use_gl else go.Scatter
                fig.add_trace(
                    trace_cls(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        name=name,
                        line=dict(width=1.4),
                        hovertemplate=f"<b>{name}</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                        )
                    )

            # Titles & axes
            fig.update_layout(
                title=f"Overlay — Cable {sn} — {e}",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=48, b=10),
                hovermode="x unified",
            )
            fig.update_xaxes(title_text="Time [s]")          # set as you prefer
            fig.update_yaxes(title_text="Resistance (Ω)")    # set unit to match your data

            st.plotly_chart(fig, use_container_width=True)