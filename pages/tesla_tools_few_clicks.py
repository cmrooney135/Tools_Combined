import streamlit as st

# --- Session-state initialization (must run before using keys) ---
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

ensure_state()

import os
import re
import ast
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px  # noqa: F401 (kept in case your methods rely on it)

# External modules provided by your repo
from Tesla import Tesla  # noqa: F401 (referenced via Cable methods)
from UploadData import process_csv


# =============================================================================
# Session-state initialization (must run before using keys)
# =============================================================================
def ensure_state():
    # Lists to hold your domain objects
    st.session_state.setdefault("cables", [])       # list[ Cable ]
    st.session_state.setdefault("tests", [])        # optional list[ Test ]

    # Files you’ve already processed (use a list for Streamlit friendliness)
    st.session_state.setdefault("processed_files", [])

    # Any caches for plots/maps per test type
    st.session_state.setdefault("continuity_figs", {})
    st.session_state.setdefault("continuity_defect_figs", {})

    st.session_state.setdefault("inv_continuity_figs", {})
    st.session_state.setdefault("inv_continuity_defect_figs", {})

    st.session_state.setdefault("DCR_figs", {})
    st.session_state.setdefault("DCR_defect_figs", {})

    st.session_state.setdefault("inv_DCR_figs", {})
    st.session_state.setdefault("inv_DCR_defect_figs", {})

    st.session_state.setdefault("leakage_figs", {})
    st.session_state.setdefault("leakage_defects", {})

    st.session_state.setdefault("leakage_1s_figs", {})
    st.session_state.setdefault("leakage_1s_defects", {})

ensure_state()


# =============================================================================
# Small utilities
# =============================================================================
def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def normalize_channel(cell):
    # Return a safe (list_a, list_b) tuple for any input
    if cell is None:
        return ([], [])
    if isinstance(cell, (list, tuple)):
        if len(cell) == 2:
            return (_to_list(cell[0]), _to_list(cell[1]))
        if len(cell) == 1:
            return (_to_list(cell[0]), [])
        if len(cell) == 0:
            return ([], [])
        # >2 entries: best effort: first two are endpoints
        return (_to_list(cell[0]), _to_list(cell[1]))
    # scalar → single endpoint
    return (_to_list(cell), [])

def widget_key(*parts) -> str:
    """Compose a globally unique Streamlit widget key from multiple parts."""
    return "§".join(str(p) for p in parts if p is not None)

def bucket_reason(text: str) -> str:
    t = (text or "").lower()
    if "short" in t:
        return "short"
    if "miswire" in t:
        return "Miswire"
    if "high" in t:
        return "High"
    if "open" in t:
        return "Open"
    if "missing" in t:
        return "Wire Missing"
    return "Other"

def has_data(cables, test_type: str, use_latest: bool = True) -> bool:
    for cable in cables:
        if cable.has_data(test_type, use_latest=use_latest):
            return True
    return False


# =============================================================================
# Channel parsing (robust)
# =============================================================================
def _first_token(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[A-Za-z0-9_]+", text)
    return m.group(0) if m else None

def _extract_endpoint(x) -> str:
    """
    x might be ['A1DIB'] or 'A1DIB' or ('A1DIB',).
    Return the first string-like token.
    """
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return str(x[0])
    return str(x)

def _parse_pair_from_obj(obj) -> Optional[Tuple[str, str]]:
    """
    Try to parse a pair from a Python object like (['A1DIB'], ['A3DIB'])
    or ['A1DIB','A3DIB'] or ('A1DIB','A3DIB').
    """
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        a = _extract_endpoint(obj[0])
        b = _extract_endpoint(obj[1])
        if a and b:
            return (a, b)
    return None

def _parse_pair_from_string(text: str) -> Optional[Tuple[str, str]]:
    """
    Handle strings like "(['A1DIB'], ['A3DIB'])" (literal-evaluable) or
    fallback: two tokens inside the text.
    """
    try:
        parsed = ast.literal_eval(text)
        pair = _parse_pair_from_obj(parsed)
        if pair:
            return pair
    except Exception:
        pass

    # Fallback: extract tokens like A13DIB, B69P1, etc.
    tokens = re.findall(r"[A-G]\d+(?:P\d+)?[A-Za-z0-9_]*", text)
    if len(tokens) >= 2:
        return (tokens[0], tokens[1])

    return None

def parse_single_channel_id(text: str) -> Optional[str]:
    """
    Parse single-channel strings:
      - 'E69 (DIB - SIGNAL)' -> 'E69DIB'
      - 'E69 DIB'            -> 'E69DIB'
      - 'B69P1'              -> 'B69P1'
    """
    if not isinstance(text, str) or not text:
        return None

    m = re.search(r"\b([A-G]\d+(?:P\d+)?)\b", text)
    if not m:
        return None

    code = m.group(1)

    paren = re.search(r"\(([^)]+)\)", text)
    if paren:
        tok = _first_token(paren.group(1))
        return code + (tok or "")

    after = text[m.end():].lstrip()
    bare = re.match(r"[A-Za-z0-9_]+", after)
    if bare:
        return code + bare.group(0)

    return code

def parse_channel_id(raw: object, *, pair_delim: str = "|", undirected: bool = True) -> Optional[str]:
    """
    Robust key normalizer:
    - Pair strings/objs like "(['A1DIB'], ['A3DIB'])" -> 'A1DIB|A3DIB'
    - Single strings like 'E69 (DIB - SIGNAL)'       -> 'E69DIB'
    - Keeps you out of collisions like 'B13P1E13DIB' by using a delimiter.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    # If it's already a pair-like object
    pair = _parse_pair_from_obj(raw)
    if pair:
        a, b = pair
        if undirected and a > b:
            a, b = b, a
        return f"{a}{pair_delim}{b}"

    # If it's a string, try pair first, then single
    if isinstance(raw, str):
        pair = _parse_pair_from_string(raw)
        if pair:
            a, b = pair
            if undirected and a > b:
                a, b = b, a
            return f"{a}{pair_delim}{b}"

        # Single channel format
        single = parse_single_channel_id(raw)
        if single:
            return single

        return None

    # Last resort: string-ify and try again
    as_str = str(raw)
    return parse_channel_id(as_str, pair_delim=pair_delim, undirected=undirected)


# =============================================================================
# Build master dataframe across cables for a given attribute
# =============================================================================
def build_master_dataframe(cables: list, attr_name: str, *, pair_delim: str = "|", undirected: bool = True):
    """Enhanced version that constructs a master DF indexed by parsed channel IDs."""
    dfs = []

    for cable in cables:
        df = getattr(cable, attr_name, None)
        if df is None or df.empty:
            continue

        # Work with first two columns: [original_key, measurement]
        # If "Measured_R (mOhm)" exists, prefer it to avoid picking Expected column by accident.
        if "Measured_R (mOhm)" in df.columns:
            tmp = df[[df.columns[0], "Measured_R (mOhm)"]].copy()
        else:
            tmp = df.iloc[:, :2].copy()

        orig_col = tmp.columns[0]
        meas_col = tmp.columns[1]

        # Parse channel ID (pair-safe)
        tmp["ParsedKey"] = tmp[orig_col].map(lambda v: parse_channel_id(v, pair_delim=pair_delim, undirected=False))

        # Drop rows where parsing failed
        tmp = tmp.dropna(subset=["ParsedKey"])

        # Convert measurement to numeric
        tmp[meas_col] = pd.to_numeric(tmp[meas_col], errors="coerce")

        # Deduplicate by parsed key
        tmp = tmp.groupby("ParsedKey", as_index=False).agg({meas_col: "max"})

        # Rename measurement column to serial number
        tmp = tmp.rename(columns={meas_col: cable.serial_number})

        dfs.append(tmp)

    if not dfs:
        # Return the expected shape even if empty
        return pd.DataFrame(columns=["ParsedKey"]), None

    # Start master
    master_df = dfs[0].copy()

    # Merge remaining
    for df_i in dfs[1:]:
        master_df = master_df.merge(df_i, on="ParsedKey", how="outer")

    # Convert numeric columns
    meas_cols = [c for c in master_df.columns if c != "ParsedKey"]
    for c in meas_cols:
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce")

    # Collapse any duplicate ParsedKey
    master_df = master_df.groupby("ParsedKey", as_index=False).max(numeric_only=True)

    # Sort by parsed key
    master_df = master_df.sort_values("ParsedKey")

    return master_df, None


# =============================================================================
# Collapse helper (kept if you call this elsewhere)
# =============================================================================
def collapse_channels(rows, *, prefer='first_non_blank'):
    """
    Convert rows like:
        ((['A2DIB'], []), 27.0, 0.0)     # 3- or 4-tuple variants both supported
    or
        ( (['A2DIB'], []), 27.0, 0 )      # last number can be flag/unused
    or
        (index, (['A2DIB'], []), 27.0, 0.0)

    into a dict: { 'A2DIB': 27.0, ... }
    """
    out = {}

    for rec in rows:
        # Normalize possible shapes
        if len(rec) == 4 and isinstance(rec[1], tuple):
            # (index, (ch1_ids, ch2_ids), v1, v2)
            _, chans, v1, v2 = rec
            ch1_ids, ch2_ids = chans
            val1 = float(v1) if v1 is not None else np.nan
            val2 = float(v2) if v2 is not None else np.nan

        elif len(rec) == 3 and isinstance(rec[0], tuple):
            # ((ch1_ids, ch2_ids), v1, maybe_flag)
            chans, v1, _ = rec
            ch1_ids, ch2_ids = chans
            val1 = float(v1) if v1 is not None else np.nan
            val2 = np.nan  # no second value present in this shape

        else:
            raise ValueError(f"Unexpected row format: {rec}")

        ch1_has = bool(ch1_ids)
        ch2_has = bool(ch2_ids)

        if ch1_has and not ch2_has:
            for site in ch1_ids:
                out[site] = val1
        elif not ch1_has and ch2_has:
            for site in ch2_ids:
                out[site] = val2
        elif ch1_has and ch2_has:
            # Both channels populated – default to ch1
            chosen_ids, chosen_val = (ch1_ids, val1) if prefer == 'first_non_blank' else (ch2_ids, val2)
            for site in chosen_ids:
                out[site] = chosen_val
        else:
            # Both blank – ignore silently
            continue

    return out


# =============================================================================
# Auto-generation helpers (no buttons)
# =============================================================================
def _leakage_split_failures(fail_df: pd.DataFrame):
    """Normalize channel and split single-endpoint vs pair/multi for leakage-type tests."""
    fail_df = (fail_df if fail_df is not None else pd.DataFrame(columns=["Channel", "Detail"])).copy()
    if "Channel" not in fail_df.columns:
        fail_df["Channel"] = None
    if "Detail" not in fail_df.columns:
        fail_df["Detail"] = ""

    fail_df["Channel"] = fail_df["Channel"].apply(normalize_channel)
    fail_df["Category"] = fail_df["Detail"].apply(bucket_reason)

    def _chan_count(pair):
        a, b = pair
        return len(_to_list(a)) + len(_to_list(b))

    fail_df["__channel_count"] = fail_df["Channel"].apply(_chan_count)
    single_mask = fail_df["__channel_count"] == 1

    single_df = fail_df[single_mask].drop(columns=["__channel_count"], errors="ignore").copy()
    pair_df   = fail_df[~single_mask].drop(columns=["__channel_count"], errors="ignore").copy()
    return single_df, pair_df


def _ensure_numeric_heatmap(
    sess_key_figs: str,
    cache_key: str,
    cable,
    test,
    value_col: str,
    title_prefix: str,
    unit: str,
):
    """Create (and cache) a numeric heatmap for analog/numeric tests."""
    fig = st.session_state[sess_key_figs].get(cache_key)
    if fig is None:
        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
            test.data,
            value_col=value_col,
            agg="max",
            return_type="list",
            verbose=False,
        )
        fig = cable.make_analog_heatmap(
            dib_vals, p1_vals, p2_vals,
            title_prefix=title_prefix,
            unit=unit,
            colorscale="Viridis",
            zmin=None,
            zmax=None,
            show_colorbar=True,
        )
        st.session_state[sess_key_figs][cache_key] = fig
    return fig


def _ensure_defect_map(
    sess_key_defects: str,
    cache_key: str,
    cable,
    fail_df: pd.DataFrame,
    title_prefix: str,
    category_col: str | None = None,
):
    """Create (and cache) a defect map from failure buckets."""
    defect_fig = st.session_state[sess_key_defects].get(cache_key)
    if defect_fig is None:
        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
            fail_df,
            category_col=category_col,   # None for continuity/DCR, "Category" for leakage types
            prefer_first=True,
            blank_label="",
            verbose=False,
        )
        defect_fig = cable.make_defect_heatmap(
            dib_cats, p1_cats, p2_cats,
            title_prefix=title_prefix,
            blank_label="—",
        )
        st.session_state[sess_key_defects][cache_key] = defect_fig
    return defect_fig


def _run_id(test):
    """Stable identifier for a test run; used to prevent cache staleness."""
    if getattr(test, "test_date", None) and getattr(test, "test_time", None):
        return f"{test.test_date}_{test.test_time}"
    return "latest"


# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Tesla Tools", page_icon = "🧹", layout="wide")
st.title("⚡🧹Tesla Tools")

uploaded_files = st.file_uploader("Upload your CSV files", type="csv", accept_multiple_files=True)

cables = st.session_state["cables"]
processed = st.session_state["processed_files"]

# ---------------------------------------------------------------------------
# Process uploads (append Cable objects; avoid duplicate serials in session)
# ---------------------------------------------------------------------------
if uploaded_files:
    for uf in uploaded_files:
        if uf.name in processed:
            continue

        cable, test = process_csv(uf, cables)

        # Only append a real Cable instance (not None)
        if cable is not None:
            # Optional: avoid duplicates by serial_number
            exists = next((c for c in st.session_state["cables"]
                           if getattr(c, "serial_number", None) == cable.serial_number), None)
            if exists is None:
                st.session_state["cables"].append(cable)

            processed.append(uf.name)

        st.session_state["processed_files"] = list(processed)

# Refresh local reference
cables = st.session_state["cables"]

# ---------------------------------------------------------------------------
# Master CSV downloads (auto)
# ---------------------------------------------------------------------------
cols = st.columns(6)
with cols[0]:
    if has_data(cables, "continuity"):
        df = build_master_dataframe(cables, "continuity")[0]
        st.download_button("Download Master Continuity CSV", df.to_csv(index=False), "continuity_master.csv", "text/csv")
with cols[1]:
    if has_data(cables, "inv_continuity"):
        df = build_master_dataframe(cables, "inv_continuity")[0]
        st.download_button("Download Master Inv Continuity CSV", df.to_csv(index=False), "inv_continuity_master.csv", "text/csv")
with cols[2]:
    if has_data(cables, "resistance"):
        df = build_master_dataframe(cables, "resistance")[0]
        st.download_button("Download Master DCR CSV", df.to_csv(index=False), "DCR_master.csv", "text/csv")
with cols[3]:
    if has_data(cables, "inv_resistance"):
        df = build_master_dataframe(cables, "inv_resistance")[0]
        st.download_button("Download Master Inverse DCR CSV", df.to_csv(index=False), "inv_DCR_master.csv", "text/csv")
with cols[4]:
    if has_data(cables, "leakage"):
        df = build_master_dataframe(cables, "leakage")[0]
        st.download_button("Download Master Leakage CSV", df.to_csv(index=False), "leakage_master.csv", "text/csv")
with cols[5]:
    if has_data(cables, "leakage_1s"):
        df = build_master_dataframe(cables, "leakage_1s")[0]
        st.download_button("Download Master Leakage 1s CSV", df.to_csv(index=False), "leakage_1s_master.csv", "text/csv")

st.divider()


# =============================================================================
# All Maps (stacked): inv_* on top of normal
# =============================================================================
st.header("All Maps (stacked)")

if not cables:
    st.info("Upload CSVs to see maps.")
else:
    for cable in cables:
        st.markdown(f"### {cable.serial_number} — {cable.length} in")
        ccols = st.columns([2, 8])  # Meta | Maps | CSV downloads
        with ccols[0]:
            st.markdown("**Serial Number**")
            st.markdown(cable.serial_number)
            st.markdown("**Length (in)**")
            st.markdown(str(cable.length))


        # ------------------------- MAPS COLUMN -------------------------
        with ccols[1]:
            # CONTINUITY (inverse on top)
            inv_cont = cable.latest_test("inv_continuity")
            cont     = cable.latest_test("continuity")

            if inv_cont and inv_cont.has_data():
                st.caption("Inverse Continuity")
                run_id = _run_id(inv_cont)
                cache_key = f"{cable.serial_number}::inv_continuity::{run_id}"

                fig = _ensure_numeric_heatmap(
                    sess_key_figs="inv_continuity_figs",
                    cache_key=cache_key,
                    cable=cable,
                    test=inv_cont,
                    value_col="Measured_R (mOhm)",
                    title_prefix="Measured Resistance",
                    unit="mΩ",
                )
                st.plotly_chart(fig, use_container_width=True, key=widget_key("inv_cont", "heat", cache_key))

                fail_df = inv_cont.failure_data if inv_cont.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                defect_fig = _ensure_defect_map(
                    sess_key_defects="inv_continuity_defect_figs",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=fail_df,
                    title_prefix="Inverse Continuity Buckets",
                    category_col=None,
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("inv_cont", "def", cache_key))

            if cont and cont.has_data():
                st.caption("Continuity")
                run_id = _run_id(cont)
                cache_key = f"{cable.serial_number}::continuity::{run_id}"

                fig = _ensure_numeric_heatmap(
                    sess_key_figs="continuity_figs",
                    cache_key=cache_key,
                    cable=cable,
                    test=cont,
                    value_col="Measured_R (mOhm)",
                    title_prefix="Measured Resistance",
                    unit="mΩ",
                )
                st.plotly_chart(fig, use_container_width=True, key=widget_key("cont", "heat", cache_key))

                fail_df = cont.failure_data if cont.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                defect_fig = _ensure_defect_map(
                    sess_key_defects="continuity_defect_figs",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=fail_df,
                    title_prefix="Continuity Buckets",
                    category_col=None,
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("cont", "def", cache_key))

            st.markdown("---")

            # DCR (inverse on top)
            inv_dcr = cable.latest_test("inv_resistance")
            dcr     = cable.latest_test("resistance")

            if inv_dcr and inv_dcr.has_data():
                st.caption("Inverse DCR")
                run_id = _run_id(inv_dcr)
                cache_key = f"{cable.serial_number}::inv_resistance::{run_id}"

                fig = _ensure_numeric_heatmap(
                    sess_key_figs="inv_DCR_figs",
                    cache_key=cache_key,
                    cable=cable,
                    test=inv_dcr,
                    value_col="Measured_R (mOhm)",
                    title_prefix="Measured Resistance",
                    unit="mΩ",
                )
                st.plotly_chart(fig, use_container_width=True, key=widget_key("inv_dcr", "heat", cache_key))

                fail_df = inv_dcr.failure_data if inv_dcr.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                defect_fig = _ensure_defect_map(
                    sess_key_defects="inv_DCR_defect_figs",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=fail_df,
                    title_prefix="Inverse Resistance Buckets",
                    category_col=None,
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("inv_dcr", "def", cache_key))

            if dcr and dcr.has_data():
                st.caption("DCR")
                run_id = _run_id(dcr)
                cache_key = f"{cable.serial_number}::resistance::{run_id}"

                fig = _ensure_numeric_heatmap(
                    sess_key_figs="DCR_figs",
                    cache_key=cache_key,
                    cable=cable,
                    test=dcr,
                    value_col="Measured_R (mOhm)",
                    title_prefix="Measured Resistance",
                    unit="mΩ",
                )
                st.plotly_chart(fig, use_container_width=True, key=widget_key("dcr", "heat", cache_key))

                fail_df = dcr.failure_data if dcr.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                defect_fig = _ensure_defect_map(
                    sess_key_defects="DCR_defect_figs",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=fail_df,
                    title_prefix="Resistance Buckets",
                    category_col=None,
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("dcr", "def", cache_key))

            st.markdown("---")

            # LEAKAGE (1s on top)
            leak1s = cable.latest_test("leakage_1s")
            leak   = cable.latest_test("leakage")

            if leak1s and leak1s.has_data():
                st.caption("Leakage 1s")
                run_id = _run_id(leak1s)
                cache_key = f"{cable.serial_number}::leakage_1s::{run_id}"

                single_df, pair_df = _leakage_split_failures(leak1s.failure_data)
                fig = st.session_state.leakage_1s_figs.get(cache_key)
                if fig is None:
                    dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                        leak1s.data,
                        value_col="Measured_pA",
                        agg="max",
                        return_type="list",
                        verbose=False,
                    )
                    fig = cable.make_analog_heatmap(
                        dib_vals, p1_vals, p2_vals,
                        title_prefix="Measured Current",
                        unit="pA",
                        colorscale="Viridis",
                        zmin=None,
                        zmax=None,
                        show_colorbar=True,
                    )
                    # overlay single-endpoint failures
                    fig = cable.overlay_failures(fig, single_df)
                    st.session_state.leakage_1s_figs[cache_key] = fig
                st.plotly_chart(fig, use_container_width=True, key=widget_key("leak1s", "heat", cache_key))

                defect_fig = _ensure_defect_map(
                    sess_key_defects="leakage_1s_defects",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=pair_df,
                    title_prefix="Leakage Buckets",
                    category_col="Category",
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("leak1s", "def", cache_key))

            if leak and leak.has_data():
                st.caption("Leakage")
                run_id = _run_id(leak)
                cache_key = f"{cable.serial_number}::leakage::{run_id}"

                single_df, pair_df = _leakage_split_failures(leak.failure_data)
                fig = st.session_state.leakage_figs.get(cache_key)
                if fig is None:
                    dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                        leak.data,
                        value_col="Measured_pA",
                        agg="max",
                        return_type="list",
                        verbose=False,
                    )
                    fig = cable.make_analog_heatmap(
                        dib_vals, p1_vals, p2_vals,
                        title_prefix="Measured Current",
                        unit="pA",
                        colorscale="Viridis",
                        zmin=None,
                        zmax=None,
                        show_colorbar=True,
                    )
                    fig = cable.overlay_failures(fig, single_df)
                    st.session_state.leakage_figs[cache_key] = fig
                st.plotly_chart(fig, use_container_width=True, key=widget_key("leak", "heat", cache_key))

                defect_fig = _ensure_defect_map(
                    sess_key_defects="leakage_defects",
                    cache_key=cache_key,
                    cable=cable,
                    fail_df=pair_df,
                    title_prefix="Leakage Buckets",
                    category_col="Category",
                )
                st.plotly_chart(defect_fig, use_container_width=True, key=widget_key("leak", "def", cache_key))

        st.divider()

