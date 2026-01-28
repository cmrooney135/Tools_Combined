import streamlit as st
import re

from Tesla import Tesla
from UploadData import process_csv
import pandas as pd
import numpy as np
import ast
import re
import pandas as pd
from typing import Optional, Tuple

from pathlib import Path

import os, tempfile

import plotly.express as px


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

cables = st.session_state["cables"] 
def bucket_reason(text: str) -> str:
    t = text.lower()
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

def make_hashable(ch):
    if isinstance(ch, tuple):
        return tuple(tuple(x) if isinstance(x, list) else x for x in ch)
    return ch




def has_data(cables, test_type: str, use_latest: bool = True) -> bool:
    for cable in cables:
        if cable.has_data(test_type, use_latest=use_latest):
            return True
    return False




import pandas as pd
import re

def parse_channel_id(text: str) -> str | None:
    """
    Extracts channel ID like 'E69DIB' or 'B69P1'.
    """
    # code + optional annotation
    m = re.search(r"\b([A-G]\d+(?:P\d+)?)\b", text)
    if not m:
        return None

    code = m.group(1)

    # Find annotation (paren or bare)
    paren = re.search(r"\(([^)]+)\)", text)
    if paren:
        # First token inside parentheses
        token = re.search(r"[A-Za-z0-9_]+", paren.group(1))
        return code + (token.group(0) if token else "")

    # Bare token case (e.g., "E69 DIB")
    after = text[m.end():].strip()
    bare = re.match(r"[A-Za-z0-9_]+", after)
    if bare:
        return code + bare.group(0)

    # No annotation
    return code


# -------- helpers --------

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

# -------- your build function, unchanged shape --------

def build_master_dataframe(cables: dict, attr_name: str, *, pair_delim: str = "|", undirected: bool = True):
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



st.set_page_config(page_title="Tesla Tools", page_icon = "🧹", layout="wide")

st.title("🧹⚡Tesla Tools")
uploaded_files = st.file_uploader("Upload your CSV files", type="csv", accept_multiple_files=True)

pattern = re.compile(r"(?<![A-Za-z0-9])0[0-4][A-Za-z0-9]{8}(?![A-Za-z0-9])", re.IGNORECASE)

cables = st.session_state["cables"]
processed = st.session_state["processed_files"]

if uploaded_files:
    for uf in uploaded_files:
        if uf.name in processed:
            continue

        cable, test = process_csv(uf, cables)

       
        # Only append a *real* Cable instance (not None)
        if cable is not None:
            # Optional: avoid duplicates by serial_number
            exists = next((c for c in st.session_state["cables"]
                           if getattr(c, "serial_number", None) == cable.serial_number), None)
            if exists is None:
                st.session_state["cables"].append(cable)

            processed.append(uf.name)

        st.session_state["processed_files"] = list(processed)

    Continuity_Tab, Inverse_Continuity_Tab, DCR_Tab, Inverse_DCR_Tab, Leakage_Tab, Leakage_1s_Tab = st.tabs(["Continuity", "Inverse Continuity", "DCR", "Inverse DCR", "Leakage", "Leakage 1s"])

    with Continuity_Tab:
        st.subheader("Continuity")
        has_continuity = has_data(cables, "continuity")
        if(has_continuity):
            continuity_master = build_master_dataframe(cables, "continuity")[0]
            csv_str = continuity_master.to_csv(index=False)
            st.download_button(
                label="Download Master Continuity CSV",
                data=csv_str,
                file_name="continuity_master.csv",
                mime="text/csv",
                key="download_continuity_csv",
            )
        st.divider()


        # Collect failure DataFrames from the *latest* continuity test per cable
        continuity_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("continuity")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                continuity_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if continuity_master_failures:
                master_df = pd.concat(continuity_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)

                # Your existing plotting function
                #fig_overall, fig_stacked = DCR_continuity_histograms(master_df)




        if "continuity_figs" not in st.session_state:
            st.session_state.continuity_figs = {}  # numeric heatmaps per serial_number
        if "continuity_defect_figs" not in st.session_state:
            st.session_state.continuity_defect_figs = {}
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)

        header_cols = st.columns(COL_LAYOUT)

        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Maps**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        continuity_master_failures = []  # list of DataFrames
        
        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("continuity")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            heatmap_exists = cable.serial_number in st.session_state.continuity_figs

            with heatmap_slot:
                btn_key = widget_key("continuity", "heatmap_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("continuity", "heatmap_fig", row_idx, cable.serial_number, run_id)
                chart_key_render = widget_key("continuity", "heatmap_render", row_idx, cable.serial_number, run_id)

                if not heatmap_exists:
                    if st.button("Generate Heatmap", key=btn_key):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_R (mOhm)",
                            agg="max",
                            return_type="list",
                            verbose=True,
                        )

                        fig = cable.make_analog_heatmap(
                            dib_vals, p1_vals, p2_vals,
                            title_prefix="Measured Resistance",
                            unit="mΩ",
                            colorscale="Viridis",
                            zmin=None,
                            zmax=None,
                            show_colorbar=True,
                        )

                        st.session_state.continuity_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True, key=chart_key_render)
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.continuity_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

            # ===== DEFECT MAP (BOTTOM) =====
            defect_exists = cable.serial_number in st.session_state.continuity_defect_figs

            with defect_slot:
                btn_key = widget_key("continuity", "defect_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("continuity", "defect_fig", row_idx, cable.serial_number, run_id)

                if not defect_exists:
                    if st.button("Generate Defect Map", key=btn_key):
                        fail_df = test.failure_data if test.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            fail_df,
                            category_col=None,
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )
                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Continuity Buckets",
                            blank_label="—",
                        )
                        st.session_state.continuity_defect_figs[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True, key=fig_key)
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.continuity_defect_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)


    with Inverse_Continuity_Tab:
        st.subheader("Inverse Continuity")
        has_inv_continuity = has_data(cables, "inv_continuity")
        if(has_inv_continuity):
            inv_continuity_master = build_master_dataframe(cables, "inv_continuity")[0]
            csv_str = inv_continuity_master.to_csv(index=False)
            st.download_button(
                label="Download Master Inv Continuity CSV",
                data=csv_str,
                file_name="inv_continuity_master.csv",
                mime="text/csv",
                key="download_inv_continuity_csv",
            )
        st.divider()


        # Collect failure DataFrames from the *latest* continuity test per cable
        inv_continuity_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("inv_continuity")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                inv_continuity_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if inv_continuity_master_failures:
                master_df = pd.concat(inv_continuity_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)

                # Your existing plotting function
                #fig_overall, fig_stacked = DCR_continuity_histograms(master_df)




        if "inv_continuity_figs" not in st.session_state:
            st.session_state.inv_continuity_figs = {}  # numeric heatmaps per serial_number
        if "inv_continuity_defect_figs" not in st.session_state:
            st.session_state.inv_continuity_defect_figs = {}
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)


        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Maps**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        
        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("inv_continuity")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            heatmap_exists = cable.serial_number in st.session_state.inv_continuity_figs

            with heatmap_slot:
                btn_key = widget_key("inv_continuity", "heatmap_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("inv_continuity", "heatmap_fig", row_idx, cable.serial_number, run_id)
                chart_key_render = widget_key("inv_continuity", "heatmap_render", row_idx, cable.serial_number, run_id)

                if not heatmap_exists:
                    if st.button("Generate Heatmap", key=btn_key):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_R (mOhm)",
                            agg="max",
                            return_type="list",
                            verbose=True,
                        )

                        fig = cable.make_analog_heatmap(
                            dib_vals, p1_vals, p2_vals,
                            title_prefix="Measured Resistance",
                            unit="mΩ",
                            colorscale="Viridis",
                            zmin=None,
                            zmax=None,
                            show_colorbar=True,
                        )

                        st.session_state.inv_continuity_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True, key=chart_key_render)
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.inv_continuity_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

            # ===== DEFECT MAP (BOTTOM) =====
            defect_exists = cable.serial_number in st.session_state.inv_continuity_defect_figs

            with defect_slot:
                btn_key = widget_key("inv_continuity", "defect_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("inv_continuity", "defect_fig", row_idx, cable.serial_number, run_id)

                if not defect_exists:
                    if st.button("Generate Defect Map", key=btn_key):
                        fail_df = test.failure_data if test.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            fail_df,
                            category_col=None,
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )
                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Inverse Continuity Buckets",
                            blank_label="—",
                        )
                        st.session_state.inv_continuity_defect_figs[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True, key=fig_key)
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.inv_continuity_defect_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

    with DCR_Tab:
        st.subheader("DC Resistance")
        has_resistance = has_data(cables, "resistance")
        if(has_resistance):
            DCR_master = build_master_dataframe(cables, "resistance")[0]
            csv_str = DCR_master.to_csv(index=False)
            st.download_button(
                label="Download Master DCR CSV",
                data=csv_str,
                file_name="DCR_master.csv",
                mime="text/csv",
                key="download_DCR_csv",
            )
        st.divider()


        # Collect failure DataFrames from the *latest* continuity test per cable
        DCR_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("resistance")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                DCR_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if DCR_master_failures:
                master_df = pd.concat(DCR_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)

                # Your existing plotting function
                #fig_overall, fig_stacked = DCR_continuity_histograms(master_df)




        if "DCR_figs" not in st.session_state:
            st.session_state.DCR_figs = {}  # numeric heatmaps per serial_number
        if "DCR_defect_figs" not in st.session_state:
            st.session_state.DCR_defect_figs = {}
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)


        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Maps**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        
        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("resistance")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            heatmap_exists = cable.serial_number in st.session_state.DCR_figs

            with heatmap_slot:
                btn_key = widget_key("resistance", "heatmap_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("resistance", "heatmap_fig", row_idx, cable.serial_number, run_id)
                chart_key_render = widget_key("resistance", "heatmap_render", row_idx, cable.serial_number, run_id)

                if not heatmap_exists:
                    if st.button("Generate Heatmap", key=btn_key):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_R (mOhm)",
                            agg="max",
                            return_type="list",
                            verbose=True,
                        )

                        fig = cable.make_analog_heatmap(
                            dib_vals, p1_vals, p2_vals,
                            title_prefix="Measured Resistance",
                            unit="mΩ",
                            colorscale="Viridis",
                            zmin=None,
                            zmax=None,
                            show_colorbar=True,
                        )

                        st.session_state.DCR_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True, key=chart_key_render)
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.DCR_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

            # ===== DEFECT MAP (BOTTOM) =====
            defect_exists = cable.serial_number in st.session_state.DCR_defect_figs

            with defect_slot:
                btn_key = widget_key("resistance", "defect_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("resistance", "defect_fig", row_idx, cable.serial_number, run_id)

                if not defect_exists:
                    if st.button("Generate Defect Map", key=btn_key):
                        fail_df = test.failure_data if test.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            fail_df,
                            category_col=None,
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )
                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Resistance Buckets",
                            blank_label="—",
                        )
                        st.session_state.DCR_defect_figs[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True, key=fig_key)
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.DCR_defect_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

    with Inverse_DCR_Tab:
        st.subheader("Inverse DC Resistance")
        has_inv_resistance = has_data(cables, "inv_resistance")
        if(has_inv_resistance):
            inv_DCR_master = build_master_dataframe(cables, "inv_resistance")[0]
            csv_str = inv_DCR_master.to_csv(index=False)
            st.download_button(
                label="Download Master Inverse DCR CSV",
                data=csv_str,
                file_name="inv_DCR_master.csv",
                mime="text/csv",
                key="download_inv_DCR_csv",
            )
        st.divider()


        # Collect failure DataFrames from the *latest* continuity test per cable
        inv_DCR_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("inv_resistance")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                inv_DCR_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if DCR_master_failures:
                master_df = pd.concat(DCR_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)



        if "inv_DCR_figs" not in st.session_state:
            st.session_state.inv_DCR_figs = {}  # numeric heatmaps per serial_number
        if "inv_DCR_defect_figs" not in st.session_state:
            st.session_state.inv_DCR_defect_figs = {}
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)


        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Maps**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        
        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("inv_resistance")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            heatmap_exists = cable.serial_number in st.session_state.inv_DCR_figs

            with heatmap_slot:
                btn_key = widget_key("inv_resistance", "heatmap_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("inv_resistance", "heatmap_fig", row_idx, cable.serial_number, run_id)
                chart_key_render = widget_key("inv_resistance", "heatmap_render", row_idx, cable.serial_number, run_id)

                if not heatmap_exists:
                    if st.button("Generate Heatmap", key=btn_key):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_R (mOhm)",
                            agg="max",
                            return_type="list",
                            verbose=True,
                        )

                        fig = cable.make_analog_heatmap(
                            dib_vals, p1_vals, p2_vals,
                            title_prefix="Measured Resistance",
                            unit="mΩ",
                            colorscale="Viridis",
                            zmin=None,
                            zmax=None,
                            show_colorbar=True,
                        )

                        st.session_state.inv_DCR_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True, key=chart_key_render)
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.inv_DCR_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

            # ===== DEFECT MAP (BOTTOM) =====
            defect_exists = cable.serial_number in st.session_state.inv_DCR_defect_figs

            with defect_slot:
                btn_key = widget_key("inv_resistance", "defect_btn", row_idx, cable.serial_number, run_id)
                fig_key = widget_key("inv_resistance", "defect_fig", row_idx, cable.serial_number, run_id)

                if not defect_exists:
                    if st.button("Generate Defect Map", key=btn_key):
                        fail_df = test.failure_data if test.failure_data is not None else pd.DataFrame(columns=["Channel", "Detail"])
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            fail_df,
                            category_col=None,
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )
                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Inverse Resistance Buckets",
                            blank_label="—",
                        )
                        st.session_state.inv_DCR_defect_figs[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True, key=fig_key)
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.inv_DCR_defect_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=fig_key)

    def collapse_channels(rows, *, prefer='first_non_blank'):
        """
        Convert rows like:
            ((['A2DIB'], []), 27.0, 0.0)     # 3- or 4-tuple variants both supported
        or
            ( (['A2DIB'], []), 27.0, 0 )      # last number can be flag/unused
        or
            (index, (['A2DIB'], []), 27.0, 0.0)

        into a dict: { 'A2DIB': 27.0, ... }

        Rules:
        - Use the value for the **first non-blank channel list**.
        - If ch1 has IDs, use value1 (or the single numeric in a 3-tuple row).
        - If ch1 is blank and ch2 has IDs, use value2.
        - If both are blank, skip the row.
        - If both are non-blank, keep ch1 by default (configurable via `prefer`).
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

    with Leakage_Tab:
        st.subheader("Leakage")
        has_leakage = has_data(cables, "leakage")
        if(has_leakage):
            leakage_master = build_master_dataframe(cables, "leakage")[0]
            csv_str = leakage_master.to_csv(index=False)
            st.download_button(
                label="Download Master Leakage CSV",
                data=csv_str,
                file_name="leakage_master.csv",
                mime="text/csv",
                key="download_leakage_csv",
            )
        st.divider()

        leakage_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("leakage")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                leakage_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if leakage_master_failures:
                master_df = pd.concat(leakage_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)


        
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")


        header_cols = st.columns(COL_LAYOUT)

        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Map**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("leakage")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            
# ======= Leakage (inside the row loop where you have `test = cable.latest_test("leakage")`) =======

            heatmap_exists = cable.serial_number in st.session_state.leakage_figs

            # build a clean fail_df per cable/test
            fail_df = test.failure_data if (test and test.failure_data is not None) else pd.DataFrame(columns=["Channel", "Detail"])
            fail_df = fail_df.copy()
            if "Channel" not in fail_df.columns:
                fail_df["Channel"] = None
            if "Detail" not in fail_df.columns:
                fail_df["Detail"] = ""

            # normalize channel to (list,list)
            fail_df["Channel"] = fail_df["Channel"].apply(normalize_channel)

            # bucket reason
            fail_df["Category"] = fail_df["Detail"].apply(bucket_reason)

            # count endpoints robustly
            def _chan_count(pair):
                a, b = pair
                return len(_to_list(a)) + len(_to_list(b))

            fail_df["__channel_count"] = fail_df["Channel"].apply(_chan_count)

            # split single-endpoint vs pair/multi
            single_mask = fail_df["__channel_count"] == 1
            single_channel_df = fail_df[single_mask].copy()
            pair_df = fail_df[~single_mask].copy()

            # clean temp column
            single_channel_df.drop(columns=["__channel_count"], inplace=True, errors="ignore")
            pair_df.drop(columns=["__channel_count"], inplace=True, errors="ignore")

            with heatmap_slot:
                if not heatmap_exists:
                    if st.button(
                        "Generate Heatmap",
                        key=widget_key("leakage", "heatmap_btn", row_idx, cable.serial_number, run_id),
                    ):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_pA",
                            agg="max",
                            return_type="list",
                            verbose=True,
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
                        fig = cable.overlay_failures(fig, single_channel_df)

                        st.session_state.leakage_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage", "heatmap_chart", row_idx, cable.serial_number, run_id))
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.leakage_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage", "heatmap_chart_cached", row_idx, cable.serial_number, run_id))

            # defect (pair) map
            defect_exists = cable.serial_number in st.session_state.leakage_defects
            with defect_slot:
                if not defect_exists:
                    if st.button(
                        "Generate Defect Map",
                        key=widget_key("leakage", "defect_btn", row_idx, cable.serial_number, run_id),
                    ):
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            pair_df,
                            category_col="Category",   # use the bucket we just built
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )

                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Leakage Buckets",
                            blank_label="—",
                        )

                        st.session_state.leakage_defects[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True,
                            key=widget_key("leakage", "defect_chart", row_idx, cable.serial_number, run_id))
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.leakage_defects.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage", "defect_chart_cached", row_idx, cable.serial_number, run_id))
                

    with Leakage_1s_Tab:
        st.subheader("1s Leakage")
        has_leakage_1s = has_data(cables, "leakage_1s")
        if(has_leakage_1s):
            leakage_1s_master = build_master_dataframe(cables, "leakage_1s")[0]
            csv_str = leakage_1s_master.to_csv(index=False)
            st.download_button(
                label="Download Master Leakage CSV",
                data=csv_str,
                file_name="leakage_1s_master.csv",
                mime="text/csv",
                key="download_leakage_1s_csv",
            )
        st.divider()

        leakage_1s_master_failures = []

        for cable in cables:  # cables = st.session_state["cables"]
            # Skip invalid entries defensively
            if cable is None or not hasattr(cable, "latest_test"):
                continue

            test = cable.latest_test("leakage_1s")  # returns a Test or None
            if not test:
                continue

            fail_df = test.failure_data
            if fail_df is not None and not fail_df.empty:
                leakage_1s_master_failures.append(fail_df.copy())

            # Build master and plots (if we collected any failures)
            if leakage_1s_master_failures:
                master_df = pd.concat(leakage_1s_master_failures, ignore_index=True)
                master_df["Category"] = master_df["Detail"].apply(bucket_reason)


        
        COL_LAYOUT = [1.5, 1.2, 7.5]  # SN | Length | Maps | Download        st.subheader("Processed Cables")


        header_cols = st.columns(COL_LAYOUT)

        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Map**")


        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        
        for row_idx, cable in enumerate(cables):
            cols = st.columns(COL_LAYOUT)

            # Get latest continuity test
            test = cable.latest_test("leakage_1s")
            if not test or not test.has_data():
                continue

            # Optional: include run id to make keys stable across multiple uploads/reruns
            run_id = f"{test.test_date}_{test.test_time}" if (test.test_date and test.test_time) else "latest"

            # ── Metadata columns ─────────────────────────────
            cols[0].markdown(cable.serial_number)
            cols[1].markdown(cable.length)

            # ── MAPS column (stacked) ─────────────────────────
            maps_col = cols[2]
            with maps_col:
                heatmap_slot = st.container()
                defect_slot = st.container()

            # ===== HEATMAP (TOP) =====
            
# ======= Leakage (inside the row loop where you have `test = cable.latest_test("leakage")`) =======

            heatmap_exists = cable.serial_number in st.session_state.leakage_1s_figs

            # build a clean fail_df per cable/test
            fail_df = test.failure_data if (test and test.failure_data is not None) else pd.DataFrame(columns=["Channel", "Detail"])
            fail_df = fail_df.copy()
            if "Channel" not in fail_df.columns:
                fail_df["Channel"] = None
            if "Detail" not in fail_df.columns:
                fail_df["Detail"] = ""

            # normalize channel to (list,list)
            fail_df["Channel"] = fail_df["Channel"].apply(normalize_channel)

            # bucket reason
            fail_df["Category"] = fail_df["Detail"].apply(bucket_reason)

            # count endpoints robustly
            def _chan_count(pair):
                a, b = pair
                return len(_to_list(a)) + len(_to_list(b))

            fail_df["__channel_count"] = fail_df["Channel"].apply(_chan_count)

            # split single-endpoint vs pair/multi
            single_mask = fail_df["__channel_count"] == 1
            single_channel_df = fail_df[single_mask].copy()
            pair_df = fail_df[~single_mask].copy()

            # clean temp column
            single_channel_df.drop(columns=["__channel_count"], inplace=True, errors="ignore")
            pair_df.drop(columns=["__channel_count"], inplace=True, errors="ignore")

            with heatmap_slot:
                if not heatmap_exists:
                    if st.button(
                        "Generate Heatmap",
                        key=widget_key("leakage_1s", "heatmap_btn", row_idx, cable.serial_number, run_id),
                    ):
                        dib_vals, p1_vals, p2_vals = cable.build_ordered_arrays(
                            test.data,
                            value_col="Measured_pA",
                            agg="max",
                            return_type="list",
                            verbose=True,
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
                        fig = cable.overlay_failures(fig, single_channel_df)

                        st.session_state.leakage_1s_figs[cable.serial_number] = fig
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage_1s", "heatmap_chart", row_idx, cable.serial_number, run_id))
                else:
                    st.markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.leakage_1s_figs.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage_1s", "heatmap_chart_cached", row_idx, cable.serial_number, run_id))

            # defect (pair) map
            defect_exists = cable.serial_number in st.session_state.leakage_1s_defects
            with defect_slot:
                if not defect_exists:
                    if st.button(
                        "Generate Defect Map",
                        key=widget_key("leakage_1s", "defect_btn", row_idx, cable.serial_number, run_id),
                    ):
                        dib_cats, p1_cats, p2_cats, categories = cable.build_bucket_arrays(
                            pair_df,
                            category_col="Category",   # use the bucket we just built
                            prefer_first=True,
                            blank_label="",
                            verbose=False,
                        )

                        defect_fig = cable.make_defect_heatmap(
                            dib_cats, p1_cats, p2_cats,
                            title_prefix="Leakage Buckets",
                            blank_label="—",
                        )

                        st.session_state.leakage_1s_defects[cable.serial_number] = defect_fig
                        st.plotly_chart(defect_fig, use_container_width=True,
                            key=widget_key("leakage_1s", "defect_chart", row_idx, cable.serial_number, run_id))
                else:
                    st.markdown("<span style='color:gray'>Defect map generated ✔</span>", unsafe_allow_html=True)
                    fig = st.session_state.leakage_1s_defects.get(cable.serial_number)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True,
                            key=widget_key("leakage_1s", "defect_chart_cached", row_idx, cable.serial_number, run_id))
                