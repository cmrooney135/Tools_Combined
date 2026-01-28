import streamlit as st 

import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st

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

ensure_state()

import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st

# ---- existing helper from earlier ----
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

# ---- build wide matrices for SKEW (delta) ----
def build_skew_wide_matrix(cables, *, end: str) -> pd.DataFrame:
    """
    Build a wide matrix for Skew where rows are 'channel site' and
    columns are '<Serial> <M/D/YYYY H:MM>', values are 'delta'.
    """
    cols = {}
    existing = set()

    for cable, test in _iter_si_tests(cables):
        if (getattr(test, "test_end", "") or "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # Find 'channel site' column and 'delta'
        ch_col = next((c for c in df.columns if str(c).strip().lower() == "channel site"), None)
        d_col  = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
        if ch_col is None or d_col is None:
            continue

        label = _format_col_label(cable.serial_number, getattr(test, "test_date", None), getattr(test, "test_time", None))
        label = _dedup_label(label, existing)
        existing.add(label)

        s = pd.to_numeric(df[d_col], errors="coerce")
        s.index = df[ch_col]
        cols[label] = s

    if not cols:
        return pd.DataFrame(columns=["Channels"])

    wide = pd.DataFrame(cols)
    wide.insert(0, "Channels", wide.index)
    wide.reset_index(drop=True, inplace=True)
    return wide

def _collect_skew_delta(cables, end: str) -> pd.DataFrame:
    """
    Gather all 'delta' values from SI_Test.skew_data for the given end ('P1' or 'P2')
    across all cables, returning a DataFrame with a single column 'value'.
    """
    deltas = []
    for cable, test in _iter_si_tests(cables):
        if getattr(test, "test_end", "").upper() != end.upper():
            continue
        df = getattr(test, "skew_data", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # be robust to column case/spacing
            colname = next((c for c in df.columns if str(c).strip().lower() == "delta"), None)
            if colname:
                series = pd.to_numeric(df[colname], errors="coerce").dropna()
                if not series.empty:
                    deltas.extend(series.tolist())
    return pd.DataFrame({"value": deltas})

def _iter_si_tests(cables):
    """Yield each SI_Test found on all cables."""
    for c in cables:
        for t in getattr(c, "TESTS", {}).get("si", []) or []:
            yield c, t

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

st.title("SI Tools - Paradise ")
st.set_page_config(page_title="SI Tools", page_icon = "🐍", layout="wide")
uploaded_files = st.file_uploader("Upload your SI files", type="DAT", accept_multiple_files=True)

cables = st.session_state["cables"]
processed = st.session_state["processed_files"]

if uploaded_files:
    for uf in uploaded_files:
        if uf.name in processed:
            continue

        cable, test = process_SI_file(uf, cables)
    
        # Only append a *real* Cable instance (not None)
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

        
st.divider()
st.subheader("Zo Histograms")

categories = [
    ("paddleboard", "Paddleboard"),
    ("cable",       "Cable"),
    ("dib",         "DIB"),
]

ends = ["P1", "P2"]  # will quietly show "No data" when an end isn’t present

# Layout: Tabs per category, inside each: two columns per end, with Max/Min stacked.
tabs = st.tabs([label for _, label in categories])
for (cat_key, cat_label), tab in zip(categories, tabs):
    with tab:
        st.write(f"**Category:** {cat_label}")
        cols = st.columns(2)
        for i, end in enumerate(ends):
            with cols[i]:
                st.markdown(f"##### End: {end}")
                # Max
                df_max = _collect_zo_values(st.session_state["cables"], category=cat_key, metric="max", end=end)
                _histogram(df_max, title=f"{cat_label} — {end} — Max Impedance", x_label="Max Impedance")
                # Min
                df_min = _collect_zo_values(st.session_state["cables"], category=cat_key, metric="min", end=end)
                _histogram(df_min, title=f"{cat_label} — {end} — Min Impedance", x_label="Min Impedance")

st.divider()
st.subheader("Skew Histograms (Δ skew [nS])")

ends = ["P1", "P2"]
cols = st.columns(2)

for i, end in enumerate(ends):
    with cols[i]:
        df_delta = _collect_skew_delta(st.session_state["cables"], end=end)
        _histogram(
            df_delta,
            title=f"Skew — {end} — Δ (nS)",
            x_label="Δ skew [nS]"
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

# SKEW: per end (P1/P2), values = delta
with st.expander("Skew — Δ (nS)", expanded=False):
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

