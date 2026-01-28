import streamlit as st
# app.py — Paradise-only
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.express as px
from _shared_ui import top_bar
from Cable import Cable
from Test import Test
from UploadData import process_csv
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

top_bar(page_icon="📈", title="⚡Tesla Summary", home_page_path="Home.py")


# --------------------------
# Single-family selector
# --------------------------
CABLE_FAMILY = "tesla"  # <---- Set to "tesla" in this version

# --- Session state init (call this FIRST) ---
def init_state():
    st.set_page_config(page_title=f"⚡Tesla Summary", page_icon= "📈", layout="wide")

    # Core containers
    st.session_state.setdefault("cables", [])
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("masters_by_type", {
        "continuity": pd.DataFrame(),
        "inv_continuity": pd.DataFrame(),
        "resistance": pd.DataFrame(),
        "inv_resistance": pd.DataFrame(),
        "leakage": pd.DataFrame(),
        "leakage_1s": pd.DataFrame(),
    })
    st.session_state.setdefault("failures_by_type", {})
    st.session_state.setdefault("_seen_run_headers", set())
    for ttype in ["continuity", "inv_continuity", "resistance", "inv_resistance"]:
        st.session_state.setdefault(f"_run_ctr_{ttype}", 0)

init_state()


# ---------- ALWAYS RENDER THE UPLOADER ----------
uploaded_files = st.file_uploader(
    f"Upload Easy-Wire CSVs for {CABLE_FAMILY.capitalize()}",
    type=["csv"],
    accept_multiple_files=True,
    key="uploader",
)

# --- Utilities ---
def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val not in (None, "", "NaT"):
                return val
    return None

def _merge_date_time(date_val, time_val):
    if date_val is not None and time_val is not None and str(date_val) != "" and str(time_val) != "":
        try:
            return pd.to_datetime(f"{date_val} {time_val}", errors="coerce")
        except Exception:
            pass
    if date_val not in (None, ""):
        try:
            return pd.to_datetime(date_val, errors="coerce")
        except Exception:
            pass
    if time_val not in (None, ""):
        try:
            today = pd.Timestamp.today().date()
            return pd.to_datetime(f"{today} {time_val}", errors="coerce")
        except Exception:
            pass
    return None

def get_serial_and_time_from_objects(cable_obj, test_obj):
    serial = _first_attr(cable_obj, [
        "serial_number", "serial", "serialNumber", "SerialNumber", "cable_serial",
        "CableSerial", "Cable_ID", "id", "ID", "sn", "SN", "name"
    ])
    date_val = _first_attr(test_obj, ["test_date", "date", "Date", "TestDate"])
    time_val = _first_attr(test_obj, ["test_time", "time", "Time", "timestamp", "start_time", "startTime"])
    test_time = _merge_date_time(date_val, time_val)
    return serial, test_time

def parse_channel_pair(s):
    def _norm_token(x):
        if x is None:
            return None
        xs = str(x).strip()
        if xs == "" or xs.lower() in ("none", "nan", "null"):
            return None
        return xs
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return _norm_token(s[0]), _norm_token(s[1])
    try:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return (None, None)
    except Exception:
        if s is None:
            return (None, None)
    t = str(s).strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    m = re.match(r"^\(\s*['\"]?([^,'\"]+)['\"]?\s*,\s*['\"]?([^,'\"]+)['\"]?\s*\)$", t)
    if not m:
        m = re.match(r"^['\"]?([^,'\"]+)['\"]?\s*,\s*['\"]?([^,'\"]+)['\"]?$", t)
    if m:
        a = _norm_token(m.group(1))
        b = _norm_token(m.group(2))
        return a, b
    return (None, None)

def _classify_failure(detail: str) -> str:
    if not isinstance(detail, str):
        return "Other"
    s = detail.strip().lower()
    patterns = [
        (r"\bwire\s*missing\b|\bmissing\s*wire\b|\bno\s*wire\b", "Wire Missing"),
        (r"\bshort(ed|s)?\b|\bshort\b", "Short"),
        (r"\bopen(ed|s)?\b|\bopen\b", "Open"),
        (r"\bmis[-\s]?wire\b|\bswapp?ed\b|\bwrong\s*wire\b", "Miswire"),
        (r"\breverse(d)?\b|\bpolarity\b", "Reverse"),
        (r"\bleak(age)?\b", "Leakage"),

        (r"\b(?:high|low)[\s\-:]*res(?:is(?:tance)?)?\b|\bout[\s\-]*of[\s\-]*specs?\b", "Resistance OOS"),
    ]

    for pat, cat in patterns:
        if re.search(pat, s):
            return cat
    return "Other"

def normalize_minimal(cable_obj, test_obj, source_name: str | None = None) -> pd.DataFrame:
    if not hasattr(test_obj, "data") or not isinstance(test_obj.data, pd.DataFrame):
        st.error(f"Test.data missing or not a DataFrame for run '{source_name or ''}'")
        return pd.DataFrame()

    df = test_obj.data.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ttype = (getattr(test_obj, "test_type", None) or getattr(test_obj, "name", None) or "").strip().lower() or "unknown"
    print(ttype)

    if "Channel" not in df.columns:
        st.error(
            f"❌ DataFrame for run '{source_name or ''}' lacks required column 'Channel'. "
            f"Columns present: {list(df.columns)}"
        )
        return pd.DataFrame()

    # ✅ More flexible measured-column resolution
    measured_candidates = [
        "Measured",                 # already normalized upstream
        "Measured_R (mOhm)",        # resistance/continuity
        "Measured_pA",          # leakage in microamps
        "Measured_pA",          # leakage with mu-symbol
        "Measured_I",               # generic current
        "Value",                    # last resort, if your parser uses a generic name
    ]
    meas_col = next((c for c in measured_candidates if c in df.columns), None)
    if not meas_col:
        st.error(
            f"❌ DataFrame for run '{source_name or ''}' lacks a measurable column. "
            f"Tried: {measured_candidates}. Columns present: {list(df.columns)}"
        )
        return pd.DataFrame()

    serial, test_time = get_serial_and_time_from_objects(cable_obj, test_obj)
    cable_type = getattr(cable_obj, "type", None)

    out = df.loc[:, ["Channel", meas_col]].copy()
    out.rename(columns={meas_col: "Measured"}, inplace=True)
    out["TestType"] = ttype
    out["CableSerial"] = serial
    out["TestTime"] = test_time
    out["CableType"] = cable_type
    out["Measured"] = pd.to_numeric(out["Measured"], errors="coerce")
    return out

def normalize_failures_minimal(cable_obj, test_obj, run_header: str | None = None, source_name: str | None = None) -> pd.DataFrame:
    if not hasattr(test_obj, "failure_data") or not isinstance(test_obj.failure_data, pd.DataFrame):
        return pd.DataFrame()
    df = test_obj.failure_data.copy()
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if "Detail" not in df.columns:
        return pd.DataFrame()
    ttype = (getattr(test_obj, "test_type", None) or getattr(test_obj, "name", None) or "").strip().lower() or "unknown"
    print(ttype)
    serial, test_time = get_serial_and_time_from_objects(cable_obj, test_obj)
    cable_type = getattr(cable_obj, "type", None)
    out = pd.DataFrame()
    out["Detail"] = df["Detail"].astype(str)
    if "Channel" in df.columns:
        out["Channel"] = df["Channel"]
        ch = df["Channel"].apply(parse_channel_pair).apply(pd.Series)
        ch.columns = ["FromPin", "ToPin"]
        out = pd.concat([out, ch], axis=1)
    else:
        out["Channel"] = None
        out["FromPin"] = None
        out["ToPin"] = None
    out["Category"] = out["Detail"].apply(_classify_failure)
    out["TestType"] = ttype
    out["CableSerial"] = serial
    out["TestTime"] = test_time
    out["CableType"] = cable_type
    if run_header:
        out["RunHeader"] = run_header
    return out

def add_failures_minimal(cable_obj, test_obj, run_header: str | None = None, source_name: str | None = None):
    norm = normalize_failures_minimal(cable_obj, test_obj, run_header=run_header, source_name=source_name)
    print("----------------failure data-------------------")
    print(norm)
    if norm.empty:
        return
    ttype = norm["TestType"].iloc[0] if "TestType" in norm.columns else (
        (getattr(test_obj, "type", None) or getattr(test_obj, "name", None) or "unknown").strip().lower()
    )
    print(ttype)
    failures = st.session_state.setdefault("failures_by_type", {})

    failures.setdefault(ttype, pd.DataFrame())
    df0 = failures[ttype]
    merged = pd.concat([df0, norm], ignore_index=True)
    subset = [c for c in ["RunHeader", "FromPin", "ToPin", "Category", "Detail"] if c in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")
    st.session_state["failures_by_type"][ttype] = merged

def build_channel_by_run_matrix(ttype: str, id_col: str = "CableSerial", time_col: str = "TestTime",
                                time_fmt: str = "%Y-%m-%d %H:%M", agg: str = "first",
                                cable_type: str | None = None) -> pd.DataFrame:
    df = st.session_state["masters_by_type"].get(ttype, pd.DataFrame())
    if df.empty:
        return pd.DataFrame(columns=["Channel"])
    df = df.copy()
    if cable_type:
        if "CableType" in df.columns:
            df = df[df["CableType"].str.lower() == cable_type.lower()]
        if df.empty:
            return pd.DataFrame(columns=["Channel"])
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            pass
    if "RunHeader" not in df.columns:
        def make_header(row):
            run_id = str(row.get(id_col, "") or "")
            t = row.get(time_col, None)
            if pd.notna(t):
                try:
                    t = pd.to_datetime(t)
                    t_str = t.strftime(time_fmt)
                except Exception:
                    t_str = str(t)
            else:
                t_str = ""
            header = f"{run_id} {t_str}".strip()
            return header if header else "UNKNOWN"
        df["RunHeader"] = df.apply(make_header, axis=1)
    wide = df.pivot_table(index="Channel", columns="RunHeader", values="Measured", aggfunc=agg)

    def parse_time_from_header(h: str):
        parts = h.rsplit(" ", 2)
        if len(parts) >= 3:
            maybe_time = f"{parts[-2]} {parts[-1]}"
            try:
                return pd.to_datetime(maybe_time)
            except Exception:
                return pd.NaT
        return pd.NaT

    ordered_cols = sorted(wide.columns, key=lambda h: (parse_time_from_header(h), str(h)))
    wide = wide[ordered_cols].reset_index()
    return wide

def add_to_master_minimal(cable_obj, test_obj, source_name: str | None = None) -> str | None:
    norm = normalize_minimal(cable_obj, test_obj, source_name=source_name)
    if norm.empty:
        return None
    if "TestType" in norm.columns and pd.notna(norm["TestType"]).any():
        ttype = str(norm["TestType"].iloc[0]).strip().lower()
    else:
        ttype = (getattr(test_obj, "type", None) or getattr(test_obj, "name", None) or "").strip().lower() or "unknown"
    serial = norm["CableSerial"].iloc[0] if "CableSerial" in norm.columns else None
    t = norm["TestTime"].iloc[0] if "TestTime" in norm.columns else None

    def fmt_time(val):
        if pd.notna(val):
            try:
                return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(val)
        return ""

    time_str = fmt_time(t)
    filename_stem = Path(source_name).stem if source_name else ""
    if serial and time_str:
        base_header = f"{serial} {time_str}"
    elif serial and filename_stem:
        base_header = f"{serial} {filename_stem}"
    elif filename_stem:
        ctr_key = f"_run_ctr_{ttype}"
        st.session_state.setdefault(ctr_key, 0)
        st.session_state[ctr_key] += 1
        base_header = f"{filename_stem} Run{st.session_state[ctr_key]:03d}"
    else:
        ctr_key = f"_run_ctr_{ttype}"
        st.session_state.setdefault(ctr_key, 0)
        st.session_state[ctr_key] += 1
        base_header = f"UNKNOWN Run{st.session_state[ctr_key]:03d}"

    seen_key = "_seen_run_headers"
    st.session_state.setdefault(seen_key, set())
    header = base_header
    bump = 2
    while header in st.session_state[seen_key]:
        header = f"{base_header} #{bump}"
        bump += 1
    st.session_state[seen_key].add(header)

    norm["RunHeader"] = header
    masters = st.session_state.setdefault("masters_by_type", {})
    masters.setdefault(ttype, pd.DataFrame())
    master = pd.concat([masters[ttype], norm], ignore_index=True)
    subset_cols = [c for c in ["RunHeader", "TestType", "Channel"] if c in master.columns]
    master = master.drop_duplicates(subset=subset_cols, keep="last") if subset_cols else master
    st.session_state["masters_by_type"][ttype] = master
    return header

def _compute_run_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["RunHeader", "Count", "Max", "Average", "StdDev"])
    work = df_long.copy()
    if "Measured" in work.columns:
        work["Measured"] = pd.to_numeric(work["Measured"], errors="coerce")
    if "RunHeader" not in work.columns:
        def _fallback_header(row):
            sid = str(row.get("CableSerial", "") or "")
            t = row.get("TestTime", None)
            try:
                t = pd.to_datetime(t)
                t_str = t.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(t) else ""
            except Exception:
                t_str = str(t) if t is not None else ""
            label = f"{sid} {t_str}".strip()
            return label if label else "UNKNOWN"
        work["RunHeader"] = work.apply(_fallback_header, axis=1)
    grp = work.groupby("RunHeader", dropna=False)["Measured"]
    stats = grp.agg(Count="count", Max="max", Average="mean", StdDev="std").reset_index()
    for c in ["Max", "Average", "StdDev"]:
        if c in stats.columns:
            stats[c] = stats[c].astype(float).round(3)
    if "TestTime" in work.columns:
        tt = (work[["RunHeader", "TestTime"]]
              .dropna(subset=["TestTime"])
              .drop_duplicates(subset=["RunHeader"]))
        stats = stats.merge(tt, on="RunHeader", how="left")
        try:
            stats["TestTime"] = pd.to_datetime(stats["TestTime"], errors="coerce")
            stats = stats.sort_values(by=["TestTime", "RunHeader"], kind="stable")
        except Exception:
            stats = stats.sort_values(by=["RunHeader"], kind="stable")
    else:
        stats = stats.sort_values(by=["RunHeader"], kind="stable")
    return stats

def _df_for(ttype: str, ctype: str) -> pd.DataFrame:
    masters = st.session_state.get("masters_by_type", {})
    df_long = masters.get(ttype, pd.DataFrame())
    if df_long.empty or "CableType" not in df_long.columns:
        return pd.DataFrame()
    df_ct = df_long[df_long["CableType"].astype(str).str.strip().str.lower() == ctype.lower()].copy()
    if df_ct.empty:
        return df_ct
    df_ct["Measured"] = pd.to_numeric(df_ct["Measured"], errors="coerce")
    df_ct = df_ct.dropna(subset=["Measured"])
    return df_ct

def _max_per_run(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["RunHeader", "MaxMeasured"])
    out = (df_long.groupby("RunHeader", dropna=False)["Measured"]
           .max().reset_index().rename(columns={"Measured": "MaxMeasured"}))
    if "TestTime" in df_long.columns:
        tm = (df_long[["RunHeader", "TestTime"]]
              .dropna(subset=["TestTime"])
              .drop_duplicates(subset=["RunHeader"]))
        out = out.merge(tm, on="RunHeader", how="left")
        try:
            out["TestTime"] = pd.to_datetime(out["TestTime"], errors="coerce")
            out = out.sort_values(by=["TestTime", "RunHeader"], kind="stable")
        except Exception:
            out = out.sort_values(by=["RunHeader"], kind="stable")
    else:
        out = out.sort_values(by=["RunHeader"], kind="stable")
    return out

def _clip_for_overflow(values: np.ndarray, overflow: float | None):
    if overflow is None or np.isnan(overflow):
        return values, ""
    clipped = np.minimum(values, overflow)
    return clipped, f"(Rightmost bin includes values ≥ {overflow:g})"

def _hist(
    data: np.ndarray,
    title: str,
    bin_size: float | None = None,
    overflow: float | None = None,
    x_label: str = "Measured",
    log_x: bool = False,
):
    clipped, overflow_note = _clip_for_overflow(np.asarray(data, dtype=float), overflow)
    hist_kwargs = {}
    if bin_size is not None and not np.isnan(bin_size) and bin_size > 0:
        xbins = dict(size=bin_size)
    else:
        xbins = None
    fig = px.histogram(
        x=clipped,
        nbins=None if xbins else 50,
        title=title + (f" {overflow_note}" if overflow_note else ""),
        labels={"x": x_label, "y": "Count"},
    )
    if xbins:
        fig.update_traces(xbins=xbins)
    if overflow is not None and not np.isnan(overflow):
        xmin = float(np.nanmin(clipped)) if clipped.size else 0.0
        fig.update_xaxes(range=[xmin, float(overflow)])
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(
        bargap=0.05,
        template="plotly_white",
        title_x=0.02,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def _processed_set() -> set[str]:
    s = st.session_state.get("processed_files", set())
    if not isinstance(s, set):
        s = set(s)
        st.session_state["processed_files"] = s
    return s

# --------------------------
# Ingest uploaded files
# --------------------------
if uploaded_files:
    for uf in uploaded_files:
       
        processed = set(st.session_state.get("processed_files", set()))
        processed = _processed_set()
        # Skip already-processed files
        if uf.name in processed:
            continue

        cable, test = process_csv(uf, st.session_state["cables"])
        if cable is not None:
            st.session_state["cables"].append(cable)
            run_header = add_to_master_minimal(cable, test, source_name=uf.name)
            add_failures_minimal(cable, test, run_header=run_header, source_name=uf.name)
            processed.add(uf.name)
            st.session_state["processed_files"] = processed


# --------------------------
# Download & Visuals (single family)
# --------------------------
TEST_TYPES = ["continuity", "inv_continuity", "resistance", "inv_resistance", "leakage", "leakage_1s"]
masters = st.session_state.get("masters_by_type", {})

def render_single_family():
    st.markdown(f"### {CABLE_FAMILY.capitalize()}")
    cols = st.columns(2)

    for idx, ttype in enumerate(TEST_TYPES):
        df_long = masters.get(ttype, pd.DataFrame())
        if not df_long.empty and "CableType" in df_long.columns:
            df_ct = df_long[
                df_long["CableType"].astype(str).str.strip().str.lower() == CABLE_FAMILY.lower()
            ]
        else:
            df_ct = pd.DataFrame()

        has_data = not df_ct.empty

        with cols[idx % 2]:
            st.write(f"**{ttype}**")

            if has_data:
                wide_df = build_channel_by_run_matrix(ttype, cable_type=CABLE_FAMILY)
                st.caption(f"{wide_df.shape[0]} channels × {max(0, wide_df.shape[1]-1)} runs")

                csv_bytes = wide_df.to_csv(index=False).encode("utf-8")
                filename = f"{CABLE_FAMILY}_{ttype}_master_wide.csv"
            else:
                csv_bytes = b""
                filename = ""

            st.download_button(
                label="Download master CSV",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
                disabled=not has_data,
                key=f"download_{CABLE_FAMILY}_{ttype}_wide"
            )

            # --- Per-run summary table ---
            if has_data:
                stats_df = _compute_run_stats(df_ct)
                st.caption("Per-run summary (one row per uploaded test run)")
                st.dataframe(
                    stats_df,
                    width="stretch",            # <-- was use_container_width=True
                    hide_index=True
                )
            else:
                st.info("No data yet for this test type.")

            # --- Histograms + controls ---
            if has_data:
                st.markdown("**Histograms**")
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

                with c1:
                    bin_size = st.number_input(
                        "Bin size",
                        min_value=0.0, value=0.0, step=0.0,
                        help="Set to 0 for auto binning. Units match the 'Measured' column.",
                        key=f"bin_{CABLE_FAMILY}_{ttype}"      # <-- UNIQUE KEY
                    )
                with c2:
                    overflow_val = st.number_input(
                        "Overflow threshold",
                        min_value=0.0, value=0.0, step=0.0,
                        help="If > 0, values ≥ threshold are grouped into the rightmost bin.",
                        key=f"overflow_{CABLE_FAMILY}_{ttype}" # <-- UNIQUE KEY
                    )
                    overflow = None if overflow_val == 0 else overflow_val


                # Build data sources
                df_ct_clean = _df_for(ttype, CABLE_FAMILY)
                df_max = _max_per_run(df_ct_clean)

                if not df_max.empty:
                    figA = _hist(
                        data=df_max["MaxMeasured"].to_numpy(),
                        title=f"{CABLE_FAMILY.capitalize()} · {ttype} — Maximum per run",
                        bin_size=bin_size if bin_size and bin_size > 0 else None,
                        overflow=overflow,
                        x_label="Max Measured (per run)",
                    )
                    st.plotly_chart(figA, width="stretch")    # <-- was use_container_width=True
                else:
                    st.info("No runs available for the max-per-run histogram.")
                # --- All measurements combined histogram ---
                if not df_ct_clean.empty:
                    all_vals = df_ct_clean["Measured"].to_numpy()

                    st.caption(f"All measurements combined: {len(all_vals):,} points")
                    figB = _hist(
                        data=all_vals,
                        title=f"{CABLE_FAMILY.capitalize()} · {ttype} — All measurements (combined)",
                        bin_size=bin_size if bin_size and bin_size > 0 else None,
                        overflow=overflow,
                        x_label="Measured",
                    )
                    st.plotly_chart(figB, width="stretch")  # <-- was use_container_width=True
                else:
                    st.info("No raw measurements available to build the combined histogram.")
                # --- Failures ---
                st.markdown("**Failure breakdown**")
                failures_all = st.session_state.get("failures_by_type", {}).get(ttype, pd.DataFrame())
                print("----------------failure data-------------------")
                print(failures_all)
                if failures_all.empty or "Category" not in failures_all.columns:
                    st.info("No failure records available yet.")
                else:
                    fct = failures_all.copy()
                    if "CableType" in fct.columns:
                        fct = fct[fct["CableType"].astype(str).str.strip().str.lower() == CABLE_FAMILY.lower()]

                    if fct.empty:
                        st.info("No failures for this cable family.")
                    else:
                        overall = (
                            fct.groupby("Category", dropna=False)
                               .size()
                               .reset_index(name="Count")
                               .sort_values(["Count", "Category"], ascending=[False, True])
                        )
                        fig_overall = px.bar(
                            overall,
                            x="Category",
                            y="Count",
                            title=f"{CABLE_FAMILY.capitalize()} · {ttype} — Failures across all runs",
                            text="Count",
                        )
                        fig_overall.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10))
                        fig_overall.update_traces(textposition="outside")
                        st.plotly_chart(fig_overall, width="stretch")  # <-- was use_container_width=True

                        # Failure table
                        tbl = fct.copy()
                        has_from_to = {"FromPin", "ToPin"}.issubset(tbl.columns)
                        if has_from_to:
                            tbl["ChannelFmt"] = tbl.apply(
                                lambda r: f"({r['FromPin']}, {r['ToPin']})"
                                if pd.notna(r.get("FromPin")) or pd.notna(r.get("ToPin")) else None,
                                axis=1,
                            )
                        else:
                            tbl["ChannelFmt"] = None

                        if "Channel" in tbl.columns:
                            tbl.loc[tbl["ChannelFmt"].isna(), "ChannelFmt"] = tbl["Channel"].astype(str)

                        show_cols = [c for c in ["RunHeader", "ChannelFmt", "Category"] if c in tbl.columns]
                        rename_map = {"ChannelFmt": "Channel"}
                        sort_cols = [c for c in ["RunHeader", "ChannelFmt", "Category"] if c in tbl.columns]
                        if sort_cols:
                            tbl = tbl.sort_values(by=sort_cols, kind="stable")
                        tbl_view = tbl[show_cols].rename(columns=rename_map)

                        st.caption("Failure records (aggregated across all runs)")
                        st.dataframe(tbl_view, width="stretch", hide_index=True)  # <-- was use_container_width=True

                        csv_fail = tbl_view.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download failures (CSV)",
                            data=csv_fail,
                            file_name=f"{CABLE_FAMILY}_{ttype}_failures.csv",
                            mime="text/csv",
                            key=f"download_{CABLE_FAMILY}_{ttype}_fail_tbl",
                        )


render_single_family()
