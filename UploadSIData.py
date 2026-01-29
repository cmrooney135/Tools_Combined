
import os
import datetime as dt
from typing import Tuple, Dict, List, Optional
from SI_Test import SI_Test
from cable_factory import create_cable
import pandas as pd



import os
import datetime as dt
from typing import Tuple, Dict, List, Optional

import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

RESULT_TOKENS = {"PASS", "FAIL"}

# --- helpers already present in your file (keeping here for completeness) ---
def _read_all_lines(obj) -> List[str]:
    if isinstance(obj, (str, bytes, os.PathLike)):
        with open(obj, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    read = getattr(obj, "read", None)
    seek = getattr(obj, "seek", None)
    if callable(read):
        if callable(seek):
            obj.seek(0)
        raw = read()
        text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if callable(seek):
            obj.seek(0)
        return text.splitlines(True)
    raise TypeError(f"Unsupported input type: {type(obj).__name__}")

def _safe_float(tok: str) -> Optional[float]:
    t = (tok or "").strip()
    up = t.upper()
    if up in {"NAN", "INF", "+INF", "-INF"}:
        return float("nan") if up == "NAN" else (float("inf") if up in {"INF", "+INF"} else float("-inf"))
    try:
        return float(t)
    except Exception:
        return None

def _normalize_channel(ch: str) -> str:
    s = (ch or "").strip()
    return s.split(".")[-1] if "." in s else s  # 'J2.A01' -> 'A01'

# ---------- ZO READER: triplets per row ----------
def read_zo_triplet_dataframes(src, data_start: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Each Zo data row layout:
      [channel], [site],  PB_avg, PB_max, PB_min,  CBL_avg, CBL_max, CBL_min,  DIB_avg, DIB_max, DIB_min,  [PASS]
    We parse the FIRST 9 numeric tokens after the first 2 columns and map to three DataFrames.
    Returns: (df_paddleboard, df_cable, df_dib) with columns ['channel', 'site', 'avg', 'max', 'min']
    """
    lines = _read_all_lines(src)
    data_lines = [ln for ln in lines[data_start:] if ln.strip() and "," in ln]

    pb_rows, cbl_rows, dib_rows = [], [], []

    for ln in data_lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue

        channel = parts[0]   # keep as-is per your preference
        site    = parts[1]

        # Collect 9 numeric values after the first two columns
        numbers: List[float] = []
        for p in parts[2:]:
            up = p.upper()
            # If at a result token and we already have enough numbers, stop
            if up in RESULT_TOKENS and len(numbers) >= 9:
                break
            val = _safe_float(p)
            if val is not None:
                numbers.append(val)
                if len(numbers) == 9:
                    break

        if len(numbers) < 9:
            # Not enough data for three triplets; skip row (or raise if you prefer)
            continue

        pb_avg, pb_max, pb_min   = numbers[0:3]
        cbl_avg, cbl_max, cbl_min = numbers[3:6]
        dib_avg, dib_max, dib_min = numbers[6:9]

        pb_rows.append({"channel": channel, "site": site, "avg": pb_avg,  "max": pb_max,  "min": pb_min})
        cbl_rows.append({"channel": channel, "site": site, "avg": cbl_avg, "max": cbl_max, "min": cbl_min})
        dib_rows.append({"channel": channel, "site": site, "avg": dib_avg, "max": dib_max, "min": dib_min})

    cols = ["channel", "site", "avg", "max", "min"]
    df_pb  = pd.DataFrame(pb_rows,  columns=cols)
    df_cbl = pd.DataFrame(cbl_rows, columns=cols)
    df_dib = pd.DataFrame(dib_rows, columns=cols)

    return df_pb, df_cbl, df_dib

# ---------- SKEW READER: pair rows ----------

import os
from typing import List
import numpy as np
import pandas as pd

RESULT_TOKENS = {"PASS", "FAIL"}

def _read_all_lines(obj) -> List[str]:
    if isinstance(obj, (str, bytes, os.PathLike)):
        with open(obj, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    read = getattr(obj, "read", None)
    seek = getattr(obj, "seek", None)
    if callable(read):
        if callable(seek):
            obj.seek(0)
        raw = read()
        text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if callable(seek):
            obj.seek(0)
        return text.splitlines(True)
    raise TypeError(f"Unsupported input type: {type(obj).__name__}")

def _safe_float(tok: str):
    t = (tok or "").strip()
    up = t.upper()
    if up == "NAN":
        return np.nan
    if up in {"INF", "+INF"}:
        return float("inf")
    if up == "-INF":
        return float("-inf")
    try:
        return float(t)
    except Exception:
        return None

def read_skew_dataframe(src, data_start: int) -> pd.DataFrame:
    """
    Parse skew rows and emit the exact columns requested:
      ['channel site', 'first', 'second', 'measurement', 'delta', 'result']

    Input example row:
      'J2.A01 / J2.A02, 1.722, 1.722, NaN, 0.0002, PASS'

    Mapping:
      - 'channel site' = the pair string (e.g., 'J2.A01 / J2.A02')
      - 'first'        = first numeric token
      - 'second'       = second numeric token
      - 'measurement'  = literal 'skew [nS]'
      - 'delta'        = last numeric token in the row (robust to extra columns)
      - 'result'       = PASS/FAIL if present, else ''

    Notes:
      - Non-numeric tokens between numbers are ignored (kept as NaN if needed).
      - If fewer than 2 numeric tokens exist, 'first'/'second' are set to NaN.
    """
    lines = _read_all_lines(src)
    data_lines = [ln for ln in lines[data_start:] if ln.strip() and "," in ln]
    out_rows = []
    for ln in data_lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue

        pair = parts[0]  # e.g., 'J2.A01 / J2.A02'
        # detect trailing result
        result = parts[-1].strip().upper()
        has_result = result in RESULT_TOKENS
        tail_idx = len(parts) - 1 if has_result else len(parts)

        # collect numeric tokens from the remainder
        nums = []
        for t in parts[1:tail_idx]:
            val = _safe_float(t)
            if val is not None:
                nums.append(val)

        first  = nums[0] if len(nums) > 0 else np.nan
        second = nums[1] if len(nums) > 1 else np.nan
        delta  = nums[-1] if len(nums) > 0 else np.nan  # last numeric (skew)

        out_rows.append({
            "channel site": pair,
            "first": first,
            "second": second,
            "measurement": "skew [nS]",
            "delta": delta,
            "result": result if has_result else "",
        })

    # ensure exact column order
    cols = ["channel site", "first", "second", "measurement", "delta", "result"]
    return pd.DataFrame(out_rows, columns=cols)

def _get_filename(obj) -> str:
    if isinstance(obj, (str, bytes, os.PathLike)):
        return os.path.basename(str(obj))
    name = getattr(obj, "name", None)
    return os.path.basename(name) if name else f"<{type(obj).__name__}>"

def _read_all_lines(obj) -> List[str]:
    if isinstance(obj, (str, bytes, os.PathLike)):
        with open(obj, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()

    read = getattr(obj, "read", None)
    seek = getattr(obj, "seek", None)
    if callable(read):
        if callable(seek):
            obj.seek(0)
        raw = read()
        text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if callable(seek):
            obj.seek(0)
        return text.splitlines(True)

    raise TypeError(f"Unsupported input type: {type(obj).__name__}")
from typing import Tuple
import re

from typing import Tuple
import re

def _normalize_for_tokens(s: str) -> str:
    """
    Uppercase and replace all non-alphanumeric with single spaces.
    This collapses weird unicode dashes/underscores, NBSP, etc.
    """
    if s is None:
        return ""
    s = s.upper()
    # Replace any char that's NOT A-Z or 0-9 with a space
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s
from typing import Tuple
import re

def _normalize_for_tokens(s: str) -> str:
    """
    Uppercase and replace all non-alphanumeric with single spaces.
    This collapses underscores, dashes, unicode spaces, etc., into separators.
    """
    if s is None:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)   # non-alnum -> space
    s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
    return s

def parse_sample_id_value(sample_id_raw: str) -> Tuple[str, str]:
    """
    Extract (serial, position_tag) from a 'Sample ID' value robustly.

    Handles examples:
      'Sample ID:  SN-01ACA3A061_P2'
      'Sample ID: SN01ACA3A061-P1'
      'Sample ID: 01ACA3A061 P2'
      '::Sample ID:  sn–01aca3a061 p1'  # unicode dash and NBSP

    Returns:
      (serial_upper, position_tag_upper)
      position_tag is 'P1'/'P2' or '' if not present.

    Guarantees:
      - Never returns 'SN' as the serial.
    """
    raw = sample_id_raw or ""
    raw_upper = raw.upper()

    # --- Detect glued SN first on the raw string (e.g., SN01ACA3A061) ---
    m_glued = re.search(r"\bSN([A-Z0-9]{3,})\b", raw_upper)
    serial_glued = m_glued.group(1) if m_glued else ""

    # --- Normalize and tokenize for robust parsing ---
    norm = _normalize_for_tokens(raw)
    tokens = norm.split()  # e.g., ["SAMPLE", "ID", "SN", "01ACA3A061", "P2"]

    # --- Position tag detection from normalized tokens ---
    pos = ""
    for t in tokens:
        if t in ("P1", "P2"):
            pos = t
            break

    # --- Serial detection ---
    serial = ""

    # 1) If we detected glued SN on raw: use it
    if serial_glued:
        serial = serial_glued
    else:
        # 2) Prefer the token immediately after 'SN'
        try:
            sn_idx = next(i for i, t in enumerate(tokens) if t == "SN")
        except StopIteration:
            sn_idx = -1

        if sn_idx >= 0 and (sn_idx + 1) < len(tokens):
            cand = tokens[sn_idx + 1]
            if cand not in ("P1", "P2") and re.fullmatch(r"[A-Z0-9]{3,}", cand):
                serial = cand

        # 3) Fallback: first plausible alnum token (skip common words and P1/P2)
        if not serial:
            for t in tokens:
                if t in ("SN", "SAMPLE", "ID", "P1", "P2"):
                    continue
                # If your serial is exactly 10 chars, you can tighten this:
                # if re.fullmatch(r"[A-Z0-9]{10}", t):
                if re.fullmatch(r"[A-Z0-9]{5,}", t):  # >=5 to avoid tiny junk tokens
                    serial = t
                    break

    if not serial:
        raise ValueError(f"Could not extract serial from Sample ID: '{sample_id_raw}'")

    return serial.upper(), pos.upper()
from typing import Tuple
import re

def _normalize_for_tokens(s: str) -> str:
    """
    Uppercase and replace all non-alphanumeric with single spaces.
    This collapses weird unicode dashes/underscores, NBSP, etc.
    """
    if s is None:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)   # non-alnum -> space
    s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
    return s


from typing import Dict, Optional, Tuple
import datetime as dt
import re

def read_header(src) -> Tuple[Dict[str, Optional[str]], int]:
    """
    Returns:
      header: {
        'test_date': 'YYYY-MM-DD' or None,
        'test_time': 'HH:MM:SS' or None,
        'cable_sn':  'SERIAL',
        'sample_id': '<raw>',
        'position_tag': 'P1'|'P2'|''
      }
      data_start: index where data begins
    """
    lines = _read_all_lines(src)
    non_empty = [(i, ln.rstrip("\n")) for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty:
        raise ValueError("Empty file")

    first_idx, first_line = non_empty[0]
    fl_norm = first_line.strip()

    # --- Skew-only or one-line variants: recognize 'Sample ID' anywhere in line (case-insensitive)
    if re.search(r'(?i)\bsample\s*id\b', fl_norm):
        # Split on first ':' if present
        if ":" in fl_norm:
            sample_id_val = fl_norm.split(":", 1)[1].strip()
        else:
            m_sid = re.search(r'(?i)\bsample\s*id\b\s*(.*)$', fl_norm)
            sample_id_val = (m_sid.group(1).strip() if m_sid else "")

        candidate = sample_id_val if sample_id_val else fl_norm
        sn, pos = parse_sample_id_value(candidate)
        header = {
            "test_date": None,
            "test_time": None,
            "cable_sn": sn,
            "sample_id": sample_id_val if sample_id_val else fl_norm,
            "position_tag": pos.upper(),
        }
        return header, first_idx + 1

    # --- Zo-style header: date, time, 'Sample ID' triple ---
    if len(non_empty) < 3:
        raise ValueError("Expected either single 'Sample ID' or date/time/sample-id triple.")

    (_, raw_date), (_, raw_time), (sid_idx, sample_line) = non_empty[:3]

    date_obj = None
    for fmt in ("%d/%b/%Y", "%d/%B/%Y"):
        try:
            date_obj = dt.datetime.strptime(raw_date.strip(), fmt).date()
            break
        except ValueError:
            pass
    if date_obj is None:
        raise ValueError(f"Unrecognized date: '{raw_date}'")

    time_obj = None
    for fmt in ("%H:%M:%S", "%I:%M:%S %p"):
        try:
            time_obj = dt.datetime.strptime(raw_time.strip(), fmt).time()
            break
        except ValueError:
            pass
    if time_obj is None:
        raise ValueError(f"Unrecognized time: '{raw_time}'")

    if ":" not in sample_line and not re.search(r'(?i)\bsample\s*id\b', sample_line):
        raise ValueError(f"Expected 'Sample ID: <value>', got: {sample_line}")

    if ":" in sample_line:
        sample_id_val = sample_line.split(":", 1)[1].strip()
    else:
        m_sid = re.search(r'(?i)\bsample\s*id\b\s*(.*)$', sample_line)
        sample_id_val = (m_sid.group(1).strip() if m_sid else "")

    sn, pos = parse_sample_id_value(sample_id_val)

    header = {
        "test_date": date_obj.isoformat(),
        "test_time": time_obj.strftime("%H:%M:%S"),
        "cable_sn": sn,
        "sample_id": sample_id_val,
        "position_tag": pos.upper(),
    }
    return header, sid_idx + 1



def read_data_frame(src, data_start: int) -> pd.DataFrame:
    lines = _read_all_lines(src)
    data_lines = [ln for ln in lines[data_start:] if ln.strip() and "," in ln]
    rows = [[p.strip() for p in ln.split(",")] for ln in data_lines]
    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)
    cols = ["channel", "site"] + [f"m{i}" for i in range(1, max_len - 2)] + ["result"]
    df = pd.DataFrame(rows, columns=cols[:max_len])

    for c in df.columns[2:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["result"] = df["result"].str.upper().str.strip()
    return df

def _latest_si_tests_for_end(cable, end: str):
    tests = [t for t in cable.get_tests("si") if getattr(t, "test_end", "").upper() == end.upper()]
    return list(reversed(tests))  # prefer latest appended

def _find_pairable_test(cable, end: str, incoming: str):
    """
    Prefer pairing with a test that already has the opposite side:
      - incoming 'zo'   -> test that has skew_data and no zo_data
      - incoming 'skew' -> test that has zo_data and no skew_data
    Fallback: any test missing the incoming side.
    """
    candidates = _latest_si_tests_for_end(cable, end)

    if incoming == "zo":
        for t in candidates:
            if getattr(t, "zo_data", None) is None and getattr(t, "skew_data", None) is not None:
                return t
    elif incoming == "skew":
        for t in candidates:
            if getattr(t, "skew_data", None) is None and getattr(t, "zo_data", None) is not None:
                return t

    for t in candidates:
        if getattr(t, f"{incoming}_data", None) is None:
            return t

    return None


def process_SI_file(uploaded_file, cables):
    # 1) Detect type
    filename = _get_filename(uploaded_file)
    name_lower = filename.lower()
    if "skew" in name_lower:
        SI_type = "skew"
    elif "zo" in name_lower:
        SI_type = "zo"
    else:
        return None, None

    # 2) Parse header
    header, data_start = read_header(uploaded_file)
    serial_norm = header["cable_sn"].strip().upper()
    end = (header.get("position_tag") or "").strip().upper()  # "P1"/"P2"
    # 3) Find/create cable
    cable = next((c for c in cables if c.serial_number.strip().upper() == serial_norm), None)
    if cable is None:
        cable = create_cable(serial_norm)
        cables.append(cable)

    # 4) Parse data with the correct reader
    if SI_type == "zo":
        # Parse Zo into three DataFrames (PB, Cable, DIB)
        df_pb, df_cable, df_dib = read_zo_triplet_dataframes(uploaded_file, data_start)
        zo_payload = {
            "paddleboard": df_pb,
            "cable": df_cable,
            "dib": df_dib,
        }
    else:
        # Skew
        df_skew = read_skew_dataframe(uploaded_file, data_start)

    # 5) Find an existing pair container for this end
    existing = _find_pairable_test(cable, end, incoming=SI_type)

    if existing is not None:
        # Update the existing object
        if SI_type == "zo":
            existing.zo_data = zo_payload
            # Anchor timestamps only from Zo
            existing.test_date = header["test_date"]
            existing.test_time = header["test_time"]
        else:
            # "skew"
            existing.skew_data = df_skew
            # no timestamps set from Skew
        existing.SI_type = SI_type
        test = existing

    else:
        # Create a new container for this end
        if SI_type == "zo":
            test = SI_Test(
                test_type="si",
                test_end=end,
                zo_data=zo_payload,
                test_date=header["test_date"],   # Zo is the anchor timestamp
                test_time=header["test_time"],
            )
        else:
            test = SI_Test(
                test_type="si",
                test_end=end,
                skew_data=df_skew,
                test_date=None,
                test_time=None,
            )
        cable.add_test(test)

    return cable, test
