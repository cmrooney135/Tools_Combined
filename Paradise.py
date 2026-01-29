
import re
from Cable import Cable
import pandas as pd
import ast
from typing import Tuple, Optional, List, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from collections import OrderedDict, defaultdict


import numpy as np
@dataclass
class Paradise(Cable):
    #Channel ORDERS GO HERE
    p1_first_row  = [f"E{i}P2" for i in range(87, 0, -1)]
    p1_second_row = [f"D{i}P2" for i in range(87, 0, -1)]
    p1_third_row  = [f"C{i}P2" for i in range(87, 0, -1)]
    p1_fourth_row = [f"B{i}P2" for i in range(87, 0, -1)]
    p1_fifth_row  = [f"A{i}P2" for i in range(87, 0, -1)]


    dib_first_row  = [f"A{i}DIB" for i in range(1, 88)]
    dib_second_row = [f"B{i}DIB" for i in range(1, 88)]
    dib_third_row  = [f"C{i}DIB" for i in range(1, 88)]
    dib_fourth_row = [f"D{i}DIB" for i in range(1, 88)]
    dib_fifth_row  = [f"E{i}DIB" for i in range(1, 88)]
    dib_sixth_row  = [f"F{i}DIB" for i in range(1, 88)]
    dib_seventh_row= [f"G{i}DIB" for i in range(1, 88)]

    p2_fifth_row  = [f"E{i}P1" for i in range(1,88)]
    p2_fourth_row = [f"D{i}P1" for i in range(1,88)]
    p2_third_row  = [f"C{i}P1" for i in range(1,88)]
    p2_second_row = [f"B{i}P1" for i in range(1,88)]
    p2_first_row  = [f"A{i}P1" for i in range(1,88)]

    p2_order = p1_first_row + p1_second_row + p1_third_row + p1_fourth_row + p1_fifth_row
    p1_order = p2_first_row + p2_second_row + p2_third_row + p2_fourth_row + p2_fifth_row
    dib_order = dib_first_row + dib_second_row + dib_third_row + dib_fourth_row + dib_fifth_row +dib_sixth_row + dib_seventh_row

    CHANNEL_WITH_ANN_RE = re.compile(
        r"""
        \b
        (?P<code>[A-G]\d+(?:P\d+)?)
        \b
        (?:
            \s*
            (?:
                \((?P<paren>[^)]+)\)
            |
                \s+(?P<bare>[A-Za-z0-9_]+)
            )
        )?
        """,
        re.VERBOSE
    )

    @property
    def type(self):
        return "paradise"
    @staticmethod
    def _extract_num(label: str) -> int:
        """Return the first integer found in a label like 'A17DIB' -> 17."""
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else None

    @staticmethod
    def _slice_row(labels, values, row_letter):
        """
        From a flat (labels, values), return the values belonging to a specific row letter,
        preserving the order already in labels.
        """
        row_vals = []
        row_nums = []
        for lab, val in zip(labels, values):
            if lab.startswith(row_letter):
                row_vals.append(val)
                row_nums.append(Paradise._extract_num(lab))
        return row_vals, row_nums
    @staticmethod
    def _first_token(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"[A-Za-z0-9_]+", text)
        return m.group(0) if m else None
    @staticmethod
    def _grid_from(values, order, row_letters):
        """
        Build a (rows x cols) grid from a 1D values array aligned to 'order'.
        - row_letters: iterable like ['A','B','C',...] in the exact order you want on Y.
        Returns: (grid, x_ticks)
            grid: list of lists of floats (NaN allowed)
            x_ticks: column tick labels (picked from the first non-empty row’s numbers)
        """
        # Use the given 'order' as the authoritative sequence
        labels = order  # already in the correct order by construction
        grid = []
        first_row_nums = None

        for letter in row_letters:
            row_vals, row_nums = Paradise._slice_row(labels, values, letter)
            grid.append(row_vals)
            # capture x ticks from the first row that has data
            if first_row_nums is None and row_nums:
                first_row_nums = row_nums

        # If we never found any numbers (completely empty data), set a default
        if first_row_nums is None:
            # try to infer column count from the longest row we built
            max_len = max((len(r) for r in grid), default=0)
            first_row_nums = list(range(1, max_len + 1))

        return grid, first_row_nums
    def bucket_reason(self, text: str) -> str:
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

    def parse_channel(self, channel: str):
        """
        Parse a channel string into (row_letter, pin_number, panel).

        Examples:
            'C58DIB' -> ('C', 58, 'DIB')
            'B12P1'  -> ('B', 12, 'P1')
            'e07p2'  -> ('E', 7, 'P2')

        Returns:
            (row_letter: str, pin_number: int, panel: str) on success
            (None, None, None) on failure
        """
        if not isinstance(channel, str):
            return None, None, None

        s = channel.strip().upper()

        # Pattern:
        #   Row: single letter A–Z (you can restrict to A–E if you want)
        #   Pin: one or more digits
        #   Panel: exactly 'DIB' or 'P1' or 'P2'
        m = re.match(r'^([A-Z])\s*0*([0-9]+)\s*(DIB|P1|P2)$', s)
        if not m:
            return None, None, None

        row = m.group(1)
        pin = int(m.group(2))          # handles leading zeros gracefully
        panel = m.group(3)

        # Optional: enforce expected row set
        # if row not in {'A','B','C','D','E'}:
        #     return None, None, None

        return row, pin, panel
    def extract_channel(self, text: str) -> Tuple[str, ...]:
        results: List[str] = []

        for m in self.CHANNEL_WITH_ANN_RE.finditer(text):
            code  = m.group("code")
            token = (
                self._first_token(m.group("paren"))
                or self._first_token(m.group("bare"))
                or ""
            )
            results.append(f"{code}{token}")

        return tuple(results)
    
    def build_ordered_arrays(
        self,
        df: pd.DataFrame,
        *,
        value_col: str | None = None,
        agg: str = "max",          # {"max","min","last"}
        fill: float = np.nan,      # fill for missing entries
        return_type: str = "list", # {"list","numpy","series"}
        verbose: bool = False,
    ):
        """
        Build three arrays in the exact order of self.dib_order, self.p1_order, self.p2_order.
        For each pair in df["Channel"], map its value into DIB and the corresponding P1/P2 array.

        Returns:
            (dib_values, p1_values, p2_values)
            - If return_type="list": lists
            - If "numpy": numpy arrays
            - If "series": pandas Series with index = the corresponding order lists
        """
        # ---- validate orders present ----
        for name in ("p1_order", "p2_order", "dib_order"):
            if not hasattr(self, name) or not getattr(self, name):
                raise AttributeError(f"Missing self.{name}. Populate order lists first.")

        p1_order = self.p1_order
        p2_order = self.p2_order
        dib_order = self.dib_order

        # ---- choose numeric column ----
        if value_col is None:
            if "Measured_R (mOhm)" in df.columns:
                value_col = "Measured_R (mOhm)"
            elif "Measured_pA" in df.columns:
                value_col = "Measured_pA"
        if value_col is None:
            raise ValueError("No numeric value column available.")
        if "Channel" not in df.columns:
            raise ValueError("Expected a 'Channel' column in the DataFrame.")

        # ---- fast label -> index maps ----
        dib_idx = {lab: i for i, lab in enumerate(dib_order)}
        p1_idx  = {lab: i for i, lab in enumerate(p1_order)}
        p2_idx  = {lab: i for i, lab in enumerate(p2_order)}

        # ---- init arrays ----
        dib_vals = [fill] * len(dib_order)
        p1_vals  = [fill] * len(p1_order)
        p2_vals  = [fill] * len(p2_order)

        # ---- aggregator ----
        def put(cur, new):
            # missing -> new
            if pd.isna(cur):
                return new
            if agg == "max":
                return max(cur, new)
            if agg == "min":
                return min(cur, new)
            if agg == "last":
                return new
            raise ValueError(f"Unknown agg: {agg}")

        # ---- parsing helpers ----
        


        def _blankify(x):
                """Normalize empty/blank representations to None ([], '', '[]', 'None', None)."""
                if x is None:
                    return None
                # Empty containers -> None
                if isinstance(x, (list, tuple, set, dict)) and len(x) == 0:
                    return None
                # Strings that are empty or read like empties -> None
                if isinstance(x, str):
                    s = x.strip().strip("\"'")
                    if s == "" or s in ("[]", "None", "none", "NULL", "null", "NaN", "nan"):
                        return None
                return x

        def parse_channel_cell(cell):
                """
                Accepts:
                (['A44DIB'], [])           # tuple/list of two lists
                ['A44DIB', 'D44P2']        # flat list of two strings
                ('A44DIB', 'D44P2')
                "(['A44DIB'], [])"         # string-serialized
                "A44DIB -> D44P2", "A44DIB|D44P2", etc.

                Returns (a, b) where each is a string label or None.
                """
                if pd.isna(cell):
                    return None, None

                # 1) Structured container first
                if isinstance(cell, (list, tuple)):
                    if len(cell) == 2:
                        a, b = cell
                        # If side is a list/tuple, pull first item if present; else treat as blank.
                        if isinstance(a, (list, tuple)):
                            a = a[0] if len(a) else None
                        if isinstance(b, (list, tuple)):
                            b = b[0] if len(b) else None
                        a = _blankify(a)
                        b = _blankify(b)
                        # Convert to str only if not None
                        a = str(a).strip().strip("\"'") if a is not None else None
                        b = str(b).strip().strip("\"'") if b is not None else None
                        return a, b

                # 2) String: try literal_eval to structured, then fallback to delimiter split
                if isinstance(cell, str):
                    s = cell.strip().strip("\"'")
                    # Try to parse literal Python representation
                    try:
                        obj = ast.literal_eval(s)
                        return parse_channel_cell(obj)
                    except Exception:
                        pass

                    # Fallback split on common separators
                    tmp = s
                    for sep in ["->", "—", "–", "|", " to ", " TO ", ":", ",", "-"]:
                        tmp = tmp.replace(sep, "|")
                    parts = [p.strip().strip("\"'") for p in tmp.split("|") if p.strip()]

                    if len(parts) == 2:
                        a = _blankify(parts[0])
                        b = _blankify(parts[1])
                        a = str(a) if a is not None else None
                        b = str(b) if b is not None else None
                        return a, b

                # Anything else
                return None, None

        

        def split_roles(a: str | None, b: str | None):
            """
            Return (dib_label, p_label) allowing single-end rows.
            - dib_label endswith 'DIB'
            - p_label  endswith 'P1' or 'P2'
            If only DIB present, returns (dib, None).
            If only P1/P2 present, returns (None, p).
            """
            def clean(s):
                if not s:
                    return None
                s = re.sub(r"\s+", "", str(s)).strip("\"'")
                # Treat blanks/empties as None
                if s in ("", "[]", "None", "none", "NULL", "null"):
                    return None
                return s

            a = clean(a)
            b = clean(b)
            labels = [lab for lab in (a, b) if lab]

            dib_label = next((lab for lab in labels if lab.endswith("DIB")), None)
            p_label   = next((lab for lab in labels if lab.endswith("P1") or lab.endswith("P2")), None)
            return dib_label, p_label




        # ---- main pass ----
        parsed_total = matched_total = wrote_dib = wrote_p1 = wrote_p2 = 0
        skipped_outside_orders = 0

        for _, row in df.iterrows():
            a, b = parse_channel_cell(row["Channel"])
            if (a is None and b is None):
                continue
            parsed_total += 1

            dib_label, p_label = split_roles(a, b)
            if (dib_label is None and p_label is None):
                continue
            matched_total += 1

            # Skip rows whose present labels are outside your orders
            if dib_label is not None and dib_label not in dib_idx:
                skipped_outside_orders += 1
                continue
            if p_label is not None and (p_label not in p1_idx and p_label not in p2_idx):
                skipped_outside_orders += 1
                continue

            try:
                val = float(row[value_col])
            except Exception:
                continue

            # Update DIB if present
            if dib_label is not None:
                di = dib_idx[dib_label]
                dib_vals[di] = put(dib_vals[di], val)
                wrote_dib += 1


            if p_label is not None:
                if p_label.endswith("P1"):
                    pi = p1_idx[p_label]
                    p1_vals[pi] = put(p1_vals[pi], val)
                    wrote_p1 += 1
                elif p_label.endswith("P2"):
                    pi = p2_idx[p_label]
                    p2_vals[pi] = put(p2_vals[pi], val)
                    wrote_p2 += 1


        if verbose:
            print(
                f"[build_ordered_arrays] parsed={parsed_total}, matched={matched_total}, "
                f"wrote_dib={wrote_dib}, wrote_p1={wrote_p1}, wrote_p2={wrote_p2}, "
                f"skipped_outside_orders={skipped_outside_orders}, value_col='{value_col}', agg='{agg}'"
            )
            # small peek
            for name, arr, order in [
                ("DIB", dib_vals, dib_order),
                ("P1", p1_vals, p1_order),
                ("P2", p2_vals, p2_order),
            ]:
                # show the first 3 non-NaN entries with labels
                sample = [(order[i], v) for i, v in enumerate(arr) if not pd.isna(v)][:3]
                print(f"  sample {name}:", sample or "<empty>")

        # ---- return in requested type ----
        if return_type == "list":
            return dib_vals, p1_vals, p2_vals
        elif return_type == "numpy":
            return np.asarray(dib_vals, dtype=float), np.asarray(p1_vals, dtype=float), np.asarray(p2_vals, dtype=float)
        elif return_type == "series":
            return (pd.Series(dib_vals, index=dib_order, name="DIB"),
                    pd.Series(p1_vals,  index=p1_order, name="P1"),
                    pd.Series(p2_vals,  index=p2_order, name="P2"))
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
    def _parse_channel_pair(self, cell: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize a 'Channel' cell into a pair of endpoint labels (left, right).

        Accepts:
            - ('A44DIB', 'D44P2')
            - (['A44DIB'], ['D44P2'])
            - "(['A44DIB'], ['D44P2'])"
            - "A44DIB|D44P2", "A44DIB -> D44P2", "A44DIB-D44P2", "A44DIB, D44P2"
            - "A44DIB"  -> ('A44DIB', None)
        """
        def _clean(s: Any) -> Optional[str]:
            if s is None:
                return None
            if isinstance(s, (list, tuple)) and len(s) > 0:
                s = s[0]
            s = str(s).strip()
            s = s.strip("\"'")
            s = s.strip(" ")
            s = re.sub(r"\s+", "", s)
            return s or None

        # 1) Already a list/tuple
        if isinstance(cell, (list, tuple)):
            if len(cell) == 2:
                a, b = _clean(cell[0]), _clean(cell[1])
                return a, b
            if len(cell) == 1:
                a = _clean(cell[0])
                return a, None
            cell = str(cell)  # fallthrough to string handling

        # 2) String: try literal_eval
        if isinstance(cell, str):
            s = cell.strip().strip("\"'")
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, (list, tuple)):
                    if len(obj) == 2:
                        a, b = _clean(obj[0]), _clean(obj[1])
                        return a, b
                    if len(obj) == 1:
                        a = _clean(obj[0])
                        return a, None
            except Exception:
                pass

            # 3) Fallback: split on common delimiters
            tmp = s
            for sep in ["->", "—", "–", "|", " to ", " TO ", ":", ",", ";", "-"]:
                tmp = tmp.replace(sep, "|")

            parts = [
            seg.strip().strip("\"'").strip("[")
                for seg in tmp.split("|")
                if seg.strip()
            ]

            if len(parts) >= 2:
                return _clean(parts[0]), _clean(parts[1])
            if len(parts) == 1:
                return _clean(parts[0]), None

        return None, None
   
    def build_bucket_arrays(
            self,
            error_df: pd.DataFrame,
            *,
            category_col: str | None = None,   # if None, derive from "Detail" via bucket_reason
            prefer_first: bool = True,         # True: first category wins; False: last wins
            blank_label: str = "—",            # use something visible so hover never looks empty
            verbose: bool = False,
        ):
            """
            Produce three categorical arrays aligned to canonical orders (DIB/P1/P2).
            Each row in error_df contributes its category to whichever endpoints exist in 'Channel'.
            """

            if error_df is None or error_df.empty:
                return (
                    [blank_label] * len(self.dib_order),
                    [blank_label] * len(self.p1_order),
                    [blank_label] * len(self.p2_order),
                    [blank_label],  # categories universe
                )

            # --- Resolve/derive Category per row ---
            if category_col is None:
                cats_per_row = []
                for _, row in error_df.iterrows():
                    detail = row.get("Detail", None)
                    cat = self.bucket_reason(detail) if pd.notna(detail) else blank_label
                    cats_per_row.append(cat)
                df = error_df.copy()
                df["__Category__"] = cats_per_row
                category_col = "__Category__"
            else:
                df = error_df.copy()
                df[category_col] = df[category_col].fillna(blank_label).astype(str)

            # --- Helpers to normalize endpoints ---
            def _blankify(x):
                if x is None: return None
                if isinstance(x, (list, tuple)) and len(x) == 0: return None
                s = str(x).strip().strip("\"'")
                if s in ("", "[]", "None", "none", "NULL", "null"): return None
                return s

            # endpoint -> category
            endpoint_cat: dict[str, str] = {}

            def put_endpoint(label, cat):
                label = _blankify(label)
                if not label:
                    return
                if prefer_first:
                    endpoint_cat.setdefault(label, cat)
                else:
                    endpoint_cat[label] = cat

            parsed_rows = 0
            for _, row in df.iterrows():
                ch = row.get("Channel")
                a, b = self._parse_channel_pair(ch)   # should return (left, right) scalars or None
                cat = row[category_col]
                put_endpoint(a, cat)
                put_endpoint(b, cat)
                parsed_rows += 1

            # --- Build arrays aligned to canonical orders ---
            def map_array(order):
                return [endpoint_cat.get(label, blank_label) for label in order]

            dib_cats = map_array(self.dib_order)
            p1_cats  = map_array(self.p1_order)
            p2_cats  = map_array(self.p2_order)

            # Universe of categories (blank first)
            uniq = pd.Series(list(endpoint_cat.values())).dropna().unique().tolist()
            uniq = [c for c in uniq if c != blank_label]
            categories = [blank_label] + uniq

            if verbose:
                print(f"[build_bucket_arrays] rows={len(df)}, parsed={parsed_rows}, "
                    f"unique_endpoints={len(endpoint_cat)}, categories={categories}")

            return dib_cats, p1_cats, p2_cats, categories
    def make_defect_heatmap(
            self,
            dib_cats, p1_cats, p2_cats,
            *,
            title_prefix: str = "Leakage Buckets",
            blank_label: str = "—",
            palette: list[str] | None = None,   # optional custom colors (len == #categories)
            legend_orientation: str = "h",      # "h" = horizontal, "v" = vertical
            legend_x: float = 0.5,              # center for horizontal legend
            legend_y: float = -0.05,            # put legend below figure
        ):
            """
            Render 3 stacked categorical heatmaps (P2 / DIB / P1) with a LEGEND, not a colorbar.
            - dib_cats / p1_cats / p2_cats: 1-D lists aligned to self.dib_order / self.p1_order / self.p2_order.
            - Empty values (''/None/NaN) -> blank_label.
            - Colors are discrete and reflected in the legend by adding one dummy Scatter per category.
            """

            # ---- Normalize incoming category arrays ----
            def norm_list(seq):
                out = []
                for v in seq:
                    if v is None:
                        out.append(blank_label)
                    elif isinstance(v, float) and np.isnan(v):
                        out.append(blank_label)
                    elif isinstance(v, str) and v.strip() == "":
                        out.append(blank_label)
                    else:
                        out.append(str(v))
                return out

            dib_cats = norm_list(dib_cats)
            p1_cats  = norm_list(p1_cats)
            p2_cats  = norm_list(p2_cats)

            # ---- Build the universe of categories with blank first ----
            cats = [blank_label]
            for seq in (dib_cats, p1_cats, p2_cats):
                for c in seq:
                    if c not in cats:
                        cats.append(c)
            cats = [blank_label] + [c for c in cats if c != blank_label]  # ensure blank is first

            cat2idx = {c: i for i, c in enumerate(cats)}

            # ---- Panel geometry ----
            dib_rows = list("ABCDEFG")
            p1_rows  = list("ABCDE")
            p2_rows  = list("EDCBA")

            # ---- Gridify category strings to 2D (same grid you use for numeric) ----
            dib_text_grid, dib_xticks = self._grid_from(dib_cats, self.dib_order, dib_rows)
            p1_text_grid,  p1_xticks  = self._grid_from(p1_cats,  self.p1_order,  p1_rows)
            p2_text_grid,  p2_xticks  = self._grid_from(p2_cats,  self.p2_order,  p2_rows)

            # ---- Map category text grids -> integer z grids ----
            def to_index_grid(text_grid):
                Z = []
                for row in text_grid:
                    Z.append([
                        cat2idx.get(
                            (blank_label if (v in (None, "") or (isinstance(v, float) and np.isnan(v))) else v),
                            0
                        )
                        for v in row
                    ])
                return Z

            dib_z = to_index_grid(dib_text_grid)
            p1_z  = to_index_grid(p1_text_grid)
            p2_z  = to_index_grid(p2_text_grid)

            # ---- Hover text + customdata with bucket strings ----
            def hover_for(xticks, row_letters, bucket_grid):
                text, custom = [], []
                for r, y in enumerate(row_letters):
                    trow, cdrow = [], []
                    for c, x in enumerate(xticks):
                        b = bucket_grid[r][c]
                        if b in (None, "") or (isinstance(b, float) and np.isnan(b)):
                            b = blank_label
                        b = str(b)
                        trow.append(b)
                        cdrow.append([y, x, b])
                    text.append(trow)
                    custom.append(cdrow)
                return text, custom

            dib_text, dib_cd = hover_for(dib_xticks, dib_rows, dib_text_grid)
            p1_text,  p1_cd  = hover_for(p1_xticks,  p1_rows,  p1_text_grid)
            p2_text,  p2_cd  = hover_for(p2_xticks,  p2_rows,  p2_text_grid)

            hovertemplate = (
                "Row %{customdata[0]}, Col %{customdata[1]}"
                "<br>Bucket: %{customdata[2]}<extra></extra>"
            )

            # ---- Build discrete color list for categories ----
            if palette is None:
                base = [
                    "#d0d0d0",  # blank
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf"
                ]
                if len(base) < len(cats):
                    need = len(cats) - len(base)
                    base += (base[1:] * ((need // len(base)) + 2))[:need]
                palette = base[:len(cats)]

            # Discrete colorscale for the heatmap (still required, but we hide the colorbar)
            zmin_i = -0.5
            zmax_i = len(cats) - 0.5
            discrete = [
                (i / (len(cats) - 1 if len(cats) > 1 else 1), palette[i])
                for i in range(len(cats))
            ]

            # ---- Build figure ----
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.08,
                subplot_titles=(f"{title_prefix} — P2", f"{title_prefix} — DIB", f"{title_prefix} — P1")
            )

            def add_trace(z, x, y, text, cd, row):
                fig.add_trace(
                    go.Heatmap(
                        z=z, x=x, y=y,
                        colorscale=discrete,
                        zmin=zmin_i, zmax=zmax_i,
                        showscale=False,                     # <- hide colorbar
                        customdata=cd, hovertemplate=hovertemplate,
                        xgap=1, ygap=1, hoverongaps=False,
                    ),
                    row=row, col=1
                )

            add_trace(p2_z, p2_xticks, p2_rows, p2_text, p2_cd, row=1)
            add_trace(dib_z, dib_xticks, dib_rows, dib_text, dib_cd, row=2)
            add_trace(p1_z, p1_xticks, p1_rows, p1_text, p1_cd, row=3)

            # ---- Add a legend (one dummy trace per category) ----
            # We'll place them on the last subplot's axes; they only serve legend entries.
            for i, cat in enumerate(cats):
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],                # no data displayed
                        mode="markers",
                        marker=dict(size=12, color=palette[i], symbol="square"),
                        name=cat,                          # legend label
                        hoverinfo="skip",
                        showlegend=True,
                    ),
                    row=3, col=1
                )

            # ---- Layout & axes ----
            fig.update_layout(
                title=f"{title_prefix} — P2 / DIB / P1",
                height=620,
                margin=dict(l=70, r=40, t=80, b=90),
                legend=dict(
                    orientation=legend_orientation,
                    x=legend_x, y=legend_y,
                    xanchor="center" if legend_orientation == "h" else "left",
                    yanchor="top" if legend_orientation == "h" else "middle",
                    title=None,
                    itemwidth=30,
                )
            )
            y2_bottom, y2_top = fig.layout.yaxis2.domain

            top_y = y2_top + 0.02
            bot_y = y2_bottom - 0.02

            arrow_end = max(dib_xticks)   
            arrow_start = dib_xticks[-15]

            fig.add_annotation(
                x=arrow_end,
                y=top_y,
                ax=arrow_start,
                ay=0,               
                xref="x2",
                yref="paper",
                axref="x2",
                ayref="pixel",
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor="blue",
                text=""
            )

            fig.add_annotation(
                x=arrow_end,
                y=bot_y,
                ax=arrow_start,
                ay=0,
                xref="x2",
                yref="paper",
                axref="x2",
                ayref="pixel",
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor="blue",
                text=""
            )
            fig.update_yaxes(title_text="P2", row=1, col=1)
            fig.update_yaxes(title_text="DIB", row=2, col=1)
            fig.update_yaxes(title_text="P1", row=3, col=1)

            for r in (1, 2, 3):
                fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, row=r, col=1)

            return fig
    def overlay_failures(self, fig, single_channel_df):
            color="red"
            size=14
            # Map panel → subplot column
            panel_to_col = {
                "DIB": 2,
                "P1":  3,
                "P2":  1,
            }

            # Collect points per panel
            points = defaultdict(lambda: {"x": [], "y": [], "text": []})

            for _, r in single_channel_df.iterrows():
                ch = (r["Channel"][0] + r["Channel"][1])[0]
                row, col, panel = self.parse_channel(ch)

                if row is None or col is None or panel not in panel_to_col:
                    continue

                points[panel]["x"].append(col)
                points[panel]["y"].append(row)
                points[panel]["text"].append(f"Single-channel failure: {ch}")

            # Add ONE trace per panel
            for panel, data in points.items():
                if not data["x"]:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=size,
                            symbol="star",
                            line=dict(width=2),
                        ),
                        name=f"Single-channel failure ({panel})",
                        hoverinfo="text",
                        text=data["text"],
                        showlegend=True,
                    ),
                    row=2,
                    col=1,   
                )

            return fig
    def make_analog_heatmap(
                self,
                dib_vals, p1_vals, p2_vals,
                #failure_vals,
                *,
                title_prefix: str = "Measured Resistance",
                unit: str = "mΩ",
                colorscale: str = "Viridis",
                zmin: float | None = None,
                zmax: float | None = None,
                show_colorbar: bool = True,
            ):
                """
                Build a single Plotly figure with 3 stacked heatmaps (DIB, P1, P2),
                sharing the same colorbar (shared coloraxis).

                - dib_vals must align with self.dib_order
                - p1_vals  must align with self.p1_order
                - p2_vals  must align with self.p2_order

                Returns:
                    fig: plotly.graph_objects.Figure
                """

                # --- Build grids from your existing orders ---
                # DIB: rows A..G, columns 1..87
                dib_row_letters = list("ABCDEFG")
                dib_grid, dib_xticks = self._grid_from(dib_vals, self.dib_order, dib_row_letters)

                # P1: rows A..E, columns 1..87
                p1_row_letters = list("ABCDE")
                p1_grid, p1_xticks = self._grid_from(p1_vals, self.p1_order, p1_row_letters)

                # P2: rows E..A, columns 87..1 (matches how you constructed p2_order)
                p2_row_letters = list("EDCBA")
                p2_grid, p2_xticks = self._grid_from(p2_vals, self.p2_order, p2_row_letters)

                # --- Determine a global zmin/zmax if not provided ---
                if zmin is None or zmax is None:
                    all_vals = []
                    for arr in (dib_vals, p1_vals, p2_vals):
                        for v in arr:
                            # Treat None/NaN as missing
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                all_vals.append(float(v))
                    if all_vals:
                        zmin = float(np.min(all_vals)) if zmin is None else zmin
                        zmax = float(np.max(all_vals)) if zmax is None else zmax
                    else:
                        # No data; leave None to let Plotly auto-scale
                        zmin = zmin
                        zmax = zmax

                # --- Create subplots with a shared coloraxis ---
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        f"{title_prefix} — P2",
                        f"{title_prefix} - DIB",
                        f"{title_prefix} — P1"
                    ),
                )

                # p1
                fig.add_trace(
                    go.Heatmap(
                        z=p1_grid,
                        x=p1_xticks,
                        y=p1_row_letters,
                        coloraxis="coloraxis",  # shared color scale
                        hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.2f} " + unit + "<extra></extra>",
                        showscale=False,  # colorbar comes from coloraxis in layout
                    ),
                    row=1, col=1
                )

                # dib
                fig.add_trace(
                    go.Heatmap(
                        z=dib_grid,
                        x=dib_xticks,
                        y=dib_row_letters,
                        coloraxis="coloraxis",
                        hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.2f} " + unit + "<extra></extra>",
                        showscale=False,
                    ),
                    row=2, col=1
                )

                # P2 (E..A rows, 87..1 cols)
                fig.add_trace(
                    go.Heatmap(
                        z=p2_grid,
                        x=p2_xticks,
                        y=p2_row_letters,
                        coloraxis="coloraxis",
                        hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.2f} " + unit + "<extra></extra>",
                        showscale=False,
                    ),
                    row=3, col=1
                )

                # Shared color axis (single colorbar)
                fig.update_layout(
                    title=f"{title_prefix} — P2 / DIB / P1",
                    height=600,  # adjust to taste
                    margin=dict(l=70, r=40, t=80, b=70),
                    coloraxis=dict(
                        colorscale=colorscale,
                        cmin=zmin,
                        cmax=zmax,
                        colorbar=dict(
                            title=unit if show_colorbar else None,
                            len=0.8,  # colorbar length relative to figure; tweak if needed
                        ),
                    ),
                )

                y2_bottom, y2_top = fig.layout.yaxis2.domain

                top_y = y2_top + 0.02
                bot_y = y2_bottom - 0.02

                arrow_end = max(dib_xticks)   
                arrow_start = dib_xticks[-15]

                fig.add_annotation(
                    x=arrow_end,
                    y=top_y,
                    ax=arrow_start,
                    ay=0,               
                    xref="x2",
                    yref="paper",
                    axref="x2",
                    ayref="pixel",
                    showarrow=True,
                    arrowhead=3,
                    arrowwidth=2,
                    arrowcolor="blue",
                    text=""
                )

                fig.add_annotation(
                    x=arrow_end,
                    y=bot_y,
                    ax=arrow_start,
                    ay=0,
                    xref="x2",
                    yref="paper",
                    axref="x2",
                    ayref="pixel",
                    showarrow=True,
                    arrowhead=3,
                    arrowwidth=2,
                    arrowcolor="blue",
                    text=""
                )
                fig.update_yaxes(
                    title_text="DIB",
                    row=2, col=1
                )
                fig.update_yaxes(
                    title_text="P2",
                    row=1, col=1
                )
                fig.update_yaxes(
                    title_text="P1",
                    row=3, col=1
                )

                fig.update_xaxes(
                    showticklabels=False,
                    ticks="",            
                    showgrid=False,
                    row=1, col=1
                )

                fig.update_xaxes(
                    showticklabels=False,
                    ticks="",
                    showgrid=False,
                    row=2, col=1
                )
                fig.update_xaxes(
                    showticklabels=False,
                    ticks="",
                    showgrid=False,
                    row=3, col=1
                )

                return fig
        