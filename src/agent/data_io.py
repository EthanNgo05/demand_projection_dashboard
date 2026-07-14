"""Shared, Streamlit-free I/O extracted from dashboard.py (Phase 2).

Single source of truth for raw/price-file discovery and raw-frame cleaning,
used by both ``dashboard.py`` (which keeps its ``@st.cache_data`` wrappers
around thin calls into this module) and the agent's ingest node.

Each function takes the pipeline module ``P`` explicitly instead of relying
on the dashboard's Streamlit-session ``pipeline_path()``/``load_pipeline()``
globals. Passing ``P=None`` falls back to the first configured model, which
is safe for discovery/cleaning because ``RAW_INPUTS_FOLDER`` /
``LIST_PRICE_GLOB`` / ``CUSTOMERS_TO_IGNORE`` / ``COMBINED_GROUPING`` are
identical across the three model files (see README, "The pipeline contract").

Must never import streamlit (directly or transitively).
"""

import glob
import os
import re
from typing import NamedTuple

import numpy as np
import openpyxl
import pandas as pd

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS, region_from_view
from agent.model_loader import load_pipeline

# Repo root (parent of src/, the folder holding raw_inputs/ + outputs/), so
# relative RAW_INPUTS_FOLDER / LIST_PRICE_GLOB paths resolve there. This file is
# src/agent/data_io.py, so climb three levels: agent -> src -> repo root.
HERE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def default_pipeline():
    """Load the first configured model (any model drives discovery/cleaning)."""
    if not MODEL_OPTIONS:
        raise FileNotFoundError(
            "No forecasting pipeline found — expected "
            "models/exponential_smoothing.py, models/xgboost.py or "
            "models/regression.py next to dashboard.py."
        )
    return load_pipeline(next(iter(MODEL_OPTIONS.values())))


def _resolve_pipeline(P):
    return default_pipeline() if P is None else P


def view_frame(df, view, P=None):
    """Rows of ``df`` belonging to ``view``: the full frame (ALL CUSTOMERS),
    one region's groups (an "All Customers - <region>" rollup), or one
    customer group. The shared view->frame step for every agent node, kept in
    exact lockstep with dashboard.compute_view's filtering.

    ``P=None`` falls back to the first configured model — safe because
    region_for_group is identical across the three model files (see the
    pipeline contract). str() on its result matches how the view string was
    built (a custom pipeline may return non-string region labels).
    """
    if view == ALL_CUSTOMERS_VIEW:
        return df
    region = region_from_view(view)
    if region is not None:
        P = _resolve_pipeline(P)
        groups = df["Customer Grouping"].map(
            lambda g: str(P.region_for_group(g))
        )
        return df[groups == region]
    return df[df["Customer Grouping"] == view]


def _raw_dir(P=None):
    """Resolve the folder holding the raw + price files.

    Honours DEMAND_RAW_DIR if set; otherwise uses the pipeline's own
    RAW_INPUTS_FOLDER constant (e.g. ``raw_inputs/demand_projections``),
    resolved relative to the repo root when it is a relative path. This means
    moving the raw folder in the pipeline is picked up here automatically.
    """
    P = _resolve_pipeline(P)
    folder = os.environ.get("DEMAND_RAW_DIR")
    if folder is None:
        folder = getattr(P, "RAW_INPUTS_FOLDER", None)
        if folder is None:
            # Older pipeline without the constant: derive it from INPUT_GLOB
            # if present, otherwise use the standard default location.
            input_glob = getattr(P, "INPUT_GLOB", None)
            folder = (
                os.path.dirname(input_glob)
                if input_glob
                else "raw_inputs/demand_projections"
            )
        if not os.path.isabs(folder):
            folder = os.path.join(HERE, folder)
    return folder


def raw_glob(P=None):
    """Build the raw-file glob, tracking the pipeline's RAW_INPUTS_FOLDER."""
    return os.path.join(_raw_dir(P), "all_demand_projections_*.xlsx")


def price_glob(P=None):
    """Build the list-price glob, mirroring the pipeline's LIST_PRICE_GLOB.

    The pipeline's glob (folder included) is used as-is, resolved relative to
    the repo root when it is a relative path — so every caller scans the same
    folder the batch pipeline does, regardless of the working directory.
    """
    P = _resolve_pipeline(P)
    pattern = getattr(
        P, "LIST_PRICE_GLOB",
        os.path.join("raw_inputs/list_prices", "list_prices_*.xlsx"),
    )
    if not os.path.isabs(pattern):
        pattern = os.path.join(HERE, pattern)
    return pattern


def discover_price_file(P=None):
    """Newest list-price file in the raw folder, or None if there isn't one."""
    matches = glob.glob(price_glob(P))
    return max(matches, key=os.path.getmtime) if matches else None


def _date_from_name(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(name))
    return m.group(1) if m else None


def discover_raw_files(P=None):
    """Return [(date_str, path)] newest first, mirroring resolve_input_file()."""
    out = []
    for path in glob.glob(raw_glob(P)):
        d = _date_from_name(path)
        if d:
            out.append((d, path))
    return sorted(out, reverse=True)


def _clean(raw_df, P):
    """Apply the exact preprocessing from the pipeline's __main__ block.

    Mirrors the updated pipeline: 'Sum of Quantity' -> Orders, and POS /
    Orders / Projection are all carried through. Falls back gracefully if an
    older file lacks the Orders column (an all-NaN Orders column is added so
    the POS-then-Orders logic still runs without a KeyError).
    """
    rename = {"'Demand'[DisplaySKU]": "SKU", "Custnmbr": "CUSTNMBR"}
    if "Sum of Quantity" in raw_df.columns:
        rename["Sum of Quantity"] = "Orders"
    df = raw_df.rename(columns=rename)

    if "Orders" not in df.columns:
        df["Orders"] = np.nan  # legacy file without an Orders/Sum of Quantity col

    df = df[["SKU", "Description", "CUSTNMBR", "WeekDate", "POS", "Orders", "Projection"]]
    # The fixed-width warehouse export space-pads its key columns (e.g.
    # 'BT1028      ', 'AMAZON-DS      '). Strip that surrounding whitespace here —
    # the single ingestion boundary both the dashboard and agent share — before
    # any key-based join or lookup runs. SKU padding made every SKU miss the
    # (stripped) list-price index (blank revenue risk) and the Plytix SKU sets
    # (active-in / discontinued checks silently ran on nothing). CUSTNMBR padding
    # made padded customers miss CUSTOMERS_TO_IGNORE and COMBINED_GROUPING, so a
    # group like AMAZON-DC fragmented across its padded/clean spellings instead of
    # folding together. Strip CUSTNMBR *before* the ignore filter and grouping map.
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["CUSTNMBR"] = df["CUSTNMBR"].astype(str).str.strip()
    df = df[~df["CUSTNMBR"].isin(P.CUSTOMERS_TO_IGNORE)]
    df["WeekDate"] = pd.to_datetime(df["WeekDate"])
    df["Customer Grouping"] = (
        df["CUSTNMBR"].map(P.COMBINED_GROUPING).fillna(df["CUSTNMBR"])
    )
    return df


def load_raw(path, P=None):
    """Read + clean a raw demand workbook from disk.

    ``header=2`` matches the PowerBI export layout (two banner rows above the
    header) — the same read dashboard.py's ``load_raw_from_path`` does.
    """
    P = _resolve_pipeline(P)
    raw = pd.read_excel(path, header=2)
    return _clean(raw, P)


# --------------------------------------------------------------------------- #
# Plytix-based SKU exclusions (single source of truth, shared by dashboard.py  #
# and the agent's ingest node). A SKU must never be forecast — or flagged by   #
# the agent — when it is discontinued/inactive, or when it appears in a region #
# it is not "Active in" per the Plytix export. Ported verbatim from            #
# dashboard.py so both paths drop the exact same rows before forecasting.      #
#                                                                              #
# The Plytix export doubles as the list-price file, so ``read_plytix`` reads   #
# the same file ``discover_price_file`` returns.                               #
# --------------------------------------------------------------------------- #

# The warehouse regions we check "Active in" against. A SKU should only be
# projected in a region it is "Active in" (per Plytix); a projection in any
# other region is flagged and excluded from the forecast.
WAREHOUSE_REGIONS = ["AU", "CA", "EU", "JP", "US"]

INACTIVE_COLS = [
    "SKU", "Location", "Region", "Active in", "Customer Grouping",
    "CUSTNMBR", "First_WeekDate", "Last_WeekDate", "Original_Projection", "Source",
]

DISCONTINUED_COLS = [
    "SKU", "SKU Status", "Region", "Customer Grouping", "CUSTNMBR",
    "First_WeekDate", "Last_WeekDate", "Original_Projection",
]


def read_plytix(path):
    """Read the raw Plytix export (for the 'Active in' / discontinued checks).

    ``path`` may be a filesystem path or a file-like object (e.g. a BytesIO of
    an uploaded workbook), so both the dashboard's on-disk and upload paths and
    the agent's ingest node share this one reader."""
    return pd.read_excel(path)


def _this_week_start():
    """Sunday-anchored start of the current week as a Timestamp.

    "This week" is deliberately the real current week (not the snapshot's
    anchor): the excluded tables report which future projections are being
    dropped going forward from now."""
    today = pd.Timestamp.today().normalize()
    return today - pd.Timedelta(days=(today.weekday() + 1) % 7)


def _active_in_list(sku_active_in, sku):
    """The list of regions a SKU is 'Active in' (e.g. ['US', 'CA', 'EU'])."""
    return [x.strip() for x in str(sku_active_in.get(sku, "")).split(",")]


def _region_code(P, grouping):
    """Two-letter region code for a customer grouping (US/CA/EU/JP/AU), or None.

    The pipeline's region_for_group returns labels like "JP (NETDEPOT)" or
    "US (LBC+NJ)"; the leading two letters are the region code we match against
    Plytix 'Active in'. Anything else (e.g. "Other") returns None.
    """
    try:
        label = P.region_for_group(grouping)
    except Exception:
        return None
    code = str(label)[:2].upper()
    return code if code in WAREHOUSE_REGIONS else None


def compute_active_products(plytix_df):
    """From the Plytix export, the set of active-product SKUs and a
    SKU -> 'Active in' string lookup.

    "Active product" mirrors inactive_projections.ipynb: SKU Status == Active,
    SKU Type == Product, and SKUs starting LS/AS excluded. Trailing '*' markers
    are stripped so SKUs line up with the demand file.

    Returns (active_sku_set, sku_active_in) or (None, None) if the Plytix export
    lacks the columns the check needs (an older list-price file).
    """
    required = {"SKU", "SKU Status", "SKU Type", "Active in"}
    if plytix_df is None or not required.issubset(plytix_df.columns):
        return None, None
    p = plytix_df.copy()
    p["SKU"] = p["SKU"].astype(str).str.rstrip("*")
    act = p[(p["SKU Status"] == "Active") & (p["SKU Type"] == "Product")]
    act = act[~act["SKU"].str.startswith(("LS", "AS"))]
    active_sku_set = set(act["SKU"])
    # Full (un-exploded) Active-in string per SKU, e.g. "US,CA,UK,SG,EU,AU".
    sku_active_in = dict(zip(p["SKU"], p["Active in"].astype(str)))
    return active_sku_set, sku_active_in


def compute_inactive_projections(df, active_sku_set, sku_active_in, P,
                                 anchors=None):
    """Active products showing up in a region they are not 'Active in'.

    This is the fix for cases like ST1082 (active in US/CA/UK/SG/EU/AU but not
    JP), which still appeared in the JP (NETDEPOT) summary: the dashboard builds
    a forward forecast for any SKU with demand history in a region. We look at
    the *demand file itself* — the same data the dashboard forecasts — map each
    customer to its region via the pipeline's region_for_group, and flag any
    active product whose region is not in its Plytix 'Active in' list.

    Returns a table (columns = INACTIVE_COLS) of the flagged
    SKU x customer x region combinations, empty if none or inputs are missing.
    """
    if not active_sku_set or not sku_active_in:
        return pd.DataFrame(columns=INACTIVE_COLS)

    frames = []

    # ----- Primary: the demand file, by customer-group region ---------------
    if df is not None and not df.empty:
        m = df.copy()
        m["SKU"] = m["SKU"].astype(str).str.rstrip("*")
        m = m[m["SKU"].isin(active_sku_set)]
        if not m.empty:
            m["Region"] = m["Customer Grouping"].map(
                lambda g: P.region_for_group(g)
            )
            m["Location"] = m["Customer Grouping"].map(
                lambda g: _region_code(P, g)
            )
            m = m[m["Location"].notna()]
            keep = [
                loc not in _active_in_list(sku_active_in, sku)
                for sku, loc in zip(m["SKU"], m["Location"])
            ]
            m = m[keep]
            m["WeekDate"] = pd.to_datetime(m["WeekDate"])
            # Only flag pairs the dashboard would actually forecast — i.e. that
            # carry a POS/Orders demand signal in the historical window (that is
            # exactly what puts a SKU in a region's summary). Without anchors we
            # fall back to any presence.
            if anchors is not None and not m.empty:
                lb, lcw, _ = anchors
                sig = (
                    (m["WeekDate"] >= lb) & (m["WeekDate"] <= lcw)
                    & (m["POS"].notna() | m.get("Orders", pd.Series(index=m.index)).notna())
                )
                live = m.loc[sig, ["SKU", "CUSTNMBR", "Location"]].drop_duplicates()
                m = m.merge(live, on=["SKU", "CUSTNMBR", "Location"], how="inner")
            if not m.empty:
                # Original projection over future weeks (this week onward) —
                # averaged per week — the projected weekly volume being excluded
                # going forward. Uses the same week boundary as the excluded
                # table's "future only" toggle.
                m["_future_proj"] = pd.to_numeric(
                    m["Projection"], errors="coerce"
                ).where(m["WeekDate"] >= _this_week_start())
                g = m.groupby(
                    ["SKU", "Location", "Region", "Customer Grouping", "CUSTNMBR"],
                    as_index=False,
                ).agg(
                    First_WeekDate=("WeekDate", "min"),
                    Last_WeekDate=("WeekDate", "max"),
                    Original_Projection=("_future_proj", "mean"),
                )
                g["Active in"] = g["SKU"].map(lambda s: sku_active_in.get(s))
                g["Source"] = "Demand file"
                frames.append(g)

    if not frames:
        return pd.DataFrame(columns=INACTIVE_COLS)

    out = pd.concat(frames, ignore_index=True)[INACTIVE_COLS]
    out = out.drop_duplicates(subset=["SKU", "Location", "CUSTNMBR"], keep="first")
    return out.sort_values(["SKU", "Location", "CUSTNMBR"]).reset_index(drop=True)


def compute_discontinued_products(plytix_df):
    """SKU -> 'SKU Status' lookup for Discontinued/Inactive products.

    Mirrors discontinued_with_projections.ipynb: keep rows whose SKU Status is
    'Discontinued' or 'Inactive'. Trailing '*' markers are stripped so SKUs line
    up with the demand file. Returns None if the Plytix export lacks the columns
    the check needs (an older list-price file).
    """
    required = {"SKU", "SKU Status"}
    if plytix_df is None or not required.issubset(plytix_df.columns):
        return None
    p = plytix_df.copy()
    p["SKU"] = p["SKU"].astype(str).str.rstrip("*")
    disc = p[p["SKU Status"].isin(["Discontinued", "Inactive"])]
    return dict(zip(disc["SKU"], disc["SKU Status"]))


def compute_discontinued_projections(df, disc_status, P):
    """Discontinued/inactive products that still carry future projections.

    Ported from discontinued_with_projections.ipynb: intersect the demand file
    with the discontinued/inactive SKU set, keep only future projection weeks
    (WeekDate after today), and aggregate to one row per SKU x customer with the
    first/last projected week. A Region column (via the pipeline's
    region_for_group) is added so a by-customer-group view can be scoped to its
    own region.

    Returns a table (columns = DISCONTINUED_COLS), empty if none or inputs are
    missing.
    """
    if not disc_status or df is None or df.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    m = df.copy()
    m["SKU"] = m["SKU"].astype(str).str.rstrip("*")
    m = m[m["SKU"].isin(disc_status)]
    if m.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    # Future projections only. Like the active-in table, "future" starts at the
    # beginning of the current week (Sunday-anchored via _this_week_start), so
    # the in-progress week is included — e.g. 7/5 counts while the 7/7 week is
    # not yet over.
    m["WeekDate"] = pd.to_datetime(m["WeekDate"])
    week_start = _this_week_start()
    m = m[m["WeekDate"] >= week_start]
    if m.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    m["Region"] = m["Customer Grouping"].map(lambda g: P.region_for_group(g))
    m["_future_proj"] = pd.to_numeric(m["Projection"], errors="coerce")
    g = m.groupby(
        ["SKU", "Region", "Customer Grouping", "CUSTNMBR"], as_index=False,
    ).agg(
        First_WeekDate=("WeekDate", "min"),
        Last_WeekDate=("WeekDate", "max"),
        Original_Projection=("_future_proj", "mean"),
    )
    g["SKU Status"] = g["SKU"].map(lambda s: disc_status.get(s))
    out = g[DISCONTINUED_COLS]
    return out.sort_values(["SKU", "CUSTNMBR"]).reset_index(drop=True)


class ExclusionResult(NamedTuple):
    """Outcome of ``apply_exclusions``: the filtered demand frame plus the two
    "excluded" tables and counts the dashboard surfaces in its own sections."""

    df: pd.DataFrame                    # demand frame with excluded rows removed
    inactive_df: pd.DataFrame           # active-SKU-in-wrong-region exclusions
    discontinued_df: pd.DataFrame       # discontinued/inactive with projections
    active_check_ran: bool              # Plytix had the 'Active in' columns
    disc_check_ran: bool                # Plytix had the 'SKU Status' column
    n_excluded_rows: int                # demand rows dropped by the active-in check
    excluded_counts_by_key: pd.Series   # per SKU||CUSTNMBR dropped-row counts
    n_disc_rows: int                    # demand rows dropped as discontinued/inactive
    n_disc_skus: int                    # distinct SKUs dropped as discontinued/inactive


def apply_exclusions(df, plytix_df, P, anchors=None):
    """Drop SKUs that must never be forecast, mirroring dashboard.main() exactly.

    Two independent Plytix-driven filters, applied to the demand frame BEFORE
    forecasting so neither the dashboard nor the agent projects or flags them:

    1. Active-in region check: an *active* product forecast in a region it is
       not "Active in" (per Plytix) has those SKU x customer rows dropped.
    2. Discontinued/inactive drop: a SKU marked Discontinued/Inactive in Plytix,
       OR carrying a trailing '*' in the demand file, is dropped entirely (the
       status is SKU-level, so every row of the SKU goes).

    With no Plytix export both checks degrade to no-ops, except the trailing-'*'
    drop, which needs no Plytix. Returns an ``ExclusionResult``; callers do the
    logging so this stays free of any logging/Streamlit dependency.
    """
    active_sku_set, sku_active_in = compute_active_products(plytix_df)
    active_check_ran = active_sku_set is not None
    inactive_df = compute_inactive_projections(
        df, active_sku_set, sku_active_in, P, anchors=anchors
    )

    disc_status = compute_discontinued_products(plytix_df)
    disc_check_ran = disc_status is not None
    discontinued_df = compute_discontinued_projections(df, disc_status, P)

    # ----- Drop the active-in exclusions (per SKU x customer) --------------
    n_excluded_rows = 0
    excluded_counts_by_key = pd.Series(dtype="int64")
    if not inactive_df.empty:
        exclude_keys = {
            f"{str(s)}||{str(c)}"
            for s, c in zip(inactive_df["SKU"], inactive_df["CUSTNMBR"])
        }
        key = df["SKU"].astype(str).str.rstrip("*") + "||" + df["CUSTNMBR"].astype(str)
        drop_mask = key.isin(exclude_keys)
        n_excluded_rows = int(drop_mask.sum())
        # Per SKU||CUSTNMBR demand-row counts, so a region-scoped excluded table
        # can report accurate row totals.
        excluded_counts_by_key = key[drop_mask].value_counts()
        if n_excluded_rows:
            df = df[~drop_mask].reset_index(drop=True)

    # ----- Drop discontinued/inactive SKUs entirely ------------------------
    # Two independent signals: a trailing '*' on the SKU code in the demand file,
    # and a Plytix 'SKU Status' of Discontinued/Inactive. Either one drops the
    # whole SKU (match at SKU level so every row goes, even rows omitting the '*').
    sku_raw = df["SKU"].astype(str)
    sku_base = sku_raw.str.rstrip("*")
    disc_bases = set(sku_base[sku_raw.str.endswith("*")])
    if disc_status:
        disc_bases |= set(disc_status)
    disc_mask = sku_base.isin(disc_bases)
    n_disc_rows = int(disc_mask.sum())
    n_disc_skus = 0
    if n_disc_rows:
        n_disc_skus = int(sku_base[disc_mask].nunique())
        df = df[~disc_mask].reset_index(drop=True)

    return ExclusionResult(
        df=df,
        inactive_df=inactive_df,
        discontinued_df=discontinued_df,
        active_check_ran=active_check_ran,
        disc_check_ran=disc_check_ran,
        n_excluded_rows=n_excluded_rows,
        excluded_counts_by_key=excluded_counts_by_key,
        n_disc_rows=n_disc_rows,
        n_disc_skus=n_disc_skus,
    )


# --------------------------------------------------------------------------- #
# Warehouse projection exports -> "missing future projections" table.          #
#                                                                              #
# Ported from active_missing_projections.py. This uses a DIFFERENT data source #
# than everything above: the warehouse projection exports (one wide grid per   #
# region, raw_inputs/warehouse_projections), NOT the demand-projection frame   #
# the dashboard forecasts. The demand file only carries SKU×customer combos    #
# that already have projections, so a *missing* projection is only visible in   #
# the warehouse grid, which lists every active-in SKU×customer×week with a NaN  #
# cell where no projection exists.                                             #
# --------------------------------------------------------------------------- #

# Region prefix on a warehouse filename (e.g. "AU_warehouse_projections_*.xlsx").
REGION_PREFIXES = ("AU", "CA", "EU", "JP", "US")

WAREHOUSE_DIRNAME = "raw_inputs/warehouse_projections"

# Long-format columns produced when a wide warehouse grid is melted.
WAREHOUSE_LONG_COLS = ["SKU", "CUSTNMBR", "WeekDate", "Projection", "Location"]

MISSING_COLS = [
    "SKU", "Location", "Region", "Active in", "CUSTNMBR",
    "First_WeekDate", "Last_WeekDate",
]


def _warehouse_dir(warehouse_dir=None):
    """Resolve the folder holding the warehouse projection exports.

    Honours WAREHOUSE_RAW_DIR if set; otherwise uses the standard location,
    resolved relative to the repo root when it is a relative path."""
    folder = warehouse_dir or os.environ.get("WAREHOUSE_RAW_DIR") or WAREHOUSE_DIRNAME
    if not os.path.isabs(folder):
        folder = os.path.join(HERE, folder)
    return folder


def warehouse_glob(warehouse_dir=None):
    """Glob matching every warehouse export in the warehouse folder."""
    return os.path.join(_warehouse_dir(warehouse_dir), "*.xlsx")


def discover_warehouse_files(warehouse_dir=None):
    """Return {snapshot_date: [paths]} for warehouse exports, newest date first.

    Each snapshot date normally has one file per region (AU/CA/EU/JP/US), so we
    group by the date embedded in the filename (files without a date land under
    "undated")."""
    groups = {}
    for path in glob.glob(warehouse_glob(warehouse_dir)):
        d = _date_from_name(path) or "undated"
        groups.setdefault(d, []).append(path)
    for paths in groups.values():
        paths.sort()
    return dict(sorted(groups.items(), reverse=True))


def _warehouse_region(name):
    """Region code (AU/CA/EU/JP/US) from a warehouse filename, or None."""
    base = os.path.basename(str(name))
    return next((p for p in REGION_PREFIXES if base.startswith(p)), None)


def warehouse_wide_to_long(source, name=None):
    """Clean one wide warehouse export into a long frame with a Location column.

    ``source`` is a filesystem path or a file-like object (e.g. a BytesIO of an
    uploaded workbook); ``name`` supplies the filename used to detect the region
    prefix (defaults to ``source`` when it is a path). Returns (long_df,
    location), or (None, None) if the name has no known region prefix.

    Ported verbatim from active_missing_projections.py: unmerge the SKU column
    and forward-fill it, cut the footer notes, then melt wide -> long. Blank
    projection cells are kept as NaN (that is exactly what "missing" means).
    """
    name = name if name is not None else source
    location = _warehouse_region(name)
    if location is None:
        return None, None

    # 1) Load with openpyxl so we can unmerge the SKU column and propagate its value
    wb = openpyxl.load_workbook(source, data_only=True)
    ws = wb[wb.sheetnames[0]]

    for merged_range in list(ws.merged_cells.ranges):
        top_left_value = ws.cell(row=merged_range.min_row, column=merged_range.min_col).value
        ws.unmerge_cells(str(merged_range))
        for row in ws.iter_rows(
            min_row=merged_range.min_row, max_row=merged_range.max_row,
            min_col=merged_range.min_col, max_col=merged_range.max_col,
        ):
            for cell in row:
                cell.value = top_left_value

    raw = pd.DataFrame(ws.values)

    # 2) Row 0 = title row with week dates starting at column index 2
    #    Row 1 = "SKU" / "CUSTNMBR" / "Proj..." labels
    #    Row 2+ = actual data, until a blank row (end of data / start of footer)
    week_dates = raw.iloc[0, 2:].tolist()
    data = raw.iloc[2:].reset_index(drop=True)

    # Stop at the first row with no customer AND no projection values at all
    # (cuts off the blank separator row + the "Applied filters..." footer note).
    def is_end_row(row):
        return pd.isna(row[1]) and row[2:].isna().all()

    end_idx = len(data)
    for i, row in data.iterrows():
        if is_end_row(row):
            end_idx = i
            break
    data = data.iloc[:end_idx]

    # Forward-fill SKU since it only appears on the first row of each merged block
    data[0] = data[0].ffill()

    # 3) Melt wide -> long
    data.columns = ["SKU", "CUSTNMBR"] + week_dates
    long_df = data.melt(
        id_vars=["SKU", "CUSTNMBR"],
        value_vars=week_dates,
        var_name="WeekDate",
        value_name="Projection",
    )
    long_df = long_df.sort_values(["SKU", "CUSTNMBR", "WeekDate"]).reset_index(drop=True)
    long_df["Location"] = location
    return long_df, location


def combine_warehouse_projections(sources):
    """Clean and concatenate many warehouse exports into one long frame.

    ``sources`` is an iterable of (source, name) pairs, where ``source`` is a
    path or file-like object and ``name`` is the filename (for region
    detection). Files whose name lacks a region prefix are skipped. Returns an
    empty (but correctly-columned) frame when nothing usable is provided.
    """
    frames = []
    for source, name in sources:
        long_df, _ = warehouse_wide_to_long(source, name)
        if long_df is not None:
            frames.append(long_df)
    if not frames:
        return pd.DataFrame(columns=WAREHOUSE_LONG_COLS)
    return pd.concat(frames, ignore_index=True)


def compute_missing_projections(projections, plytix_df, df, P):
    """Active SKUs missing future projections in regions they ARE 'Active in'.

    Mirrors active_missing_projections.py: from the combined warehouse grid
    (``projections``), keep the NaN projection cells, intersect with active
    products in a region that IS in their Plytix 'Active in' list, restrict to
    the coming 15-week window, and roll up to one row per SKU×Location×customer
    with the first/last missing week. A Region label (via the pipeline's
    region_for_group) is added from ``df``'s customer groupings so a
    by-customer-group view can scope to its own region, matching the sibling
    excluded tables.

    Returns a table (columns = MISSING_COLS), empty if none or inputs missing.
    """
    active_sku_set, sku_active_in = compute_active_products(plytix_df)
    if not active_sku_set or projections is None or projections.empty:
        return pd.DataFrame(columns=MISSING_COLS)

    # Missing projection cells only.
    missing = projections[projections["Projection"].isna()].copy()
    if missing.empty:
        return pd.DataFrame(columns=MISSING_COLS)

    # Active (SKU, Location) pairs, restricted to the warehouse regions.
    pairs = [
        (sku, loc)
        for sku in active_sku_set
        for loc in _active_in_list(sku_active_in, sku)
        if loc in WAREHOUSE_REGIONS
    ]
    if not pairs:
        return pd.DataFrame(columns=MISSING_COLS)
    active_pairs = pd.DataFrame(pairs, columns=["SKU", "Location"]).drop_duplicates()

    m = active_pairs.merge(missing, on=["SKU", "Location"], how="inner")
    if m.empty:
        return pd.DataFrame(columns=MISSING_COLS)

    # Coming 15-week window (today < WeekDate <= today + 15 weeks), per notebook.
    m["WeekDate"] = pd.to_datetime(m["WeekDate"])
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(weeks=15)
    m = m[(m["WeekDate"] > today) & (m["WeekDate"] <= cutoff)]
    if m.empty:
        return pd.DataFrame(columns=MISSING_COLS)

    g = m.groupby(["SKU", "Location", "CUSTNMBR"], as_index=False).agg(
        First_WeekDate=("WeekDate", "min"),
        Last_WeekDate=("WeekDate", "max"),
    )
    # Full (un-exploded) 'Active in' string per SKU, e.g. "US,CA,UK,SG,EU,AU".
    g["Active in"] = g["SKU"].map(lambda s: sku_active_in.get(s))

    # Region label from the Location code, consistent with the sibling tables
    # (e.g. Location "JP" -> "JP (NETDEPOT)"). Fall back to the raw code when a
    # region has no customer group in the demand frame.
    code_to_label = {}
    if df is not None and not df.empty:
        for grp in df["Customer Grouping"].dropna().unique():
            code = _region_code(P, grp)
            if code is not None:
                code_to_label[code] = P.region_for_group(grp)
    g["Region"] = g["Location"].map(code_to_label).fillna(g["Location"])

    return g[MISSING_COLS].sort_values(
        ["SKU", "Location", "CUSTNMBR"]
    ).reset_index(drop=True)
