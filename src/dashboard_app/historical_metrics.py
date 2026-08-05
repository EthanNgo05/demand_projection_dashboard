"""Historical (backward-looking) aggregation for the Historical Summary view.

Deliberately streamlit-free and side-effect-free so every definition below can be
unit-tested in plain pytest -- see tests/test_historical_metrics.py. The view layer
(historical_summary.py) and the chart layer (historical_charts.py) hold no maths.

Every number this module produces follows these definitions. They are written down
because each is a place where two reasonable people would compute a different
figure, and a dashboard that silently picks one is worse than one that says which:

Revenue
    Sum of (units x that SKU's Plytix list price). A retail-value PROXY, not
    invoiced revenue -- there is no price actually paid anywhere in the demand
    data. A SKU with no price entry is EXCLUDED from revenue, never counted as
    $0, matching the convention in kpis.py; ``price_coverage`` reports how much of
    the window that leaves uncovered so a thin price file cannot read as a slump.

Demand
    POS where it exists, else Orders -- resolved PER RAW ROW, i.e. per
    (Customer, SKU, week), before anything is summed. Every row contributes
    exactly one of the two signals, so nothing is double-counted and nothing is
    dropped.

    The forecast views instead pick one signal per SKU for a whole window, which
    is right for fitting a model (a series must not switch signals mid-stream) and
    wrong here, where the job is a complete account of what sold. Measured against
    the live snapshot, per-SKU labelling silently discarded 135,327 units across
    6,156 SKU-weeks -- 2,688 customer/SKU pairs report POS in some weeks and
    Orders in others, and every off-label week vanished.

    Row-level resolution also has to happen BEFORE customers are aggregated: if
    customer A reports POS and customer B reports Orders for the same SKU-week,
    coalescing after the sum would keep A's POS and lose B's Orders entirely.

    ``Data Source`` survives as a descriptive per-pair label -- "POS", "Orders" or
    "Mixed" -- so a planner can see which signal underlies a number without it
    changing what was counted.

Weeks and months
    WeekDate is the SUNDAY starting a 7-day week. A week belongs wholly to the
    calendar month of that Sunday -- no proration across a month boundary. All
    windows end at ``lcw`` (the last COMPLETE week); the in-progress week is never
    counted, since its partial sell-through would read as a collapse.

Year over year
    The prior-year comparable ends at ``lcw - 364 days``. 364 is exactly 52 weeks,
    so the comparison lands on the same weekday and compares whole weeks against
    whole weeks; ISO week numbers would be ambiguous for a Sunday-anchored date
    (a Sunday closes an ISO week rather than opening one). In a 53-week year this
    drifts by about a week, which is the accepted cost of exact weekday alignment.

Discontinued SKUs
    Included, and counted for the years they were active. This view alone reads
    ``ExclusionResult.df_with_discontinued``; every forecast view keeps using the
    frame those SKUs are dropped from. A SKU appears in the raw data both with and
    without its trailing '*', so codes are normalised before aggregation -- one
    product must be one series, not two.
"""
import numpy as np
import pandas as pd

# Folded-tail bucket. The categorical palette is eight slots and is never cycled,
# so anything past the top N lands here rather than reusing a leading colour.
OTHER_LABEL = "Other"

# Value shown for a SKU with no Plytix "SKU Type" -- distinct from a real type, and
# never silently merged into one.
UNSPECIFIED_TYPE = "Unspecified"

# Analysis windows offered by the view's window selector. The labels double as the
# selector's option list, so a new window means one entry here and one branch in
# ``window_bounds``.
WINDOW_YTD = "Year to date"
WINDOW_4W = "Last 4 weeks"
WINDOW_13W = "Last 13 weeks"
WINDOW_26W = "Last 26 weeks"
WINDOW_52W = "Last 52 weeks"
WINDOW_2Y = "Last 2 years"
WINDOW_3Y = "Last 3 years"
# A fixed CALENDAR window, not a trailing one -- "how did last year finish", which a
# trailing 52 weeks cannot answer once the new year is under way.
WINDOW_LAST_YEAR = "Last full calendar year"
# Everything on record: the frame's earliest week through the last complete week.
# Self-adjusting, so it stays correct however far back the snapshot reaches -- which
# matters because the extract's history anchor is a date someone can move.
WINDOW_ALL = "All history"

# UI sentinel only. Its bounds come from the date picker via ``snap_window``, so
# ``window_bounds`` deliberately RAISES on it rather than guessing.
WINDOW_CUSTOM = "Custom range…"

# Everything the selector offers, in display order.
WINDOW_OPTIONS = (
    WINDOW_YTD, WINDOW_4W, WINDOW_13W, WINDOW_26W, WINDOW_52W,
    WINDOW_2Y, WINDOW_3Y, WINDOW_LAST_YEAR, WINDOW_ALL, WINDOW_CUSTOM,
)
# The subset ``window_bounds`` can resolve from ``lcw`` ALONE. The two it excludes
# both depend on something else: WINDOW_CUSTOM on the date picker, WINDOW_ALL on how
# far back the data itself goes. Exported so callers that iterate windows (notably
# the tests) can't accidentally include one that raises.
DATA_DEPENDENT_WINDOWS = (WINDOW_ALL, WINDOW_CUSTOM)
NAMED_WINDOWS = tuple(w for w in WINDOW_OPTIONS
                      if w not in DATA_DEPENDENT_WINDOWS)

# Whole weeks spanned by each rolling window, inclusive of lcw itself.
_ROLLING_WEEKS = {
    WINDOW_4W: 4, WINDOW_13W: 13, WINDOW_26W: 26,
    WINDOW_52W: 52, WINDOW_2Y: 104, WINDOW_3Y: 156,
}

_FRAME_COLS = ["Customer Grouping", "SKU", "WeekDate", "POS", "Orders",
               "Projection", "Description"]


# --------------------------------------------------------------------------- #
# Building the weekly frame                                                    #
# --------------------------------------------------------------------------- #
def historical_weekly_frame(df, P):
    """Per-(Customer Grouping, SKU) weekly frame RETAINING discontinued SKUs.

    The historical counterpart to ``exceptions.sku_week_by_group``. Both delegate
    the actual arithmetic to the same model-agnostic ``P.aggregate_to_sku_week``,
    so any SKU present in both frames carries identical numbers; the two
    differences are deliberate and are the whole reason this exists separately
    rather than as a flag on that function (which stays untouched):

    1. Trailing-'*' SKUs are KEPT. They are real history.
    2. SKU codes are '*'-stripped BEFORE aggregating, so 'ST1082' and 'ST1082*'
       collapse into one series. Stripping afterwards would leave two rows for the
       same SKU-week that later groupbys would double-count.

    Returns an empty, correctly-typed frame when there is nothing to aggregate.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_FRAME_COLS)

    normalised = df.copy()
    normalised["SKU"] = normalised["SKU"].astype(str).str.rstrip("*").str.strip()

    # Row-level coalesce, BEFORE any summing: each (Customer, SKU, week) row
    # contributes exactly one signal. Doing this after the customer aggregation
    # would drop customer B's Orders whenever customer A reported POS for the same
    # SKU-week -- see the module docstring.
    pos = pd.to_numeric(normalised["POS"], errors="coerce")
    orders = pd.to_numeric(normalised.get("Orders"), errors="coerce")
    normalised["_demand"] = pos.where(pos.notna(), orders)

    frames = []
    for group, sub in normalised.groupby("Customer Grouping"):
        # POS / Orders / Projection / Description come from the shared pipeline
        # primitive so they tie out exactly with exceptions.sku_week_by_group.
        ag = P.aggregate_to_sku_week(sub)
        demand = (sub.groupby(["SKU", "WeekDate"])["_demand"]
                  .sum(min_count=1).reset_index(name="demand"))
        ag = ag.merge(demand, on=["SKU", "WeekDate"], how="left")
        ag["Customer Grouping"] = group
        frames.append(ag)
    if not frames:
        return pd.DataFrame(columns=_FRAME_COLS)

    out = pd.concat(frames, ignore_index=True)
    out["WeekDate"] = pd.to_datetime(out["WeekDate"])
    return out


def full_history_data_source(frame):
    """Per-(Customer Grouping, SKU) label: 'POS', 'Orders' or 'Mixed'.

    DESCRIPTIVE ONLY -- it reports which signal underlies a pair's history and
    does NOT decide what gets counted (``historical_weekly_frame`` already
    resolved that per row). 'Mixed' is the honest answer for the 2,688 pairs in
    the live snapshot that report POS in some weeks and Orders in others; calling
    those "POS" is what used to make their Orders weeks disappear.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Customer Grouping", "SKU", "Data Source"])
    flags = (
        frame.assign(_pos=frame["POS"].notna(),
                     _ord=frame["Orders"].notna() if "Orders" in frame.columns
                     else False)
        .groupby(["Customer Grouping", "SKU"])[["_pos", "_ord"]].any()
        .reset_index()
    )
    flags["Data Source"] = np.select(
        [flags["_pos"] & flags["_ord"], flags["_pos"]],
        ["Mixed", "POS"],
        default="Orders",
    )
    return flags.drop(columns=["_pos", "_ord"])


def coalesce_demand(frame, source=None):
    """Attach the descriptive ``Data Source`` label to an already-coalesced frame.

    ``demand`` itself is produced by ``historical_weekly_frame`` at row level; this
    only labels it. Kept as a named step so the enrichment chain reads in the order
    the docstring describes. Rows with neither signal stay NaN rather than becoming
    0 -- "no data" and "sold nothing" are different facts, and ``min_count=1``
    downstream keeps them apart.
    """
    if frame is None or frame.empty:
        out = frame.copy() if frame is not None else pd.DataFrame(columns=_FRAME_COLS)
        out["demand"] = pd.Series(dtype="float64")
        out["Data Source"] = pd.Series(dtype="object")
        return out
    if source is None:
        source = full_history_data_source(frame)
    out = frame.merge(source, on=["Customer Grouping", "SKU"], how="left")
    if "demand" not in out.columns:   # defensive: a frame built by another path
        pos = pd.to_numeric(out["POS"], errors="coerce")
        orders = pd.to_numeric(out.get("Orders"), errors="coerce")
        out["demand"] = pos.where(pos.notna(), orders)
    return out


def normalise_prices(prices):
    """Any SKU->price input (dict, pandas Series, or None) as ``{str: float}``.

    Prices reach this module as a Series almost everywhere in the app but as a
    dict in tests, and ``if not prices`` raises on a Series ("truth value is
    ambiguous") -- so the normalisation happens here rather than at each call
    site. SKUs are '*'-stripped to match ``historical_weekly_frame``'s codes, and
    unparseable prices are dropped so they read as "unknown", never as $0.
    """
    if prices is None:
        return {}
    items = prices.items() if isinstance(prices, (pd.Series, dict)) else None
    if items is None:
        return {}
    out = {}
    for key, value in items:
        val = pd.to_numeric(value, errors="coerce")
        if not pd.isna(val):
            out[str(key).rstrip("*").strip()] = float(val)
    return out


def attach_revenue(frame, prices):
    """Add ``List Price`` and ``revenue`` (= demand x price) columns.

    Unpriced SKUs get NaN in both, so they drop out of revenue sums instead of
    dragging them down as zeroes.
    """
    out = frame.copy()
    price_map = normalise_prices(prices)
    if not price_map:
        out["List Price"] = np.nan
        out["revenue"] = np.nan
        return out
    out["List Price"] = out["SKU"].astype(str).map(price_map)
    out["revenue"] = pd.to_numeric(out["demand"], errors="coerce") * out["List Price"]
    return out


def attach_region(frame, P):
    """Add a ``Region`` column derived from Customer Grouping.

    Region is not a column on the demand data -- it is a function of the customer
    group (``P.region_for_group``). Mapped over the DISTINCT groups rather than
    every row, since there are a handful of groups and hundreds of thousands of
    rows.
    """
    out = frame.copy()
    if out.empty:
        out["Region"] = pd.Series(dtype="object")
        return out
    groups = out["Customer Grouping"].dropna().unique()
    lookup = {g: P.region_for_group(g) for g in groups}
    out["Region"] = out["Customer Grouping"].map(lookup)
    return out


def attach_sku_type(frame, plytix_df):
    """Add a ``SKU Type`` column from the Plytix export, or all-Unspecified.

    SKUs are '*'-stripped on the Plytix side too, so a discontinued code still
    matches its attribute row.
    """
    out = frame.copy()
    if (plytix_df is None or getattr(plytix_df, "empty", True)
            or "SKU Type" not in getattr(plytix_df, "columns", [])):
        out["SKU Type"] = UNSPECIFIED_TYPE
        return out
    types = (
        plytix_df.dropna(subset=["SKU"])
        .assign(_sku=lambda d: d["SKU"].astype(str).str.rstrip("*").str.strip())
        .drop_duplicates("_sku")
        .set_index("_sku")["SKU Type"]
    )
    out["SKU Type"] = out["SKU"].astype(str).map(types).fillna(UNSPECIFIED_TYPE)
    return out


def build_frame(df, P, prices, plytix_df=None):
    """The whole enrichment chain in one call: weekly -> demand -> $ -> dimensions.

    ``prices`` may be a dict or a pandas Series (see ``normalise_prices``).
    """
    frame = historical_weekly_frame(df, P)
    frame = coalesce_demand(frame)
    frame = attach_revenue(frame, prices)
    frame = attach_region(frame, P)
    frame = attach_sku_type(frame, plytix_df)
    return frame


# --------------------------------------------------------------------------- #
# Windows                                                                      #
# --------------------------------------------------------------------------- #
def window_bounds(kind, lcw):
    """(start, end) Timestamps for a named analysis window, ending at ``lcw``.

    ``lcw`` is the last COMPLETE week, so no window can reach into the
    part-elapsed current week.
    """
    lcw = pd.Timestamp(lcw)
    if kind == WINDOW_YTD:
        return pd.Timestamp(year=lcw.year, month=1, day=1), lcw
    if kind == WINDOW_LAST_YEAR:
        # A calendar window, so it gets its own branch rather than a week count. It
        # needs no clamp against lcw: Dec 31 of lcw.year - 1 is by definition before
        # Jan 1 of lcw.year, which is on or before lcw.
        year = lcw.year - 1
        return (pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31))
    if kind in DATA_DEPENDENT_WINDOWS:
        # Neither span is derivable from lcw: WINDOW_CUSTOM comes from the date
        # picker (snap_window) and WINDOW_ALL from the frame (all_history_bounds).
        # Raising beats returning a plausible-but-wrong window silently.
        resolver = ("snap_window()" if kind == WINDOW_CUSTOM
                    else "all_history_bounds()")
        raise ValueError(
            f"{kind!r} has no bounds derivable from lcw alone — resolve it through "
            f"{resolver} and pass the result explicitly."
        )
    weeks = _ROLLING_WEEKS.get(kind)
    if weeks is None:
        raise ValueError(f"Unknown analysis window: {kind!r}")
    return lcw - pd.Timedelta(weeks=weeks - 1), lcw


def all_history_bounds(frame, lcw):
    """(earliest week on record, ``lcw``) — the full span of the snapshot.

    Reads the floor off the DATA rather than assuming a lookback, so it follows the
    extract's history anchor wherever that is set. Returns None for an empty frame,
    matching ``snap_window``'s "nothing to measure" contract.
    """
    if frame is None or frame.empty or "WeekDate" not in frame.columns:
        return None
    weeks = pd.to_datetime(frame["WeekDate"])
    # Clip to lcw so a snapshot's forward projection weeks never leak into a
    # historical window (the frame carries 15 weeks of them).
    earliest = weeks.min()
    lcw = pd.Timestamp(lcw)
    if pd.isna(earliest) or earliest > lcw:
        return None
    return pd.Timestamp(earliest), lcw


def snap_window(start, end, lcw):
    """Snap an arbitrary date pair to whole Sunday-anchored weeks, or None.

    WeekDate is the Sunday that STARTS a week, and every total in this module is a
    sum of whole weeks. A custom range that kept its raw edges would silently count
    a part-week as a full one, so its totals would not be comparable with any other
    window, nor with the 364-day year-over-year shift.

    * ``start`` moves FORWARD to the first week beginning on or after it.
    * ``end`` moves BACK to the last week whose Saturday falls on or before it, so a
      partially-elapsed trailing week is excluded rather than counted whole.
    * ``end`` is then clamped to ``lcw``, so no custom range can reach into the
      in-progress week.

    Returns ``(start, end)`` or ``None`` when the range cannot contain one complete
    week -- the caller warns rather than rendering a screen of zeroes.
    """
    start, end, lcw = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(lcw)
    # pandas weekday(): Mon=0 .. Sun=6, so days to the next Sunday is (6 - wd) % 7
    # and a Sunday snaps to itself.
    snapped_start = start + pd.Timedelta(days=(6 - start.weekday()) % 7)
    # The Sunday opening end's own week; step back a week if that week's Saturday
    # (Sunday + 6 days) runs past `end`.
    week_of_end = end - pd.Timedelta(days=(end.weekday() + 1) % 7)
    if week_of_end + pd.Timedelta(days=6) > end:
        week_of_end -= pd.Timedelta(weeks=1)
    snapped_end = min(week_of_end, lcw)
    if snapped_start > snapped_end:
        return None
    return snapped_start, snapped_end


def prior_year_window(start, end, anchor_to_year_start=False):
    """The same window shifted back one year for a like-for-like comparison.

    Shifted by 364 days (52 whole weeks) so the endpoint keeps its weekday and
    whole weeks line up against whole weeks. ``anchor_to_year_start`` is for YTD,
    where the comparable must start at the prior January 1st rather than 364 days
    before this January 1st.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    prior_end = end - pd.Timedelta(days=364)
    if anchor_to_year_start:
        prior_start = pd.Timestamp(year=prior_end.year, month=1, day=1)
    else:
        prior_start = start - pd.Timedelta(days=364)
    return prior_start, prior_end


def _anchor_weeks(start, end):
    """How many Sunday WeekDates fall inside ``[start, end]``.

    The number of weeks a window actually TOTALS, which is not the same as its
    endpoints' day span: a rolling window runs Sunday-to-Sunday, so its endpoints sit
    ``(weeks - 1) * 7`` days apart while it covers ``weeks`` of them. A calendar
    window ("Last full calendar year") has edges that fall mid-week, where the two
    readings differ again. Uses the same snapping arithmetic as ``snap_window``.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first = start + pd.Timedelta(days=(6 - start.weekday()) % 7)
    last = end - pd.Timedelta(days=(end.weekday() + 1) % 7)
    return 0 if first > last else int((last - first).days // 7) + 1


def prior_period_window(start, end):
    """The equal-length stretch of whole weeks immediately BEFORE ``[start, end]``.

    The sequential comparison -- momentum -- as opposed to ``prior_year_window``'s
    seasonal one. It abuts the window with no gap and no overlap: the prior period
    ends on the Saturday before ``start`` and reaches back as many WEEKS as the
    window itself contains, so like-for-like week counts meet. Shifting by the raw
    day span instead would hand a calendar-year window a 53-week comparable for its
    52 weeks.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # max(1) so a window too narrow to hold a week still yields a real span rather
    # than an empty one that would read as "no prior data".
    weeks = max(_anchor_weeks(start, end), 1)
    return start - pd.Timedelta(weeks=weeks), start - pd.Timedelta(days=1)


def clip(frame, start, end):
    """Rows whose WeekDate falls in [start, end]."""
    if frame is None or frame.empty:
        return frame
    weeks = frame["WeekDate"]
    return frame[(weeks >= pd.Timestamp(start)) & (weeks <= pd.Timestamp(end))]


def pct_change(current, prior):
    """Percent change, or None when there is no meaningful base to compare to.

    None rather than 0 or inf: a tile with no prior-year data must render an
    absent delta, not a confident "+0%".
    """
    if prior is None or current is None:
        return None
    if not np.isfinite(prior) or not np.isfinite(current) or prior == 0:
        return None
    return (current - prior) / abs(prior) * 100.0


def _total(series):
    """Sum treating all-NaN as 0.0 (nothing sold) but preserving float type."""
    if series is None or len(series) == 0:
        return 0.0
    total = pd.to_numeric(series, errors="coerce").sum(min_count=1)
    return 0.0 if pd.isna(total) else float(total)


def window_totals(frame, start, end):
    """Headline totals for one window: units, revenue, weeks, SKUs, customers."""
    win = clip(frame, start, end)
    if win is None or win.empty:
        return {"units": 0.0, "revenue": 0.0, "weeks": 0, "skus": 0, "customers": 0}
    return {
        "units": _total(win["demand"]),
        "revenue": _total(win["revenue"]) if "revenue" in win.columns else 0.0,
        "weeks": int(win["WeekDate"].nunique()),
        "skus": int(win.loc[win["demand"].notna(), "SKU"].nunique()),
        "customers": int(
            win.loc[win["demand"].notna(), "Customer Grouping"].nunique()
        ),
    }


def price_coverage(frame, start, end):
    """How much of a window's demand carries a list price.

    Returns (priced_skus, total_skus, pct_of_units_priced). The view captions this
    so a partial price file reads as missing data rather than as lost revenue.
    """
    win = clip(frame, start, end)
    if win is None or win.empty:
        return 0, 0, 0.0
    sold = win[win["demand"].notna() & (win["demand"] != 0)]
    if sold.empty:
        return 0, 0, 0.0
    total_skus = int(sold["SKU"].nunique())
    priced = sold[sold["List Price"].notna()] if "List Price" in sold.columns \
        else sold.iloc[0:0]
    priced_skus = int(priced["SKU"].nunique())
    total_units = _total(sold["demand"])
    pct = (_total(priced["demand"]) / total_units * 100.0) if total_units else 0.0
    return priced_skus, total_skus, pct


def _sold(frame):
    """Rows representing an actual sale -- a real, non-zero demand figure.

    The single definition of "sold", so every count and every list agrees. NaN is
    "no data" and 0 is "stocked but sold nothing"; neither makes a SKU active.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame(columns=_FRAME_COLS)
    return frame[frame["demand"].notna() & (frame["demand"] != 0)]


def breadth(frame, start, end):
    """Assortment health: active / new / dormant counts for a window.

    Every count here is ``len()`` of the corresponding breakdown frame, NOT a
    separate calculation. That is deliberate and load-bearing: these numbers are
    tiles a planner clicks to see the underlying list, and a tile reading 47 above a
    list of 45 rows would discredit the whole view. Sharing the implementation makes
    them agree by construction rather than by two pieces of code being kept in step.
    """
    return {
        "active_skus": len(active_skus_breakdown(frame, start, end)),
        "active_customers": len(active_customers_breakdown(frame, start, end)),
        "new_skus": len(new_skus_breakdown(frame, start, end)),
        "dormant_skus": len(dormant_skus_breakdown(frame, start, end)),
    }


def concentration(frame, start, end, n=10):
    """Share of window revenue earned by the top ``n`` SKUs, as a percent.

    Reads the share straight off ``top_share_breakdown`` so the tile and the list
    it opens cannot disagree (see ``breadth``). Returns None when nothing in the
    window is priced -- 0% would read as "no concentration" when the truth is
    "unknown".
    """
    top = top_share_breakdown(frame, start, end, n=n)
    if top.empty:
        return None
    return float(top["Share %"].sum())


# --------------------------------------------------------------------------- #
# Breakdowns -- the lists behind the KPI tiles                                 #
# --------------------------------------------------------------------------- #
# Each returns a display-ready frame for the tile's click-through modal. They are
# also the source of truth for the tile numbers themselves (see breadth /
# concentration above), which is why they live here beside the window maths rather
# than in the view layer.

def _with_key(frame):
    """Copy of ``frame`` with a normalised string SKU column ``_k`` to group on.

    Grouping by a Series derived from a DIFFERENT (unfiltered) frame raises on the
    length mismatch, so the key travels with the rows instead.
    """
    out = frame.copy()
    out["_k"] = out["SKU"].astype(str)
    return out


def _descriptions(frame):
    """``{SKU: Description}`` from whichever rows carry one."""
    if frame is None or frame.empty or "Description" not in frame.columns:
        return {}
    named = _with_key(frame).dropna(subset=["Description"])
    if named.empty:
        return {}
    return named.groupby("_k")["Description"].first().to_dict()


def active_skus_breakdown(frame, start, end):
    """SKUs that sold inside the window, biggest revenue first."""
    cols = ["SKU", "Description", "Units", "Revenue", "Weeks with sales"]
    win = _sold(clip(frame, start, end))
    if win is None or win.empty:
        return pd.DataFrame(columns=cols)
    win = _with_key(win)
    out = win.groupby("_k").agg(
        Units=("demand", _total),
        Revenue=("revenue", _total),
        **{"Weeks with sales": ("WeekDate", "nunique")},
    ).reset_index().rename(columns={"_k": "SKU"})
    out["Description"] = out["SKU"].map(_descriptions(win))
    return out.sort_values("Revenue", ascending=False)[cols].reset_index(drop=True)


def active_customers_breakdown(frame, start, end):
    """Customer groups that sold inside the window, biggest revenue first."""
    cols = ["Customer Grouping", "Region", "SKUs", "Units", "Revenue"]
    win = _sold(clip(frame, start, end))
    if win is None or win.empty:
        return pd.DataFrame(columns=cols)
    out = win.groupby("Customer Grouping").agg(
        SKUs=("SKU", "nunique"),
        Units=("demand", _total),
        Revenue=("revenue", _total),
    ).reset_index()
    if "Region" in win.columns:
        region = win.groupby("Customer Grouping")["Region"].first()
        out["Region"] = out["Customer Grouping"].map(region)
    else:
        out["Region"] = None
    return out.sort_values("Revenue", ascending=False)[cols].reset_index(drop=True)


def new_skus_breakdown(frame, start, end):
    """SKUs whose FIRST EVER sale in this snapshot falls inside the window.

    "First ever" is measured against the whole frame, not the window -- otherwise
    every SKU would look new in every window.
    """
    cols = ["SKU", "Description", "First sale week", "Units", "Revenue"]
    sold = _sold(frame)
    if sold is None or sold.empty:
        return pd.DataFrame(columns=cols)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sold = _with_key(sold)
    first = sold.groupby("_k")["WeekDate"].min()
    launched = first[(first >= start) & (first <= end)]
    if launched.empty:
        return pd.DataFrame(columns=cols)

    win = clip(sold, start, end)
    win = win[win["_k"].isin(launched.index)]
    out = win.groupby("_k").agg(
        Units=("demand", _total), Revenue=("revenue", _total),
    ).reset_index().rename(columns={"_k": "SKU"})
    out["Description"] = out["SKU"].map(_descriptions(sold))
    out["First sale week"] = out["SKU"].map(launched)
    return out.sort_values("First sale week")[cols].reset_index(drop=True)


def dormant_skus_breakdown(frame, start, end):
    """SKUs that sold in the 52 weeks before the window but nothing inside it.

    Assortment quietly drifting away -- invisible to an active-SKU count. Units and
    revenue describe that PRIOR stretch (there is nothing to report inside the
    window, by definition), and "Weeks since last sale" is counted from ``end`` so
    it reads as of the as-of date.
    """
    cols = ["SKU", "Description", "Last sale week", "Weeks since last sale",
            "Units (prior 52 wks)", "Revenue (prior 52 wks)"]
    sold = _sold(frame)
    if sold is None or sold.empty:
        return pd.DataFrame(columns=cols)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    sold = _with_key(sold)
    inside = set(clip(sold, start, end)["_k"])
    prior = clip(sold, start - pd.Timedelta(weeks=52), start - pd.Timedelta(days=1))
    if prior is None or prior.empty:
        return pd.DataFrame(columns=cols)
    prior = prior[~prior["_k"].isin(inside)]
    if prior.empty:
        return pd.DataFrame(columns=cols)

    out = prior.groupby("_k").agg(**{
        "Units (prior 52 wks)": ("demand", _total),
        "Revenue (prior 52 wks)": ("revenue", _total),
        "Last sale week": ("WeekDate", "max"),
    }).reset_index().rename(columns={"_k": "SKU"})
    out["Description"] = out["SKU"].map(_descriptions(sold))
    out["Weeks since last sale"] = (
        (end - out["Last sale week"]).dt.days // 7
    ).astype("int64")
    return out.sort_values("Last sale week", ascending=False)[cols] \
              .reset_index(drop=True)


def top_share_breakdown(frame, start, end, n=10):
    """The ``n`` biggest-revenue SKUs with their share and running total.

    Shares are computed against ALL priced window revenue, so "Share %" answers
    "how much of the business is this SKU" rather than "how much of this table".
    Built on ``pareto`` so the cumulative maths exists in exactly one place.
    """
    cols = ["SKU", "Description", "Units", "Revenue", "Share %", "Cumulative %"]
    ranked = pareto(frame, start, end)
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    total = float(ranked["revenue"].sum())
    top = ranked.head(n).copy()

    win = clip(frame, start, end)
    units = win.groupby(win["SKU"].astype(str))["demand"].sum(min_count=1)
    desc = (win.dropna(subset=["Description"])
            .groupby(win["SKU"].astype(str))["Description"].first())

    top["Units"] = top["SKU"].map(units)
    top["Description"] = top["SKU"].map(desc)
    top["Revenue"] = top["revenue"]
    top["Share %"] = top["revenue"] / total * 100.0 if total else 0.0
    top["Cumulative %"] = top["cum_share"]
    return top[cols].reset_index(drop=True)


def monthly_breakdown(frame, start, end, prior_start=None, prior_end=None):
    """Calendar-month units and revenue for a window, with a year-ago column.

    The prior-year revenue is aligned by CALENDAR MONTH (Jan against Jan), which is
    the comparison a month table implies -- distinct from the 364-day shift the KPI
    tiles use, where whole-week alignment is what matters. Months absent from the
    prior period read as blank, not 0.
    """
    cols = ["Month", "Units", "Revenue", "Revenue (prior year)", "YoY %"]
    win = clip(frame, start, end)
    if win is None or win.empty:
        return pd.DataFrame(columns=cols)
    cur = monthly_totals(win, "revenue").merge(
        monthly_totals(win, "demand")[["MonthStart", "demand"]],
        on="MonthStart", how="left",
    )
    out = pd.DataFrame({
        "MonthStart": cur["MonthStart"],
        "Month": cur["MonthStart"].dt.strftime("%b %Y"),
        "Units": cur["demand"],
        "Revenue": cur["revenue"],
    })

    out["Revenue (prior year)"] = np.nan
    if prior_start is not None and prior_end is not None:
        prior = monthly_totals(clip(frame, prior_start, prior_end), "revenue")
        if not prior.empty:
            # Key on (month-of-year, year+1) so Jan lines up against Jan.
            lookup = {(int(y) + 1, int(m)): v for y, m, v in zip(
                prior["Year"], prior["MonthNum"], prior["revenue"])}
            out["Revenue (prior year)"] = [
                lookup.get((int(ms.year), int(ms.month)))
                for ms in cur["MonthStart"]
            ]
    out["YoY %"] = [
        pct_change(c, p) for c, p in zip(out["Revenue"],
                                        out["Revenue (prior year)"])
    ]
    return out.sort_values("MonthStart")[cols].reset_index(drop=True)


def weekly_breakdown(frame, start, end):
    """Week-by-week units and revenue, most recent first."""
    cols = ["Week", "Units", "Revenue"]
    win = clip(frame, start, end)
    if win is None or win.empty:
        return pd.DataFrame(columns=cols)
    rev = weekly_totals(win, "revenue")
    units = weekly_totals(win, "demand")
    out = rev.merge(units, on="WeekDate", how="outer").sort_values(
        "WeekDate", ascending=False)
    return pd.DataFrame({
        "Week": out["WeekDate"],
        "Units": out["demand"],
        "Revenue": out["revenue"],
    })[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Chart-shaped aggregations                                                    #
# --------------------------------------------------------------------------- #
def weekly_totals(frame, value_col="revenue"):
    """WeekDate -> total, as a two-column frame ready to plot."""
    if frame is None or frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=["WeekDate", value_col])
    out = (frame.groupby("WeekDate")[value_col].sum(min_count=1)
           .reset_index().sort_values("WeekDate"))
    return out


def monthly_totals(frame, value_col="revenue"):
    """Calendar-month totals with Year / Month / MonthNum, sorted chronologically.

    A week counts wholly toward the month of its Sunday -- see the module
    docstring on why there is no proration.
    """
    cols = ["MonthStart", "Year", "MonthNum", value_col]
    if frame is None or frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=cols)
    out = frame.copy()
    out["MonthStart"] = out["WeekDate"].values.astype("datetime64[M]")
    out = (out.groupby("MonthStart")[value_col].sum(min_count=1)
           .reset_index().sort_values("MonthStart"))
    out["Year"] = out["MonthStart"].dt.year
    out["MonthNum"] = out["MonthStart"].dt.month
    return out[cols]


def seasonality_frame(frame, value_col="revenue"):
    """Year x week-of-year totals for overlaying seasons on one axis.

    Week-of-year is an ordinal counted from each year's own January 1st
    (``(WeekDate - Jan 1).days // 7 + 1``) rather than an ISO week number, so
    every year's week 1 starts at the same point in the calendar and the curves
    are comparable. ISO numbering would offset Sunday-anchored dates by a week.
    """
    cols = ["Year", "WeekOfYear", value_col]
    if frame is None or frame.empty or value_col not in frame.columns:
        return pd.DataFrame(columns=cols)
    out = frame.copy()
    year_start = pd.to_datetime(out["WeekDate"].dt.year.astype(str) + "-01-01")
    out["Year"] = out["WeekDate"].dt.year
    out["WeekOfYear"] = ((out["WeekDate"] - year_start).dt.days // 7) + 1
    return (out.groupby(["Year", "WeekOfYear"])[value_col].sum(min_count=1)
            .reset_index().sort_values(["Year", "WeekOfYear"]))[cols]


def fold_to_top_n(totals, n=8, label_col=None, value_col="revenue"):
    """Keep the ``n`` largest rows and sum the rest into a single OTHER_LABEL row.

    The categorical palette is eight slots and is never cycled, so the tail has to
    collapse rather than borrow a leading colour.
    """
    if totals is None or totals.empty or len(totals) <= n:
        return totals
    label_col = label_col or totals.columns[0]
    top = totals.nlargest(n, value_col)
    rest = totals[~totals[label_col].isin(top[label_col])]
    if rest.empty:
        return top
    tail = pd.DataFrame([{label_col: OTHER_LABEL,
                          value_col: _total(rest[value_col])}])
    return pd.concat([top, tail], ignore_index=True)


def by_dimension(frame, dim, start, end, value_col="revenue", top_n=None):
    """Window totals grouped by one dimension, largest first.

    ``top_n`` folds the tail into OTHER_LABEL (see ``fold_to_top_n``).
    """
    win = clip(frame, start, end)
    if win is None or win.empty or dim not in win.columns:
        return pd.DataFrame(columns=[dim, value_col])
    out = (win.groupby(win[dim].fillna(OTHER_LABEL).astype(str))[value_col]
           .sum(min_count=1).reset_index())
    out.columns = [dim, value_col]
    out = out.dropna(subset=[value_col])
    out = out[out[value_col] > 0].sort_values(value_col, ascending=False)
    if top_n is not None:
        out = fold_to_top_n(out, n=top_n, label_col=dim, value_col=value_col)
    return out.reset_index(drop=True)


def weekly_by_dimension(frame, dim, value_col="revenue", top_n=8):
    """Week x dimension totals for a stacked area, tail folded into OTHER_LABEL.

    Membership of the top ``n`` is decided once over the WHOLE frame, not per
    week, so a category cannot flicker in and out of "Other" week to week.
    """
    cols = ["WeekDate", dim, value_col]
    if frame is None or frame.empty or dim not in frame.columns:
        return pd.DataFrame(columns=cols)
    out = frame.copy()
    out[dim] = out[dim].fillna(OTHER_LABEL).astype(str)
    ranked = (out.groupby(dim)[value_col].sum(min_count=1)
              .dropna().nlargest(top_n).index)
    out[dim] = np.where(out[dim].isin(ranked), out[dim], OTHER_LABEL)
    return (out.groupby(["WeekDate", dim])[value_col].sum(min_count=1)
            .reset_index().sort_values(["WeekDate", dim]))[cols]


def top_skus(frame, start, end, n=15):
    """The ``n`` highest-revenue SKUs in a window, with units and description."""
    cols = ["SKU", "Description", "units", "revenue"]
    win = clip(frame, start, end)
    if win is None or win.empty:
        return pd.DataFrame(columns=cols)
    grouped = win.groupby(win["SKU"].astype(str)).agg(
        units=("demand", lambda s: _total(s)),
        revenue=("revenue", lambda s: _total(s)),
    ).reset_index().rename(columns={"SKU": "SKU"})
    desc = (win.dropna(subset=["Description"])
            .groupby(win["SKU"].astype(str))["Description"].first())
    grouped["Description"] = grouped["SKU"].map(desc)
    grouped = grouped[grouped["revenue"] > 0]
    return grouped.nlargest(n, "revenue")[cols].reset_index(drop=True)


def yoy_movers(frame, start, end, prior=None, n=10):
    """Biggest year-over-year revenue gainers and decliners across a window.

    Compares ``[start, end]`` with the same weeks a year earlier. ``prior`` accepts
    an explicit ``(start, end)`` so a caller can hand over the SAME comparison its
    KPI tiles used -- notably the year-start anchoring a Year-to-date window needs
    -- rather than this function deriving a second, subtly different one. Left None
    it falls back to ``prior_year_window`` (a 364-day shift, so whole weeks meet
    whole weeks).

    SKUs absent from one side count as 0 there -- a launch or a discontinuation IS
    the movement, and dropping them would hide the largest swings.

    Returns one frame of the top ``n`` gainers and top ``n`` decliners, sorted by
    delta descending.
    """
    cols = ["SKU", "Description", "current", "prior", "delta"]
    if frame is None or frame.empty or "revenue" not in frame.columns:
        return pd.DataFrame(columns=cols)
    cur_start, cur_end = pd.Timestamp(start), pd.Timestamp(end)
    pri_start, pri_end = (prior if prior is not None
                          else prior_year_window(cur_start, cur_end))

    def _by_sku(a, b):
        win = clip(frame, a, b)
        if win is None or win.empty:
            return pd.Series(dtype="float64")
        return win.groupby(win["SKU"].astype(str))["revenue"].sum(min_count=1)

    current = _by_sku(cur_start, cur_end)
    prior = _by_sku(pri_start, pri_end)
    if current.empty and prior.empty:
        return pd.DataFrame(columns=cols)

    out = pd.concat([current.rename("current"), prior.rename("prior")], axis=1)
    out = out.fillna(0.0).reset_index().rename(columns={"index": "SKU"})
    out["delta"] = out["current"] - out["prior"]
    out = out[out["delta"] != 0]
    if out.empty:
        return pd.DataFrame(columns=cols)

    desc = (frame.dropna(subset=["Description"])
            .groupby(frame["SKU"].astype(str))["Description"].first())
    out["Description"] = out["SKU"].map(desc)

    gainers = out.nlargest(n, "delta")
    decliners = out.nsmallest(n, "delta")
    both = pd.concat([gainers, decliners]).drop_duplicates("SKU")
    return both.sort_values("delta", ascending=False)[cols].reset_index(drop=True)


def pareto(frame, start, end):
    """SKUs ranked by window revenue with a cumulative-share curve.

    Returns (SKU, revenue, rank, cum_share_pct) -- how few SKUs carry how much of
    the business.
    """
    cols = ["SKU", "revenue", "rank", "cum_share"]
    win = clip(frame, start, end)
    if win is None or win.empty or "revenue" not in win.columns:
        return pd.DataFrame(columns=cols)
    by_sku = (win.groupby(win["SKU"].astype(str))["revenue"]
              .sum(min_count=1).dropna())
    by_sku = by_sku[by_sku > 0].sort_values(ascending=False)
    if by_sku.empty:
        return pd.DataFrame(columns=cols)
    out = by_sku.reset_index()
    out.columns = ["SKU", "revenue"]
    out["rank"] = np.arange(1, len(out) + 1)
    out["cum_share"] = out["revenue"].cumsum() / out["revenue"].sum() * 100.0
    return out[cols]


def month_year_matrix(frame, value_col="revenue"):
    """Month x Year pivot for the seasonal heatmap (rows = months 1-12)."""
    monthly = monthly_totals(frame, value_col=value_col)
    if monthly.empty:
        return pd.DataFrame()
    return monthly.pivot(index="MonthNum", columns="Year",
                         values=value_col).reindex(range(1, 13))
