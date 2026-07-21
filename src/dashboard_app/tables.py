"""Summary-table styling and the Excel-style add-filter-chip table filters."""
import pandas as pd
import streamlit as st

from dashboard_app.config import MODEL_USED_COL, PRICE_COL, RISK_COL, fmt_dollar


# --------------------------------------------------------------------------- #
# Summary table styling                                                       #
# --------------------------------------------------------------------------- #
def style_summary(summary_df):
    """Format numbers and colour the up/down columns (up green / down red)."""
    df = summary_df.copy()
    int_cols = [c for c in [
        "Weeks with data", "Initial Projection Average",
        "Updated Projection Average", "Projection Difference",
    ] if c in df.columns]
    fmt = {c: "{:,.0f}" for c in int_cols}
    # Format every descriptive-average column to one decimal. Single-group views
    # carry one; the Optimal Projections combined view carries two (All-History
    # and 8-Week POS/Orders Average).
    for c in df.columns:
        if c.endswith("POS/Orders Average") and pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:,.1f}"
    if PRICE_COL in df.columns:
        fmt[PRICE_COL] = lambda v: fmt_dollar(v, decimals=2)
    if RISK_COL in df.columns:
        fmt[RISK_COL] = lambda v: fmt_dollar(v, decimals=0)

    def colour_diff(v):
        if pd.isna(v):
            return ""
        if v > 0:
            return "color:#15803d;font-weight:600"
        if v < 0:
            return "color:#b91c1c;font-weight:600"
        return "color:#64748b"

    sty = df.style.format(fmt, na_rep="—")
    # Colour both the unit difference and the dollar revenue risk by direction.
    diff_cols = [c for c in ["Projection Difference", RISK_COL] if c in df.columns]
    if diff_cols:
        sty = sty.map(colour_diff, subset=diff_cols)
    return sty


# --------------------------------------------------------------------------- #
# Excel-style add-filter chips (searchable checklist / date range per field)   #
# --------------------------------------------------------------------------- #
# Only these fields are ever offered as filters — no continuous-number columns
# (e.g. Revenue Risk) that would make a useless hundreds-long checkbox list.
_SMALL_LIST = 15    # lists this short show every value up-front; bigger ones search
_CAP = 200          # max checkboxes rendered at once (keeps big fields snappy)
_ADD_PLACEHOLDER = "➕ Add filter…"
# Recognised week/date columns across the summary and data-quality tables.
_DATE_COLS = ["First_WeekDate", "Last_WeekDate",
              "First Projected Week", "Last Projected Week",
              "First Missing Week", "Last Missing Week"]


def _cb_key(wkey, opt):
    return f"{wkey}__cb__{opt}"


def _set_many(wkey, options, value):
    """Callback: check/uncheck a batch of options (runs before the rerun renders
    the checkboxes, so setting their session_state keys is safe)."""
    for o in options:
        st.session_state[_cb_key(wkey, o)] = value


def _build_fields(df, key, P):
    """Whitelist of filterable fields for ``df``, in a fixed order.

    Only SKU / Customer / Data Source / Model Used / Region / Date range /
    Active In are ever offered, and only when the underlying column exists (and,
    for checklists, varies). Each field is a dict describing how to read options
    and build a mask; ``kind`` is ``checklist``, ``active_in`` or ``date``.
    """
    fields = []

    def add_checklist(label, series):
        if series is not None and series.nunique(dropna=True) > 1:
            fields.append({
                "label": label, "wkey": f"{key}::{label}", "kind": "checklist",
                "values": series,
                "options": sorted(series.dropna().unique(), key=str),
            })

    if "SKU" in df.columns:
        add_checklist("SKU", df["SKU"])

    cust_col = next((c for c in ("Customer Grouping", "Customer")
                     if c in df.columns), None)
    if cust_col:
        add_checklist("Customer", df[cust_col])

    if "Data Source" in df.columns:
        add_checklist("Data Source", df["Data Source"])

    # Model Used: only present on the Optimized Projections table, where each
    # customer group carries its own backtest-winning model.
    if MODEL_USED_COL in df.columns:
        add_checklist("Model Used", df[MODEL_USED_COL])

    # Region: an explicit Region/Region Code column if present, else derived from
    # the customer grouping via the loaded pipeline (summary/KPI tables).
    if "Region" in df.columns:
        add_checklist("Region", df["Region"])
    elif "Region Code" in df.columns:
        add_checklist("Region", df["Region Code"])
    elif P is not None and "Customer Grouping" in df.columns:
        add_checklist("Region",
                      df["Customer Grouping"].map(lambda g: str(P.region_for_group(g))))

    date_cols = [c for c in _DATE_COLS if c in df.columns]
    if date_cols:
        parsed = {c: pd.to_datetime(df[c], errors="coerce") for c in date_cols}
        allv = pd.concat(parsed.values())
        lo, hi = allv.min(), allv.max()
        if pd.notna(lo) and pd.notna(hi) and lo.date() != hi.date():
            firsts = [c for c in date_cols if "First" in c] or date_cols
            lasts = [c for c in date_cols if "Last" in c] or date_cols
            fields.append({
                "label": "Date range", "wkey": f"{key}::Date", "kind": "date",
                "first": pd.concat([parsed[c] for c in firsts], axis=1).min(axis=1),
                "last": pd.concat([parsed[c] for c in lasts], axis=1).max(axis=1),
                "min_d": lo.date(), "max_d": hi.date(),
            })

    if "Active in" in df.columns:
        codes = sorted({x.strip() for s in df["Active in"].dropna()
                        for x in str(s).split(",") if x.strip()})
        if len(codes) > 1:
            fields.append({
                "label": "Active In", "wkey": f"{key}::ActiveIn",
                "kind": "active_in", "values": df["Active in"], "options": codes,
            })

    return fields


def _selection(field):
    """Read a field's current selection from session_state (persists across
    reruns): a set of checked values/codes, or a ``(start, end)`` date tuple."""
    wkey = field["wkey"]
    if field["kind"] == "date":
        cur = st.session_state.get(f"{wkey}__di")
        if isinstance(cur, (tuple, list)) and len(cur) == 2:
            return (cur[0], cur[1])
        return None
    return {o for o in field["options"]
            if st.session_state.get(_cb_key(wkey, o), False)}


def _field_mask(df, field, selection):
    """Boolean row mask for one field's selection (empty selection → all True)."""
    kind = field["kind"]
    if kind == "checklist":
        if not selection:
            return pd.Series(True, index=df.index)
        return field["values"].isin(selection)
    if kind == "active_in":
        if not selection:
            return pd.Series(True, index=df.index)
        sel = set(selection)
        return field["values"].apply(
            lambda s: bool({x.strip() for x in str(s).split(",")} & sel)
        )
    # date: keep rows whose [first, last] interval overlaps the picked window.
    if not selection:
        return pd.Series(True, index=df.index)
    start, end = selection
    if start == field["min_d"] and end == field["max_d"]:
        return pd.Series(True, index=df.index)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    overlap = (field["last"] >= start_ts) & (field["first"] <= end_ts)
    return overlap.fillna(True)  # keep rows with unknown dates


def _popover_checklist(label, options, wkey):
    """A filter chip: a popover button opening a searchable checkbox list.

    Values are OR-ed within the field. Small lists render in full; large ones
    show a search box (and only the values you've already checked) until you
    type, so the dropdown stays fast and uncluttered. Each checkbox owns its
    state via a stable session-state key, so a choice survives reruns and being
    searched out of view. Returns the set of checked values (empty = no filter).
    """
    options = list(options)

    def selected():
        return [o for o in options if st.session_state.get(_cb_key(wkey, o), False)]

    n = len(selected())
    with st.popover(f"{label} ({n})" if n else label, use_container_width=True):
        query = st.text_input(
            f"Search {label}", key=f"{wkey}__q", placeholder="Type to search…",
            label_visibility="collapsed",
        ).strip().lower()

        if query:
            matches = [o for o in options if query in str(o).lower()]
        elif len(options) <= _SMALL_LIST:
            matches = options
        else:
            matches = None  # big list, no query: show only what's already picked

        batch = matches if matches is not None else options
        c_all, c_clear = st.columns(2)
        c_all.button("Select all", key=f"{wkey}__all", use_container_width=True,
                     on_click=_set_many, args=(wkey, batch, True))
        c_clear.button("Clear", key=f"{wkey}__clear", use_container_width=True,
                       on_click=_set_many, args=(wkey, batch, False))

        if matches is None:
            for o in sorted(selected(), key=str):
                st.checkbox(str(o), key=_cb_key(wkey, o))
            st.caption(f"Type to search {len(options):,} values.")
        else:
            for o in matches[:_CAP]:
                st.checkbox(str(o), key=_cb_key(wkey, o))
            if len(matches) > _CAP:
                st.caption(f"Showing {_CAP:,} of {len(matches):,} — refine the search.")
            elif not matches:
                st.caption("No matches.")

    return set(selected())


def _popover_daterange(label, field):
    """A date-range filter chip. Returns the picked ``(start, end)`` or None."""
    wkey, dikey = field["wkey"], f"{field['wkey']}__di"
    lo, hi = field["min_d"], field["max_d"]
    cur = st.session_state.get(dikey)
    narrowed = isinstance(cur, (tuple, list)) and len(cur) == 2 and tuple(cur) != (lo, hi)
    with st.popover(f"{label} ✓" if narrowed else label, use_container_width=True):
        kwargs = {} if dikey in st.session_state else {"value": (lo, hi)}
        val = st.date_input(label, min_value=lo, max_value=hi, key=dikey,
                            label_visibility="collapsed", **kwargs)
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return (val[0], val[1])
    return None  # mid-selection (only a start picked) → treat as no filter


def _add_filter(key, active_key):
    """Callback: activate the field chosen in the "Add filter" selectbox."""
    choice = st.session_state.get(f"{key}__add")
    if choice and choice != _ADD_PLACEHOLDER:
        active = list(st.session_state.get(active_key, []))
        if choice not in active:
            active.append(choice)
        st.session_state[active_key] = active
    st.session_state[f"{key}__add"] = _ADD_PLACEHOLDER  # reset for the next add


def _remove_filter(active_key, label, wkey, kind, options):
    """Callback: drop a filter chip and clear whatever it had selected."""
    st.session_state[active_key] = [
        l for l in st.session_state.get(active_key, []) if l != label
    ]
    if kind == "date":
        st.session_state.pop(f"{wkey}__di", None)
    else:
        for o in options:
            st.session_state[_cb_key(wkey, o)] = False


def filter_table(df, key, P=None):
    """Add-filter-chip filtering: start clean, add only the fields you want.

    An "Add filter" picker activates a field; each active filter shows as a chip
    — a searchable checkbox dropdown (or a date-range picker) plus a ✕ to remove
    it. Excel semantics (OR within a field, AND across fields) with
    cross-filtering, so the active dropdowns only offer values that still yield
    rows. Only the whitelist SKU / Customer / Data Source / Model Used / Region /
    Date range / Active In is offered. ``key`` namespaces the widgets.
    """
    fields = _build_fields(df, key, P)
    if not fields:
        return df
    by_label = {f["label"]: f for f in fields}
    labels_in_order = [f["label"] for f in fields]

    active_key = f"{key}__active"
    active = [l for l in st.session_state.get(active_key, []) if l in by_label]
    st.session_state[active_key] = active  # sanitised (columns change per view)
    active_fields = [by_label[l] for l in active]

    # "Add filter" picker — only fields not already active.
    addable = [l for l in labels_in_order if l not in active]
    add_col, _ = st.columns([1, 2])
    with add_col:
        if addable:
            st.selectbox(
                "Add filter", [_ADD_PLACEHOLDER] + addable, key=f"{key}__add",
                label_visibility="collapsed", on_change=_add_filter,
                args=(key, active_key),
            )
        else:
            st.caption("All filters added.")

    # Current selections, read from session_state (persist across reruns).
    sel = {f["label"]: _selection(f) for f in active_fields}

    def available(target):
        """Options of the target field still reachable under every OTHER active
        filter — Excel's narrowed dropdown. (Date fields have no option list.)"""
        mask = pd.Series(True, index=df.index)
        for f in active_fields:
            if f["label"] == target["label"]:
                continue
            mask &= _field_mask(df, f, sel[f["label"]])
        if target["kind"] == "checklist":
            return set(target["values"][mask].dropna().unique())
        if target["kind"] == "active_in":
            codes = set()
            for s in target["values"][mask].dropna():
                codes |= {x.strip() for x in str(s).split(",") if x.strip()}
            return codes
        return None

    # Clamp away any checked value that no longer yields rows, so an empty
    # combination can't persist (date fields aren't clamped).
    for f in active_fields:
        if f["kind"] == "date":
            continue
        reachable = available(f)
        for o in list(sel[f["label"]]):
            if o not in reachable:
                st.session_state[_cb_key(f["wkey"], o)] = False
                sel[f["label"]].discard(o)

    # Render active filters as compact chips — [ control ][✕] — several per row.
    selections = {}
    per_row, chip_w, x_w = 3, 5, 1
    unit = chip_w + x_w
    for start in range(0, len(active_fields), per_row):
        chunk = active_fields[start:start + per_row]
        widths = [chip_w, x_w] * len(chunk)
        if len(chunk) < per_row:
            widths.append(unit * (per_row - len(chunk)))  # spacer keeps chips small
        cols = st.columns(widths)
        for i, f in enumerate(chunk):
            with cols[i * 2]:
                if f["kind"] == "date":
                    selections[f["label"]] = _popover_daterange(f["label"], f)
                else:
                    selections[f["label"]] = _popover_checklist(
                        f["label"], sorted(available(f), key=str), f["wkey"]
                    )
            with cols[i * 2 + 1]:
                st.button("✕", key=f"{f['wkey']}__rm",
                          help=f"Remove the {f['label']} filter",
                          on_click=_remove_filter,
                          args=(active_key, f["label"], f["wkey"], f["kind"],
                                f.get("options", [])))

    mask = pd.Series(True, index=df.index)
    for f in active_fields:
        mask &= _field_mask(df, f, selections[f["label"]])

    out = df[mask]
    if len(out) != len(df):
        st.caption(f"{len(out):,} of {len(df):,} rows match the filters.")
    return out


@st.fragment
def render_filtered_table(df, key, P=None, *, style=True):
    """Render the add-filter chips + the table in an isolated fragment.

    The fragment scopes a filter click to just this block, so filtering never
    reruns the whole dashboard — it stays quick and clean, like Excel. ``style``
    applies the summary formatting/colouring (summary & KPI tables); pass
    ``style=False`` for the data-quality tables, which render plainly. ``df``
    (the unfiltered frame) is captured as a fragment arg and reused each rerun.
    """
    filtered = filter_table(df, key, P)
    st.dataframe(
        style_summary(filtered) if style else filtered,
        width="stretch", hide_index=True,
    )
