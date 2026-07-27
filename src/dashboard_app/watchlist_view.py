"""The Watchlist view: star (SKU, Customer Grouping) combinations on named,
shared watchlists and jump straight to their projection detail.

Detail numbers reuse the best-model-per-group table (the same source as Optimized
Projections), and each detail card reuses the Exceptions view's actuals-vs-plan
chart + date-range picker + "Calculate Optimal Projection" (5-model backtest)
button, so a starred row reads exactly like an Exceptions card.
"""
import functools

import pandas as pd
import streamlit as st

from dashboard_app.compute import (
    ALL_HIST_AVG_COL,
    EIGHT_WK_AVG_COL,
    _agent_summaries_mtime,
    _agent_summaries_oldest_at,
    compute_by_customer_best,
)
from dashboard_app.config import BEST_MODEL_COMBINED_VIEW, PRICE_COL
from dashboard_app.exceptions import _render_exception_chart, sku_week_by_group
from dashboard_app.tables import render_selectable_table
from dashboard_app.watchlist import (
    DEFAULT_NAME, active_name, active_pairs, create_watchlist, delete_watchlist,
    is_starred, list_names, remove_star, rename_watchlist, set_active, toggle_star,
)

# st.dataframe stores its row selection under "<key>__sel"; the watchlist table
# uses key "watchlist". Removing a row shifts positional indices, so this is
# cleared on removal to stop a stale selection reopening the wrong card.
_WATCHLIST_SEL_KEY = "watchlist__sel"

# Detail-card field order for the watchlist (decoupled from the best-model table's
# full column set). Renders as three rows:
#   Customer Grouping · Region · Data Source
#   All-History Avg · 8-Week Avg · Current Projection Average
#   List Price · Weeks with data
# The projection change (new value, difference, revenue risk) comes from the card's
# "Calculate Optimal Projection" button, so it's intentionally left off the grid.
WATCHLIST_CARD_COLS = [
    "Customer Grouping", "Region", "Data Source",
    ALL_HIST_AVG_COL, EIGHT_WK_AVG_COL, "Current Projection Average",
    PRICE_COL, "Weeks with data",
]


def _best_model_table(df, today_ts, today_str, prices, n_excluded_rows):
    """The best-model-per-group combined table, sharing the Optimized view's
    session cache so switching between the two views never recomputes it.

    Returns the 6-tuple ``compute_by_customer_best`` yields (``combined`` first),
    or a tuple of Nones when nothing resolved.
    """
    price_marker = None if prices is None else int(len(prices))
    sig = (BEST_MODEL_COMBINED_VIEW, today_str, price_marker, n_excluded_rows,
           _agent_summaries_mtime())
    if st.session_state.get("bestmix_structural") != sig:
        with st.spinner("Forecasting each group with its best model…"):
            result = compute_by_customer_best(df, today_ts, prices, min_weeks=None)
        st.session_state["bestmix_result"] = result
        # Oldest stamp — see kpis.py: the shared "bestmix_generated_at" cache must
        # mean the same thing whichever view populated it (the freshness caption
        # reads it as "everything here is at least this fresh").
        st.session_state["bestmix_generated_at"] = _agent_summaries_oldest_at()
        st.session_state["bestmix_structural"] = sig
    else:
        result = st.session_state.get("bestmix_result")
    return result if result is not None else (None, None, None, None, None, [])


def _watchlist_agg(df, P, today_str):
    """Per-SKU-week aggregate for the detail-card charts, cached in session by
    snapshot so the (now infrequent) full reruns don't re-aggregate every group."""
    sig = (today_str, int(len(df)))
    if st.session_state.get("wl_agg_sig") != sig:
        st.session_state["wl_agg"] = sku_week_by_group(df, P)
        st.session_state["wl_agg_sig"] = sig
    return st.session_state["wl_agg"]


# --------------------------------------------------------------------------- #
# List-management modals (clean confirmation windows) + the controls row       #
# --------------------------------------------------------------------------- #
# All three mirror agent_summary._confirm_run_all_dialog: a two-column
# Cancel / primary-action footer, closing the modal with st.rerun().
@st.dialog("New watchlist")
def _new_dialog():
    name = st.text_input("Name", key="wl_new_name", placeholder="e.g. Q3 Priorities")
    left, right = st.columns(2)
    if left.button("Cancel", key="wl_new_cancel", width="stretch"):
        st.rerun()
    if right.button("Create", key="wl_new_go", type="primary", width="stretch"):
        if create_watchlist(name):
            st.rerun()
        st.warning("Enter a name for the watchlist.")


@st.dialog("Rename watchlist")
def _rename_dialog(active):
    name = st.text_input("New name", value=active, key="wl_ren_name")
    left, right = st.columns(2)
    if left.button("Cancel", key="wl_ren_cancel", width="stretch"):
        st.rerun()
    if right.button("Rename", key="wl_ren_go", type="primary", width="stretch"):
        if rename_watchlist(active, name):
            st.rerun()
        st.warning("Choose a new name that isn't already in use.")


@st.dialog("Delete watchlist")
def _delete_dialog(active):
    st.warning(f"Delete **{active}** and its starred items? This can't be undone.")
    # Colour the destructive confirm button red (Streamlit has no danger button
    # type; scope the style to this button's key wrapper, as the app does elsewhere).
    st.markdown(
        "<style>.st-key-wl_del_go button{background-color:#dc2626;"
        "border-color:#dc2626;color:#fff;}"
        ".st-key-wl_del_go button:hover{background-color:#b91c1c;"
        "border-color:#b91c1c;color:#fff;}</style>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    if left.button("Cancel", key="wl_del_cancel", width="stretch"):
        st.rerun()
    if right.button("Delete", key="wl_del_go", type="primary", width="stretch"):
        delete_watchlist(active)
        st.rerun()


@st.dialog("Remove from watchlist")
def _remove_dialog(pairs, active):
    """Confirm removing one or more ``(sku, customer)`` pairs from ``active``.

    Shared by the detail-card "Remove" button (a single pair) and the orphan-pair
    "Remove selected" control (a list). Confirming removes each pair, clears the
    watchlist table's stale row selection, and triggers a full app rerun so the
    table and the ★ marker in the other views refresh."""
    pairs = list(pairs)
    if len(pairs) == 1:
        s, c = pairs[0]
        st.warning(f"Remove **{s} / {c}** from **{active}**?")
    else:
        st.warning(f"Remove **{len(pairs)}** combinations from **{active}**?")
    # Red confirm button — same scoped-style trick as _delete_dialog.
    st.markdown(
        "<style>.st-key-wl_rm_go button{background-color:#dc2626;"
        "border-color:#dc2626;color:#fff;}"
        ".st-key-wl_rm_go button:hover{background-color:#b91c1c;"
        "border-color:#b91c1c;color:#fff;}</style>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    if left.button("Cancel", key="wl_rm_cancel", width="stretch"):
        st.rerun()
    if right.button("Remove", key="wl_rm_go", type="primary", width="stretch"):
        for s, c in pairs:
            remove_star(s, c, active)
        # Reset widgets whose stored value now references removed rows: the table's
        # positional selection (a shifted index could reopen the wrong card) and the
        # orphan multiselect (Streamlit rejects stored values absent from options).
        # Guarded — these are widget-owned keys.
        for k in (_WATCHLIST_SEL_KEY, "wl_missing_ms"):
            try:
                st.session_state.pop(k, None)
            except Exception:
                pass
        st.rerun()


@st.fragment
def _list_controls():
    """Watchlist selector + always-visible icon actions (New / Rename / Delete),
    each opening a modal confirmation window. In a fragment so opening a modal
    reruns only this block (no full page reload); changing the active list
    triggers a full rerun so the table and the ★ marker in the other views
    refresh. Assumes ≥1 watchlist exists (empty-state prompt lives in
    render_watchlist)."""
    names = list_names()
    active = active_name()

    # Red trash glyph — the delete affordance reads as destructive at a glance.
    st.markdown(
        "<style>.st-key-wl_del_btn button{color:#dc2626;}</style>",
        unsafe_allow_html=True,
    )

    # [ selector (wide) ][ add ][ rename ][ delete ]
    c_sel, c_new, c_ren, c_del = st.columns([6, 1, 1, 1],
                                            vertical_alignment="bottom")
    with c_sel:
        # Keyless: the index is re-seeded from active_name() each run and a change
        # writes through set_active + full rerun, so the widget can't hold a stale
        # (e.g. just-deleted) name.
        choice = st.selectbox("Watchlist", names,
                              index=names.index(active) if active in names else 0)

    def _icon(container, glyph, key, help_text, on_click):
        with container:
            if st.button(f":material/{glyph}:", key=key, help=help_text,
                         width="stretch"):
                on_click()

    _icon(c_new, "add", "wl_new_btn", "New watchlist", _new_dialog)
    _icon(c_ren, "edit", "wl_ren_btn", "Rename this watchlist",
          lambda: _rename_dialog(active))
    _icon(c_del, "delete", "wl_del_btn", "Delete this watchlist",
          lambda: _delete_dialog(active))

    if choice != active_name():
        set_active(choice)
        st.rerun()  # full: the table + ★ marker elsewhere must refresh


def render_watchlist(df, today_ts, today_str, prices, n_excluded_rows, anchors, P):
    """Render the WATCHLIST_VIEW. Signature mirrors render_exceptions /
    _render_best_model_combined so main()'s dispatch call is uniform."""
    st.subheader("Watchlist")
    st.caption(
        "Star SKU / customer-group combinations to pin their projection detail. "
        "Create multiple named lists; the one selected here is the *active* list "
        "that the ★ marker and Starred filter reflect across the other tables. "
        "Watchlists are shared across everyone using this dashboard."
    )

    # Empty state (no lists yet) — the create prompt lives here, outside the
    # fragment, so the fragment can assume at least one list exists.
    if not list_names():
        st.info("No watchlists yet — create one to get started.")
        new = st.text_input("New watchlist name", value=DEFAULT_NAME,
                             key="wl_new_first")
        if st.button("Create watchlist", type="primary",
                     key="wl_new_first_btn") and create_watchlist(new):
            st.rerun()
        return

    _list_controls()
    active = active_name()

    # ----- Add / remove control (acts on the active list) ------------------- #
    # Cascade the pickers so only real (SKU, group) pairs that actually have
    # demand data can be selected — the customer list is narrowed to the groups
    # the chosen SKU sells to. Prevents starring nonexistent combinations (which
    # would have no detail card and get stranded in the "not in best-model table"
    # list below).
    skus = sorted(df["SKU"].dropna().astype(str).unique().tolist())
    c_sku, c_cust = st.columns(2)
    sel_sku = c_sku.selectbox("SKU", skus, help="Type to search") if skus else None
    groups = (sorted(df.loc[df["SKU"].astype(str) == sel_sku, "Customer Grouping"]
                     .dropna().astype(str).unique().tolist())
              if sel_sku is not None else [])
    sel_cust = c_cust.selectbox("Customer Grouping", groups) if groups else None
    if sel_sku is not None and sel_cust is not None:
        starred = is_starred(sel_sku, sel_cust)
        label = (f"★ Remove from “{active}”" if starred
                 else f"☆ Add to “{active}”")
        if starred:
            # Destructive red styling for the remove mode (scoped to this button).
            st.markdown(
                "<style>.st-key-wl_toggle button{background-color:#dc2626;"
                "border-color:#dc2626;color:#fff;}"
                ".st-key-wl_toggle button:hover{background-color:#b91c1c;"
                "border-color:#b91c1c;color:#fff;}</style>",
                unsafe_allow_html=True,
            )
        if st.button(label, key="wl_toggle",
                     type="secondary" if starred else "primary"):
            toggle_star(sel_sku, sel_cust)
            st.rerun()

    st.divider()

    # ----- Watchlist table -------------------------------------------------- #
    pairs = active_pairs()
    if not pairs:
        st.info(
            f"**{active}** is empty. Pick a SKU and customer group above and click "
            "**Add**. Starred rows are also marked with a ★ on every other table."
        )
        return

    combined, *_ = _best_model_table(df, today_ts, today_str, prices, n_excluded_rows)
    if combined is None or getattr(combined, "empty", True):
        st.warning(
            "No customer group has a recommended model yet, so watchlist detail "
            "can't be shown. Click **Recommend models (all views)** (or run "
            "`python -m agent.batch`), then reopen this view."
        )
        return

    keys = list(zip(combined["SKU"].astype(str),
                    combined["Customer Grouping"].astype(str)))
    present = set(keys)
    in_watchlist = pd.Series([k in pairs for k in keys], index=combined.index)
    watch_df = combined[in_watchlist]

    if watch_df.empty:
        st.info(
            "None of the starred combinations on this list are in the current "
            "best-model table (they may be discontinued or awaiting a model "
            "recommendation)."
        )
    else:
        # Detail cards reuse the Exceptions chart: actuals-vs-plan + date range +
        # the "Calculate Optimal Projection" (5-model backtest) button. Bind its
        # leading args; the (row, key_base) tail matches the detail_chart contract.
        agg = _watchlist_agg(df, P, today_str)
        chart_cb = functools.partial(
            _render_exception_chart, agg, anchors, df, prices, today_ts
        )
        # Region isn't in the best-model table; derive it for the detail card.
        # .assign copies, so the shared best-model cache stays untouched.
        watch_df = watch_df.assign(
            Region=watch_df["Customer Grouping"].map(
                lambda g: str(P.region_for_group(g))
            )
        )
        render_selectable_table(
            watch_df, "watchlist", P,
            condensed_cols=["SKU", "Customer Grouping"],
            detail_chart=chart_cb,
            detail_cols=WATCHLIST_CARD_COLS,
            row_action={
                "label": f"🗑 Remove from “{active}”",
                "help": "Remove this combination from the active watchlist",
                "danger": True,
                "callback": lambda row: _remove_dialog(
                    [(str(row["SKU"]), str(row["Customer Grouping"]))], active),
            },
        )

    # Orphaned pins: on the list but absent from the best-model table (discontinued,
    # remapped, or awaiting a recommendation), so they have no row/detail card to
    # remove from. Offer a dedicated remover so they're never stranded.
    missing = sorted(p for p in pairs if p not in present)
    if missing:
        st.caption(
            f"{len(missing)} starred combination(s) on this list aren't in the "
            "current best-model table (discontinued, remapped, or awaiting a model "
            "recommendation). Select any below to remove them."
        )
        by_label = {f"{s} / {c}": (s, c) for s, c in missing}
        picked = st.multiselect(
            "Combinations not in the best-model table", list(by_label),
            key="wl_missing_ms",
            placeholder="Select combinations to remove — type to search",
            label_visibility="collapsed",
        )
        if st.button("🗑 Remove selected", key="wl_missing_rm",
                     disabled=not picked):
            _remove_dialog([by_label[p] for p in picked], active)
