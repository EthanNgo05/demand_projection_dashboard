"""Data-quality section renderers (inactive / missing / discontinued / no-POS)."""
import pandas as pd
import streamlit as st

from dashboard_app.config import ALL_CUSTOMERS_VIEW, region_from_view
from dashboard_app.datasources import _this_week_start
from dashboard_app.tables import render_filtered_table
from dashboard_app.compute import summary_to_excel


def render_inactive_section(view, region, check_ran, inactive_df,
                            excluded_counts_by_key, n_excluded_rows, today_str,
                            key_skus=None, key_suffix="", show_header=True):
    """Table of active products projected in regions they are not 'Active in'.

    The rows behind these SKU × customer × region combos were dropped from every
    summary table above (the SKU isn't 'Active in' that region). Surface them so
    the exclusion is visible and auditable.

    ``key_skus`` (a set of key SKUs) filters the table to those SKUs only;
    ``key_suffix`` disambiguates widget keys when the section is rendered twice on
    one page (e.g. once per Exceptions tab); ``show_header`` suppresses the ``###``
    title when the caller already labels the section (e.g. an expander).
    """
    HEADER = "### SKUs with forecasts in locations they are not active in"
    if show_header:
        st.markdown(HEADER)
    if not check_ran:
        st.info(
            "Upload a Plytix export with an 'Active in' column (sidebar) to run "
            "the active-in check."
        )
        return

    # In a "By customer group" view, mirror the summary table above: only
    # show rows whose region matches the selected region (e.g. a US view
    # shouldn't list "JP (NETDEPOT)" rows). ALL CUSTOMERS shows every region.
    region_scoped = view != ALL_CUSTOMERS_VIEW and region is not None
    table_df = inactive_df
    if region_scoped:
        table_df = inactive_df[inactive_df["Region"] == region]
    if key_skus is not None:
        table_df = table_df[
            table_df["SKU"].astype(str).str.rstrip("*").isin(key_skus)
        ]

    # Always show only non-zero future projections: rows whose Last_WeekDate is
    # this week and onward (Sunday-anchored via _this_week_start, matching the
    # "future avg/wk" projection column) with a non-zero projection.
    week_start = _this_week_start().date()
    fdf = table_df.copy()
    fdf["First_WeekDate"] = pd.to_datetime(fdf["First_WeekDate"]).dt.date
    fdf["Last_WeekDate"] = pd.to_datetime(fdf["Last_WeekDate"]).dt.date
    fdf["Original_Projection"] = pd.to_numeric(
        fdf["Original_Projection"], errors="coerce"
    ).round(0)
    fdf = fdf[
        (fdf["Last_WeekDate"] >= week_start)
        & (fdf["Original_Projection"].notna())
        & (fdf["Original_Projection"] != 0)
    ]

    if fdf.empty:
        if region_scoped:
            st.success(
                f"None found for {region} — every active product here is "
                "only forecast in regions it is active in."
            )
        else:
            st.success(
                "None found — every active product is only forecast in "
                "regions it is active in."
            )
        return

    n_skus = fdf["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Excluded from the forecast above{scope_note}: "
        f"{n_skus:,} distinct SKUs. Each is an active product being "
        "forecast in a region (US/CA/EU/JP/AU) that is not in its Plytix "
        "'Active in' list."
    )

    show = fdf[[
        'SKU', 'Region', 'Region Code', 'Active in', 'Customer Grouping',
        'First_WeekDate', 'Last_WeekDate', 'Original_Projection',
    ]].rename(columns={
        "First_WeekDate": "First Projected Week",
        "Last_WeekDate": "Last Projected Week",
        "Original_Projection": "Original Projection (future avg/wk)",
    })
    render_filtered_table(show, f"filter_inactive{key_suffix}", style=False)
    st.download_button(
        "⬇️ Download the excluded (inactive-region) projections table",
        data=summary_to_excel(show, sheet_name="inactive_projections"),
        file_name=f"inactive_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_inactive_projections{key_suffix}",
    )


def render_missing_section(view, region, warehouse_df, check_ran, missing_df,
                           today_str, cust_source=None, P=None,
                           key_skus=None, key_suffix="", show_header=True):
    """Table of active products MISSING future projections in active regions.

    Ported from active_missing_projections.py. The inverse of the inactive
    section above: these are active SKUs that ARE 'Active in' a region but have
    no projection for one or more of the coming 15 weeks there. Sourced from the
    warehouse projection grid (sidebar), the only place a blank/missing week is
    visible. In a "By customer group" view, only rows whose region matches the
    selected region are shown; ALL CUSTOMERS shows every region.

    ``key_skus``/``key_suffix``/``show_header`` behave as in
    ``render_inactive_section``.
    """
    HEADER = "### SKUs missing forecasts in locations they are active in"
    if show_header:
        st.markdown(HEADER)
    if warehouse_df is None or warehouse_df.empty:
        st.info(
            "Upload the warehouse projection files (AU/CA/EU/JP/US) in the "
            "sidebar to run the missing-projections check."
        )
        return
    if not check_ran:
        st.info(
            "Upload a Plytix export with an 'Active in' column (sidebar) to run "
            "the missing-projections check."
        )
        return

    # Each Customer folds to its forecast customer group (e.g. AMAZON-DS ->
    # AMAZON-DC); used both to scope a by-customer view and to look up the source.
    grouping = getattr(P, "COMBINED_GROUPING", {}) if P is not None else {}
    row_group = missing_df["Customer"].map(lambda c: grouping.get(c, c))

    # A by-customer-group view shows only that group's rows (not every customer
    # in the region); a per-region "All Customers" rollup shows every group in
    # its region; ALL CUSTOMERS shows everything.
    group_scoped = view != ALL_CUSTOMERS_VIEW
    region_all = region_from_view(view)
    table_df = missing_df
    if region_all is not None and P is not None:
        table_df = missing_df[
            row_group.map(lambda g: str(P.region_for_group(g))) == region_all
        ]
    elif group_scoped:
        table_df = missing_df[row_group == view]
    if key_skus is not None:
        table_df = table_df[
            table_df["SKU"].astype(str).str.rstrip("*").isin(key_skus)
        ]

    if table_df.empty:
        if group_scoped:
            st.success(
                f"None found for {view} — every active product here has "
                "future projections in the regions it is active in."
            )
        else:
            st.success(
                "None found — every active product has future projections in "
                "the regions it is active in."
            )
        return

    n_skus = table_df["SKU"].nunique()
    scope_note = f" for {view}" if group_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct SKUs. Each is an active "
        "product (Plytix) with no projection for one or more of the coming 15 "
        "weeks in a region (US/CA/EU/JP/AU) it IS 'Active in'."
    )

    show = table_df[[
        'SKU', 'Region', 'Region Code', 'Active in', 'Customer',
        'First_WeekDate', 'Last_WeekDate',
    ]].rename(columns={
        "First_WeekDate": "First Missing Week",
        "Last_WeekDate": "Last Missing Week",
    })
    # Data source (POS/Orders) from the summary table, keyed by (customer, SKU).
    src_lookup = cust_source or {}
    show.insert(
        show.columns.get_loc("Customer") + 1,
        "Data Source",
        [
            src_lookup.get((grouping.get(c, c), str(s).rstrip("*")))
            for s, c in zip(show["SKU"], show["Customer"])
        ],
    )
    render_filtered_table(show, f"filter_missing{key_suffix}", P, style=False)
    st.download_button(
        "⬇️ Download the missing-projections table",
        data=summary_to_excel(show, sheet_name="missing_projections"),
        file_name=f"missing_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_missing_projections{key_suffix}",
    )


def render_discontinued_section(view, region, disc_check_ran, discontinued_df,
                                today_str,
                                key_skus=None, key_suffix="", show_header=True):
    """Table of Discontinued/Inactive products that still carry projections.

    Ported from discontinued_with_projections.ipynb. In a "By customer group"
    view, only rows whose region matches the selected region are shown (e.g. an
    EU view won't list AAFES, a US customer); ALL CUSTOMERS shows every region.

    ``key_skus``/``key_suffix``/``show_header`` behave as in
    ``render_inactive_section``.
    """
    HEADER = "### Inactive/discontinued SKUs with forecasts"
    if show_header:
        st.markdown(HEADER)
    if not disc_check_ran:
        st.info(
            "Upload a Plytix export with a 'SKU Status' column (sidebar) to run "
            "the discontinued-product check."
        )
        return

    region_scoped = view != ALL_CUSTOMERS_VIEW and region is not None
    table_df = discontinued_df
    if region_scoped:
        table_df = discontinued_df[discontinued_df["Region"] == region]
    if key_skus is not None:
        table_df = table_df[
            table_df["SKU"].astype(str).str.rstrip("*").isin(key_skus)
        ]

    # Apply the non-zero future-projection filter BEFORE the empty check, so a
    # scope whose rows all zero out still shows the "None found" message rather
    # than an empty table (mirrors render_inactive_section above).
    disc = table_df.copy()
    disc["First_WeekDate"] = pd.to_datetime(disc["First_WeekDate"]).dt.date
    disc["Last_WeekDate"] = pd.to_datetime(disc["Last_WeekDate"]).dt.date
    disc["Original_Projection"] = pd.to_numeric(
        disc["Original_Projection"], errors="coerce"
    ).round(0)
    disc = disc[
        disc["Original_Projection"].notna() &
        (disc["Original_Projection"] != 0)
    ]

    if disc.empty:
        if region_scoped:
            st.success(
                f"None found for {region} — no discontinued or inactive "
                "products carry future projections here."
            )
        else:
            st.success(
                "None found — no discontinued or inactive products carry "
                "future projections."
            )
        return

    n_skus = disc["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct SKUs marked Discontinued or "
        "Inactive in Plytix that still carry future projections (future weeks "
        "only)."
    )

    disc = disc[[
        'SKU', 'SKU Status', 'Region', 'Region Code', 'Customer Grouping',
        'First_WeekDate', 'Last_WeekDate', 'Original_Projection',
    ]].rename(columns={
        "First_WeekDate": "First Projected Week",
        "Last_WeekDate": "Last Projected Week",
        "Original_Projection": "Original Projection (future avg/wk)",
    })

    render_filtered_table(disc, f"filter_discontinued{key_suffix}", style=False)
    st.download_button(
        "⬇️ Download the discontinued/inactive projections table",
        data=summary_to_excel(disc, sheet_name="discontinued_projections"),
        file_name=f"discontinued_with_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_discontinued_projections{key_suffix}",
    )


def render_missing_pos_section(view, region, missing_pos_df, today_str,
                               key_skus=None, key_suffix="", show_header=True):
    """Table of active SKUs (incl. Parts) missing POS/Orders where they're active.

    Ported from missing_pos.ipynb. Flags SKU x customer combos that have stopped
    receiving (or never received) POS/Orders data in a region the SKU is "Active
    in", over full history. In a "By customer group" view only rows whose region
    matches the selected region are shown; ALL CUSTOMERS shows every region.

    ``key_skus``/``key_suffix``/``show_header`` behave as in
    ``render_inactive_section``.
    """
    HEADER = "### SKUs missing POS/Orders data in locations they are active in"
    if show_header:
        st.markdown(HEADER)
    st.caption(
        "Note: this table only surfaces combos that sold within the past 3 months "
        "and have since gone silent. It deliberately excludes customer/SKU "
        "combinations that were never part of the assortment (no POS or Orders "
        "ever) and long-dead combinations that haven't sold in over 3 months."
    )
    if missing_pos_df is None:
        st.info(
            "Upload a Plytix export with an 'Active in' column (sidebar) to run "
            "the missing POS/Orders check."
        )
        return

    region_scoped = view != ALL_CUSTOMERS_VIEW and region is not None
    table_df = missing_pos_df
    if region_scoped:
        table_df = missing_pos_df[missing_pos_df["Region"] == region]
    if key_skus is not None:
        table_df = table_df[
            table_df["SKU"].astype(str).str.rstrip("*").isin(key_skus)
        ]

    if table_df.empty:
        if region_scoped:
            st.success(
                f"None found for {region} — every active SKU has recent "
                "POS/Orders data in its active channels here."
            )
        else:
            st.success(
                "None found — every active SKU has recent POS/Orders data in "
                "its active channels."
            )
        return

    n_skus = table_df["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct active SKUs (Parts included) "
        "that sold in a location they're active in within the past 3 months but have "
        "since stopped receiving POS/Orders data (trailing-3-month look-back)."
    )

    show = table_df.copy()
    show["First Missing Week"] = pd.to_datetime(show["First Missing Week"]).dt.date
    show["Last Missing Week"] = pd.to_datetime(show["Last Missing Week"]).dt.date

    render_filtered_table(show, f"filter_missing_pos{key_suffix}", style=False)
    st.download_button(
        "⬇️ Download the missing POS/Orders table",
        data=summary_to_excel(show, sheet_name="missing_pos_orders"),
        file_name=f"missing_pos_orders_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_missing_pos_orders{key_suffix}",
    )
