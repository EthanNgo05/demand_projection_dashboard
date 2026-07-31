"""The detail-card KPI tiles: one order, one treatment, nothing dropped.

Detail-card KPIs used to render three different ways — flat markdown in the card's
field grid, shaded ``st.metric`` tiles beside the projections chart, and a
hand-rolled coloured ``<span>`` in the Exceptions card — and each view ordered its
own card. None of it was covered by a test, so the labels and layout could drift
silently. These are the guards for the consolidated version:

  * every field a card asks for is known to ``KPI_ORDER`` (an unknown name sorts
    silently to the end, which is nearly always a typo);
  * the ordering itself is canonical and independent of the order a view lists in;
  * the Recent Trend's three interesting cases (growth, death, no baseline);
  * and — the load-bearing one — that no KPI a card used to show has gone missing,
    which is what the consolidation could most easily break.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("streamlit")

from dashboard_app.config import (  # noqa: E402
    ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, KPI_HELP, KPI_ORDER, KPI_TEXT_FIELDS,
    MODEL_USED_COL, ONHAND_COL, PRICE_COL, RISK_COL, TREND_COL, WOS_COL, kpi_sort,
)


def _all_card_col_lists():
    """Every shipped ``*_CARD_COLS`` list, by name."""
    import dashboard
    from dashboard_app import exceptions, kpis, watchlist_view

    return {
        "QUICK_CARD_COLS": dashboard.QUICK_CARD_COLS,
        "BEST_MIX_CARD_COLS": kpis.BEST_MIX_CARD_COLS,
        "EXCEPTION_CARD_COLS": exceptions.EXCEPTION_CARD_COLS,
        "SPIKE_CARD_COLS": exceptions.SPIKE_CARD_COLS,
        "WATCHLIST_CARD_COLS": watchlist_view.WATCHLIST_CARD_COLS,
    }


# --------------------------------------------------------------------------- #
# Ordering                                                                    #
# --------------------------------------------------------------------------- #
def test_every_card_field_is_known_to_kpi_order():
    """A card field missing from KPI_ORDER sorts to the end instead of erroring.

    That silent fallback is deliberate (a view may pass a bespoke column), which is
    exactly why the shipped lists need a test: a renamed or typo'd column would
    otherwise drift to the bottom of every card with no failure anywhere.
    """
    known = set(KPI_ORDER) | {"Note"}   # Note is peeled out and rendered full-width
    for name, cols in _all_card_col_lists().items():
        unknown = [c for c in cols if c not in known]
        assert not unknown, f"{name}: {unknown} not in KPI_ORDER (typo, or add them)"


def test_kpi_sort_is_canonical_not_caller_order():
    """Two views listing the same fields differently must produce ONE order.

    This is the whole point of the shared order: a planner moving between the
    Projections and Exceptions cards should find a field in the same place, rather
    than wherever that view happened to list it.
    """
    fields = [RISK_COL, "Customer Grouping", TREND_COL, PRICE_COL, ALL_TIME_AVG_COL]
    assert kpi_sort(fields) == kpi_sort(list(reversed(fields)))
    # And it really is the KPI_ORDER sequence, identity before money.
    assert kpi_sort(fields) == [
        "Customer Grouping", ALL_TIME_AVG_COL, TREND_COL, PRICE_COL, RISK_COL,
    ]


def test_kpi_sort_keeps_unknown_fields_last_and_stable():
    out = kpi_sort(["zzz-custom", RISK_COL, "aaa-custom", "Customer Grouping"])
    assert out[:2] == ["Customer Grouping", RISK_COL]
    # Unknowns keep the order they were given (sorted() is stable) — not alphabetical.
    assert out[2:] == ["zzz-custom", "aaa-custom"]


def test_identity_fields_are_text_not_stat_tiles():
    """Long identity values must not render as big tabular numbers.

    "Holt-Winters (triple) exponential smoothing" in a 1/4-width tile at the stat
    size is the case that forced the two tile kinds.
    """
    assert MODEL_USED_COL in KPI_TEXT_FIELDS
    assert "Customer Grouping" in KPI_TEXT_FIELDS
    assert "Data Source" in KPI_TEXT_FIELDS
    # Measurements are never text tiles — they need the tabular figures.
    for c in (ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL, RISK_COL, WOS_COL):
        assert c not in KPI_TEXT_FIELDS, f"{c} would lose tabular-nums alignment"


def test_measurement_fields_carry_help_text():
    """Every number a planner has to interpret gets a tooltip.

    Identity fields are self-explanatory; a bare "12.4" under "WOS Impact" is not.
    """
    for c in (ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL, PRICE_COL, RISK_COL,
              ONHAND_COL, WOS_COL, "Current Projection Average",
              "Updated Projection Average", "Projection Difference"):
        assert KPI_HELP.get(c), f"{c} has no tooltip"


# --------------------------------------------------------------------------- #
# Recent Trend                                                                #
# --------------------------------------------------------------------------- #
TREND_TODAY = pd.Timestamp("2026-07-30")   # last complete week = 2026-07-19


def _weeks(start, n):
    return pd.date_range(start, periods=n, freq="W-SUN")


def _trend_frame(cases):
    """Per-group SKU-week frame from {sku: [(start, n_weeks, pos_per_week), ...]}."""
    rows = []
    for sku, spans in cases.items():
        for start, n, pos in spans:
            for w in _weeks(start, n):
                rows.append({"Customer Grouping": "AMAZON-DC", "SKU": sku,
                             "WeekDate": w, "POS": pos, "Orders": np.nan,
                             "Projection": np.nan})
    return pd.DataFrame(rows)


def test_recent_trend_growth_decline_death_and_no_baseline():
    """The four cases, on adjacent 8-week windows.

    The prior window ends the week before the recent one begins, so the two never
    overlap — otherwise a change would be measured partly against itself.
    """
    from dashboard_app import compute

    # recent window: 2026-05-31 .. 2026-07-19 ; prior: 2026-04-05 .. 2026-05-24
    agg = _trend_frame({
        "UP":   [("2026-04-05", 8, 10), ("2026-05-31", 8, 20)],   # doubled
        "DOWN": [("2026-04-05", 8, 20), ("2026-05-31", 8, 10)],   # halved
        "DEAD": [("2026-04-05", 8, 30)],                          # stopped selling
        "NEW":  [("2026-05-31", 8, 50)],                          # no baseline
    })
    out = compute._descriptive_averages(agg, TREND_TODAY).set_index("SKU")

    assert out.loc["UP", TREND_COL] == 100.0
    assert out.loc["DOWN", TREND_COL] == -50.0
    # A SKU that WAS selling and has stopped is -100%, not blank. This is the single
    # most actionable trend value on the board, so it must not be swallowed by the
    # missing recent average (absent week = zero, as everywhere else in the app).
    assert out.loc["DEAD", TREND_COL] == -100.0
    # No prior sales is genuinely undefined, not +infinity — blank, rendered "New".
    assert pd.isna(out.loc["NEW", TREND_COL])


def test_recent_trend_is_not_published_as_a_third_average():
    """Only the trend %, never the prior-8-week average it is derived from.

    A third average column beside All-Time and 8-Week would recreate exactly the
    same-window/different-number confusion the All-Time rename removed.
    """
    from dashboard_app import compute

    agg = _trend_frame({"A": [("2026-04-05", 8, 10), ("2026-05-31", 8, 20)]})
    out = compute._descriptive_averages(agg, TREND_TODAY)
    assert compute._PRIOR_8WK_COL not in out.columns
    avg_cols = [c for c in out.columns if c.endswith("POS/Orders Average")]
    assert sorted(avg_cols) == sorted([ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL])


def test_trend_renders_new_rather_than_a_dash_when_there_is_no_baseline():
    """"—" reads as missing data; "New" says why it's blank."""
    from dashboard_app.tables import _tile_value

    assert _tile_value(TREND_COL, np.nan) == "New"
    assert _tile_value(TREND_COL, -100.0) == "-100.0%"
    # Signed and one decimal: the sign carries the direction without relying on colour.
    assert _tile_value(TREND_COL, 12.4) == "+12.4%"
    assert _tile_value(TREND_COL, 1234.5) == "+1,234.5%"   # thousands separator
    # Every other column keeps the shared em-dash for a blank.
    assert _tile_value(ONHAND_COL, np.nan) == "—"


# --------------------------------------------------------------------------- #
# Weeks of Supply                                                             #
# --------------------------------------------------------------------------- #
def test_attach_supply_columns_matches_the_spikes_definition():
    """One WOS definition: total On Hand ÷ total CURRENT projection across ALL
    customers. SKU-level, so identical on every one of a SKU's customer rows."""
    from dashboard_app.compute import attach_supply_columns

    summary = pd.DataFrame({
        "SKU": ["A-1", "A-1", "B-2"],
        "Customer Grouping": ["AMAZON-DC", "COSTCO", "AMAZON-DC"],
        "Current Projection Average": [60.0, 40.0, 0.0],
    })
    out = attach_supply_columns(summary, {"A-1": 500.0, "B-2": 90.0})

    # A-1: 500 / (60 + 40) = 5.0 weeks, the same on both of its rows.
    assert list(out[WOS_COL][:2]) == [5.0, 5.0]
    assert list(out[ONHAND_COL][:2]) == [500.0, 500.0]
    # B-2 has stock but no plan to divide by: cover is undefined, not infinite.
    assert pd.isna(out.loc[2, WOS_COL])
    assert out.loc[2, ONHAND_COL] == 90.0


def test_attach_supply_columns_leaves_the_frame_alone_without_a_map():
    """No On Hand data => no tiles, rather than tiles reading 0.

    "Unknown stock" and "no stock" lead to opposite decisions, so a zero-fill here
    would be actively misleading.
    """
    from dashboard_app.compute import attach_supply_columns

    summary = pd.DataFrame({"SKU": ["A-1"], "Customer Grouping": ["AMAZON-DC"],
                            "Current Projection Average": [10.0]})
    out = attach_supply_columns(summary, None)
    assert ONHAND_COL not in out.columns and WOS_COL not in out.columns


# --------------------------------------------------------------------------- #
# Nothing dropped                                                             #
# --------------------------------------------------------------------------- #
def test_projections_card_still_shows_every_kpi_it_used_to():
    """The consolidation moved 7 st.metric calls into the shared tile grid.

    Six were read straight off the row and must now appear in the card's field set;
    the seventh (Projected Revenue) is derived and comes from projection_kpi_extras.
    This is the direct guard on "don't lose a KPI" — the risk the consolidation
    carries. Data Source must appear ONCE: it used to be in both zones.
    """
    import dashboard
    from dashboard_app import kpis

    for name, cols in (("QUICK_CARD_COLS", dashboard.QUICK_CARD_COLS),
                       ("BEST_MIX_CARD_COLS", kpis.BEST_MIX_CARD_COLS)):
        for moved in ("Data Source", "Current Projection Average",
                      "Updated Projection Average", "Projection Difference",
                      PRICE_COL, RISK_COL):
            assert moved in cols, f"{name} lost {moved!r} in the consolidation"
        assert cols.count("Data Source") == 1, f"{name} lists Data Source twice"
        # And the additions.
        assert EIGHT_WK_AVG_COL in cols and ALL_TIME_AVG_COL in cols
        assert TREND_COL in cols
        assert ONHAND_COL in cols and WOS_COL in cols


def test_projected_revenue_and_difference_delta_are_derived_not_lost():
    from dashboard_app.kpis import (
        projection_difference_delta, projection_kpi_extras,
    )

    row = pd.Series({
        "SKU": "A-1", PRICE_COL: 12.5, "Updated Projection Average": 40.0,
        "Current Projection Average": 50.0, "Projection Difference": -10.0,
    })
    labels = [t[0] for t in projection_kpi_extras(row)]
    assert "Projected Revenue" in labels          # 12.5 * 40 = $500
    assert projection_kpi_extras(row)[0][1] == "$500"
    assert projection_difference_delta(row) == "-20.0%"

    # No list price -> no tile at all, matching how Revenue Risk degrades.
    assert projection_kpi_extras(row.drop(PRICE_COL)) == []
    # No base to be a percentage OF -> no delta, rather than a misleading 0.0%.
    assert projection_difference_delta(
        pd.Series({"Projection Difference": -10.0, "Current Projection Average": 0.0})
    ) is None


def test_exceptions_card_gained_the_columns_the_view_ranks_on():
    """The gap and % deviation are what the Exceptions view sorts by, and were
    somehow never on the card a planner opens to understand a flagged row."""
    from dashboard_app import exceptions as EX

    for c in (EX.GAP_COL, EX.PCT_COL, TREND_COL, ONHAND_COL, WOS_COL):
        assert c in EX.EXCEPTION_CARD_COLS
    # Still carries everything it did before.
    for c in ("Customer Grouping", "Region", "Data Source", EX.STATUS_COL,
              EX.RECENT_COL, EX.PROJ_COL, PRICE_COL, EX.WEEKS_COL, EX.FLAG_COL):
        assert c in EX.EXCEPTION_CARD_COLS


def test_spike_card_omits_the_trend_on_purpose():
    """Every spike row would read "New" — a column carrying no information."""
    from dashboard_app import exceptions as EX

    assert TREND_COL not in EX.SPIKE_CARD_COLS
    assert ONHAND_COL in EX.SPIKE_CARD_COLS      # but On Hand is useful here
    assert WOS_COL in EX.SPIKE_CARD_COLS
