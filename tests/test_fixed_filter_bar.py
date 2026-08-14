"""The fixed filter bar, and the SKU detail section it sits beneath.

Both are shared by Quick Projections and Optimized Projections, and both are driven
here through ``AppTest.from_function`` over a synthetic frame rather than through the
whole dashboard: the behaviour under test is arithmetic and widget wiring, neither of
which needs a real forecast, and Optimized in particular depends on agent-batch
outputs that a test environment need not have.

``AppTest.from_function`` ships only the function's OWN source to the script runner,
so each app function below builds its frames inline rather than calling a module-level
helper — a shared fixture would come back as a NameError inside the app.

The frame is deliberately at SKU x customer grain with a region split that makes the
cross-filter question concrete — AMAZON-DC sells only in US, so a Region = AU pick
must not leave it selectable. That is the exact case the bar exists to prevent.
"""

import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard_app.config import MIXED_SOURCE, PRICE_COL, RISK_COL  # noqa: E402
from dashboard_app.config import EIGHT_WK_AVG_COL  # noqa: E402

SKU_MS = "t::SKU__ms"
CUST_MS = "t::Customer__ms"
REGION_MS = "t::Region__ms"
KEY_ON = "t::Key SKU__on"
KEY_BTN = "t::Key SKU__btn"


def _frame():
    """One row per (SKU, customer group), the grain both main tables carry.

    US-only and AU-only groups, so "narrow by Region" has a wrong answer available.
    Kept at module level for the tests that call the pure helpers directly; the
    AppTest app functions rebuild it inline (see the module docstring).
    """
    df = pd.DataFrame(
        [
            ("SKU-A", "Widget A", "AMAZON-DC", "US"),
            ("SKU-A", "Widget A", "ACR", "AU"),
            ("SKU-B", "Widget B", "AMAZON-DC", "US"),
            ("SKU-C", "Widget C", "ACR", "AU"),
        ],
        columns=["SKU", "Description", "Customer Grouping", "Region"],
    )
    df["Data Source"] = ["POS", "Orders", "POS", "POS"]
    df["Weeks with data"] = 20
    df[EIGHT_WK_AVG_COL] = [10.0, 20.0, 30.0, 40.0]
    df["Current Projection Average"] = [8.0, 18.0, 28.0, 38.0]
    df["Updated Projection Average"] = [11.0, 21.0, 31.0, 41.0]
    df["Projection Difference"] = (
        df["Updated Projection Average"] - df["Current Projection Average"]
    )
    df[PRICE_COL] = 5.0
    df[RISK_COL] = df["Projection Difference"] * df[PRICE_COL]
    return df


# --------------------------------------------------------------------------- #
# The bar itself                                                              #
# --------------------------------------------------------------------------- #
def _bar_app():
    """A page that is nothing but the fixed bar over the synthetic frame."""
    import pandas as pd
    import streamlit as st

    from dashboard_app.config import PRICE_COL, RISK_COL, EIGHT_WK_AVG_COL
    from dashboard_app.tables import FIXED_FILTER_LABELS, filter_table

    df = pd.DataFrame(
        [
            ("SKU-A", "Widget A", "AMAZON-DC", "US"),
            ("SKU-A", "Widget A", "ACR", "AU"),
            ("SKU-B", "Widget B", "AMAZON-DC", "US"),
            ("SKU-C", "Widget C", "ACR", "AU"),
        ],
        columns=["SKU", "Description", "Customer Grouping", "Region"],
    )
    df["Data Source"] = ["POS", "Orders", "POS", "POS"]
    df["Weeks with data"] = 20
    df[EIGHT_WK_AVG_COL] = [10.0, 20.0, 30.0, 40.0]
    df["Current Projection Average"] = [8.0, 18.0, 28.0, 38.0]
    df["Updated Projection Average"] = [11.0, 21.0, 31.0, 41.0]
    df["Projection Difference"] = (
        df["Updated Projection Average"] - df["Current Projection Average"]
    )
    df[PRICE_COL] = 5.0
    df[RISK_COL] = df["Projection Difference"] * df[PRICE_COL]

    out = filter_table(df, "t", None, fixed=FIXED_FILTER_LABELS)
    st.session_state["_rows"] = sorted(
        f"{s}/{c}" for s, c in zip(out["SKU"], out["Customer Grouping"])
    )


def _run_bar(**state):
    at = AppTest.from_function(_bar_app, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_three_dropdowns_are_there_before_anything_is_clicked():
    """No "Add filter" step: the bar is the first thing on screen.

    That is the whole point of ``fixed`` — the add-filter picker made every filter a
    pick-then-configure discovery problem, and a planner who wants "this SKU at this
    customer" should not have to find the filters first.
    """
    at = _run_bar()
    assert [m.key for m in at.multiselect] == [SKU_MS, CUST_MS, REGION_MS], (
        "SKU / Customer / Region, in that reading order and nothing else"
    )
    assert not at.selectbox, "the ➕ Add filter picker must not render in fixed mode"
    assert all(m.value == [] for m in at.multiselect), "opens unfiltered"


def test_region_narrows_the_customer_dropdown_to_reachable_groups():
    """Region = AU must not leave AMAZON-DC (a US-only group) selectable.

    Offering a value that yields zero rows is the failure this bar exists to prevent:
    the dropdown would be advertising a combination the data cannot honour.
    ``available()`` runs each field's options through every OTHER field's mask.
    """
    at = _run_bar(**{REGION_MS: ["AU"]})
    assert at.multiselect(key=CUST_MS).options == ["ACR"]
    # And the SKU list narrows the same way — SKU-B is US-only.
    assert at.multiselect(key=SKU_MS).options == ["SKU-A — Widget A",
                                                  "SKU-C — Widget C"]
    assert at.session_state["_rows"] == ["SKU-A/ACR", "SKU-C/ACR"]


def test_clearing_region_brings_the_options_back():
    """Cross-filtering must be a view of the data, not a one-way narrowing."""
    assert _run_bar(**{REGION_MS: ["AU"]}).multiselect(key=CUST_MS).options == ["ACR"]
    assert _run_bar().multiselect(key=CUST_MS).options == ["ACR", "AMAZON-DC"]


def test_the_sku_dropdown_is_labelled_with_descriptions():
    """Typing a product name has to find the SKU; the stored value stays the raw SKU.

    The standalone dropdown this bar replaced carried the description, and losing it
    would have been a quiet regression for anyone who searches by name.
    """
    assert _run_bar().multiselect(key=SKU_MS).options[0] == "SKU-A — Widget A"
    at = _run_bar(**{SKU_MS: ["SKU-A"]})
    assert at.session_state[SKU_MS] == ["SKU-A"], "raw SKU stored, not the label"


def test_key_skus_is_a_button_that_holds_its_state(monkeypatch):
    """One click, not open-tick-dismiss — and the state key stays the shared one.

    ``key_only_active`` and anything saved under the old popover form both read
    ``{wkey}__on``, so the button has to write that and not a key of its own.

    ``AppTest.from_function`` runs the app in this same process, so patching the
    module-level ``current_key_skus`` is what ``key_sku_mask`` resolves — no key-SKU
    workbook has to exist for this to run.
    """
    monkeypatch.setattr("dashboard_app.keyskus.current_key_skus",
                        lambda: frozenset({"SKU-A"}))
    at = _run_bar()
    keybtn = [b for b in at.button if b.key == KEY_BTN]
    assert keybtn, f"no key-SKU button; got {[b.key for b in at.button]}"
    assert keybtn[0].label == "⭐ Key SKUs only"

    at = _run_bar(**{KEY_ON: True})
    assert at.session_state["_rows"] == ["SKU-A/ACR", "SKU-A/AMAZON-DC"]
    # ...and it cross-filters like any other field: only key SKUs stay selectable.
    assert at.multiselect(key=SKU_MS).options == ["SKU-A — Widget A"]


def test_a_frame_with_no_key_skus_gets_no_button(monkeypatch):
    """A button that can only ever empty the table is worse than no button.

    This is the one "would it narrow anything?" gate ``fixed`` keeps, and the reason
    it keeps it: unlike the dropdowns, a key-SKU toggle over a frame with no key SKUs
    has no honest state to be in.
    """
    monkeypatch.setattr("dashboard_app.keyskus.current_key_skus", frozenset)
    assert not [b for b in _run_bar().button if b.key == KEY_BTN]


def test_a_single_valued_field_still_renders():
    """A fixed control must not vanish because it happens not to narrow anything.

    The add-filter menu hides single-valued fields — there, a menu entry that does
    nothing is noise. The bar makes the opposite promise: these fields are always on
    screen, so on a one-region view Region renders with its single option rather than
    leaving a hole where the reader expects a control.
    """
    at = _run_bar(**{REGION_MS: ["AU"]})
    assert at.multiselect(key=CUST_MS).options == ["ACR"], "one option, still drawn"
    assert at.multiselect(key=REGION_MS) is not None


# --------------------------------------------------------------------------- #
# The SKU detail section (shared by both views)                               #
# --------------------------------------------------------------------------- #
def _detail_app():
    """The SKU detail section, fed the SKU x customer frame Optimized passes."""
    import pandas as pd

    from dashboard_app.config import PRICE_COL, RISK_COL, EIGHT_WK_AVG_COL
    from dashboard_app.kpis import render_sku_detail_section

    df = pd.DataFrame(
        [
            ("SKU-A", "Widget A", "AMAZON-DC", "US"),
            ("SKU-A", "Widget A", "ACR", "AU"),
            ("SKU-B", "Widget B", "AMAZON-DC", "US"),
            ("SKU-C", "Widget C", "ACR", "AU"),
        ],
        columns=["SKU", "Description", "Customer Grouping", "Region"],
    )
    df["Data Source"] = ["POS", "Orders", "POS", "POS"]
    df["Weeks with data"] = 20
    df[EIGHT_WK_AVG_COL] = [10.0, 20.0, 30.0, 40.0]
    df["Current Projection Average"] = [8.0, 18.0, 28.0, 38.0]
    df["Updated Projection Average"] = [11.0, 21.0, 31.0, 41.0]
    df["Projection Difference"] = (
        df["Updated Projection Average"] - df["Current Projection Average"]
    )
    df[PRICE_COL] = 5.0
    df[RISK_COL] = df["Projection Difference"] * df[PRICE_COL]

    hist = pd.date_range("2026-01-05", periods=12, freq="W-MON")
    fcst = pd.date_range("2026-04-06", periods=4, freq="W-MON")
    agg = pd.DataFrame([
        {"SKU": s, "WeekDate": w, "POS": 10.0, "Orders": None, "Projection": 9.0}
        for s in ("SKU-A", "SKU-B", "SKU-C") for w in hist
    ])
    weekly = pd.DataFrame([
        {"SKU": s, "WeekDate": w, "projected_pos": 11.0}
        for s in ("SKU-A", "SKU-B", "SKU-C") for w in fcst
    ])
    anchors = (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-30"),
               pd.Timestamp("2026-04-06"))
    render_sku_detail_section(df, agg, weekly, df, anchors, None, key="best")


def test_sku_detail_totals_the_skus_customer_rows():
    """The tiles are the SUM of that SKU's (SKU, customer) rows — the one grain.

    This is what "rolled up sums of projections made at SKU x customer level" has to
    mean numerically, and it is why Optimized passes ``combined`` for both the KPI
    frame and the breakdown frame instead of growing a per-SKU summary of its own:
    ``_render_kpis`` sums the rows, so the frame scoped to one SKU already IS the
    roll-up. A second path here would be a second number.
    """
    at = AppTest.from_function(_detail_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="best_sku").value == "SKU-A", "first SKU by default"

    tiles = {m.label: m.value for m in at.metric}
    # SKU-A sells at both AMAZON-DC and ACR: 11.0 + 21.0 updated, 8.0 + 18.0 current.
    assert tiles["Updated Forecast (avg/wk)"] == "32"
    assert tiles["Current Forecast (avg/wk)"] == "26"
    assert tiles["Projection Difference (avg/wk)"] == "+6"
    # 6 units x $5, summed over the SKU's two rows.
    assert tiles["Revenue Risk (avg/wk)"] == "+$30"
    assert "SKUs Forecasted" not in tiles, (
        "the count can only read 1 for a single SKU; the tile was dropped"
    )


def test_sku_detail_renders_its_chart_and_its_breakdown():
    """Chart plus donut, the shape Quick's section already had.

    The donut is the itemised form of the tiles beside it — same number, once summed
    and once broken out — so a section with only one of the two would leave the
    reader no way to see where a SKU's volume comes from.
    """
    at = AppTest.from_function(_detail_app, default_timeout=60).run()
    assert not at.exception, at.exception
    headings = [m.value for m in at.markdown]
    assert "### SKU detail" in headings
    assert "#### Customer group breakdown" in headings
    assert len(at.get("plotly_chart")) == 2, "the SKU chart and the share donut"


def test_sku_detail_reports_a_mixed_source_when_the_groups_disagree():
    """SKU-A is POS at one group and Orders at another.

    Quick never hits this — ``roll_up_summary`` has already collapsed its frame to
    ``MIXED_SOURCE`` — but Optimized passes the un-rolled SKU x customer frame, where
    picking ``.iloc[0]`` would have labelled the chart with whichever group sorted
    first. ``_sku_detail_source`` is the reduction that makes one function safe for
    both grains.
    """
    from dashboard_app.kpis import _sku_detail_source

    df = _frame()
    assert _sku_detail_source(df[df["SKU"] == "SKU-A"]) == MIXED_SOURCE
    assert _sku_detail_source(df[df["SKU"] == "SKU-B"]) == "POS"
    assert _sku_detail_source(df.drop(columns=["Data Source"])) == "POS"
