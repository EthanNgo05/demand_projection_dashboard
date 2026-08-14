"""Phase 5: Streamlit integration via the headless AppTest harness.

Two things must hold:
  1. Clicking "Run Agent Summary" runs the graph (on a background thread so the
     UI doesn't freeze) and the result is rendered in the UI.
  2. A plain rerun (changing any other widget) does NOT invoke the LLM — the
     agent is button-triggered only. This is the load-bearing test: it enforces
     the "never on rerun" rule rather than trusting it by inspection.

The graph is faked so no real LLM/backtest runs. The dashboard imports
build_graph lazily inside the button handler (`from agent.graph import
build_graph`), so patching `agent.graph.build_graph` is what the click resolves.

These run the full dashboard against the real raw_inputs workbook, so the
initial forecast compute can take a few seconds — hence the generous timeout.
"""

import inspect
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = os.path.join(REPO_ROOT, "src", "dashboard.py")
HAS_RAW = bool(
    __import__("glob").glob(
        os.path.join(REPO_ROOT, "raw_inputs", "demand_projections", "*.xlsx")
    )
)
needs_data = pytest.mark.skipif(
    not HAS_RAW, reason="no raw_inputs workbook to drive the full dashboard"
)


class _FakeGraph:
    """Stands in for the compiled LangGraph graph. Its stream() writes the same
    summary JSON publish would (so the dashboard's cached-render path shows it),
    records that it was called, and yields one {node: delta} update per node the
    way LangGraph's stream(stream_mode="updates") does, so the dashboard's
    progress bar advances."""

    def __init__(self, recorder):
        self._recorder = recorder

    def stream(self, state, config=None):
        self._recorder["called_with"] = state
        self._recorder["config"] = config
        import dashboard  # the module under test

        view = state["view"]
        path = dashboard._agent_summary_path(view)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "view": view,
                    "generated_at": "2026-07-08T12:00:00",
                    "best_model": "XGBoost",
                    "mase_by_model": {"XGBoost": 0.85},
                    "narrative": "Demand is flat.",
                    "anomalies": ["- SKU-1 spiked"],
                    "confidence_flag": False,
                    "errors": [],
                },
                f,
            )
        yield {"ingest": {}}
        yield {"run_all_models": {}}
        yield {"evaluate_models": {}}
        yield {"select_best_model": {"best_model": "XGBoost", "confidence_flag": False}}
        yield {"flag_anomalies": {"anomalies": ["- SKU-1 spiked"]}}
        yield {"summarize": {"narrative": "Demand is flat."}}
        yield {"publish": {"window_excluded_skus": []}}


@needs_data
def test_run_agent_button_triggers_graph(monkeypatch):
    recorder = {}
    monkeypatch.setattr(
        "agent.graph.build_graph", lambda: _FakeGraph(recorder)
    )

    # Clean any stale summary for the default (ALL CUSTOMERS) view first.
    import dashboard

    default_path = dashboard._agent_summary_path(dashboard.ALL_CUSTOMERS_VIEW)
    if os.path.exists(default_path):
        os.remove(default_path)

    try:
        # 300s: on a cold cache the real workbook's load + exclusion checks
        # alone take ~60s and the first ALL CUSTOMERS compute (incl. the
        # per-customer breakdown) another ~60s+ (outgrew the old 60s budget,
        # 2026-07-14). The timeout is a cap, not a wait — a passing run
        # returns as soon as the script finishes.
        at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
        assert not at.exception
        # The click starts the pipeline on a background thread (non-blocking)
        # and reruns; wait for that thread before asserting on the summary.
        at.button(key="run_agent_summary").click().run()
        assert "agent_job_thread" in at.session_state, (
            "background agent thread was not started"
        )
        thread = at.session_state["agent_job_thread"]
        thread.join(timeout=30)
        assert not thread.is_alive(), "agent thread did not finish in time"
        # Re-run so the finished job is finalized and its summary rendered.
        at.run()
        assert not at.exception

        assert recorder.get("called_with"), "graph.stream was never called"
        assert recorder["called_with"]["view"] == dashboard.ALL_CUSTOMERS_VIEW
        # The rendered summary reports the best model from the written JSON.
        assert "XGBoost" in " ".join(m.value for m in at.success)
    finally:
        if os.path.exists(default_path):
            os.remove(default_path)


@needs_data
def test_smoothing_params_survive_model_round_trip():
    """Selecting Holt autofits tuned α/β/φ; switching model away and back must
    re-establish them, not silently fall back to the file defaults.

    Regression test for the autofit_tried / autofit_params desync: switching
    model dropped ``autofit_params`` but left the ``autofit_tried`` marker set,
    so returning to the smoothing model saw "already tried", skipped the
    backtest, and computed the forecast with file-default α/β/φ instead of the
    tuned ones — changing the displayed forecast for an unchanged view/snapshot.

    ``autofit_params`` is now a per-(model, view, snapshot) map — its keys are
    ``(pipeline_path, view, today_str)`` tuples — so "has Holt params" means some
    key carries the Holt pipeline path.
    """
    import dashboard

    HOLT = "Holt's (double) exponential smoothing"
    OTHER = "8-Week Moving Average"
    assert HOLT in dashboard.MODEL_OPTIONS and OTHER in dashboard.MODEL_OPTIONS

    def _has_holt_params(state):
        params = state["autofit_params"] if "autofit_params" in state else {}
        return any(key[0] == dashboard.MODEL_OPTIONS[HOLT] for key in params)

    at = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    assert not at.exception

    # Select Holt -> autofit runs and stores tuned params for this view.
    # The model widget is now a top-of-page dropdown (selectbox), not a radio.
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception
    assert _has_holt_params(at.session_state), (
        "autofit did not run on first Holt selection"
    )

    # Leave to another model, then come back to Holt.
    at.selectbox(key="model_choice").set_value(OTHER).run()
    assert not at.exception
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception

    # Returning to Holt must re-establish autofit params for this view; otherwise
    # the forecast is silently computed with file defaults -> a different number.
    assert _has_holt_params(at.session_state), (
        "autofit params lost after a model round-trip: the forecast falls back "
        "to file-default alpha/beta/phi and changes for an unchanged view"
    )


@needs_data
def test_exceptions_view_renders():
    """Selecting the Exceptions scope renders its own (model-agnostic) view
    without error — the routing branch and render_exceptions wiring both work.

    Key SKUs used to be a second tab; they are now an attribute (a blue "Key" chip
    plus a per-table filter), so the view is a single page. Pins that: no tabs, and
    exactly one of each control that used to be duplicated per tab.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    # "scope" is now a top-of-page segmented control, which AppTest has no direct
    # accessor for; drive it through session state (the key is unchanged).
    at.session_state["scope"] = dashboard.EXCEPTIONS_VIEW
    at.run()
    assert not at.exception
    assert any("Exceptions" == s.value for s in at.subheader)
    assert not at.tabs, "the Key SKUs / All Exceptions tabs should be gone"
    # One Group-by control and one spikes threshold, not one per (former) tab.
    assert {ni.label for ni in at.number_input} >= {"Min % deviation", "Min revenue risk / wk"}
    assert sum(ni.label == "Minimum container impact" for ni in at.number_input) == 1


@needs_data
def test_historical_summary_view_renders():
    """Selecting the Historical Summary scope renders its forecast-free view.

    Pins the routing branch, the render_historical_summary wiring, the filter bar
    and the four chart tabs. The tiles are plain st.metric calls (deliberately not
    tables._render_kpi_tiles), so this also catches a duplicate-container-key
    regression in the three stacked KPI rows.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    tab_labels = {t.label for t in at.tabs}
    assert {"Trend & Seasonality", "Mix & Breakdown", "Movers & Concentration",
            "Seasonal Heatmap"} <= tab_labels

    metric_labels = {m.label for m in at.metric}
    assert {"Revenue", "Units", "Revenue / Week", "Active SKUs",
            "Top-10 Revenue Share"} <= metric_labels
    # No tile may name a period of its own: every one is measured over the analysis
    # window, whose dates are stated once above the grid. Labels like "YTD Revenue"
    # or "Trailing 52-Wk Revenue" are exactly what made the selector look broken.
    assert not [lbl for lbl in metric_labels
                if "YTD" in lbl or "52" in lbl or "13" in lbl], metric_labels

    # The filter bar and the window selector both render. There is deliberately no
    # SKU-type filter — planners never slice by it — but the SKU Type COLUMN must
    # survive, because the Mix & Breakdown donut reads it.
    assert "hist_window" in at.session_state
    filter_labels = {ms.label for ms in at.multiselect}
    assert {"Region", "Customer group"} <= filter_labels
    assert "SKU type" not in filter_labels
    assert "SKU Type" in at.session_state["hist_base"].columns


@needs_data
def test_historical_summary_tile_grid_is_two_by_four():
    """Eight tiles in two sections of four, each with a click target.

    The uniform grid is the fix for the tiles reading as a scattered list, so its
    shape is pinned: a stray 9th tile or a missing button would reintroduce the
    ragged row this replaced. It was three rows of four while seven tiles carried
    fixed spans of their own; those spans are window OPTIONS now, so the tiles that
    existed only to hold them are gone.
    """
    import dashboard
    from dashboard_app import historical_summary as hs

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    assert [len(ids) for _, ids in hs._SECTIONS] == [4, 4]
    tile_ids = [t for _, ids in hs._SECTIONS for t in ids]
    assert len(tile_ids) == len(set(tile_ids)) == 8
    # Every tile in the grid must have a spec, and every spec must be on the grid —
    # otherwise a tile opens the wrong breakdown or a KPI silently disappears.
    assert set(tile_ids) == set(hs._TILES)

    rendered = {m.label for m in at.metric}
    for tile_id in tile_ids:
        label = hs._tile_label(tile_id, at.session_state["hist_window"])
        assert label in rendered, f"{tile_id} tile did not render"

    button_keys = {b.key for b in at.button}
    for tile_id in tile_ids:
        assert f"histkpi-go-{tile_id}" in button_keys, f"{tile_id} has no click target"

    # Section captions anchor the grid visually.
    captions = " ".join(c.value for c in at.caption)
    assert all(name in captions for name, _ in hs._SECTIONS)


@needs_data
@pytest.mark.parametrize("tile_id", ["dormant_skus", "top10_share", "revenue",
                                     "active_customers", "revenue_per_week"])
def test_clicking_a_tile_opens_its_breakdown(tile_id):
    """Clicking a tile must open a modal with a table, not raise.

    dormant_skus and top10_share are the two the request named explicitly;
    revenue_per_week is the one whose modal is a week-by-week table.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    n_tables_before = len(at.dataframe)
    at.button(key=f"histkpi-go-{tile_id}").click().run()
    assert not at.exception, f"clicking {tile_id} raised"
    assert len(at.dataframe) > n_tables_before, \
        f"{tile_id} modal rendered no breakdown table"


@needs_data
def test_historical_summary_shows_the_windows_actual_dates():
    """The window's real dates must appear above the tiles, and match m['bounds'].

    A named window ("Last 13 weeks") never says WHICH weeks, so this line is the
    only place a planner can see the period a number covers.
    """
    import pandas as pd
    import dashboard
    from dashboard_app import historical_metrics as hm

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    P = dashboard.load_pipeline(dashboard.pipeline_path())
    _, lcw, _ = P.week_anchors(pd.Timestamp(at.session_state["hist_base_sig"][0]))
    start, end = hm.window_bounds(at.session_state["hist_window"], lcw)

    body = " ".join(m.value for m in at.markdown)
    assert "Analysis window" in body, "the window's dates are not shown"
    # Both endpoints, formatted as the view formats them.
    assert f"{end:%b %d, %Y}" in body, f"end date {end:%b %d, %Y} missing"
    assert f"{start:%b %d}" in body, f"start date {start:%b %d} missing"

    # The tiles' deltas are percentages, and a percentage whose base is unstated is
    # not a fact -- so both comparison periods are named with their real dates too.
    captions = " ".join(c.value for c in at.caption)
    ly_start, ly_end = hm.prior_year_window(
        start, end,
        anchor_to_year_start=(at.session_state["hist_window"] == hm.WINDOW_YTD))
    prior_start, prior_end = hm.prior_period_window(start, end)
    assert "vs the period before" in captions
    assert f"{prior_start:%b %d, %Y}" in captions or \
           f"{prior_start:%b %d}" in captions, "prior period's dates missing"
    assert "vs the same weeks last year" in captions
    assert f"{ly_end:%b %d, %Y}" in captions, "prior year's dates missing"


@needs_data
def test_changing_the_analysis_window_changes_every_tile():
    """The reported complaint, end to end on live data.

    Seven of the twelve original tiles were pinned to fixed spans, so selecting
    "Last 4 weeks" left most of the grid showing trailing-52-week figures. Every
    tile now has to move -- except the two assortment COUNTS that can legitimately
    coincide (a customer list rarely changes between windows), which are checked
    only for not raising.
    """
    import dashboard
    from dashboard_app import historical_metrics as hm

    def _values(window):
        at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
        at.session_state["scope"] = dashboard.HISTORICAL_VIEW
        at.session_state["hist_window"] = window
        at.run()
        assert not at.exception, f"{window} raised"
        return {m.label: m.value for m in at.metric}

    short = _values(hm.WINDOW_4W)
    long = _values(hm.WINDOW_52W)
    for label in ("Revenue", "Units", "Revenue / Week"):
        assert short[label] != long[label], (
            f"{label!r} reads {short[label]} over 4 weeks and over 52 — it is not "
            f"measured over the analysis window"
        )


@needs_data
def test_all_history_window_reports_the_true_earliest_week():
    """'All history' must follow the data, not a fixed lookback."""
    import pandas as pd
    import dashboard
    from dashboard_app import historical_metrics as hm

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.session_state["hist_window"] = hm.WINDOW_ALL
    at.run()
    assert not at.exception

    base = at.session_state["hist_base"]
    P = dashboard.load_pipeline(dashboard.pipeline_path())
    _, lcw, _ = P.week_anchors(pd.Timestamp(at.session_state["hist_base_sig"][0]))
    earliest = hm.all_history_bounds(base, lcw)[0]

    body = " ".join(m.value for m in at.markdown)
    assert f"{earliest:%b %d, %Y}" in body, (
        f"expected the snapshot's true floor {earliest:%b %d, %Y} in the window line"
    )


@needs_data
def test_mix_tab_no_longer_charts_sku_type():
    """The SKU-type donut is gone; the SKU Type column is deliberately retained."""
    import dashboard
    from dashboard_app import historical_summary as hs

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    # Strip comments and the docstring: both legitimately DISCUSS SKU Type (they
    # record that the column is retained on purpose), so only executable lines can
    # be checked for a chart of it.
    body = inspect.getsource(hs._tab_mix).split('"""')[-1]
    code = "\n".join(line.split("#")[0] for line in body.splitlines())
    assert "SKU Type" not in code, "_tab_mix still charts SKU Type"
    assert code.count("share_donut") == 1, "expected exactly one donut (Region)"

    # The column stays available for future use even though nothing plots it.
    assert "SKU Type" in at.session_state["hist_base"].columns


@needs_data
def test_chart_tabs_pick_years_and_default_sensibly():
    """Trend opens on every year; Mix and Movers open on the latest one.

    The tabs used to carry free date-range pickers. What a planner compares here is
    calendar years, so Trend takes a multiselect (years overlay on one Jan-Dec axis)
    and Mix/Movers take one year at a time (a donut or a Pareto over several years
    merges them rather than comparing them).
    """
    import dashboard
    from dashboard_app import historical_summary as hs

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    years = at.session_state["hist_trend_years"]
    assert years, "the trend tab should open with every year selected"
    assert sorted(years) == list(years), "years should arrive oldest first"

    latest = max(years)
    assert at.session_state["hist_mix_year"] == latest
    assert at.session_state["hist_movers_year"] == latest

    # The retired date-range keys must be gone, not merely unused.
    for stale in ["hist_trend_preset", "hist_mix_preset", "hist_movers_preset",
                  "hist_heatmap_preset", "hist_heatmap_year"]:
        assert stale not in at.session_state, f"{stale} survived the rewrite"

    # Every year selection renders: a single year, all years, and none at all.
    for value in [hs.ALL_YEARS, latest, min(years)]:
        at.session_state["hist_mix_year"] = value
        at.session_state["hist_movers_year"] = value
        at.run()
        assert not at.exception, f"year selection {value!r} broke the view"

    at.session_state["hist_trend_years"] = []
    at.run()
    assert not at.exception, "deselecting every year must render an empty panel"


@needs_data
def test_year_pickers_do_not_move_the_kpi_tiles():
    """The decoupling: a tab's year selection must leave every tile untouched."""
    import dashboard
    from dashboard_app import historical_summary as hs

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception
    before = {m.label: m.value for m in at.metric}
    assert before, "no tiles rendered"

    # The widest and the narrowest chart selections available — if the tiles
    # followed a chart's period at all, one of these would show it.
    at.session_state["hist_mix_year"] = hs.ALL_YEARS
    at.session_state["hist_movers_year"] = min(at.session_state["hist_trend_years"])
    at.session_state["hist_trend_years"] = []
    at.run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric} == before, \
        "a chart's year selection moved the KPI tiles — they must follow only the " \
        "analysis window"


@needs_data
@pytest.mark.parametrize("window", ["Last 4 weeks", "Last 26 weeks",
                                    "Last full calendar year", "All history"])
def test_historical_summary_renders_every_new_window(window):
    """Each added window must render — a 4-week span is the likeliest to be sparse."""
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.session_state["hist_window"] = window
    at.run()
    assert not at.exception
    assert {m.label for m in at.metric}, "no tiles rendered"


@needs_data
def test_custom_window_renders_and_reports_the_snapped_range():
    """A custom range must render and caption the weeks it ACTUALLY covers."""
    import pandas as pd
    import dashboard
    from dashboard_app import historical_metrics as hm

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.session_state["hist_window"] = hm.WINDOW_CUSTOM
    at.run()
    assert not at.exception

    P = dashboard.load_pipeline(dashboard.pipeline_path())
    _, lcw, _ = P.week_anchors(pd.Timestamp(at.session_state["hist_base_sig"][0]))
    # Mid-week edges, so snapping is actually exercised.
    at.session_state["hist_custom_range"] = (
        (lcw - pd.Timedelta(weeks=8, days=3)).date(),
        (lcw + pd.Timedelta(days=2)).date(),
    )
    at.run()
    assert not at.exception
    # The snapped range is stated in the window line ABOVE the tiles (st.markdown),
    # not in the caption below them — you need the period before reading the numbers.
    body = " ".join(m.value for m in at.markdown)
    assert "snapped to whole weeks" in body, \
        "the window line must state the range actually measured, not what was typed"
    assert "Analysis window" in body


@needs_data
def test_breakdown_row_counts_match_their_tiles_on_real_data():
    """The tile number and its modal's row count must agree on the live snapshot.

    Unit tests pin this on a synthetic frame; this catches a real-data-only
    divergence (odd SKU codes, all-NaN rows, discontinued duplicates).
    """
    import pandas as pd
    import dashboard
    from dashboard_app import historical_metrics as hm
    from dashboard_app import historical_summary as hs

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    at.session_state["scope"] = dashboard.HISTORICAL_VIEW
    at.run()
    assert not at.exception

    base = at.session_state["hist_base"]
    # Derive lcw the way the app does rather than re-deriving the week maths here —
    # the snapshot date is the first element of the base-frame signature.
    P = dashboard.load_pipeline(dashboard.pipeline_path())
    _, lcw, _ = P.week_anchors(pd.Timestamp(at.session_state["hist_base_sig"][0]))
    start, end = hm.window_bounds(hm.WINDOW_YTD, lcw)

    counts = hm.breadth(base, start, end)
    for key, fn in [("active_skus", hm.active_skus_breakdown),
                    ("active_customers", hm.active_customers_breakdown),
                    ("new_skus", hm.new_skus_breakdown),
                    ("dormant_skus", hm.dormant_skus_breakdown)]:
        assert counts[key] == len(fn(base, start, end)), (
            f"{key} tile would show {counts[key]} above a list of "
            f"{len(fn(base, start, end))} rows"
        )
    assert set(hs._TILES) >= set(counts)


@needs_data
def test_historical_summary_counts_discontinued_skus():
    """The view must read the pre-discontinued-drop frame.

    This is the whole reason ExclusionResult grew a df_with_discontinued field: the
    exclusion step deletes a retired SKU's ENTIRE history, so a Historical Summary
    built from the forecast frame would understate every prior year and flatter
    year-over-year growth. Asserted at the data layer rather than through the UI so
    the failure message points at the cause.
    """
    import dashboard
    from agent import data_io
    from dashboard_app import historical_metrics as hm

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    excl = at.session_state["_excl_result"]
    if not excl.n_disc_rows:
        pytest.skip("this snapshot has no discontinued SKUs to distinguish")

    assert len(excl.df_with_discontinued) > len(excl.df)
    P = dashboard.load_pipeline(dashboard.pipeline_path())
    kept = set(hm.historical_weekly_frame(excl.df_with_discontinued, P)["SKU"])
    forecast = set(hm.historical_weekly_frame(excl.df, P)["SKU"])
    assert kept - forecast, "discontinued SKUs must survive into the historical frame"
    assert not any(str(s).endswith("*") for s in kept), "'*' must be normalised away"
    assert data_io  # the field is defined on data_io.ExclusionResult


def _headings(at):
    """The view body's "### ..." section headings, as one searchable string."""
    return " ".join(m.value for m in at.markdown if m.value.startswith("###"))


@needs_data
def test_quick_default_view_is_all_customers_in_the_new_order():
    """The default Quick Projections landing view, and its section order.

    Region defaults to the ALL_REGIONS sentinel and Customer group to the raw
    ALL_CUSTOMERS_VIEW string (raw, not the prettified label — several callers,
    including the agent-run button, read the selected value as a view ID). The
    body then reads KPIs -> total demand -> Customer detail -> SKU detail -> the
    by-SKU view-total table -> the by-SKU-and-customer table.

    The view-total table sits directly under SKU detail because it is the table form
    of what that section just charted, and is the ONLY one of the two in a (collapsed)
    expander — so it is titled by its label rather than an "###" heading and does not
    show up in _headings. The by-customer table below it is the page's main content
    and always renders, heading and all.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    assert at.session_state["quick_region"] == dashboard.ALL_REGIONS
    assert at.selectbox(key="quick_group").value == dashboard.ALL_CUSTOMERS_VIEW

    headings = _headings(at)
    assert "### Customer detail" in headings
    # The two drill-downs, one per axis: SKU detail is the view TOTAL for one SKU
    # across every customer group (the table's row detail cards remain the
    # per-(SKU, customer group) view), so it is gated to the aggregate scopes.
    assert "### SKU detail" in headings
    assert "### Summary table by SKU and customer" in headings
    # Both drill-downs come before the table, customer first.
    assert headings.index("### Customer detail") < \
        headings.index("### SKU detail") < \
        headings.index("### Summary table by SKU and customer")

    expanders = list(at.expander)
    labels = [e.label or "" for e in expanders]
    hits = [e for e, lbl in zip(expanders, labels)
            if "Summary table by SKU (view total)" in lbl]
    assert len(hits) == 1, labels
    # AppTest's Expander wrapper surfaces only label/icon; the open/closed flag
    # lives on the underlying Expandable proto.
    assert not hits[0].proto.expanded, "the view-total table must stay collapsed"


# The fixed filter bar's SKU dropdown, as Streamlit keys it: tables._ms_key over
# "<table key>::SKU". Driving the widget through session_state (rather than
# .set_value) is how the rest of this file reaches inside the table's fragment.
BY_CUST_SKU_FILTER = "filter_by_customer::SKU__ms"


def _undecorate(cell):
    """The raw SKU behind a rendered cell. ``mark_starred_sku`` prefixes ``★ `` to
    watchlist rows and ``mark_key_sku`` turns key SKUs into a ``[sku, "Key"]`` chip
    list; neither reaches the frame the filters run on, so a test that feeds a
    rendered cell back into the SKU filter has to strip both first."""
    if not isinstance(cell, str) and hasattr(cell, "__getitem__") and len(cell):
        cell = cell[0]
    return str(cell).lstrip("★ ")


@needs_data
def test_by_customer_sku_filter_narrows_the_table_but_not_the_download():
    """Filtering to a SKU drills the on-screen table; the Excel export stays whole.

    The two must not track each other. A planner who drilled to one SKU and then
    hit download expects the same workbook they'd have got without touching the
    filter — every SKU x customer combination in the view — because once the file
    is off the page a silently narrowed export is indistinguishable from a
    complete one. So `data=` reads `by_cust_table` (the full, sorted frame) while
    only `render_selectable_table` sees the filtered one.

    AppTest does not surface download buttons as elements, so the export side is
    checked at the source (as test_sku_detail_... does for its donut) and the
    display side through the rendered table.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    sku_filter = at.multiselect(key=BY_CUST_SKU_FILTER)
    assert sku_filter.value == [], "must open showing every row"
    if len(sku_filter.options) < 2:
        pytest.skip("no SKUs in the by-customer table for this snapshot")

    def _by_cust_rows(app):
        """SKU values of the by-customer table, undecorated.

        Identified by its CONDENSED column set: the view-total table above it also
        carries SKU and Customer Grouping columns, but renders the full frame, so
        "has both columns" would match that one first.
        """
        condensed = set(dashboard.QUICK_CONDENSED_COLS)
        for df in app.dataframe:
            data = getattr(df.value, "data", df.value)
            if "SKU" in data.columns and set(data.columns) <= condensed:
                return [_undecorate(s) for s in data["SKU"]]
        return None

    before = _by_cust_rows(at)
    assert before, "did not find the by-customer table among the rendered frames"

    # The multiselect's .options are FORMATTED labels ("<SKU> — <description>") while
    # the stored value is the raw SKU, so pick a raw SKU off the rendered rows and
    # write it to the widget's own session-state key. That is legal here for the same
    # reason filter_table's clamp loop does it: the write lands before the widget
    # instantiates on the next run.
    target = before[0]
    at.session_state[BY_CUST_SKU_FILTER] = [target]
    at.run()
    assert not at.exception, at.exception

    after = _by_cust_rows(at)
    if after is None:
        # The picked SKU left exactly ONE row, so the derived focus dropped the table
        # and opened that row's card outright — see the focus test below. Nothing to
        # assert about narrowing here; the export assertion still applies.
        assert any(m.label == "Customer Grouping" for m in at.metric)
    else:
        assert set(after) == {target}, f"table still shows {sorted(set(after))}"
        assert len(after) < len(before) or len(set(before)) == 1

    body = inspect.getsource(dashboard.main)
    assert "with_export_flags(by_cust_table)" in body, (
        "the by-customer download must export the FULL frame, not the filtered one"
    )


def _condensed_by_cust_table(app):
    """The by-customer table's rendered frame, or None when it is not on the page.

    Identified by its CONDENSED column set, the same way
    ``test_by_customer_sku_picker_narrows_the_table_but_not_the_download`` does: the
    view-total table above it also carries SKU and Customer Grouping, but renders the
    full frame, so "has both columns" would match that one first.
    """
    import dashboard

    condensed = set(dashboard.QUICK_CONDENSED_COLS)
    for df in app.dataframe:
        data = getattr(df.value, "data", df.value)
        if "SKU" in data.columns and set(data.columns) <= condensed:
            return data
    return None


def _single_group_view(at):
    """Drive the app to a single customer group, or skip. Returns the AppTest.

    A single-group view has exactly one Customer Grouping, so each of its SKUs has
    exactly ONE row in the by-customer table — which is the situation focus_single
    exists for. Mirrors the region/group walk in
    ``test_quick_single_group_renders_the_by_customer_table``: which groups carry
    demand depends on the snapshot, so both loops are needed.
    """
    import dashboard

    for region in [r for r in at.selectbox(key="quick_region").options
                   if r != dashboard.ALL_REGIONS]:
        at.selectbox(key="quick_region").set_value(region).run()
        assert not at.exception
        for i in range(1, len(at.selectbox(key="quick_group").options)):
            at.selectbox(key="quick_group").select_index(i).run()
            assert not at.exception
            if not at.error and len(at.multiselect(key=BY_CUST_SKU_FILTER).options) > 1:
                return at
    pytest.skip("no individual customer group has demand in this snapshot")


@needs_data
def test_one_row_left_opens_its_card_and_drops_the_table():
    """A SKU that leaves one row needs no click: the card opens, the table goes.

    The table earns its place by being a CHOICE. Narrowed to a single row it is not
    one — it is a click standing between the reader and something they have already
    asked for, and its five condensed columns are all repeated by the card's tiles.
    So on a single-group view (one Customer Grouping, therefore one row per SKU)
    picking a SKU must leave the card and nothing else.

    The signal is DERIVED now (``tables.sku_filter_narrowed`` — exactly one SKU picked
    in the fixed bar) rather than passed in as ``focus_single`` by a caller-side
    dropdown, so this pins the behaviour to the filter that replaced that dropdown.

    The ✕ goes with the table: it deselects a table row, and closing the card would
    otherwise strand the reader with no way to re-open it.
    """
    at = _single_group_view(
        AppTest.from_file(DASHBOARD, default_timeout=300).run()
    )
    table = _condensed_by_cust_table(at)
    assert table is not None, "table must show when nothing is filtered"
    before = [m.label for m in at.metric]

    at.session_state[BY_CUST_SKU_FILTER] = [_undecorate(table["SKU"].iloc[0])]
    at.run()
    assert not at.exception, at.exception

    assert _condensed_by_cust_table(at) is None, (
        "the one-row table is redundant with the card and must not render"
    )
    added = [m.label for m in at.metric if m.label not in before]
    assert "Customer Grouping" in added, f"the card did not open; got {added}"
    assert not [b for b in at.button if b.label == "✕"], (
        "no table to deselect from, so the card must not offer a ✕"
    )


@needs_data
def test_changing_a_filter_closes_the_open_detail_cards():
    """A filter change must drop the table's open cards, not re-point them.

    ``render_selectable_table`` keys its selection by POSITION, so row 0 of the
    unfiltered frame and row 0 of a filtered one are different (SKU, customer) pairs.
    Leaving the selection alone would silently swap an open card's subject — a card
    labelled with one customer group showing another's numbers. Every control on the
    fixed bar therefore clears ``{key}__sel`` (``tables._clear_row_selection``); this
    used to be one dropdown's bespoke callback.
    """
    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    table = _condensed_by_cust_table(at)
    if table is None or table.empty:
        pytest.skip("no rows in the by-customer table for this snapshot")

    at.session_state["filter_by_customer__sel"] = {"selection": {"rows": [0]}}
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Customer Grouping" for m in at.metric), "card did not open"

    # Streamlit fires on_change only when the widget's value actually changes, so
    # this drives the callback rather than writing the key straight through.
    at.multiselect(key=BY_CUST_SKU_FILTER).set_value(
        [_undecorate(table["SKU"].iloc[0])]
    ).run()
    assert not at.exception, at.exception
    assert at.session_state["filter_by_customer__sel"]["selection"]["rows"] == [], (
        "the filter change must clear the positional row selection"
    )


@needs_data
def test_sku_detail_is_a_chart_a_kpi_stack_and_a_share_donut():
    """What the section carries after the three things that didn't earn their space.

    The section is scoped to ONE SKU, which is what makes each removal safe:
    "SKUs Forecasted" could only ever read 1; the weekly-forecast table restated
    the chart beside it row by row; and the customer-group breakdown asked a share
    question a table made the reader compute. The breakdown's columns are not gone
    — they moved into the donut's per-slice hover (see test_sku_breakdown_pie.py).
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception

    headings = _headings(at)
    assert "### SKU detail" in headings
    assert "#### Customer group breakdown" in headings
    assert "#### Weekly forecast" not in headings
    # AppTest does not surface download buttons as elements, so the export that sat
    # under the table is checked at the source (as test_mix_tab does for its donut).
    # The section itself lives in kpis.render_sku_detail_section — shared with
    # Optimized Projections — so that, not dashboard.main, is where it is read from.
    assert "dl_sku_weekly" not in inspect.getsource(dashboard.main), (
        "the weekly-forecast export went with its table"
    )
    assert "customer_share_donut" in inspect.getsource(
        dashboard.render_sku_detail_section
    )
    # Twice, not three times: the page-top row and Customer detail (one group, many
    # SKUs) both still count SKUs. SKU detail no longer does.
    assert sum(m.label == "SKUs Forecasted" for m in at.metric) == 2
    # The other stacked tiles stay — only the count was dropped.
    assert any(m.label == "Updated Forecast (avg/wk)" for m in at.metric)


@needs_data
def test_quick_region_change_resets_the_customer_group():
    """Switching Region must re-option the Customer-group selectbox cleanly.

    Both are keyed, which keeps the options list out of the widget's element
    identity — so the stored group survives the switch and Streamlit is the one
    that has to notice it is no longer offered and fall back to the first option
    (that region's rollup), writing the corrected value back in the same run. If a
    Streamlit upgrade changes that behaviour, this fails loudly rather than the
    page throwing "value not in options" for a planner.

    Only the two selectboxes matter here — they render into region_slot before the
    body runs, so a region whose rollup has no recent demand (and therefore stops
    the body with a "nothing to forecast" error) is still a valid probe.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    regions = [r for r in at.selectbox(key="quick_region").options
               if r != dashboard.ALL_REGIONS]
    # A region with at least one individual group, so there is something to hold
    # across the switch.
    for region in regions:
        at.selectbox(key="quick_region").set_value(region).run()
        assert not at.exception
        # .options are FORMATTED labels; .value is the raw view string.
        assert at.selectbox(key="quick_group").options[0] == \
            dashboard.quick_group_label(dashboard.region_all_view(region))
        if len(at.selectbox(key="quick_group").options) > 1:
            break
    else:
        pytest.skip("no region has an individual customer group")

    # Hold a specific group (index >= 1, where the label IS the raw group name).
    at.selectbox(key="quick_group").select_index(1).run()
    assert not at.exception
    held = at.selectbox(key="quick_group").value
    assert dashboard.region_from_view(held) is None, "expected a bare group"

    # Switch to a DIFFERENT region: the held group is no longer offered, so
    # Streamlit must fall back to the new region's rollup.
    other = next(r for r in regions if r != region)
    at.selectbox(key="quick_region").set_value(other).run()
    assert not at.exception
    assert at.selectbox(key="quick_group").value == dashboard.region_all_view(other)


@needs_data
def test_quick_single_group_renders_the_by_customer_table():
    """A single customer group still gets the main by-SKU-and-customer table.

    It used to be gated to the combined / region-rollup views, so this branch had
    no coverage at all. It now runs through single_group_frames (the fast path that
    reuses compute_view's frames). Customer detail, SKU detail and the view-total
    expander are deliberately absent: with one group all three would just restate
    the totals above (the chart and KPI row at the top of the page ARE that group,
    and the table's row detail cards already hold its per-SKU charts).

    Groups are probed rather than assumed — a group with no demand in the model's
    window legitimately stops the body with a "nothing to forecast" error, and
    which groups those are depends on the snapshot.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    for region in [r for r in at.selectbox(key="quick_region").options
                   if r != dashboard.ALL_REGIONS]:
        at.selectbox(key="quick_region").set_value(region).run()
        assert not at.exception
        for i in range(1, len(at.selectbox(key="quick_group").options)):
            at.selectbox(key="quick_group").select_index(i).run()
            assert not at.exception
            if not at.error:
                break
        else:
            continue
        break
    else:
        pytest.skip("no individual customer group has demand in this snapshot")

    headings = _headings(at)
    assert "### Summary table by SKU and customer" in headings
    assert "### Customer detail" not in headings
    assert "### SKU detail" not in headings
    labels = [e.label or "" for e in at.expander]
    assert not any("Summary table by SKU (view total)" in lbl for lbl in labels)
    # The fixed filter bar renders here too — the table is on every Quick view, not
    # just the view-total ones — and opens with nothing picked, which is what keeps
    # this branch showing the whole table on arrival.
    assert at.multiselect(key=BY_CUST_SKU_FILTER).value == []
    assert at.multiselect(key="filter_by_customer::Customer__ms").value == [], (
        "Customer must be on the bar even on a single-group view, where it has "
        "exactly one option — a fixed control that vanishes reads as a bug"
    )


@needs_data
def test_quick_detail_card_renders_its_kpis_as_tiles():
    """Opening a row's detail card renders its KPIs as shaded metric tiles.

    Detail-card KPIs are ``st.metric`` widgets now — that is what makes them inherit
    the stylesheet's tile treatment instead of being flat markdown. Selecting a row
    is driven through session_state because AppTest has no dataframe-selection API.

    The card's tiles are additional metrics beyond the page's 7-KPI row, so the
    assertion is on the DELTA between before and after selection: whatever the KPI
    row contributes, opening a card must add the card's own fields on top.
    """
    import dashboard
    from dashboard_app.config import PRICE_COL as PRICE_COL_LABEL

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    before = [m.label for m in at.metric]

    # Select the first row of the by-customer table (key "filter_by_customer").
    at.session_state["filter_by_customer__sel"] = {"selection": {"rows": [0]}}
    at.run()
    assert not at.exception, at.exception

    after = [m.label for m in at.metric]
    added = [lbl for lbl in after if lbl not in before]
    if not added:
        pytest.skip("no rows in the by-customer table for this snapshot")

    # Every field the card asks for that exists on the row shows up as a tile.
    for expected in ("Customer Grouping", "Data Source",
                     "Current Projection Average", "Projection Difference"):
        assert expected in added, (
            f"{expected!r} missing from the card's tiles; got {added}"
        )
    # Data Source appeared in BOTH the old field grid and the old metrics column.
    assert after.count("Data Source") == 1, (
        f"Data Source rendered {after.count('Data Source')} times: {after}"
    )
    # The rendered tiles really are in KPI_ORDER — including the DERIVED ones. Sorting
    # only the column-backed tiles left "Projected Revenue" stranded at the end of the
    # grid, away from the List Price and Revenue Risk it is read against.
    from dashboard_app.config import kpi_sort

    assert added == kpi_sort(added), (
        f"card tiles are not in canonical order:\n  got      {added}\n"
        f"  expected {kpi_sort(added)}"
    )
    if "Projected Revenue" in added and PRICE_COL_LABEL in added:
        assert added.index("Projected Revenue") > added.index(PRICE_COL_LABEL)
    # The retired chart-side labels must not come back alongside the column names.
    for retired in ("Current Forecast (avg/wk)", "Updated Forecast (avg/wk)",
                    "Projection Difference (avg/wk)"):
        assert retired not in added, (
            f"{retired!r} is the old chart-column label; the tile uses the column name"
        )


def test_historical_window_label_covers_every_models_avg_column():
    """The window prefix on the weekly-demand metrics, per model.

    The figure's window follows the selected model, so the label has to be driven
    off that model's own average-column name. Read the labels out of the real model
    files rather than hardcoding them, so a model whose LOOKBACK_WEEKS changes (or a
    new model file) shows up here instead of silently rendering a bare label.
    """
    from dashboard_app.compute import ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL
    from dashboard_app.config import MODEL_OPTIONS
    from dashboard_app.summaries import historical_window_label
    from agent.model_loader import load_pipeline

    # The canonical columns (compute.py's constants), which the model files match.
    assert historical_window_label(EIGHT_WK_AVG_COL) == "8-Week"
    assert historical_window_label(ALL_TIME_AVG_COL) == "All-Time"
    # Legacy space-spelled column from a frame persisted by an older build.
    assert historical_window_label("8 Week POS/Orders Average") == "8-Week"

    seen = set()
    for label, path in MODEL_OPTIONS.items():
        P = load_pipeline(path)
        avg_col = getattr(P, "AVG_COL_LABEL", EIGHT_WK_AVG_COL)
        window = historical_window_label(avg_col)
        assert window in {"8-Week", "All-Time"} or window.endswith("-Week"), (
            f"{label}: {avg_col!r} produced an unusable window label {window!r}"
        )
        # Never leak the raw column name into a metric label.
        assert "POS/Orders" not in window
        seen.add(window)
    assert seen >= {"8-Week", "All-Time"}, (
        f"expected both windows across the model catalog, saw {seen}"
    )


def test_every_model_label_matches_a_canonical_average_column():
    """A model's AVG_COL_LABEL must be one of the two canonical spellings.

    ``attach_descriptive_averages`` replaces the model's average column with the
    centrally-computed one BY NAME. A model spelling it differently (the old
    "All-History ..." / space-spelled "8 Week ...") leaves both on the same table:
    two columns, two definitions, one window — the exact confusion planners hit.
    """
    from dashboard_app.compute import ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL
    from dashboard_app.config import MODEL_OPTIONS
    from agent.model_loader import load_pipeline

    canonical = {ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL}
    for label, path in MODEL_OPTIONS.items():
        P = load_pipeline(path)
        avg_col = getattr(P, "AVG_COL_LABEL", None)
        if avg_col is None:                    # regression: hardcoded DISPLAY_NAMES
            avg_col = P.DISPLAY_NAMES["8_week_pos_avg"]
        assert avg_col in canonical, (
            f"{label}: average column {avg_col!r} is not one of {canonical} — "
            "the dashboard would carry two copies of the same figure"
        )


@needs_data
def test_quick_kpi_row_names_the_total_weekly_demand_window():
    """The KPI row must say WHICH window its total weekly demand covers.

    With the default 8-Week Moving Average model this reads "Total Weekly Demand
    (8-Week avg)"; the other four models make it "(All-Time avg)". Without the
    window the two are indistinguishable on screen even though they can differ
    substantially.

    It must also NOT be called "Historical Demand" any more: that name was shared
    with a per-SKU metric on the detail card, which computed a different number
    (weeks-with-demand denominator). This KPI is a view TOTAL and is deliberately
    not the sum of the per-SKU average column, so its name says "Total".
    """
    from dashboard_app.compute import EIGHT_WK_AVG_COL
    from dashboard_app.summaries import historical_window_label
    from dashboard_app.pipeline import load_pipeline, pipeline_path

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception

    P = load_pipeline(pipeline_path())
    expected = historical_window_label(
        getattr(P, "AVG_COL_LABEL", EIGHT_WK_AVG_COL)
    )
    labels = [m.label for m in at.metric]
    assert f"Total Weekly Demand ({expected} avg)" in labels, (
        f"expected a {expected}-labelled total-weekly-demand metric, got {labels}"
    )
    # No un-windowed one, and none of the retired "Historical Demand" wording.
    assert "Total Weekly Demand" not in labels
    assert not any("Historical Demand" in (lbl or "") for lbl in labels), (
        f"the retired 'Historical Demand' wording is back: {labels}"
    )


@needs_data
def test_provider_selector_change_does_not_call_llm(monkeypatch):
    """Moving the reasoning-LLM selector (a plain rerun) must not fire the LLM.
    The agent only runs on the button click, so any non-button interaction is a
    valid probe; the provider radio is the on-point one."""
    called = {"llm": False}

    def _boom(*a, **k):
        called["llm"] = True
        raise AssertionError("LLM must not be invoked on a plain rerun")

    monkeypatch.setattr("agent.llm.get_llm", _boom)

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    at.radio(key="agent_llm_provider").set_value("Local LLM").run()
    assert called["llm"] is False
    assert not at.exception


def test_agent_scores_prefers_mase_and_falls_back():
    """_agent_scores reads the current mase_by_model key, falls back to the
    legacy mae_by_model for stale pre-MASE JSONs, and degrades to empty."""
    import dashboard

    assert dashboard._agent_scores({"mase_by_model": {"A": 0.9}}) == ({"A": 0.9}, True)
    assert dashboard._agent_scores({"mae_by_model": {"A": 22.1}}) == ({"A": 22.1}, False)
    # Both present -> mase wins (a regenerated file never carries both, but
    # prefer-current is the documented contract).
    assert dashboard._agent_scores(
        {"mase_by_model": {"A": 0.9}, "mae_by_model": {"A": 22.1}}
    ) == ({"A": 0.9}, True)
    assert dashboard._agent_scores({}) == ({}, False)
