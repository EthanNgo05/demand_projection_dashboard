"""Golden-master parity guard for the performance work.

The performance changes (vectorised rolling MAD, hoisted densification, caching,
the on-disk forecast cache) must not move a single number. This suite pins the
output of every compute entry point against a committed golden and asserts
*exact* equality — the same bar `test_phase2_parity` / `test_phase6_full_parity`
hold the agent-vs-dashboard numbers to.

Goldens live in ``tests/fixtures/golden/`` as Parquet, generated from the seeded
multi-year frame in ``make_perf_fixture.py`` (long enough that outlier
cleansing, Holt-Winters seasonality and gap densification all actually engage —
the tiny Phase-2 fixture is not).

Both sides of every comparison are round-tripped through Parquet, so
serialisation fidelity (e.g. object-dtype ``datetime.date`` columns in the
weekly frames) can never masquerade as a numeric difference.

Regenerate deliberately, and only when a change to the numbers is intended:

    REGEN_GOLDENS=1 pytest tests/test_perf_parity.py

A missing golden is written rather than failed, so the first run on a fresh
checkout bootstraps itself. An *existing* golden is never silently overwritten.
"""

import os
import sys
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
GOLDEN_DIR = os.path.join(FIXTURE_DIR, "golden")
if FIXTURE_DIR not in sys.path:
    sys.path.insert(0, FIXTURE_DIR)

from make_perf_fixture import TODAY, build_frame, build_prices  # noqa: E402

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models"
)
MODEL_FILES = [
    "regression",
    "exponential_smoothing",
    "holt_winters",
    "xgboost",
    "tsb",
]

REGEN = os.environ.get("REGEN_GOLDENS") == "1"


# --------------------------------------------------------------------------
# Golden plumbing
# --------------------------------------------------------------------------
def _roundtrip(df):
    """Send a frame through Parquet and back.

    Applied to the golden *and* the freshly computed frame so both carry the
    identical dtype coercions Parquet performs. Without this, an object-dtype
    column of ``datetime.date`` (which the models' weekly frames carry) would
    read back as datetime64 and every comparison would fail for a reason that
    has nothing to do with the forecast.
    """
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return pd.read_parquet(buf)


def assert_golden(name, df):
    """Compare ``df`` against the committed golden ``name``, exactly."""
    assert df is not None, f"{name}: computed frame is None"
    path = os.path.join(GOLDEN_DIR, f"{name}.parquet")

    # Column order is part of the contract (it drives the Excel exports), so
    # compare as-is rather than sorting. Row order is deterministic too: every
    # producer sorts or groups its way to a stable order.
    if REGEN or not os.path.exists(path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        df.to_parquet(path, index=False)
        # Warn rather than skip: a test can pin several goldens, and skipping on
        # the first write would abort before the rest were written.
        warnings.warn(
            f"wrote golden {name}.parquet ({len(df):,} rows) — "
            "rerun without REGEN_GOLDENS to compare",
            stacklevel=2,
        )
        return

    expected = pd.read_parquet(path)
    actual = _roundtrip(df)
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=True, check_dtype=True,
        obj=f"golden:{name}",
    )


@pytest.fixture(scope="module")
def raw():
    return build_frame()


@pytest.fixture(scope="module")
def prices():
    return build_prices()


_PIPELINES = {}


def pipeline(name):
    """Load a model module by file path, exactly as the app does (cached here)."""
    if name not in _PIPELINES:
        from agent.model_loader import load_pipeline

        _PIPELINES[name] = load_pipeline(os.path.join(MODELS_DIR, f"{name}.py"))
    return _PIPELINES[name]


_CLEANED = {}


def cleaned(name, raw_df):
    """``_clean`` output for one pipeline (cleaning rules are per-pipeline)."""
    if name not in _CLEANED:
        from agent import data_io

        _CLEANED[name] = data_io._clean(raw_df, pipeline(name))
    return _CLEANED[name]


# --------------------------------------------------------------------------
# Per-model forecast goldens
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", MODEL_FILES)
def test_model_combined_view_golden(model, raw, prices):
    """summary + weekly for the all-customers view, per model."""
    P = pipeline(model)
    df = cleaned(model, raw)
    agg = P.aggregate_to_sku_week(df)

    kwargs = {}
    import inspect

    if "list_prices" in inspect.signature(P.fit_regression).parameters:
        kwargs["list_prices"] = prices

    summary, weekly = P.fit_regression(
        agg, TODAY, grouping_label="ALL CUSTOMERS", breakdown_df=df, **kwargs
    )
    assert not summary.empty, f"{model}: nothing forecast — fixture too small?"
    assert_golden(f"{model}__combined__summary", summary)
    assert_golden(f"{model}__combined__weekly", weekly)


@pytest.mark.parametrize("model", MODEL_FILES)
def test_model_single_group_golden(model, raw, prices):
    """summary + weekly for one customer group, per model."""
    P = pipeline(model)
    df = cleaned(model, raw)
    sub = df[df["Customer Grouping"] == "AMAZON-DC"]
    assert not sub.empty, "fixture lost the AMAZON-DC grouping"
    agg = P.aggregate_to_sku_week(sub)

    summary, weekly = P.fit_regression(agg, TODAY, grouping_label="AMAZON-DC")
    assert_golden(f"{model}__amazon__summary", summary)
    assert_golden(f"{model}__amazon__weekly", weekly)


@pytest.mark.parametrize("model", MODEL_FILES)
def test_aggregate_to_sku_week_golden(model, raw):
    """The shared SKU-week aggregate feeding every fit."""
    P = pipeline(model)
    agg = P.aggregate_to_sku_week(cleaned(model, raw))
    assert_golden(f"{model}__agg", agg)


def test_autofit_golden(raw):
    """The exponential-smoothing grid search — the one autofit path."""
    P = pipeline("exponential_smoothing")
    agg = P.aggregate_to_sku_week(cleaned("exponential_smoothing", raw))
    fitted = P.autofit_smoothing(agg, TODAY)
    assert fitted is not None, "autofit returned nothing"
    # A dict of scalars, not a frame — pin it as a one-row table.
    assert_golden("autofit__params", pd.DataFrame([{
        k: v for k, v in sorted(fitted.items()) if isinstance(v, (int, float, str))
    }]))


# --------------------------------------------------------------------------
# Dashboard compute goldens
# --------------------------------------------------------------------------
def test_compute_by_customer_golden(raw, prices):
    """The per-(SKU, group) stitched table behind the ALL CUSTOMERS view."""
    from dashboard_app import compute

    df = cleaned("regression", raw)
    by_cust = compute.compute_by_customer(
        df, TODAY, os.path.join(MODELS_DIR, "regression.py"), prices,
    )
    assert_golden("compute_by_customer__regression", by_cust)


# --------------------------------------------------------------------------
# compute_by_customer_frames / single_group_frames / the averages helper
# --------------------------------------------------------------------------
REG_PATH = os.path.join(MODELS_DIR, "regression.py")


def test_frames_change_only_the_average_columns(raw, prices):
    """``compute_by_customer_frames`` adds the all-time average and touches nothing else.

    The Quick Projections table reads the _frames variant; this pins that the only
    difference from the golden-mastered ``compute_by_customer`` output is the
    average columns — All-Time added, 8-Week replaced in place by the identical
    central value (see the test below).
    """
    from dashboard_app import compute

    df = cleaned("regression", raw)
    plain = compute.compute_by_customer(df, TODAY, REG_PATH, prices)
    framed, weekly_by_group, agg_by_group = compute.compute_by_customer_frames(
        df, TODAY, REG_PATH, prices,
    )

    shared = [c for c in plain.columns if not c.endswith("POS/Orders Average")]
    pd.testing.assert_frame_equal(framed[shared], plain[shared], check_exact=True)
    # The all-time average and the recent trend are new. The 8-week column already
    # exists under its canonical name — the model files spell it exactly as
    # compute.py's constant, so attach_descriptive_averages overwrites it in place
    # rather than leaving a second, differently-spelled copy of the same figure.
    assert set(framed.columns) - set(plain.columns) == {
        compute.ALL_TIME_AVG_COL, compute.TREND_COL,
    }
    assert set(plain.columns) - set(framed.columns) == set()
    # All three land together right after "Weeks with data".
    pos = list(framed.columns).index("Weeks with data")
    assert list(framed.columns)[pos + 1:pos + 4] == [
        compute.ALL_TIME_AVG_COL, compute.EIGHT_WK_AVG_COL, compute.TREND_COL,
    ]
    # The per-group frames cover the same groups as the table, un-summed.
    for frame in (weekly_by_group, agg_by_group):
        assert set(frame["Customer Grouping"]) == set(framed["Customer Grouping"])


def test_central_8wk_matches_the_regression_models_own_average(raw, prices):
    """The claim the Quick Projections table now rests on.

    Its 8-Week column is the centrally-computed one, overwriting whatever the model
    reported. ``_descriptive_averages``' docstring asserts the two are equal for the
    8-Week Moving Average model (same POS-then-Orders source, same total /
    weeks-since-first span, and that model applies no outlier cleansing); if they
    ever diverge, every row of that column silently changes value.
    """
    from dashboard_app import compute

    df = cleaned("regression", raw)
    plain = compute.compute_by_customer(df, TODAY, REG_PATH, prices)
    framed, _, _ = compute.compute_by_customer_frames(df, TODAY, REG_PATH, prices)

    key = ["SKU", "Customer Grouping"]
    merged = plain[key + [compute.EIGHT_WK_AVG_COL]].merge(
        framed[key + [compute.EIGHT_WK_AVG_COL]], on=key, how="outer",
        suffixes=("_model", "_central"), indicator=True,
    )
    assert (merged["_merge"] == "both").all(), "row sets diverged"
    pd.testing.assert_series_equal(
        merged[f"{compute.EIGHT_WK_AVG_COL}_central"],
        merged[f"{compute.EIGHT_WK_AVG_COL}_model"],
        check_names=False, check_exact=True,
    )


def test_single_group_frames_matches_the_loop(raw, prices):
    """The fast path that avoids fitting a single-group view twice.

    ``compute_view``'s single-group branch and ``_forecast_one_group`` make the
    same three calls, so reusing the former's frames must equal running the loop
    over just that group's rows.
    """
    from dashboard_app import compute

    df = cleaned("regression", raw)
    group = "AMAZON-DC"
    summary, weekly, agg = compute.compute_view(
        df, group, TODAY, REG_PATH, prices,
    )
    reused, reused_wk, reused_ag = compute.single_group_frames(
        summary, weekly, agg, group, TODAY,
    )
    looped, loop_wk, loop_ag = compute.compute_by_customer_frames(
        df[df["Customer Grouping"] == group], TODAY, REG_PATH, prices,
    )
    pd.testing.assert_frame_equal(reused, looped, check_exact=True)
    pd.testing.assert_frame_equal(reused_wk, loop_wk, check_exact=True)
    pd.testing.assert_frame_equal(reused_ag, loop_ag, check_exact=True)


def _summary_frame(rows, cols):
    """Minimal summary frame: rows are dicts keyed by column name."""
    return pd.DataFrame(rows, columns=cols)


def test_attach_descriptive_averages_drops_the_space_spelled_column():
    """The legacy "8 Week ..." spelling, from a frame persisted by an older build.

    The model files now spell it exactly as EIGHT_WK_AVG_COL, so this path only
    matters for old cached frames — but it must not leave both spellings behind.
    """
    from dashboard_app import compute

    agg = _agg_frame([("AMAZON-DC", "A-1", "2026-06-21", 8.0, np.nan)])
    summary = _summary_frame(
        [{"SKU": "A-1", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3,
          "8 Week POS/Orders Average": 8.0, "Updated Projection Average": 5.0}],
        ["SKU", "Customer Grouping", "Weeks with data",
         "8 Week POS/Orders Average", "Updated Projection Average"],
    )
    out = compute.attach_descriptive_averages(summary, agg, pd.Timestamp("2026-07-01"))
    assert "8 Week POS/Orders Average" not in out.columns
    # All three descriptive columns present, slotted after "Weeks with data".
    assert list(out.columns) == [
        "SKU", "Customer Grouping", "Weeks with data",
        compute.ALL_TIME_AVG_COL, compute.EIGHT_WK_AVG_COL, compute.TREND_COL,
        "Updated Projection Average",
    ]


def test_attach_descriptive_averages_central_all_time_overrides_the_model():
    """The central observed average WINS over whatever the model reported.

    Four of the five models report the mean of the outlier-CLEANSED series they fit.
    Displaying that made one column mean different things depending on the selected
    model, and disagree with the demand plotted beside it. So the central value
    overwrites it; the model's figure survives only where ``_descriptive_averages``
    produced no row for the pair at all, so such a row shows something not a blank.
    """
    from dashboard_app import compute

    agg = _agg_frame([
        ("AMAZON-DC", "A-1", "2026-06-21", 8.0, np.nan),
        ("AMAZON-DC", "A-2", "2026-06-21", 4.0, np.nan),
    ])
    summary = _summary_frame(
        # A-1 carries a model value the central pass also covers -> overridden.
        # A-2 carries none -> filled centrally. ORPHAN is absent from `agg`, so the
        # central pass has no row for it and its model value survives.
        [{"SKU": "A-1", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3,
          compute.ALL_TIME_AVG_COL: 123.4},
         {"SKU": "A-2", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3,
          compute.ALL_TIME_AVG_COL: np.nan},
         {"SKU": "ORPHAN", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3,
          compute.ALL_TIME_AVG_COL: 77.7}],
        ["SKU", "Customer Grouping", "Weeks with data", compute.ALL_TIME_AVG_COL],
    )
    out = compute.attach_descriptive_averages(summary, agg, pd.Timestamp("2026-07-01"))
    by_sku = out.set_index("SKU")
    assert by_sku.loc["A-1", compute.ALL_TIME_AVG_COL] == 8.0    # central overrode 123.4
    assert by_sku.loc["A-2", compute.ALL_TIME_AVG_COL] == 4.0    # filled centrally
    assert by_sku.loc["ORPHAN", compute.ALL_TIME_AVG_COL] == 77.7   # no central row


def test_attach_descriptive_averages_zero_fills_missing_recent_pairs():
    """A pair the central averages never saw gets a genuine 0.0 recent run-rate."""
    from dashboard_app import compute

    # A-2 sells only outside the 8-week window, so it has an all-history average
    # but no recent one.
    agg = _agg_frame([
        ("AMAZON-DC", "A-1", "2026-06-21", 8.0, np.nan),
        ("AMAZON-DC", "A-2", "2025-01-05", 40.0, np.nan),
    ])
    summary = _summary_frame(
        [{"SKU": "A-1", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3},
         {"SKU": "A-2", "Customer Grouping": "AMAZON-DC", "Weeks with data": 3},
         {"SKU": "GHOST", "Customer Grouping": "AMAZON-DC", "Weeks with data": 0}],
        ["SKU", "Customer Grouping", "Weeks with data"],
    )
    out = compute.attach_descriptive_averages(summary, agg, pd.Timestamp("2026-07-01"))
    by_sku = out.set_index("SKU")
    assert by_sku.loc["A-2", compute.EIGHT_WK_AVG_COL] == 0.0
    assert by_sku.loc["A-2", compute.ALL_TIME_AVG_COL] > 0
    # A pair absent from the aggregate entirely: 0.0 recent, blank all-history.
    assert by_sku.loc["GHOST", compute.EIGHT_WK_AVG_COL] == 0.0
    assert pd.isna(by_sku.loc["GHOST", compute.ALL_TIME_AVG_COL])


def test_attach_descriptive_averages_without_weeks_with_data():
    """No "Weeks with data" anchor => no reorder, and no ValueError from index()."""
    from dashboard_app import compute

    agg = _agg_frame([("AMAZON-DC", "A-1", "2026-06-21", 8.0, np.nan)])
    summary = _summary_frame(
        [{"SKU": "A-1", "Customer Grouping": "AMAZON-DC"}],
        ["SKU", "Customer Grouping"],
    )
    out = compute.attach_descriptive_averages(summary, agg, pd.Timestamp("2026-07-01"))
    assert list(out.columns) == [
        "SKU", "Customer Grouping", compute.ALL_TIME_AVG_COL,
        compute.EIGHT_WK_AVG_COL, compute.TREND_COL,
    ]


def test_compute_by_customer_best_golden(raw, prices, monkeypatch):
    """The Optimized Projections table: each group fit with its own best model.

    The winner resolution is monkeypatched because it reads
    ``outputs/agent_summary_<group>.json``, which the fixture has none of — the
    real call would return all-Nones and the golden would be vacuous.

    Two models are alternated across the groups on purpose. The 8-Week Moving
    Average model reports only an 8-week average while Holt's reports only an
    all-time one, so a mixed table exercises both sides of the descriptive-average
    merge (override-the-model-value and fill-the-gap) as well as the concat of
    summaries with differing columns. It is also the case that matters most for
    this change: before it, one column on this very table held Holt's cleansed
    figure on some rows and the central observed figure on others.
    Alternation is by sorted-group index, not ``hash()``, which is salted per
    process and would make the golden non-reproducible.
    """
    from dashboard_app import compute

    df = cleaned("regression", raw)
    models = [
        ("8-Week Moving Average", os.path.join(MODELS_DIR, "regression.py")),
        ("Holt's (double) exponential smoothing",
         os.path.join(MODELS_DIR, "exponential_smoothing.py")),
    ]
    order = {g: i for i, g in
             enumerate(sorted(df["Customer Grouping"].dropna().unique()))}
    monkeypatch.setattr(
        compute, "_best_model_for_group", lambda g: models[order[g] % 2]
    )

    combined, weekly_all, agg_all, weekly_by_group, agg_by_group, excluded = (
        compute.compute_by_customer_best(df, TODAY, prices)
    )
    assert not excluded, f"every group should resolve, got excluded={excluded}"
    assert combined[compute.MODEL_USED_COL].nunique() == 2, "both models should win"
    assert_golden("compute_by_customer_best__combined", combined)
    assert_golden("compute_by_customer_best__weekly_all", weekly_all)
    assert_golden("compute_by_customer_best__agg_all", agg_all)
    assert_golden("compute_by_customer_best__weekly_by_group", weekly_by_group)
    assert_golden("compute_by_customer_best__agg_by_group", agg_by_group)


def test_descriptive_averages_golden(raw):
    """The all-time / 8-week demand averages every by-customer table now shows."""
    from dashboard_app import compute

    P = pipeline("regression")
    df = cleaned("regression", raw)
    agg_by_group = pd.concat(
        [
            P.aggregate_to_sku_week(g).assign(**{"Customer Grouping": name})
            for name, g in df.groupby("Customer Grouping", sort=True)
        ],
        ignore_index=True,
    )
    out = compute._descriptive_averages(agg_by_group, TODAY)
    assert_golden("descriptive_averages", out)


def _descriptive_averages_reference(agg_by_group, today_ts):
    """The original per-(group, SKU) Python-loop implementation, kept verbatim.

    ``_descriptive_averages`` was vectorised for speed (~10s -> ~0.4s on the live
    snapshot); this is what it must keep reproducing exactly. Held here rather
    than only as a golden file so the equivalence is checked against real logic on
    adversarial inputs, not just one recorded output.
    """
    days_since_sunday = (today_ts.weekday() + 1) % 7
    current_week_start = today_ts - pd.Timedelta(days=days_since_sunday)
    last_complete_week = current_week_start - pd.Timedelta(weeks=1)
    eight_wk_start = last_complete_week - pd.Timedelta(weeks=7)
    prior_8wk_end = eight_wk_start - pd.Timedelta(weeks=1)
    prior_8wk_start = prior_8wk_end - pd.Timedelta(weeks=7)
    hist_start = today_ts - pd.DateOffset(years=3)

    from dashboard_app.compute import (
        ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL, _PRIOR_8WK_COL,
    )

    A = agg_by_group.copy()
    A["SKU"] = A["SKU"].astype(str)
    A = A[~A["SKU"].str.endswith("*")]
    A["WeekDate"] = pd.to_datetime(A["WeekDate"])

    def _avg(start, out_col, end=None):
        end = last_complete_week if end is None else end
        win = A[(A["WeekDate"] >= start) & (A["WeekDate"] <= end)]
        rows = []
        for (grp, sku), g in win.groupby(["Customer Grouping", "SKU"], sort=False):
            pos = g[g["POS"].notna()]
            if not pos.empty:
                vals, weeks = pos["POS"], pos["WeekDate"]
            else:
                orders = g[g["Orders"].notna()]
                if orders.empty:
                    continue
                vals, weeks = orders["Orders"], orders["WeekDate"]
            weeks_span = int(round((end - weeks.min()).days / 7)) + 1
            rows.append({"Customer Grouping": grp, "SKU": sku,
                         out_col: round(vals.sum() / max(weeks_span, 1), 1)})
        return pd.DataFrame(rows, columns=["Customer Grouping", "SKU", out_col])

    key = ["Customer Grouping", "SKU"]
    out = (_avg(hist_start, ALL_TIME_AVG_COL)
           .merge(_avg(eight_wk_start, EIGHT_WK_AVG_COL), on=key, how="outer")
           .merge(_avg(prior_8wk_start, _PRIOR_8WK_COL, end=prior_8wk_end),
                  on=key, how="outer"))
    # Trend, spelled out row-by-row rather than vectorised: a missing RECENT average
    # is a real zero (the pair sold nothing lately), a missing PRIOR average is no
    # baseline at all and stays blank. See _descriptive_averages for why.
    trend = []
    for _, r in out.iterrows():
        prior, recent = r[_PRIOR_8WK_COL], r[EIGHT_WK_AVG_COL]
        recent = 0.0 if pd.isna(recent) else float(recent)
        if pd.isna(prior) or float(prior) <= 0:
            trend.append(np.nan)
        else:
            trend.append(round((recent - float(prior)) / float(prior) * 100.0, 1))
    out[TREND_COL] = pd.Series(trend, index=out.index, dtype="float64")
    return out.drop(columns=[_PRIOR_8WK_COL])


def _agg_frame(rows):
    """Build a minimal per-group SKU-week frame from (group, sku, week, pos, ord)."""
    return pd.DataFrame(
        [{"Customer Grouping": g, "SKU": s, "WeekDate": pd.Timestamp(w),
          "POS": p, "Orders": o, "Projection": 1.0} for g, s, w, p, o in rows]
    )


@pytest.mark.parametrize("label,rows", [
    # The two cases the multi-year fixture never produces, and which broke the
    # first vectorised draft: a window with NO POS rows anywhere, and one with no
    # Orders rows anywhere. Both used to leave an empty float64 Series standing in
    # for a datetime64 column, which numpy refuses to promote.
    ("orders only", [
        ("AMAZON-DC", "A-1", "2026-06-21", np.nan, 5.0),
        ("AMAZON-DC", "A-1", "2026-06-14", np.nan, 7.0),
        ("COSTCO", "B-2", "2026-06-21", np.nan, 3.0),
    ]),
    ("pos only", [
        ("AMAZON-DC", "A-1", "2026-06-21", 5.0, np.nan),
        ("COSTCO", "B-2", "2026-06-14", 8.0, np.nan),
    ]),
    ("mixed per pair", [
        ("AMAZON-DC", "A-1", "2026-06-21", 5.0, 99.0),   # POS wins
        ("AMAZON-DC", "A-2", "2026-06-21", np.nan, 4.0),  # Orders fallback
        ("COSTCO", "B-2", "2026-06-21", np.nan, np.nan),  # dropped entirely
    ]),
    ("single row", [("AMAZON-DC", "A-1", "2026-06-21", 5.0, np.nan)]),
    ("discontinued sku dropped", [
        ("AMAZON-DC", "A-1*", "2026-06-21", 5.0, np.nan),
        ("AMAZON-DC", "A-2", "2026-06-21", 6.0, np.nan),
    ]),
    ("all outside the window", [
        ("AMAZON-DC", "A-1", "2020-01-05", 5.0, np.nan),
    ]),
])
def test_descriptive_averages_matches_reference(label, rows):
    """Edge cases, checked against the original loop rather than a golden file."""
    from dashboard_app import compute

    agg = _agg_frame(rows)
    today = pd.Timestamp("2026-07-01")
    compute._descriptive_averages.clear()
    got = compute._descriptive_averages(agg, today)
    want = _descriptive_averages_reference(agg, today)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want.reset_index(drop=True),
        check_exact=True, check_dtype=False, obj=label,
    )


def test_descriptive_averages_matches_reference_on_fixture(raw):
    """And on the full multi-year fixture, dtypes included."""
    from dashboard_app import compute

    P = pipeline("regression")
    df = cleaned("regression", raw)
    agg_by_group = pd.concat(
        [P.aggregate_to_sku_week(g).assign(**{"Customer Grouping": name})
         for name, g in df.groupby("Customer Grouping", sort=True)],
        ignore_index=True,
    )
    today = TODAY
    compute._descriptive_averages.clear()
    pd.testing.assert_frame_equal(
        compute._descriptive_averages(agg_by_group, today),
        _descriptive_averages_reference(agg_by_group, today),
        check_exact=True, check_dtype=True,
    )


def test_compute_exceptions_golden(raw, prices):
    """The actuals-vs-plan exception scan."""
    from dashboard_app import exceptions as EX

    P = pipeline("regression")
    frame = EX.compute_exceptions(cleaned("regression", raw), TODAY, prices, P)
    assert_golden("compute_exceptions", frame)


def test_compute_spikes_golden(raw, prices):
    """The 'selling with no projections' spike scan."""
    from dashboard_app import exceptions as EX

    P = pipeline("regression")
    agg = EX.sku_week_by_group(cleaned("regression", raw), P)
    assert_golden("sku_week_by_group", agg)

    frame = EX.compute_spikes(agg, TODAY, prices, P, min_container_impact=0.0)
    assert_golden("compute_spikes", frame)


# --------------------------------------------------------------------------
# Rolling-MAD primitive: property test against the pandas reference
# --------------------------------------------------------------------------
def _reference_rolling(y, window, min_periods=3):
    """The original implementation: pandas rolling + a Python MAD callback.

    Kept verbatim here as the thing the vectorised helper must reproduce, so the
    property test stays meaningful after the model files stop using it.
    """
    def _mad(a):
        a = np.asarray(a, dtype="float64")
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.nan
        return np.median(np.abs(a - np.median(a)))

    s = pd.Series(np.asarray(y, dtype="float64"))
    med = s.rolling(window, center=True, min_periods=min_periods).median().to_numpy()
    mad = s.rolling(window, center=True, min_periods=min_periods).apply(
        _mad, raw=True
    ).to_numpy()
    return med, mad


MAD_MODELS = ["exponential_smoothing", "holt_winters", "xgboost", "tsb"]


def test_rolling_helper_identical_across_models():
    """``_rolling_median_mad`` must be byte-identical in all four model files.

    The models are deliberately standalone — shared helpers are duplicated on
    purpose so a model can be swapped in via DEMAND_PIPELINE with no package
    imports (see CLAUDE.md's pipeline contract). That makes drift the real risk,
    so pin it: one source of truth checked four ways. This is also what licenses
    the property test below to exercise a single model rather than all four.
    """
    import inspect as _inspect

    sources = {}
    for name in MAD_MODELS:
        fn = getattr(pipeline(name), "_rolling_median_mad", None)
        assert fn is not None, f"{name} is missing _rolling_median_mad"
        sources[name] = _inspect.getsource(fn)

    first = sources[MAD_MODELS[0]]
    for name, src in sources.items():
        assert src == first, (
            f"_rolling_median_mad in {name}.py has drifted from "
            f"{MAD_MODELS[0]}.py — the model files must stay identical"
        )


def test_rolling_median_mad_matches_pandas():
    """The vectorised rolling median/MAD must equal the pandas version exactly.

    800 seeded random series x 6 window sizes, deliberately including NaNs,
    all-zero runs, single-element series and series shorter than the window —
    the edge cases where a hand-rolled sliding window is easiest to get wrong
    (notably the centred-padding split, which differs for even windows).
    Asserted with ``array_equal(equal_nan=True)``: bitwise, not approximate.

    Runs against one model only; the test above proves the other three carry the
    identical function, and each model's end-to-end ``cleanse_series`` golden
    covers its integration.
    """
    model = MAD_MODELS[0]
    helper = getattr(pipeline(model), "_rolling_median_mad", None)
    assert helper is not None, f"{model} is missing _rolling_median_mad"

    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # np.nanmedian on all-NaN slices
        for _ in range(800):
            n = int(rng.integers(1, 200))
            y = rng.choice(
                [0.0, 1.0, 5.0, 100.0, np.nan], size=n, p=[.45, .2, .2, .1, .05]
            ).astype("float64")
            for w in (4, 5, 6, 7, 12, 13):
                exp_med, exp_mad = _reference_rolling(y, w)
                got_med, got_mad = helper(y, w, 3)
                assert np.array_equal(got_med, exp_med, equal_nan=True), \
                    f"{model}: median mismatch n={n} window={w}"
                assert np.array_equal(got_mad, exp_mad, equal_nan=True), \
                    f"{model}: MAD mismatch n={n} window={w}"


@pytest.mark.parametrize("model", ["exponential_smoothing", "holt_winters",
                                   "xgboost", "tsb"])
def test_cleanse_series_golden(model, raw):
    """cleanse_series end-to-end on every fixture SKU's real densified series.

    Guards the integration, not just the primitive: flags, method labels and the
    replaced values all have to match, for every SKU shape the fixture carries.
    """
    P = pipeline(model)
    df = cleaned(model, raw)
    agg = P.aggregate_to_sku_week(df)

    rows = []
    for sku, g in agg.groupby("SKU", sort=True):
        g = g.sort_values("WeekDate")
        y = g["POS"].fillna(g["Orders"]).fillna(0.0).to_numpy(dtype="float64")
        cleanedy, flags, method = P.cleanse_series(g["WeekDate"], y)
        rows.append(pd.DataFrame({
            "SKU": sku,
            "WeekDate": g["WeekDate"].to_numpy(),
            "y": y,
            "cleaned": cleanedy,
            "flag": flags,
            "method": [str(m) for m in method],
        }))
    assert_golden(f"{model}__cleanse_series", pd.concat(rows, ignore_index=True))
