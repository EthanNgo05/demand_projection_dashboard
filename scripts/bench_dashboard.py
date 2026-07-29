"""Time the dashboard's compute paths against the live snapshot.

Run before and after a performance change and diff the two tables. Every number
is wall-clock seconds on the newest snapshot in ``raw_inputs/demand_projections``
(so it reflects real data volume, not the test fixture).

    python scripts/bench_dashboard.py                  # everything
    python scripts/bench_dashboard.py --skip-slow      # drop the 60-group loops
    python scripts/bench_dashboard.py --json out.json  # machine-readable too

Deliberately measures the *cold* cost of each step: the Streamlit caches are not
active outside a Streamlit runtime, and the on-disk forecast cache is bypassed
via DEMAND_FORECAST_CACHE=0 so a warmed cache can't flatter the fit timings.
Use the app itself to judge warm/cached behaviour.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time

# Single-threaded so timings are comparable run to run (and match how the batch
# workers actually execute). Must be set before NumPy/XGBoost import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "XGB_N_JOBS"):
    os.environ.setdefault(_v, "1")
os.environ["DEMAND_FORECAST_CACHE"] = "0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd  # noqa: E402

MODELS = ["regression", "exponential_smoothing", "tsb", "holt_winters", "xgboost"]

results = {}


@contextlib.contextmanager
def step(label, quiet=True):
    """Time a block, record it, and print the row. Model files print progress
    notes to stdout on every fit; swallow that so the table stays readable."""
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            yield
    finally:
        dt = time.perf_counter() - t0
        results[label] = round(dt, 3)
        print(f"  {label:<52} {dt:8.2f}s", flush=True)


def load_model(name):
    path = os.path.join(SRC, "models", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"bench_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip the ~60-group loops (compute_by_customer/_best)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args()

    from agent import data_io
    from dashboard_app import compute, exceptions as EX

    files = data_io.discover_raw_files()
    if not files:
        sys.exit("No snapshot found under raw_inputs/demand_projections/")
    snapshot_date, snapshot = files[0]
    print(f"\nSnapshot: {os.path.basename(snapshot)}\n")

    print("Ingest")
    with step("read_raw_frame (parquet sidecar)"):
        raw = data_io.read_raw_frame(snapshot)
    P = load_model("regression")
    with step("_clean"):
        cleaned = data_io._clean(raw, P)
    today = pd.Timestamp.today().normalize()
    print(f"  -> {len(raw):,} raw rows, {len(cleaned):,} cleaned, "
          f"{cleaned['Customer Grouping'].nunique()} groups, "
          f"{cleaned['SKU'].nunique()} SKUs\n")

    print("Per-model fit (all-customers view)")
    for name in MODELS:
        mod = load_model(name)
        cl = data_io._clean(raw, mod)
        with step(f"{name}: aggregate_to_sku_week"):
            agg = mod.aggregate_to_sku_week(cl)
        with step(f"{name}: fit_regression"):
            mod.fit_regression(agg, today, grouping_label="ALL CUSTOMERS",
                               breakdown_df=cl)
    es = load_model("exponential_smoothing")
    es_cl = data_io._clean(raw, es)
    es_agg = es.aggregate_to_sku_week(es_cl)
    with step("exponential_smoothing: autofit_smoothing"):
        es.autofit_smoothing(es_agg, today)
    print()

    print("Exceptions view")
    with step("sku_week_by_group"):
        agg_by_group = EX.sku_week_by_group(cleaned, P)
    with step("compute_exceptions"):
        exc = EX.compute_exceptions(cleaned, today, None, P)
    with step("compute_spikes"):
        EX.compute_spikes(agg_by_group, today, None, P)
    with step("_descriptive_averages"):
        compute._descriptive_averages(agg_by_group, today)
    print()

    print("Excel export builders")
    reg_agg = P.aggregate_to_sku_week(cleaned)
    with step("(setup) regression fit for export frames"):
        summary, weekly = P.fit_regression(reg_agg, today,
                                           grouping_label="ALL CUSTOMERS",
                                           breakdown_df=cleaned)
    with step("view_to_excel (summary + weekly)"):
        compute.view_to_excel(summary, weekly)
    with step("summary_to_excel (exceptions frame)"):
        compute.summary_to_excel(exc)
    print()

    if not args.skip_slow:
        print("Multi-group loops")
        with step("compute_by_customer (regression, all groups)"):
            compute.compute_by_customer(
                cleaned, today, os.path.join(SRC, "models", "regression.py"))
        with step("compute_by_customer_best (Optimized Projections)"):
            compute.compute_by_customer_best(cleaned, today)
        print()

    total = round(sum(results.values()), 2)
    print(f"  {'TOTAL':<52} {total:8.2f}s\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"snapshot": snapshot_date, "seconds": results,
                       "total": total}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
