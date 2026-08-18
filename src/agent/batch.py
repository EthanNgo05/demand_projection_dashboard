"""Parallel batch runner: precompute every view's agent summary.

    cd src && python -m agent.batch                 # all views, all cores-1
    cd src && python -m agent.batch --workers 6
    cd src && python -m agent.batch --no-llm         # skip LLM prose (fast/cheap)
    cd src && python -m agent.batch --provider anthropic

The single-view CLI (``agent.run``) runs one view; within a view the four
models are fit serially, because one model (Holt-Winters) dominates the runtime
so per-model parallelism doesn't pay off. Parallelism lives HERE instead: this
runner fans the ~114 views out across a process pool and makes each worker
single-threaded, so views run concurrently without the per-view dominant-model
bottleneck and without CPU oversubscription. Every view writes its own
``outputs/agent_summary_<view>.json`` (see ``agent/nodes/publish.py``), which
the dashboard reads back instantly.

Design:
  * The parent ingests **once** (reads the snapshot — fast via the Parquet
    sidecar — and fetches Plytix a single time), then hands the cleaned frame to
    every worker via one temp Parquet file. ``ingest`` short-circuits when the
    state already carries ``cleaned_df``, so no worker re-reads or re-fetches.
  * Thread caps (``OMP_NUM_THREADS`` etc., ``XGB_N_JOBS=1``) are set in the
    parent BEFORE the pool is created, so each spawned worker inherits them and
    imports NumPy/XGBoost single-threaded. N single-threaded workers then use
    the cores without contention.
"""

import argparse
import os
import sys
import tempfile
import time

import pandas as pd

from agent.config import (
    ALL_CUSTOMERS_VIEW,
    MODEL_OPTIONS,
    region_all_view,
)
from agent.data_io import default_pipeline
from agent.nodes.ingest import ingest

# Set once per worker process by the pool initializer, then reused across every
# view that worker handles (loaded lazily so the Parquet read happens at most
# once per process, not once per view).
_CLEANED_PATH = None
_TODAY = None
_CLEANED_CACHE = None
_PRICES = None
_DATA_SIG = None


def enumerate_views(cleaned_df, P):
    """Every view the dashboard offers, in a stable order.

    ``ALL_CUSTOMERS_VIEW`` + one per-region rollup (``All Customers - <region>``)
    for each region present + every individual Customer Grouping. Mirrors
    dashboard.list_views' bucketing via the pipeline's ``region_for_group``, but
    without importing streamlit.
    """
    groups = sorted(cleaned_df["Customer Grouping"].dropna().unique().tolist())
    regions = sorted({str(P.region_for_group(g)) for g in groups})
    return [ALL_CUSTOMERS_VIEW] + [region_all_view(r) for r in regions] + groups


def _worker_init(cleaned_path, prices, today_ts, data_sig=None):
    """Pool initializer: stash the shared inputs for this worker process."""
    global _CLEANED_PATH, _TODAY, _PRICES, _DATA_SIG
    _CLEANED_PATH = cleaned_path
    _TODAY = today_ts
    _PRICES = prices
    _DATA_SIG = data_sig


def _cleaned():
    """The cleaned demand frame for this worker, read once and cached."""
    global _CLEANED_CACHE
    if _CLEANED_CACHE is None:
        _CLEANED_CACHE = pd.read_parquet(_CLEANED_PATH)
    return _CLEANED_CACHE


def _warm_forecast_cache(view, results):
    """Persist this view's already-computed frames for the dashboard to reuse.

    ``run_all_models`` fits EVERY model for the view and keeps each one's
    ``summary_df``/``weekly_df``/``agg`` in ``state["results"]`` — numbers the
    batch previously discarded, publishing only which model won. Those frames are
    exactly what ``dashboard_app.compute.compute_view`` /
    ``_forecast_one_group`` produce for the same view and model (the Phase-2 and
    Phase-6 parity suites assert that with exact-match equality), so writing them
    into the shared on-disk cache turns the dashboard's first open of the day into
    a Parquet read instead of a recompute — the ~55s Optimized Projections build
    most of all.

    Both cache "kinds" are warmed for a plain customer group, because the two
    front-door paths key differently: Quick Projections asks for ``kind="view"``,
    while the Optimized/Watchlist per-group loop asks for ``kind="group"``. The
    inputs are identical for a group view (``view_frame`` for a group is the same
    ``df["Customer Grouping"] == group`` slice ``_forecast_one_group`` receives),
    so one computation legitimately satisfies both keys.

    Smoothing parameters are left at None, matching the models' own module
    constants. That is what ``compute_view`` uses for models with no autofit
    (XGBoost / TSB / Holt-Winters); the Holt pipeline's autofitted alpha/beta/phi
    produce a DIFFERENT key, so the dashboard correctly misses and refits rather
    than reading a default-parameter forecast. Never guessed, never approximated.

    Entirely best-effort: any failure is swallowed, since the summary JSON this
    batch exists to write has already been published by the time we get here.
    """
    if not results:
        return
    try:
        from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS, region_from_view
        from dashboard_app import forecast_cache

        if not _DATA_SIG or not forecast_cache.enabled():
            return
        today_str = pd.Timestamp(_TODAY).date().isoformat()
        # A group view feeds both the per-view and per-group cache keys; the ALL
        # and region rollups only ever exist as a "view".
        is_rollup = view == ALL_CUSTOMERS_VIEW or region_from_view(view) is not None
        kinds = ("view",) if is_rollup else ("view", "group")

        for label, bundle in results.items():
            path = MODEL_OPTIONS.get(label)
            summary = (bundle or {}).get("summary_df")
            weekly = (bundle or {}).get("weekly_df")
            agg = (bundle or {}).get("agg")
            if path is None or summary is None or weekly is None or agg is None:
                continue
            frames = {"summary": summary, "weekly": weekly, "agg": agg}
            for kind in kinds:
                key = forecast_cache.forecast_key(
                    _DATA_SIG, view, path, today_str, kind=kind,
                )
                forecast_cache.put(
                    key, frames,
                    {"view": view, "model": os.path.basename(path),
                     "kind": kind, "source": "agent.batch"},
                )
    except Exception:  # noqa: BLE001 - warming must never fail a view
        pass


def _run_view(view):
    """Run the full agent graph for one view. Returns (view, ok, error_str).

    Imports build_graph lazily so the (NumPy-pulling) import happens inside the
    already-thread-capped worker. The state is pre-seeded with the cleaned frame
    + prices, so ``ingest`` skips the read/clean/exclusions and Plytix fetch.
    """
    from agent.graph import build_graph

    try:
        state = {
            "view": view,
            "today_ts": _TODAY,
            "cleaned_df": _cleaned(),
            "prices": _PRICES,
        }
        final = build_graph().invoke(state)
        _warm_forecast_cache(view, final.get("results"))
        errs = final.get("errors") or []
        return view, True, ("; ".join(errs) if errs else None)
    except Exception as e:  # one view must never sink the batch
        return view, False, f"{type(e).__name__}: {e}"


def _default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Precompute every view's agent summary in parallel."
    )
    ap.add_argument("--workers", type=int, default=_default_workers(),
                    help="Worker processes (default: CPU count - 1).")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM anomaly/narrative nodes (pure numeric "
                         "refresh — avoids one LLM call per view).")
    ap.add_argument("--provider", choices=["anthropic", "local"], default=None,
                    help="LLM provider for the reasoning nodes (overrides "
                         "LLM_PROVIDER); ignored with --no-llm.")
    ap.add_argument("--views", nargs="+", default=None,
                    help="Run only these views (default: all). Used by the "
                         "dashboard's 'Retry failed views' button.")
    args = ap.parse_args(argv)

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.no_llm:
        os.environ["AGENT_SKIP_LLM"] = "1"

    today_ts = pd.Timestamp.today().normalize()

    # --- Ingest once in the parent -----------------------------------------
    seed = ingest({"today_ts": today_ts})
    cleaned = seed.get("cleaned_df")
    if cleaned is None or cleaned.empty:
        errs = seed.get("errors") or ["ingest produced no data"]
        print("Batch aborted:", "; ".join(errs), file=sys.stderr)
        return 1
    prices = seed.get("prices")

    # --- Forecast-cache signature (must match the dashboard's exactly) ------
    # Built here in the parent, from the same inputs dashboard.py uses, and
    # handed to every worker so each can persist its view's frames under a key
    # the dashboard will actually look up. If this drifts from dashboard.py's
    # data_sig block the warm-up silently stops being found (a miss, never a
    # wrong number), so the two are covered by test_forecast_cache.py.
    data_sig = None
    try:
        from dashboard_app import forecast_cache

        data_sig = forecast_cache.snapshot_signature(
            seed.get("raw_path"),
            n_excluded_rows=seed.get("n_excluded_rows"),
            n_rows=len(cleaned),
            prices=forecast_cache.content_signature(prices),
        )
    except Exception as exc:  # noqa: BLE001 - warming is optional
        print(f"Forecast-cache warming disabled ({type(exc).__name__}: {exc})")

    P = default_pipeline()
    views = enumerate_views(cleaned, P)
    if args.views:
        # Retry / subset run: keep only the requested views that still exist
        # (a view can vanish between runs if the underlying data changed).
        wanted = set(args.views)
        views = [v for v in views if v in wanted]
        missing = wanted - set(views)
        if missing:
            print(f"Skipping {len(missing)} unknown view(s): "
                  + ", ".join(sorted(missing)))
        if not views:
            print("Batch aborted: none of the requested views exist.",
                  file=sys.stderr)
            return 1
    print(f"Precomputing {len(views)} view(s) with {args.workers} worker(s)"
          + (" [no-llm]" if args.no_llm else ""))

    # --- Hand the cleaned frame to workers via one temp Parquet ------------
    tmpdir = tempfile.mkdtemp(prefix="agent_batch_")
    cleaned_path = os.path.join(tmpdir, "cleaned.parquet")
    cleaned.to_parquet(cleaned_path, index=False)

    # --- Cap per-process threads BEFORE spawning, so children inherit it ---
    # Each worker runs one view with its models serial (that is already the
    # only mode), so cap each worker to a single thread — otherwise XGBoost
    # (n_jobs=-1) and BLAS would each grab every core and N workers would
    # oversubscribe. BLAS reads these at import time, so they must be set before
    # the pool spawns fresh interpreters.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "XGB_N_JOBS"):
        os.environ[var] = "1"

    # ProcessPoolExecutor imported here so the thread-cap env is set first.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    started = time.time()
    ok = fail = 0
    failures = []
    # Emit the total up front so the dashboard's progress reader (batch_progress)
    # shows "0 of N done" immediately, instead of sitting on "getting started"
    # until the first (slow) view finishes. flush is belt-and-suspenders — the
    # child already runs with PYTHONUNBUFFERED=1.
    print(f"  [0/{len(views)}] starting…", flush=True)
    try:
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_worker_init,
            initargs=(cleaned_path, prices, today_ts, data_sig),
        ) as ex:
            # submit + as_completed, NOT ex.map: map yields results in SUBMISSION
            # order, so the counter would stall on the first-listed view (the slow
            # combined/regional rollups from enumerate_views) even while a dozen
            # small views have already finished and written their JSON. Iterating
            # as they COMPLETE makes "k of N done" track the files actually on
            # disk. _run_view never raises (it wraps its own errors), so
            # fut.result() is a plain (view, ok, error) unpack.
            futures = [ex.submit(_run_view, v) for v in views]
            for fut in as_completed(futures):
                view, succeeded, error = fut.result()
                if succeeded:
                    ok += 1
                    tag = "ok" if not error else f"ok (warnings: {error})"
                else:
                    fail += 1
                    failures.append((view, error))
                    tag = f"FAILED: {error}"
                print(f"  [{ok + fail}/{len(views)}] {view} -> {tag}", flush=True)
    finally:
        try:
            os.remove(cleaned_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    # Keep outputs/.cache bounded: this run just wrote up to 5 models x 2 kinds
    # per view, so without a prune the warm-up would grow without limit as
    # snapshots roll over night after night. Least-recently-USED entries go first
    # (get() touches the marker), so views people actually open survive.
    if data_sig:
        try:
            from dashboard_app import forecast_cache

            removed = forecast_cache.prune()
            n_entries, n_bytes = forecast_cache.stats()
            print(f"Forecast cache: {n_entries} entries, {n_bytes / 1e6:.0f} MB"
                  + (f" ({removed} pruned)" if removed else ""))
        except Exception as exc:  # noqa: BLE001 - housekeeping is optional
            print(f"Forecast-cache prune skipped ({type(exc).__name__}: {exc})")

    elapsed = time.time() - started
    print(f"\nDone: {ok} ok, {fail} failed in {elapsed:.0f}s.")
    if failures:
        print("Failures:")
        for view, error in failures:
            print(f"  {view}: {error}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
