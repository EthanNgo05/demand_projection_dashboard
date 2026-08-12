# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About simplehuman

[simplehuman](https://www.simplehuman.com) designs and sells premium home products — sensor trash cans, soap dispensers, sensor mirrors, dish racks, paper towel pumps, and related accessories. This repo is an internal supply-chain tool: it forecasts 15-week SKU-level demand so the planning team can spot where existing projections are too high or too low.

Scale of the data (per the latest warehouse snapshot):

- **~700 active SKUs**, after the SQL excludes samples, promos, credits, and other non-sellable item classes.
- **~57 raw customers** (`Custnmbr`), consolidated into a smaller set of **Customer Groupings** across 5 fulfillment regions: US (LBC+NJ), EU (SH-CTS), AU (ACR), CA (YYZ5), JP (NETDEPOT). Several raw customers fold into one group — e.g. AMAZON-DC / AMAZON-DS / MARVAL-FBM all become `AMAZON-DC`; Web + Warranty US become one group. Customers with no explicit mapping keep their own name as a single-member group, and region "Others - <country>" buckets catch the long tail.

## Common commands

Run everything **from the repo root** so the `raw_inputs/` / `outputs/` relative paths resolve (the agent CLIs are the exception — run those from `src/`).

```bash
pip install -r requirements.txt

# Interactive dashboard
streamlit run src/dashboard.py

# Batch forecast (each model file is also a standalone script; writes per-group + combined .xlsx to outputs/)
python src/models/exponential_smoothing.py   # or regression.py / holt_winters.py / xgboost.py / tsb.py

# Warehouse pulls (or the scheduled wrapper for all four nightly steps: ./refresh_demand_data.ps1)
python src/extract_demand_details.py          # ~10 min full 36-month pull -> dated all_demand_projections_<date>.xlsx + .parquet sidecar
python src/extract_demand_details.py --incremental   # fast: recent weeks + forward projections, merged into the latest snapshot
python src/extract_warehouse_projections.py   # ~2 min -> 5 regional <REGION>_warehouse_projections_<date>.xlsx
python src/extract_key_skus.py                # ~seconds -> dated key_skus/key_skus_<date>.xlsx (the "Key SKUs" watchlist)
python src/active_missing_projections.py      # batch: active SKUs missing forward projections -> outputs/missing_projections

# Tests (~290 fast tests). The agent package lives under src/; pytest.ini puts src/ on sys.path.
pytest tests/ -v
pytest tests/test_phase3_select.py::test_name   # single test
pytest --runslow                                # include the 7 slow full-matrix parity tests

# Performance: time every compute path against the live snapshot (before/after a change)
python scripts/bench_dashboard.py               # flags: --skip-slow, --json out.json

# Audit a view's totals: per-SKU old (top-down fit) vs new (sum of parts) -> outputs/*.xlsx
python scripts/reconcile_grain_change.py        # flags: --view, --model, --out

# Regenerate the golden masters — ONLY when a change to the numbers is intended
REGEN_GOLDENS=1 pytest tests/test_perf_parity.py

# Agent end-to-end for one view (run from src/ so `python -m agent.run` resolves the package)
cd src && python -m agent.run --view "All customers (combined)"
cd src && python -m agent.run --view "AMAZON-DC"

# Precompute every view's agent summary in parallel (what the nightly job runs)
cd src && python -m agent.batch               # flags: --workers N, --no-llm (skip narrative prose)
```

## Architecture

Two front-ends run over one shared forecasting core:

1. **`src/dashboard.py` + `src/dashboard_app/`** — Streamlit + Plotly UI. `dashboard.py` is now a **thin entrypoint/facade**: it configures logging and re-exports every helper from the `dashboard_app/` package as `dashboard.<name>` (so tests and `main()`/`_run()` resolve unchanged), then `main()` drives the UI. All implementation lives in `dashboard_app/` — see "The dashboard_app package" below. The UI loads the selected model **by file path** via `importlib` (chosen through the `DEMAND_PIPELINE` env var) and runs it live per Customer Grouping; its "🔄 Refresh data" button spawns the extract scripts on demand (incremental demand pull + warehouse pull + key-SKUs pull); its agent section reads the precomputed `outputs/agent_summary_<view>.json` files and can also run one view live.
2. **`src/agent/`** — a LangGraph pipeline (`ingest → run_all_models → evaluate_models → select_best_model →` conditional `→ flag_anomalies/summarize` or `flag_low_confidence` `→ publish`) that runs all the models, backtests to pick the best per view, uses an LLM to flag anomalies and write a narrative, and publishes `outputs/agent_summary_<view>.json` + logs. See `docs/agentic_workflow/` for the phased design (state schema in `00-overview.md`). `agent/config.py` mirrors `dashboard_app/config.py`'s `MODEL_OPTIONS` and `ALL_CUSTOMERS_VIEW` and must stay in sync with them.

Forecast **views** offered by both front-ends: `All customers (combined)`, one `All Customers - <region>` rollup per region, and every individual Customer Grouping (`agent/batch.py`'s `enumerate_views` mirrors `dashboard.list_views` without importing streamlit). The dashboard also has three **dashboard-only top-level scopes** that never reach the compute path and are never returned by `list_views`/`enumerate_views` (so the agent never forecasts them): **Optimized Projections** (`BEST_MODEL_COMBINED_VIEW` — each group with its own backtest-winning model, stitched into one table; depends on the agent batch), **Exceptions** (`EXCEPTIONS_VIEW` — model-agnostic actuals-vs-plan scan), and **Watchlist** (`WATCHLIST_VIEW` — starred SKU/group pins). Standard single-model views live under a fourth **Quick Projections** (`QUICK_VIEW`) tab, whose two dropdowns (**Region**, including the UI-only `ALL_REGIONS` sentinel, then **Customer group**) resolve it to one of the three real view-string shapes. `config.py`'s `SCOPE_LABELS`/`SCOPE_CAPTIONS` map these internal IDs to the tab strip's friendly labels/captions, and `quick_group_label` prettifies a view string for display without ever closing over the selected region.

Quick Projections and Optimized Projections deliberately share one page shape — KPI row → total weekly demand → `Customer detail` → `SKU detail` → a condensed `Summary table by SKU and customer` whose rows expand into a detail card (`kpis.render_sku_detail_card`, shared by both). Quick keeps a view-level per-SKU table in a collapsed expander below: it is the same numbers rolled up to the grain an order is placed at, plus the one column unique to it (`Top Volume Customer Groups`).

### One grain: a total is the sum of its SKU × customer parts

**Forecasts are fit per (SKU, Customer Grouping), and every view-level figure is the SUM of those rows.** This is the single most important invariant in the app; `tests/test_rollup_ties.py` pins it and `scripts/reconcile_grain_change.py` audits it against the live snapshot.

Quick Projections used to break it. For `All customers (combined)` and the region rollups it called `compute_view`, which sums every customer's history into one series per SKU and fits *that* ("top-down"), and showed the result as the KPI row / total chart — directly above a `Summary table by SKU and customer` built from the per-customer fits ("bottom-up"). The two ran **+19.9% apart** on the live snapshot (97,312 vs 116,637 units/wk; only 27% of SKUs tied exactly). **90% of that gap was demand the top-down fit could not see at all:** POS-vs-Orders is chosen per series (`regression.py`'s `pos_grp = grp[grp["POS"].notna()]` … else Orders) while `aggregate_to_sku_week` sums POS and Orders into *separate* columns — so a SKU with POS at any one customer resolved to POS for the whole aggregate and every Orders-only customer's demand was silently dropped. The remaining 10% is per-series span denominators, the zero floor and per-row rounding, none of which commute with summation.

The render path now runs the per-group loop only and rolls up, exactly as Optimized Projections always has, through shared helpers so the two views provably agree:

- **`compute.roll_up_to_sku_week`** — the one (SKU, WeekDate) roll-up. Customer groups are disjoint, so summing is a plain total. Used by both views.
- **`summaries.resolve_demand`** — resolves each (SKU, customer)'s OWN POS/Orders signal into a `demand` column *before* the roll-up, so the actuals total covers every customer too. `historical_window` honours a precomputed `demand` column rather than re-deriving one per SKU, which after a roll-up is impossible.
- **`compute.roll_up_summary`** — per-SKU totals. `Data Source` becomes `config.MIXED_SOURCE` where a SKU's groups disagree (363 of 566 SKUs); `Weeks with data` is recounted so a week two customers both sold in counts once.
- **`compute.attach_current_projection`** — `Current Projection Average` is raw snapshot data with no model in it, yet it was **not additive**: the models take a mean over only the horizon weeks a series *has* a plan for, so each customer divided by a different denominator (1–15). Recomputed over a fixed horizon-length denominator, deliberately **unrounded** (rounding 4,000 rows then summing ≠ rounding the sum). `Projection Difference` / `Revenue Risk` are re-derived from it.
- **`compute.sku_grain_demand_frame`** — hands the resolved `demand` to `_descriptive_averages` **as the POS column** with Orders blanked. That helper picks its own source per (group, SKU), which at SKU grain would re-drop the Orders-only customers; neutralising the choice reuses its golden-mastered span logic instead of reimplementing it.
- **`compute.attach_top_volume`** — `Top Volume Customer Groups` used to arrive as a side effect of the combined fit's `breakdown_df`. `top_volume_customers` was always a standalone rollup of the raw frame, present on all five models, so it is called directly and the column survives.
- **`dashboard._coverage_note`** — because a total is now a sum, a row that never got forecast is missing units. Groups the per-group loop skipped, blank-Customer-Grouping rows, and (SKU, customer) pairs with a forward plan but no recent demand are counted and reported under the KPI row.

**`compute_view` itself is unchanged and must stay that way.** It is still the single-group path (where top-down and bottom-up are the same fit), and three things pin its combined-view output to exact equality: `tests/test_region_all_view.py` (fast), the `test_phase2_parity` / `test_phase6_full_parity` suites against `agent/nodes/forecast.py`'s line-by-line reproduction, and `agent/batch.py`'s cache warm-up, which writes agent frames under `compute_view`'s own `kind="view"` key — change its semantics and the nightly batch silently serves stale numbers with no test to catch it.

Still on the old basis, and deliberately out of scope for that change: the agent's ALL/region anomaly bullets, the batch's `ALL_CUSTOMERS_*.xlsx` (a separate combined fit; the per-group concatenation beside it is already bottom-up), Exceptions' `Group by = SKU` roll-up, and Historical Summary (which intentionally uses a different frame — it retains discontinued SKUs — so its actuals run higher than every projection view's).

### The dashboard_app package

`dashboard.py` re-exports everything from these modules, so most helpers are reachable as either `dashboard.<name>` or `dashboard_app.<module>.<name>`. Streamlit-free modules are marked — those carry no `st.*` and are safe to import from the agent or tests without a Streamlit runtime.

- **`config.py`** *(streamlit-free)* — constants, the `MODEL_OPTIONS` catalog, the view/scope ID constants (`ALL_CUSTOMERS_VIEW`, `BEST_MODEL_COMBINED_VIEW`, `EXCEPTIONS_VIEW`, `WATCHLIST_VIEW`, `QUICK_VIEW`, region-rollup prefix) with `SCOPE_LABELS`/`SCOPE_CAPTIONS`, and pure format/view helpers. `HERE` = `src/`, `REPO_ROOT` = its parent (code vs. data anchoring).
- **`pipeline.py`** — load the forecasting model by file path (cached, mtime-keyed) and introspect `fit_regression`'s signature (`_supports_smoothing`/`_min_weeks`/`_prices`/`_autofit`) to decide which sidebar controls show.
- **`compute.py`** — the forecasting compute core: `list_views`, `compute_view` (now the SINGLE-GROUP path only, plus the agent's parity contract — see "One grain" above), `compute_by_customer`, `compute_by_customer_frames` (what Quick Projections uses — the per-group loop plus the un-summed per-group chart frames and both descriptive averages), `single_group_frames` (the fast path when the view already *is* one group, so it isn't fit twice under `compute_view`'s and `_forecast_one_group`'s separate cache keys), `compute_by_customer_best` (best-model-per-group), the roll-up family (`roll_up_to_sku_week`, `roll_up_summary`, `attach_current_projection`, `sku_grain_demand_frame`, `attach_top_volume`), agent-summary readers, Excel exporters. One `_by_customer_frames` loop backs all three by-customer entry points, and one `attach_descriptive_averages` gives every row both the 8-Week run-rate and the All-Time average regardless of which model produced it. **Both come from `_descriptive_averages` and OVERRIDE whatever the model reported** — the models report the mean of the series they fit, which for four of the five is outlier-cleansed, so leaving that in place made one column mean different things per model and disagree with the chart beside it. The models' own cleansed figure still ships in their standalone `.xlsx` output and in `compute_view`'s summary (which the agent parity tests hold to exact equality with `fit_regression`, so it must not be touched), but **it no longer reaches the screen anywhere**: the view-total expander is now a roll-up carrying the central observed averages, so the `... (model fit)` display rename it used to need is gone. `compute_by_customer` stays the raw stitched concatenation on purpose — that is what the golden master pins.

`attach_descriptive_averages` also derives **`TREND_COL`** (`"Recent Trend"`) — the percent change from the 8 weeks before last to the last 8 completed weeks, i.e. "is this accelerating or decaying". Its prior-8-week baseline is computed by the same `_avg` (which takes an `end` bound for it) and then **dropped**: publishing a third average column beside the other two would rebuild exactly the confusion the All-Time rename removed. A missing *recent* average is a real zero, so a SKU that was selling and has stopped reads −100%; a missing *prior* average is no baseline at all, stays `NaN`, and the tile renders **"New"**.

**`attach_supply_columns`** adds `ONHAND_COL` / `WOS_COL` and is deliberately **not** part of `attach_descriptive_averages` — On Hand comes from a separately loaded map (`data_io.onhand_by_sku`), not from the fit, so folding it in would put a second data source into a forecast-cache key that only tracks forecast inputs. It is applied in the render paths instead. WOS uses the **same** definition as `exceptions.compute_spikes` (total On Hand ÷ total *current* weekly projection across all customers, SKU-level); there is deliberately no "WOS vs the updated forecast" variant.

**One name per window.** `ALL_TIME_AVG_COL` = `"All-Time POS/Orders Average"` and `EIGHT_WK_AVG_COL` = `"8-Week POS/Orders Average"` are the only spellings the UI shows, and the five model files' `AVG_COL_LABEL` / `DISPLAY_NAMES` match them **exactly** so the central value replaces the model's column in place. These two, plus `TREND_COL` / `ONHAND_COL` / `WOS_COL`, live in **`config.py`** (re-exported from `compute.py` for the callers that always imported them from there) so `KPI_ORDER` can name them without `config` importing the streamlit-dependent `compute`. `summaries.historical_window_label` is a pass-through, not a translation: the KPI label and the table column must use the same word for the same window. It previously rendered the internal `"All-History"` as the display word `"All-Time"`, which put two names for one window on screen.
- **`datasources.py`** / **`summaries.py`** *(summaries is streamlit-free)* — cached file discovery + readers over `agent.data_io`, and pure summary/column/timestamp helpers.
- **`charts.py`** / **`tables.py`** / **`kpis.py`** — Plotly builders + date-range control; summary-table styling and the Excel-style add-filter-chip table filter (`render_filtered_table`); the KPI row and the Optimized (best-model-combined) render.

**Detail cards have ONE shape, everywhere.** `tables._render_row_detail` is the single renderer behind every click-to-expand card (Quick, Optimized, Exceptions, Spikes, Watchlist): title → a shaded KPI tile grid (`_render_kpi_tiles`, 4 per row) → the chart full-width → optional row-action button. Three rules keep the cards consistent:

- **Tiles are `st.metric`**, so they inherit the `[data-testid="stMetric"]` styling in `dashboard.py`'s one stylesheet rather than restating it — the cards and the page-top KPI row cannot drift apart. Detail-card-scoped rules step the value down and let it wrap; `KPI_TEXT_FIELDS` values (Customer, Region, `Model Used`, …) are wrapped in a `st-key-kpitile-text-*` container so CSS can render them as captions instead of headlines.
- **`config.KPI_ORDER` decides placement, not the view.** Each view's `*_CARD_COLS` list is a *set* of fields to show; `kpi_sort` orders them (identity → history → forecast → money → supply), so a field sits in the same spot on every card. `config.KPI_HELP` supplies the tooltips. Derived tiles from `extra_kpis` are folded in **before** sorting, so e.g. `Projected Revenue` lands beside List Price rather than at the end.
- **A card has exactly one KPI zone.** The projections card used to be chart-left / metrics-right *on top of* the field grid — two zones, with `Data Source` in both. `render_sku_detail_card` now draws only the chart; its seven metrics are tiles. The Exceptions card's "Calculate Optimal Projection" result is the one exception: it's computed on a button click, so it stays beside that button, as an `st.metric` pair so it still looks like a tile.

`extra_kpis` (derived tiles) and `kpi_deltas` (a secondary figure under an existing tile, e.g. Δ% under `Projection Difference`) are the two per-view hooks, bound with `functools.partial` and threaded through `render_selectable_table`.
- **`exceptions.py`** — the Exceptions view: `compute_exceptions` (8-week POS/Orders run-rate vs. the same 15 forward weeks the models use → gap/pct/USD impact) + `render_exceptions` with per-SKU detail cards. Also `compute_spikes` (the "Recent spikes in POS/Orders with no projections" table: SKUs projected 0 that have started selling in the last 8 weeks — the first selling week is the spike onset — with First Week Spike / Weeks Since Spike; filtered by an adjustable "Minimum container impact" threshold), rendered below the Under/Over sections in both tabs via `_render_spikes_section`. That table also carries two **SKU-level** columns (constant across a SKU's customer rows): **Container Impact** = the SKU's total cumulative spike units ÷ its `Container Load` (from Plytix, via `data_io.container_load_from_plytix`), and **WOS Impact** (Weeks of Supply) = the SKU's total On Hand ÷ its total weekly projection across all customers. The row shows Container Impact; WOS + List Price live in the click-to-expand detail card (SKU — Description title, chart, and Calculate Optimal Projection). On Hand comes from the raw demand frame (dropped by `_clean`) via `data_io.onhand_by_sku`, loaded by `datasources.load_onhand_by_sku_from_path/_from_bytes` and threaded into `render_exceptions` as `onhand_by_sku` (like `allocation_pairs`).
- **`watchlist.py`** / **`watchlist_view.py`** — named shared watchlists persisted to `outputs/watchlist.json` (atomic writes; active list is per-session state) and the ★ marker used across tables; the Watchlist view reuses the best-model numbers plus the Exceptions detail card.
- **`dataquality.py`** — the inactive / missing-projection / discontinued / no-POS data-quality section renderers.
- **`refresh.py`** — subprocess-backed manual refresh (demand / warehouse / key-SKUs / agent batch) with lock/log files and progress polling.
- **`agent_summary.py`** — read + render the precomputed agent-summary JSON, the live single-view run, and the `_model_fit_callout` reconciliation.
- **`forecast_cache.py`** *(streamlit-free)* — the **persistent** forecast cache under `outputs/.cache/forecasts/<key>/` (`summary`/`weekly`/`agg` as Parquet + a `meta.json` marker written last; `params.json` for autofit's scalars). Keyed on `snapshot_signature` (basename + `st_mtime_ns` + size of the snapshot, so an in-place `--incremental` rewrite invalidates) folded with a content hash of the list prices, the view, the model file's mtime, α/β/φ/min-weeks and the run date. Streamlit-free so `agent/batch.py` can warm it from a worker. Every read/write is best-effort: a missing pyarrow, a corrupt file or a half-written entry is a **miss** (recompute), never an exception — deleting `outputs/.cache` at any moment is safe. Disable with `DEMAND_FORECAST_CACHE=0` (what `scripts/bench_dashboard.py` does so a warm cache can't flatter a timing run).

### Caching layers (why a revisited view is instant)

Three tiers sit in front of every forecast, cheapest first:

1. **`st.session_state`** — `dashboard.py`'s `fc_cache` (per view, bounded by `FC_CACHE_MAX`), `autofit_params`, `exceptions_structural`/`exceptions_spikes`, `bestmix_*`. Per browser session; lost on refresh.
2. **`@st.cache_data`** — per process. Note `compute_view`/`_forecast_one_group` take the full frame as a hashed argument, which is why tier 1 exists on top.
3. **`forecast_cache`** — on disk, survives restarts, shared across sessions, and **warmed nightly**: `agent.batch`'s `_warm_forecast_cache` persists the frames `run_all_models` already computed for all five models per view (previously discarded — only `best_model` was published). The dashboard's `data_sig` and the batch's must stay identical or the warm-up silently stops being found; `tests/test_forecast_cache.py` pins that, and `AgentState.n_excluded_rows` exists solely so the batch can reproduce the dashboard's key.

`data_sig` is threaded explicitly through `compute_view` / `_forecast_one_group` / `compute_by_customer{,_best}` / `run_autofit` / `optimal_projection_for` rather than held in module state — Streamlit runs each session's script on its own thread against the same module objects, so a shared global could let one session's snapshot key another session's forecast. `None` (an upload override) disables the disk tier.

**Cache-key gotcha:** Streamlit **excludes underscore-prefixed parameters** from a `@st.cache_data` key. `datasources.py`'s invalidation args (`mtime`/`mtimes`/`nonce`/`week_key`/`data`) must therefore carry no underscore — they once did, and the cache silently never invalidated (an incremental refresh kept serving the pre-refresh frame; "Refresh from Plytix" never re-fetched). `tests/test_datasource_cache_keys.py` guards it.

### The agent pipeline (target architecture)

```
raw_inputs/*.xlsx (demand) ─┐
list_prices/*.xlsx (Plytix) ─┤
                             ▼
                    ┌─────────────────┐
                    │     ingest      │  discover files, load, clean, apply Plytix exclusions
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ run_all_models  │  fit all 5 models for the view (serial; see Parallelism)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ evaluate_models │  one shared walk-forward backtest → pooled MASE per model
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
              ┌─────┤ select_best_model├─────┐   winner = lowest MASE; confidence_flag if MASE > threshold
              │     └─────────────────┘      │
   confidence ok                      low confidence / no scoreable backtest
              ▼                               ▼
     ┌─────────────────┐            ┌────────────────────┐
     │ flag_anomalies  │  (LLM)     │ flag_low_confidence │  (LLM)
     └────────┬────────┘            └──────────┬─────────┘
              ▼                                 │
     ┌─────────────────┐                        │   both LLM nodes also emit the
     │    summarize    │  (LLM)                  │   expected-best-model reasoning
     └────────┬────────┘                        │   (see below)
              └────────────────┬────────────────┘
                               ▼
                     ┌───────────────────┐
                     │      publish      │  write outputs/agent_summary_<view>.json + app.log
                     └───────────────────┘
```

**Model-fit reasoning (expected vs. actual best model).** `select_best_model` picks the winner purely by lowest backtest MASE. The two LLM nodes (`summarize` and `flag_low_confidence`) additionally record what model the LLM would *expect* to fit best given the view's demand character, reconciled against the MASE winner — so a view that is clearly intermittent yet won by XGBoost surfaces that mismatch rather than hiding it. This is **grounded, not guessed**: `agent/demand_profile.py` deterministically computes Syntetos-Boylan demand-classification features (% zero-weeks, average demand interval / ADI, lumpiness CV², weeks of history, SKU count, and a `smooth/intermittent/erratic/lumpy` `pattern`) and `reasoning._fit_block` folds them plus the per-model MASE table into the *same* LLM call (no extra call per view). The response is pinned to three parseable sections (`EXPECTED_MODEL` / `FIT_NOTE` / `SUMMARY`); `reasoning._parse_model_fit` validates the expected label against `MODEL_OPTIONS` and degrades to plain narrative if a weak local model ignores the format. The publish payload gains `expected_best_model` (a `MODEL_OPTIONS` label or `null`) and `model_fit_note` (a concise expected-vs-actual sentence) alongside `best_model`/`mase_by_model`; both are `null` on `--no-llm` runs. The dashboard's agent section renders the reconciliation via `dashboard._model_fit_callout`.

### The pipeline contract (most important thing to understand)

Each model file in `src/models/` is **deliberately standalone and self-contained** — shared constants (customer groupings, ignore lists) are **duplicated in every model file on purpose** so a model can be swapped in via `DEMAND_PIPELINE` with no package imports. Both the dashboard and the agent talk to a model through this convention:

- Functions: `week_anchors`, `aggregate_to_sku_week`, `fit_regression` (aliased per module — e.g. `fit_regression = fit_exponential_smoothing`), `region_for_group`
- Constants: `RAW_INPUTS_FOLDER`, `LIST_PRICE_GLOB`, `CUSTOMERS_TO_IGNORE`, `COMBINED_GROUPING`

The dashboard **introspects `fit_regression`'s signature** to decide which sidebar controls to show: `alpha`/`beta`/`phi` args → smoothing sliders; `min_weeks_for_trend` → min-weeks slider; `list_prices` → revenue-risk columns; an `autofit_smoothing` function → the Autofit button. This is why XGBoost's sliders hide automatically — its signature carries no smoothing params.

**⚠️ If you change the customer groupings or ignore lists, edit all five model files identically** (`regression.py`, `exponential_smoothing.py`, `holt_winters.py`, `xgboost.py`, `tsb.py`). `src/agent/data_io.py`'s `_clean` is the shared cleaning step and must stay in sync too (see the sync comment in `regression.py`'s `__main__`).

### The five models (`src/models/`)

- **`regression.py`** — 8-week moving average nudged by a dampened linear-regression slope (`TREND_WEIGHT = 0.25`). Labeled "8-Week Moving Average" in the UI.
- **`exponential_smoothing.py`** — Holt's double exponential smoothing (level + trend, damped by `PHI`). The only model with outlier cleansing, promo uplift, and an `autofit_smoothing` grid search.
- **`holt_winters.py`** — Holt-Winters triple exponential smoothing: level + damped trend + **additive seasonality** (`SEASONAL_PERIODS = 52`, annual), fit via `statsmodels` (self-tunes α/β/γ/φ, so no smoothing sliders/autofit, like XGBoost). Labeled "Holt-Winters (triple) exponential smoothing". Needs ≥2 full annual cycles (`MIN_WEEKS_FOR_SEASONAL = 104`); short-history SKUs and non-converging fits fall back to non-seasonal damped Holt. Reuses `exponential_smoothing.py`'s cleansing / window / flatten-to-week-1 behaviour. By far the slowest model — this shapes the parallelism design below.
- **`xgboost.py`** — gradient-boosted trees, **pooled per Customer Grouping** (SKU histories are too short to train per-SKU), each SKU scaled by its own mean, forecast 15 weeks recursively. Falls back to sklearn's `HistGradientBoostingRegressor` if `xgboost` isn't installed.
- **`tsb.py`** — TSB (Teunter–Syntetos–Babai) for intermittent/lumpy demand (the majority of SKUs here): a smoothed demand *probability* (updated every week, so dead SKUs decay to 0) × a smoothed demand *size* (updated on non-zero weeks); forecast = probability × size, an intrinsically flat rate. Fixed `ALPHA_P`/`ALPHA_Z`, no sliders/autofit (like XGBoost); `FILL_GAPS_WITH_ZERO` must stay True (zeros are TSB's signal). Labeled "TSB (intermittent demand)".

### Parallelism model (`src/agent/batch.py`)

Within one view the models fit **serially** — Holt-Winters dominates the runtime, so per-model parallelism doesn't pay. Instead `agent.batch` fans the ~60 views across a `ProcessPoolExecutor` of **single-threaded** workers: thread-cap env vars (`OMP_NUM_THREADS` etc., `XGB_N_JOBS=1`) are set in the parent *before* the pool spawns, so workers import NumPy/XGBoost single-threaded and N workers use N cores without contention. The parent ingests once (snapshot read + Plytix fetch) and hands every worker the cleaned frame via one temp Parquet file — `ingest` short-circuits when the state already carries `cleaned_df`, so no worker re-reads or re-fetches.

### Data flow & inputs

- **Nightly job** (`refresh_demand_data.ps1`, registered with Windows Task Scheduler): full demand pull → warehouse pull (independent — runs even if the demand pull failed) → key-SKUs pull (`extract_key_skus.py`) → `agent.batch` precompute (only if the demand pull succeeded). The nightly demand pull is deliberately the **full 36-month pull** — the self-healing baseline that picks up restated actuals, item renames, and customer remaps; the dashboard's refresh button runs the fast `--incremental` pull instead. Worst exit code wins so Task Scheduler flags a failure in any step. Logs to `logs/<date>/logs_refresh.txt`. The batch step also **warms `forecast_cache`** (and prunes it), so the first person in each morning reads forecasts off disk instead of recomputing. Warm hits require the batch's `today_ts` (wall clock) to match the snapshot's date label, which the pull-then-batch ordering gives; if they diverge the dashboard simply recomputes and writes its own entry — a miss, never a wrong number.
- `sql/demand_details_optimized.sql` is the **default** query behind the demand extract (`DEFAULT_SQL` in `extract_demand_details.py`); `--incremental` only works against it (it rewrites a marker line in the batch to narrow the date window). The legacy `sql/demand_details.sql` is **UTF-16 encoded** (opens as garbled/spaced text in some tools — that's expected, per `.gitattributes`). Region "Others - <country>" buckets attach at `Custnmbr` grain via `MIN(Customer)` — don't drop them when touching the SQL.
- **Parquet sidecars**: the demand extract writes a `.parquet` sidecar next to each snapshot `.xlsx` (same basename). The `.xlsx` stays the source of truth; `data_io.read_raw_frame` prefers the sidecar when it's at least as new, else reads the xlsx and backfills the sidecar. Sidecar writes are best-effort (no pyarrow → logged and skipped). Snapshot pruning keeps the newest `KEEP_SNAPSHOTS` files (default 3, `DEMAND_KEEP_SNAPSHOTS` env var) and deletes each pruned snapshot's sidecar with it. **`extract_demand_details.load_previous_snapshot` reads through the same sidecar rule** (`_read_snapshot_frame`), so an `--incremental` pull loads its history half in ~0.15s instead of re-parsing 34 MB of xlsx.
  - **Measure columns must stay float64.** pyodbc returns `decimal.Decimal` for SQL decimal columns, which pandas stores as *object*, and pyarrow refuses to convert that — which silently broke the sidecar write on **every** run for at least ten days (the xlsx write never noticed, because openpyxl serialises Decimal as a number). The only symptom was slowness: no fresh sidecar meant `read_raw_frame` re-parsed the workbook (~45–66s vs 0.15s) on the first page load after every refresh, and again inside every incremental pull. `select_and_rename` now normalises via `coerce_numeric_columns` (`NUMERIC_COLUMNS` is derived from `SQL_TO_POWERBI_FORMAT`, so a new column is classified automatically); this also stops `merge_snapshots`' `pd.concat` producing an object column from float64-previous + object-fresh.
  - **The sidecar must equal `pd.read_excel(path, header=2)`** — `read_raw_frame` serves them interchangeably, so a divergence would make displayed numbers depend on which file was read. Two normalisations are needed and one difference is irreducible: blank text → NaN (Excel writes a zero-length string as an empty cell) is applied by `_blank_text_to_na` **inside `write_parquet_sidecar` only** — applying it upstream would change which rows ship, because `_apply_output_filters` drops on `Custnmbr.notna()`. Excel's ~15-significant-digit storage and its data-dependent int64 inference for null-free whole-number columns are documented, value-preserving, and pinned by `tests/test_parquet_sidecar.py` (verified exact on the real 709k-row snapshot).
- Raw inputs live at the repo root under `raw_inputs/`: `demand_projections/all_demand_projections_<date>.xlsx` (written by the extract; PowerBI exports also work), `list_prices/list_prices_*.xlsx` (Plytix export — drives revenue-risk columns *and* the two data-quality checks below), `warehouse_projections/<REGION>_*.xlsx` (normally written by `extract_warehouse_projections.py` from `sql/warehouse_projections.sql`; manual PowerBI exports also work — `data_io.warehouse_wide_to_long` sniffs whether a file is the legacy wide matrix or the long table layout, and for long files reconstructs the missing SKU×customer×week cells that drive the missing-projections table), and `key_skus/key_skus_<date>.xlsx` (single `SKU` column from `extract_key_skus.py` via `sql/key_skus.sql`; the dashboard discovers the newest via `data_io.discover_key_skus_file`/`read_key_skus` to populate the "Key SKUs" watchlist).
- **Data-quality checks** (dashboard, need the Plytix export): SKUs projected into a region they aren't "Active in", and Discontinued/Inactive SKUs still carrying projections — both flagged, excluded from the forecast, and listed in their own tables.
- Only Python code lives under `src/`. Data/log/doc folders (`raw_inputs/`, `outputs/`, `logs/`, `sql/`, `docs/`, `notebooks/`) stay at the repo root — `outputs/` and `logs/` are gitignored.

## Configuration (`.env`, see `.env.example`)

- `LLM_PROVIDER` = `anthropic` (needs `ANTHROPIC_API_KEY`, default model `claude-sonnet-5`) or `local` (Google Gemma Model). Only the agent's reasoning nodes call an LLM; forecasting math is fully deterministic and needs no key.
- SQL Server connection for the extract: `SQL_SERVER` and `SQL_DATABASE` are **required** (no hardcoded defaults). Blank `SQL_USER` → Windows trusted auth.
- `DEMAND_PIPELINE` (path to the model file to load) and `DEMAND_RAW_DIR` (raw-data folder) override the dashboard/extract defaults. `DEMAND_PYTHON` points the nightly `.ps1` at a specific interpreter/venv.

## Testing notes

- `pytest.ini` puts `src/` on `sys.path` so `import dashboard`, `from agent ...` resolve.
- Phases 1–3 are deterministic; parity tests (`test_phase2_parity`, `test_phase6_full_parity`) diff the agent's numbers against `dashboard.compute_view` with **exact-match** assertions (both call the same `fit_regression`). These are marked `slow` and skipped unless you pass `--runslow`.
- Phase 4 (LLM) tests mock the model; one API-key-gated live smoke test exists for manual use.
- Beyond the phase suites: `test_warehouse_extract` / `test_warehouse_reader` cover the regional pull and wide/long layout sniffing, `test_incremental_refresh` covers the `--incremental` SQL rewrite and snapshot merge, `test_region_all_view` covers the per-region rollup views, and `test_datawarehouse_integration` covers the demand extract end-to-end.
- **`test_rollup_ties.py` is the guard for the one-grain invariant.** Fast and fully synthetic (no fixture workbook, no fit), so the arithmetic that makes a view total equal the sum of its (SKU, customer) rows is checked directly: each customer's own POS/Orders signal surviving into the total, `Current Projection Average` summing across ragged plan coverage, `MIXED_SOURCE` labelling, `Weeks with data` counting a shared week once, and the column order the Excel exports depend on. If a change makes a KPI disagree with the table under it, this fails before anyone opens the app.
- **`test_perf_parity.py` is the guard for performance work.** Golden masters in `tests/fixtures/golden/*.parquet` pin the exact output of every model's `fit_regression`/`aggregate_to_sku_week`/`cleanse_series`, plus `compute_exceptions`, `compute_spikes`, `_descriptive_averages` and `compute_by_customer`, at `check_exact=True`. They are built from `tests/fixtures/make_perf_fixture.py` — a seeded ~3-year, 24-SKU frame with promo spikes, stockout dips, dead SKUs and late starters, so outlier cleansing, Holt-Winters seasonality (≥104 weeks) and gap densification all actually engage. The tiny Phase-2 fixture (9 weeks) reaches none of that. Both sides of each comparison are round-tripped through Parquet so serialisation can't masquerade as a numeric diff. Two functions that were optimised also keep their **original implementation** in the test file as an executable reference (`_reference_rolling`, `_descriptive_averages_reference`) and are checked against it on adversarial inputs — that is what caught the empty-POS/empty-Orders dtype case the goldens missed.
- If you optimise anything on the compute path, run `pytest tests/test_perf_parity.py` **before and after**. Regenerate goldens only with `REGEN_GOLDENS=1` and only when a number change is intended.
- Dashboard tests use Streamlit's `AppTest`; its `session_state` is a proxy without `.get()` — use `in` checks / subscripting.
