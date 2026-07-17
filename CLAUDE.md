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

# Warehouse pulls (or the scheduled wrapper for all three nightly steps: ./refresh_demand_data.ps1)
python src/extract_demand_details.py          # ~10 min full 36-month pull -> dated all_demand_projections_<date>.xlsx + .parquet sidecar
python src/extract_demand_details.py --incremental   # fast: recent weeks + forward projections, merged into the latest snapshot
python src/extract_warehouse_projections.py   # ~2 min -> 5 regional <REGION>_warehouse_projections_<date>.xlsx

# Tests (~120 fast tests). The agent package lives under src/; pytest.ini puts src/ on sys.path.
pytest tests/ -v
pytest tests/test_phase3_select.py::test_name   # single test
pytest --runslow                                # include the 7 slow full-matrix parity tests

# Agent end-to-end for one view (run from src/ so `python -m agent.run` resolves the package)
cd src && python -m agent.run --view "All customers (combined)"
cd src && python -m agent.run --view "AMAZON-DC"

# Precompute every view's agent summary in parallel (what the nightly job runs)
cd src && python -m agent.batch               # flags: --workers N, --no-llm (skip narrative prose)
```

## Architecture

Two front-ends run over one shared forecasting core:

1. **`src/dashboard.py`** — Streamlit + Plotly UI. Loads the selected model **by file path** via `importlib` (chosen through the `DEMAND_PIPELINE` env var) and runs it live per Customer Grouping. Its "🔄 Refresh data" button spawns the extract scripts on demand (incremental demand pull + warehouse pull); its agent section reads the precomputed `outputs/agent_summary_<view>.json` files and can also run one view live.
2. **`src/agent/`** — a LangGraph pipeline (`ingest → run_all_models → evaluate_models → select_best_model →` conditional `→ flag_anomalies/summarize` or `flag_low_confidence` `→ publish`) that runs all the models, backtests to pick the best per view, uses an LLM to flag anomalies and write a narrative, and publishes `outputs/agent_summary_<view>.json` + logs. See `docs/agentic_workflow/` for the phased design (state schema in `00-overview.md`). `agent/config.py` mirrors `dashboard.py`'s `MODEL_OPTIONS` and `ALL_CUSTOMERS_VIEW` and must stay in sync with them.

Views offered by both front-ends: `All customers (combined)`, one `All Customers - <region>` rollup per region, and every individual Customer Grouping (`agent/batch.py`'s `enumerate_views` mirrors `dashboard.list_views` without importing streamlit).

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

- **Nightly job** (`refresh_demand_data.ps1`, registered with Windows Task Scheduler): full demand pull → warehouse pull (independent — runs even if the demand pull failed) → `agent.batch` precompute (only if the demand pull succeeded). The nightly demand pull is deliberately the **full 36-month pull** — the self-healing baseline that picks up restated actuals, item renames, and customer remaps; the dashboard's refresh button runs the fast `--incremental` pull instead. Worst exit code wins so Task Scheduler flags a failure in any step. Logs to `logs/<date>/logs_refresh.txt`.
- `sql/demand_details_optimized.sql` is the **default** query behind the demand extract (`DEFAULT_SQL` in `extract_demand_details.py`); `--incremental` only works against it (it rewrites a marker line in the batch to narrow the date window). The legacy `sql/demand_details.sql` is **UTF-16 encoded** (opens as garbled/spaced text in some tools — that's expected, per `.gitattributes`). Region "Others - <country>" buckets attach at `Custnmbr` grain via `MIN(Customer)` — don't drop them when touching the SQL.
- **Parquet sidecars**: the demand extract writes a `.parquet` sidecar next to each snapshot `.xlsx` (same basename). The `.xlsx` stays the source of truth; `data_io.read_raw_frame` prefers the sidecar when it's at least as new, else reads the xlsx and backfills the sidecar. Sidecar writes are best-effort (no pyarrow → logged and skipped). Snapshot pruning keeps the newest `KEEP_SNAPSHOTS` files (default 3, `DEMAND_KEEP_SNAPSHOTS` env var) and deletes each pruned snapshot's sidecar with it.
- Raw inputs live at the repo root under `raw_inputs/`: `demand_projections/all_demand_projections_<date>.xlsx` (written by the extract; PowerBI exports also work), `list_prices/list_prices_*.xlsx` (Plytix export — drives revenue-risk columns *and* the two data-quality checks below), and `warehouse_projections/<REGION>_*.xlsx` (normally written by `extract_warehouse_projections.py` from `sql/warehouse_projections.sql`; manual PowerBI exports also work — `data_io.warehouse_wide_to_long` sniffs whether a file is the legacy wide matrix or the long table layout, and for long files reconstructs the missing SKU×customer×week cells that drive the missing-projections table).
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
- Dashboard tests use Streamlit's `AppTest`; its `session_state` is a proxy without `.get()` — use `in` checks / subscripting.
