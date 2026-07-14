# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About simplehuman

[simplehuman](https://www.simplehuman.com) designs and sells premium home products — sensor trash cans, soap dispensers, sensor mirrors, dish racks, paper towel pumps, and related accessories. This repo is an internal supply-chain tool: it forecasts 15-week SKU-level demand so the planning team can spot where existing projections are too high or too low.

Scale of the data (per the latest warehouse snapshot):

- **~700 active SKUs**, after the SQL excludes samples, promos, credits, and other non-sellable item classes.
- **~57 raw customers** (`Custnmbr`), consolidated into a smaller set of **Customer Groupings** across 5 fulfillment regions: US (LBC+NJ), EU (SH-CTS), AU (ACR), CA (YYZ5), JP (NETDEPOT). Several raw customers fold into one group — e.g. AMAZON-DC / AMAZON-DS / MARVAL-FBM all become `AMAZON-DC`; Web + Warranty US become one group. Customers with no explicit mapping keep their own name as a single-member group, and region "Others - <country>" buckets catch the long tail.

## Common commands

Run everything **from the repo root** so the `raw_inputs/` / `outputs/` relative paths resolve.

```bash
pip install -r requirements.txt

# Interactive dashboard
streamlit run src/dashboard.py

# Batch forecast (each model file is also a standalone script; writes per-group + combined .xlsx to outputs/)
python src/models/exponential_smoothing.py   # or regression.py / xgboost.py

# Nightly warehouse pulls (or the scheduled wrapper for both: ./refresh_demand_data.ps1)
python src/extract_demand_details.py          # ~10 min -> dated all_demand_projections_<date>.xlsx
python src/extract_warehouse_projections.py   # ~2 min  -> 5 regional <REGION>_warehouse_projections_<date>.xlsx

# Tests (fast suite, 14 tests). The agent package lives under src/; pytest.ini puts src/ on sys.path.
pytest tests/ -v
pytest tests/test_phase3_select.py::test_name   # single test
pytest --runslow                                # include the slow full-matrix parity suites

# Agent end-to-end for one view (run from src/ so `python -m agent.run` resolves the package)
cd src && python -m agent.run --view "ALL CUSTOMERS (combined)"
cd src && python -m agent.run --view "AMAZON-DC"
```

## Architecture

Two front-ends run over one shared forecasting core:

1. **`src/dashboard.py`** — Streamlit + Plotly UI. Loads the selected model **by file path** via `importlib` (chosen through the `DEMAND_PIPELINE` env var) and runs it live per Customer Grouping.
2. **`src/agent/`** — a LangGraph pipeline (`ingest → run_all_models → evaluate_models → select_best_model →` conditional `→ flag_anomalies/summarize` or `flag_low_confidence` `→ publish`) that runs all three models, backtests to pick the best per view, uses an LLM to flag anomalies and write a narrative, and publishes to `outputs/` + logs. See `docs/agentic_workflow/` for the phased design (state schema in `00-overview.md`).

### The pipeline contract (most important thing to understand)

Each model file in `src/models/` is **deliberately standalone and self-contained** — shared constants (customer groupings, ignore lists) are **duplicated in every model file on purpose** so a model can be swapped in via `DEMAND_PIPELINE` with no package imports. Both the dashboard and the agent talk to a model through this convention:

- Functions: `week_anchors`, `aggregate_to_sku_week`, `fit_regression` (aliased per module — e.g. `fit_regression = fit_exponential_smoothing`), `region_for_group`
- Constants: `RAW_INPUTS_FOLDER`, `LIST_PRICE_GLOB`, `CUSTOMERS_TO_IGNORE`, `COMBINED_GROUPING`

The dashboard **introspects `fit_regression`'s signature** to decide which sidebar controls to show: `alpha`/`beta`/`phi` args → smoothing sliders; `min_weeks_for_trend` → min-weeks slider; `list_prices` → revenue-risk columns; an `autofit_smoothing` function → the Autofit button. This is why XGBoost's sliders hide automatically — its signature carries no smoothing params.

**⚠️ If you change the customer groupings or ignore lists, edit all three model files identically** (`regression.py`, `exponential_smoothing.py`, `xgboost.py`). `src/agent/data_io.py`'s `_clean` is the shared cleaning step and must stay in sync too (see the sync comment in `regression.py`'s `__main__`).

### The three models (`src/models/`)

- **`regression.py`** — 8-week moving average nudged by a dampened linear-regression slope (`TREND_WEIGHT = 0.25`). Labeled "8-Week Moving Average" in the UI.
- **`exponential_smoothing.py`** — Holt's double exponential smoothing (level + trend, damped by `PHI`). The only model with outlier cleansing, promo uplift, and an `autofit_smoothing` grid search.
- **`xgboost.py`** — gradient-boosted trees, **pooled per Customer Grouping** (SKU histories are too short to train per-SKU), each SKU scaled by its own mean, forecast 15 weeks recursively. Falls back to sklearn's `HistGradientBoostingRegressor` if `xgboost` isn't installed.

### Data flow & inputs

- `sql/demand_details.sql` is the warehouse query behind the nightly pull. It's **UTF-16 encoded** (opens as garbled/spaced text in some tools — that's expected, per `.gitattributes`). `demand_details_optimized.sql` is a work-in-progress optimized rewrite. Region "Others - <country>" buckets attach at `Custnmbr` grain via `MIN(Customer)` — don't drop them when optimizing.
- Raw inputs live at the repo root under `raw_inputs/`: `demand_projections/all_demand_projections_<date>.xlsx` (PowerBI export, the main POS/projection snapshot), `list_prices/list_prices_*.xlsx` (Plytix export — drives revenue-risk columns *and* the two data-quality checks below), and `warehouse_projections/<REGION>_*.xlsx` (normally written by `extract_warehouse_projections.py` from `sql/warehouse_projections.sql`; manual PowerBI exports also work — `data_io.warehouse_wide_to_long` sniffs whether a file is the legacy wide matrix or the long table layout, and for long files reconstructs the missing SKU×customer×week cells that drive the missing-projections table).
- **Data-quality checks** (dashboard, need the Plytix export): SKUs projected into a region they aren't "Active in", and Discontinued/Inactive SKUs still carrying projections — both flagged, excluded from the forecast, and listed in their own tables.
- Only Python code lives under `src/`. Data/log/doc folders (`raw_inputs/`, `outputs/`, `logs/`, `sql/`, `docs/`, `notebooks/`) stay at the repo root — `outputs/` and `logs/` are gitignored.

## Configuration (`.env`, see `.env.example`)

- `LLM_PROVIDER` = `anthropic` (needs `ANTHROPIC_API_KEY`, default model `claude-sonnet-5`) or `local` (Google Gemma Model). Only the agent's reasoning nodes call an LLM; forecasting math is fully deterministic and needs no key.
- SQL Server connection for the extract: `SQL_SERVER` and `SQL_DATABASE` are **required** (no hardcoded defaults). Blank `SQL_USER` → Windows trusted auth.
- `DEMAND_PIPELINE` (path to the model file to load) and `DEMAND_RAW_DIR` (raw-data folder) override the dashboard/extract defaults.

## Testing notes

- `pytest.ini` puts `src/` on `sys.path` so `import dashboard`, `from agent ...` resolve.
- Phases 1–3 are deterministic; parity tests (`test_phase2_parity`, `test_phase6_full_parity`) diff the agent's numbers against `dashboard.compute_view` with **exact-match** assertions (both call the same `fit_regression`). These are marked `slow` and skipped unless you pass `--runslow`.
- Phase 4 (LLM) tests mock the model; one API-key-gated live smoke test exists for manual use.
