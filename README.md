# Streamlit Demand Forecasting Dashboard

A Streamlit + Plotly dashboard for SKU-level demand forecasting. It can dynamically run any of five models:

- **8-Week Moving Average** (8-week average + a light dampened trend)
- **Holt's (double) exponential smoothing** (level + damped trend, with outlier cleansing and promo uplift)
- **Holt-Winters (triple) exponential smoothing** (level + damped trend + additive annual seasonality)
- **XGBoost** (pooled gradient-boosted trees, with the same cleansing/uplift)
- **TSB (intermittent demand)** (Teunter–Syntetos–Babai — demand probability × size, for the many intermittent/lumpy SKUs)

Alongside the interactive dashboard, a **LangGraph agent** runs the whole pipeline
headlessly per view: it fits every model, backtests to pick the best, uses an LLM to
flag anomalies + write a narrative + reason about model fit, and publishes a JSON
summary the dashboard reads back. See [Agentic workflow](#agentic-workflow).

## Project layout

```
src/                              # all importable app code
├── dashboard.py                  # Streamlit + Plotly front-end; runs any model live
├── log_config.py                 # shared date-organized logging helpers
├── extract_demand_details.py     # nightly SQL-warehouse pull -> dated .xlsx
├── active_missing_projections.py # batch "active SKUs missing forecasts" report
├── agent/                        # LangGraph forecasting/reasoning pipeline
└── models/
    ├── regression.py             # 8-week average + dampened linear-regression slope
    ├── exponential_smoothing.py  # Double exponential smoothing (level + trend, damped)
    ├── holt_winters.py           # Triple exponential smoothing (+ additive annual seasonality)
    ├── xgboost.py                # Pooled gradient-boosted trees (XGBoost)
    └── tsb.py                    # TSB — intermittent/lumpy demand (probability × size)
tests/                            # pytest suite (adds src/ to sys.path)
notebooks/                        # EDA + the checks later ported into the dashboard
docs/agentic_workflow/            # Design notes
sql/                              # demand_details.sql (the warehouse query)
raw_inputs/
├── demand_projections/           # all_demand_projections_YYYY-MM-DD.xlsx (PowerBI export)
├── list_prices/                  # list_prices_*.xlsx (Plytix export: prices + statuses)
└── warehouse/                    # per-region warehouse projections
outputs/                          # batch-mode output (gitignored)
```

The data folders (`raw_inputs/`, `outputs/`, `logs/`, `sql/`, `docs/`,
`notebooks/`) stay at the repo root; only the Python code lives under `src/`.

## Running it

Dashboard (interactive):

```
pip install -r requirements.txt
streamlit run src/dashboard.py
```

Batch mode — each model file is also a standalone script that picks up the newest
raw file and writes per-group + combined Excel forecasts under `outputs/`. Run
these from the repo root so the `raw_inputs/` / `outputs/` paths resolve:

```
python src/models/exponential_smoothing.py     # or regression.py / holt_winters.py / xgboost.py
```

## The pipeline contract

`dashboard.py` loads the selected model **by file path** (`importlib`) and talks to
it through a small convention: each model file is deliberately **standalone and
self-contained** (shared constants like the customer groupings are repeated in
each file on purpose — a model file can be swapped in via the `DEMAND_PIPELINE`
env var without any package imports). A pipeline must expose:

- `week_anchors`, `aggregate_to_sku_week`, `fit_regression`, `region_for_group`
- constants such as `RAW_INPUTS_FOLDER`, `LIST_PRICE_GLOB`, `CUSTOMERS_TO_IGNORE`, `COMBINED_GROUPING`

The dashboard inspects `fit_regression`'s signature to decide which sidebar
controls to show: `alpha`/`beta`/`phi` args → smoothing sliders,
`min_weeks_for_trend` → min-weeks slider, `list_prices` → revenue-risk columns,
and an `autofit_smoothing` function → the Autofit button. **If you edit the
groupings or ignore lists, change all four model files identically.**

Environment overrides: `DEMAND_PIPELINE` (path to an extra/custom model file,
offered as the default) and `DEMAND_RAW_DIR` (raw-data folder).

## Agentic workflow

The `src/agent/` package is a [LangGraph](https://langchain-ai.github.io/langgraph/)
pipeline that runs the *same* forecasting core headlessly for one view, picks the best
model by backtest accuracy, and uses an LLM to explain the result. The dashboard's
agent section reads back the JSON it publishes (and can also run one view live).

### Target architecture

```
raw_inputs/*.xlsx (demand) ─┐
list_prices/*.xlsx (Plytix) ─┤
                             ▼
                    ┌─────────────────┐
                    │     ingest      │  discover files, load, clean, apply Plytix exclusions
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ run_all_models  │  fit all 5 models for the view
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ evaluate_models │  one shared walk-forward backtest → pooled MASE per model
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
              ┌─────┤ select_best_model├─────┐   winner = lowest MASE
              │     └─────────────────┘      │   (confidence_flag if MASE > threshold)
   confidence ok                      low confidence / no scoreable backtest
              ▼                               ▼
     ┌─────────────────┐            ┌────────────────────┐
     │ flag_anomalies  │  (LLM)     │ flag_low_confidence │  (LLM)
     └────────┬────────┘            └──────────┬─────────┘
              ▼                                 │   both LLM nodes also emit the
     ┌─────────────────┐                        │   expected-best-model reasoning
     │    summarize    │  (LLM)                  │
     └────────┬────────┘                        │
              └────────────────┬────────────────┘
                               ▼
                     ┌───────────────────┐
                     │      publish      │  write outputs/agent_summary_<view>.json + app.log
                     └───────────────────┘
```

Phases 1–3 (ingest → select) are fully deterministic and need no API key; only the
reasoning nodes call an LLM (Claude via `langchain-anthropic`, or any
OpenAI-compatible local server — set by `LLM_PROVIDER`). Every model is scored through
one shared walk-forward backtest and scaled by a plain 8-week-average baseline, giving
a **pooled MASE** that is comparable across models *and* views (< 1 beats the baseline;
the winner is the lowest MASE).

### Expected vs. actual best model

The winner is chosen purely by MASE, but that number says *which* model won, not *why*.
The LLM nodes additionally record the model they would **expect** to fit best from the
view's demand character and reconcile it against the actual winner — e.g. *"Web Sales-CA
has intermittent demand, so I'd expect TSB, but XGBoost scored the best MASE and is used
instead."* This is **grounded, not guessed**: `agent/demand_profile.py` deterministically
computes the [Syntetos–Boylan](https://en.wikipedia.org/wiki/Demand_forecasting)
demand-classification features and hands them to the LLM:

| Feature | Meaning | Model signal |
|---|---|---|
| `pct_zero_weeks` | intermittency | high → TSB territory |
| `avg_demand_interval` (ADI) | mean gap between demand weeks | ≥ 1.32 → intermittent/lumpy |
| `cv2_demand_size` | lumpiness of demand size (CV²) | ≥ 0.49 → erratic/lumpy |
| `weeks_of_history` | history length | < ~104 → Holt-Winters can't fit annual seasonality |
| `sku_count` | pooling scale | more SKUs favour pooled XGBoost |
| `pattern` | `smooth` / `intermittent` / `erratic` / `lumpy` quadrant | derived from ADI + CV² |

These are folded into the *same* LLM call (no extra call per view), which returns three
parseable sections (`EXPECTED_MODEL` / `FIT_NOTE` / `SUMMARY`). The published
`agent_summary_<view>.json` gains two fields — `expected_best_model` (a model label or
`null`) and `model_fit_note` (the reconciling sentence) — next to `best_model` and
`mase_by_model`; both are `null` on `--no-llm` runs.

### Running the agent

```
cd src                                        # the agent package lives under src/

# one view, end-to-end (prints the summary)
python -m agent.run --view "All customers (combined)"
python -m agent.run --view "AMAZON-DC"

# precompute every view's summary in parallel (what the nightly job runs)
python -m agent.batch                         # flags: --workers N, --no-llm (skip LLM prose)
```

Configure the LLM via `.env` (see `.env.example`): `LLM_PROVIDER=anthropic`
(needs `ANTHROPIC_API_KEY`) or `local` (an OpenAI-compatible endpoint). The forecasting
math is fully deterministic and needs no key.

## Models

### `models/regression.py`

Anchors each SKU to its 8-week average demand and nudges it by a dampened
linear-regression slope:

$$
\text{projected pos}(k) = \text{avg}_{8w} + \text{slope} \cdot \text{TREND\_WEIGHT} \cdot k, \qquad k=1,\ldots,15
$$

`TREND_WEIGHT = 0.25` (0 → pure average, 1 → pure trend). Always fits exactly
the last 8 completed weeks.

### `models/exponential_smoothing.py`

Double exponential smoothing (level + trend), with dampening so long-run slopes taper rather than run away.

$$
\text{level}_t = \alpha y_t + (1-\alpha)\left(\text{level}_{t-1}+\phi\,\text{trend}_{t-1}\right)
$$

$$
\text{trend}_t = \beta(\text{level}_t-\text{level}_{t-1}) +(1-\beta)\phi\,\text{trend}_{t-1}
$$

$$
\text{projected position}(h) = \text{level}_T +(\phi+\phi^2+\cdots+\phi^h)\,\text{trend}_T, \qquad h=1,\ldots,15
$$

| Parameter | Range | Role |
|-----------|-------|------|
| `ALPHA` | 0–1 | Level smoothing — how fast the level tracks recent demand |
| `BETA`  | 0–1 | Trend smoothing — how fast the slope adapts (the ES analogue of the old `TREND_WEIGHT`; exposed as `TREND_WEIGHT` for dashboard compatibility) |
| `PHI`   | 0–1 | Trend damping — values < 1 flatten the trend the further out we forecast, so a short-run slope is not extrapolated indefinitely (`PHI = 1` → plain Holt; `PHI = 0` → flat at the level) |

Extras: fits all completed history by default (`LOOKBACK_WEEKS = None`),
zero-fills gap weeks inside a SKU's active span (`FILL_GAPS_WITH_ZERO`),
cleanses promo spikes/stockout dips before fitting (`CLEANSE_OUTLIERS`,
`PROMO_WEEKS`), re-adds promo uplift onto future promo weeks (`PROMO_UPLIFT`),
and can grid-search α/β/φ by backtesting (`autofit_smoothing`, the dashboard's
Autofit button).

### `models/holt_winters.py`

Holt-Winters (triple exponential smoothing): the level + damped trend of the
Holt model **plus an additive seasonal component**, so a recurring annual pattern
(the Q4 holiday peak, summer lulls) is modelled explicitly rather than averaged
away. Only viable now that ~3 years of weekly history has accumulated — a seasonal
fit needs at least two full annual cycles.

$$
\text{projected position}(h) = \text{level}_T +(\phi+\phi^2+\cdots+\phi^h)\,\text{trend}_T + \text{season}_{T+h}, \qquad h=1,\ldots,15
$$

The fit is delegated to `statsmodels.tsa.holtwinters.ExponentialSmoothing`
(`seasonal="add"`, `seasonal_periods=52`, damped trend), which **optimises its own**
`α`/`β`/`γ`/`φ` by maximum likelihood — so there are **no smoothing sliders and no
Autofit button** for this model (its `fit_regression` carries no α/β/φ args and it
defines no `autofit_smoothing`, so the dashboard hides those controls, as it does
for XGBoost).

| Constant | Default | Role |
|----------|---------|------|
| `SEASONAL_PERIODS` | 52 | Length of one seasonal cycle in weeks (annual) |
| `MIN_WEEKS_FOR_SEASONAL` | 104 | Minimum history (2 full cycles) before a seasonal fit is attempted |

Additive (not multiplicative) seasonality is used deliberately — demand has many
zero / near-zero weeks, on which a multiplicative season degenerates. SKUs with
less than `MIN_WEEKS_FOR_SEASONAL` weeks of history (new/short-lived items), or
whose statsmodels fit fails to converge, **fall back** to the non-seasonal
damped-Holt forecast, so one awkward series never sinks a run. Like the Holt
model, the published forecast is flattened to the upcoming week's value (the app
re-runs weekly and only that value is used); seasonality adjusts it via that
week's seasonal index. Shares the Holt model's history window, gap zero-filling,
and promo/outlier cleansing.

### `models/xgboost.py`

Gradient-boosted trees (XGBoost), **pooled**: one model is trained per Customer Grouping across all of its SKUs, then each SKU is forecast 15 weeks ahead **recursively** (each predicted week is appended to history to build the next week's features).

```
features(SKU, week t) = [ lag_1 .. lag_6,
                          rolling means (4wk, 8wk),
                          weeks since first sale,
                          position in history,
                          week-of-year sin/cos ]

target(SKU, week t)   = demand at week t, scaled by the SKU's own mean
```

**Design notes:**

- **Pooling matters** because individual SKU histories are short (often 10–40 weeks) — too little to train a tree ensemble per SKU — while the pooled group easily has hundreds or thousands of rows.
- **Per-SKU mean scaling**: each SKU is scaled by its own mean before pooling, so big and small SKUs share one model without the big ones dominating the squared-error loss.
- **Fixed hyperparameters**: `XGB_PARAMS` (`n_estimators`, `learning_rate`, `max_depth`, etc.) are fixed module-level constants, not exposed to the dashboard — `fit_xgboost`'s signature carries no alpha/beta/phi, so the dashboard's smoothing sliders hide themselves automatically.
- **`MIN_TRAIN_ROWS`**: when the weeks available in data is less than this value, projection falls back to a flat-mean forecast
- **Fallback**: uses sklearn's `HistGradientBoostingRegressor` if the `xgboost` package isn't installed.

### `models/tsb.py`

TSB (Teunter–Syntetos–Babai) for **intermittent / lumpy demand** — the majority of SKUs
here, which sell in occasional bursts with many zero weeks. Classic level/trend models
chase those gaps to zero (or extrapolate a spurious slope); TSB instead tracks two
separately-smoothed quantities and multiplies them:

$$
\hat{y} = p_t \cdot z_t
$$

- $p_t$ — a smoothed **demand probability**, updated *every* week (so a SKU that stops
  selling decays toward 0 instead of freezing at its last value).
- $z_t$ — a smoothed **demand size**, updated *only* on non-zero weeks (so the size
  estimate isn't dragged down by the zeros).

The forecast is an intrinsically flat rate (probability × size), which is the right shape
for intermittent series. Smoothing constants `ALPHA_P` / `ALPHA_Z` are fixed module-level
constants — like XGBoost, `fit_tsb`'s signature carries no α/β/φ, so the dashboard hides
the smoothing sliders and Autofit. `FILL_GAPS_WITH_ZERO` must stay `True`: the zeros are
TSB's signal, not missing data.

## Data-quality checks (dashboard)

Ported from the notebooks and run automatically when a Plytix export is loaded:

- **Active-in check** — active products projected in a region they are not
  "Active in" are flagged, excluded from the forecast, and listed in their own table.
- **Discontinued check** — SKUs marked Discontinued/Inactive that still carry
  future projections are flagged and excluded.

Both need the Plytix `list_prices_*.xlsx` export (which also drives the
revenue-risk columns). Unhandled dashboard errors are logged to
`logs/<date>/app.log` (gitignored) with a friendly message shown to the user.

## Testing

Running the test suite (~150 fast tests):

```
pip install -r requirements.txt
pytest tests/ -v
pytest --runslow          # include the slow full-matrix parity tests
````

Run end-to-end and print a row count for all 3 models (the `agent` package lives
under `src/`, so run these from that folder — data paths still resolve to the
repo root automatically):

```
cd src
python -m agent.run --view "All customers (combined)"
python -m agent.run --view "AMAZON-DC"
python -m agent.run --view "ANOTHER-CUSTOMER-GROUP"
```
