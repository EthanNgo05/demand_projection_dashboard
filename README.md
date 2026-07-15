# Streamlit Demand Forecasting Dashboard

A Streamlit + Plotly dashboard for SKU-level demand forecasting. It can dynamically run any of three models:

- **8-Week Moving Average** (8-week average + a light dampened trend)
- **Holt's Exponential Smoothing** (damped trend, with outlier cleansing and promo uplift)
- **XGBoost** (pooled gradient-boosted trees, with the same cleansing/uplift)

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
    └── xgboost.py                # Pooled gradient-boosted trees (XGBoost)
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

Runnng the 14 tests:

```
pip install -r requirements.txt
pytest tests/ -v
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
