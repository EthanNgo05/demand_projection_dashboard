# Streamlit Demand Forecasting Dashboard

A Streamlit + Plotly dashboard for SKU-level demand forecasting. It can dynamically run either of two models:

- **Holt-Winters Exponential Smoothing** (damped trend)
- **XGBoost** (pooled gradient-boosted trees)

## Architecture

```
dashboard.py                      # Streamlit + Plotly front-end; dynamically runs either model
models/
├── exponential_smoothing.py      # Double exponential smoothing (level + trend, damped)
└── xgboost.py                    # Pooled gradient-boosted trees (XGBoost)
```

## Models

### `models/exponential_smoothing.py`

Double exponential smoothing (level + trend), with dampening so long-run slopes taper rather than run away.

$$
\text{level}_t = \alpha y_t + (1-\alpha)\left(\text{level}_{t-1}+\phi\,\text{trend}_{t-1}\right)
$$

$$
\text{trend}_t = \beta(\text{level}_t-\text{level}_{t-1}) +(1-\beta)\phi\,\text{trend}_{t-1}
$$

$$
\text{projected\_pos}(h) = \text{level}_T +(\phi+\phi^2+\cdots+\phi^h)\,\text{trend}_T, \qquad h=1,\ldots,15
$$

| Parameter | Range | Role |
|-----------|-------|------|
| `ALPHA` | 0–1 | Level smoothing — how fast the level tracks recent demand |
| `BETA`  | 0–1 | Trend smoothing — how fast the slope adapts (the ES analogue of the old `TREND_WEIGHT`; exposed as `TREND_WEIGHT` for dashboard compatibility) |
| `PHI`   | 0–1 | Trend damping — values < 1 flatten the trend the further out we forecast, so a short-run slope is not extrapolated indefinitely (`PHI = 1` → plain Holt; `PHI = 0` → flat at the level) |

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