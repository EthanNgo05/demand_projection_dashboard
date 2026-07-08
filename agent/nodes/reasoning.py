"""Phase 4 LLM reasoning nodes: anomaly flagging + narrative summaries.

Everything upstream (ingest, forecast, evaluate, select) stays deterministic —
these nodes only reason over numbers that are already computed; they never
touch the forecasting math. All LLM access goes through agent.llm.safe_invoke,
which is provider-agnostic (Claude API or a local OpenAI-compatible server)
and never raises: failures degrade into state["errors"].
"""

import numpy as np
import pandas as pd

from agent.llm import safe_invoke
from agent.logging_util import logger
from agent.state import AgentState

# Cap the anomaly-prompt table. ALL CUSTOMERS summaries can run to hundreds of
# SKU rows; we pre-filter in pandas to the largest absolute % swings instead of
# dumping the whole summary_df into the prompt.
MAX_ANOMALY_ROWS = 40

ANOMALY_PROMPT = """You are reviewing a SKU-level demand forecast for "{view}".
Below is the summary table (columns: SKU, Description, recent 8-week average,
projected next-15-week average, % change, weeks of history). It is pre-sorted
by absolute % change and capped at the top {max_rows} movers. Flag only genuine
outliers — large swings, sign flips (growth to decline or vice versa), or SKUs
where the forecast looks implausible given a short history. Do not comment on
normal variation.

{table}

Return a short bullet list (max 6 bullets). If nothing stands out, say so in one line."""

LOW_CONFIDENCE_PROMPT = """The demand forecast for "{view}" did not clear the
backtest accuracy threshold (best model: {best_model}, MAE {mae:.1f} vs a
threshold of {threshold}). Likely causes to check: short SKU history, a recent
promo/outlier week, or genuinely volatile demand. Write 2-3 sentences a
planner can act on — do not just restate the numbers."""

SUMMARY_PROMPT = """Write a 3-4 sentence executive summary of the demand
forecast for "{view}" using {best_model} (backtest MAE {mae:.1f}). Mention the
overall direction (growing/flat/declining), and call out the anomalies below
if any are notable. Plain language, no bullet points.

Anomalies flagged:
{anomalies}"""


def _recent_avg_column(summary_df: pd.DataFrame):
    """The recent-average column name varies by model: '8 Week POS/Orders
    Average' (regression), 'All-History POS/Orders Average' or
    '{N} Week POS/Orders Average' (ES/XGBoost). Match by suffix."""
    for col in summary_df.columns:
        if str(col).endswith("POS/Orders Average"):
            return col
    return None


def _anomaly_table(summary_df: pd.DataFrame, max_rows: int = MAX_ANOMALY_ROWS) -> str:
    """Compact, pre-filtered markdown table for the anomaly prompt.

    Computes % change in pandas (deterministic-in), sorts by absolute swing,
    and keeps only the top ``max_rows`` rows to respect the token budget.
    """
    recent_col = _recent_avg_column(summary_df)
    if recent_col is None or "Updated Projection Average" not in summary_df.columns:
        # Unknown summary schema — still cap the prompt, just don't rank.
        return summary_df.head(max_rows).to_markdown(index=False)
    recent = pd.to_numeric(summary_df[recent_col], errors="coerce")
    projected = pd.to_numeric(summary_df["Updated Projection Average"], errors="coerce")
    pct = (projected - recent) / recent.abs().replace(0, np.nan) * 100

    compact = pd.DataFrame(
        {
            "SKU": summary_df["SKU"],
            "Description": summary_df["Description"],
            "Recent 8wk avg": recent.round(1),
            "Projected 15wk avg": projected.round(1),
            "% change": pct.round(1),
        }
    )
    if "Weeks with data" in summary_df.columns:
        compact["Weeks with data"] = summary_df["Weeks with data"]

    order = pct.abs().sort_values(ascending=False, na_position="last").index
    return compact.reindex(order).head(max_rows).to_markdown(index=False)


def flag_anomalies(state: AgentState) -> dict:
    best = state["best_model"]
    summary = state["results"][best]["summary_df"]
    table = _anomaly_table(summary)
    text, err = safe_invoke(
        ANOMALY_PROMPT.format(view=state["view"], table=table, max_rows=MAX_ANOMALY_ROWS)
    )
    if err:
        logger.warning("flag_anomalies [%s]: %s", state.get("view", "?"), err)
        return {"anomalies": [], "errors": state.get("errors", []) + [err]}
    anomalies = [line.strip() for line in text.splitlines() if line.strip()]
    logger.info(
        "flag_anomalies [%s]: %d line(s) flagged", state.get("view", "?"), len(anomalies)
    )
    return {"anomalies": anomalies}


def flag_low_confidence(state: AgentState) -> dict:
    best = state.get("best_model")
    if best is None:
        # Phase 3's select routes here with best_model=None when NO model
        # produced a scoreable backtest (thin history). There's no MAE to
        # explain — don't call the LLM, just say so deterministically.
        return {
            "narrative": (
                f'No model produced a scoreable backtest for "{state["view"]}" — '
                "history is too short for a holdout. Review this view manually."
            )
        }
    text, err = safe_invoke(
        LOW_CONFIDENCE_PROMPT.format(
            view=state["view"],
            best_model=best,
            mae=state["results"][best]["mae"],
            threshold=state.get("mae_confidence_threshold", 50),
        )
    )
    if err:
        logger.warning("flag_low_confidence [%s]: %s", state.get("view", "?"), err)
        return {"narrative": None, "errors": state.get("errors", []) + [err]}
    return {"narrative": text}


def summarize(state: AgentState) -> dict:
    best = state["best_model"]
    text, err = safe_invoke(
        SUMMARY_PROMPT.format(
            view=state["view"],
            best_model=best,
            mae=state["results"][best]["mae"],
            anomalies="\n".join(state.get("anomalies", [])) or "None",
        )
    )
    if err:
        logger.warning("summarize [%s]: %s", state.get("view", "?"), err)
        return {"narrative": None, "errors": state.get("errors", []) + [err]}
    return {"narrative": text}
