"""Phase 4 LLM reasoning nodes: anomaly flagging + narrative summaries.

Everything upstream (ingest, forecast, evaluate, select) stays deterministic —
these nodes only reason over numbers that are already computed; they never
touch the forecasting math. All LLM access goes through agent.llm.safe_invoke,
which is provider-agnostic (Claude API or a local OpenAI-compatible server)
and never raises: failures degrade into state["errors"].
"""

import os
import re

import numpy as np
import pandas as pd

from agent import config
from agent.demand_profile import demand_profile
from agent.llm import safe_invoke
from agent.logging_util import logger
from agent.state import AgentState


def _skip_llm() -> bool:
    """True when LLM narrative/anomaly nodes should be skipped.

    Set ``AGENT_SKIP_LLM=1`` (the batch runner's ``--no-llm``) for a pure
    numeric refresh — every view still gets its best model + backtest MASEs, just
    no generated prose, avoiding one LLM call per view (~57× on a full batch)."""
    return bool(os.environ.get("AGENT_SKIP_LLM"))

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
backtest accuracy threshold (best model: {best_model}, MASE {mase:.2f} vs a
threshold of {threshold}; MASE < 1 beats a plain 8-week moving-average
forecast). Likely causes to check: short SKU history, a recent promo/outlier
week, or genuinely volatile demand.
{fit_block}
Fill the SUMMARY section with 2-3 sentences a planner can act on — do not just
restate the numbers."""

SUMMARY_PROMPT = """You are reviewing the demand forecast for "{view}", selected
model {best_model} (backtest MASE {mase:.2f}, vs a plain 8-week moving average;
< 1 beats that baseline).

Anomalies flagged:
{anomalies}
{fit_block}
Fill the SUMMARY section with a 3-4 sentence executive summary: the overall
direction (growing/flat/declining), calling out the anomalies above if any are
notable. Plain language, no bullet points."""

# Appended to both prompts above. It hands the LLM the deterministically-computed
# demand pattern (so its "expected best model" is grounded, not guessed) plus the
# per-model MASE table, and pins the response to three parseable sections. The
# EXPECTED judgment is explicitly asked for from the demand pattern ALONE, so a
# surprising MASE winner (e.g. XGBoost beating TSB on clearly intermittent demand)
# surfaces as a real observation rather than being rationalised after the fact.
MODEL_FIT_BLOCK = """
--- MODEL FIT ---
Demand pattern for this view (computed from history, not guessed):
{profile}

Backtest MASE by model (lower is better; < 1 beats a plain 8-week moving average;
"n/a" = the model produced no scoreable backtest):
{mase_table}

Selected model (lowest MASE): {best_model}

Considering ONLY the demand pattern above (ignore the MASE scores when forming
this expectation), which single model would you EXPECT to fit best? Guidance:
- "TSB (intermittent demand)" — built for intermittent/lumpy demand: many zero
  weeks, average demand interval (ADI) >= 1.32.
- "Holt-Winters (triple) exponential smoothing" — needs long history (~104+
  weeks) to fit annual seasonality.
- "Holt's (double) exponential smoothing" / "8-Week Moving Average" — smoother,
  steadier series with a trend and few zero weeks.
- "XGBoost" — pooled across SKUs; benefits from many SKUs sharing a pattern.
Pick exactly one of these labels: {labels}

Respond in EXACTLY this format — three sections, each starting with its label on
its own line, nothing before EXPECTED_MODEL:
EXPECTED_MODEL: <one label from the list above>
FIT_NOTE: <1-2 sentences naming your expected model; if it differs from the
selected model, note that the selected one won on backtest MASE>
SUMMARY: <see instruction below>"""

# Regex-anchored labels for parsing the three-section response back apart.
_FIT_LABEL_RE = re.compile(
    r"^\s*(EXPECTED_MODEL|FIT_NOTE|SUMMARY)\s*:\s*(.*)$", re.IGNORECASE
)


def _format_profile(prof: dict) -> str:
    """Render a demand_profile dict as a compact bullet block for the prompt."""
    def fmt(v, suffix=""):
        return "n/a" if v is None else f"{v}{suffix}"

    return (
        f"- pattern: {prof.get('pattern', 'unknown')}\n"
        f"- SKUs: {prof.get('sku_count', 0)}; weeks of history: "
        f"{prof.get('weeks_of_history', 0)}; total volume: {fmt(prof.get('total_volume'))}\n"
        f"- zero-demand weeks: {fmt(prof.get('pct_zero_weeks'), '%')}\n"
        f"- avg demand interval (ADI): {fmt(prof.get('avg_demand_interval'))} "
        f"(>= 1.32 = intermittent)\n"
        f"- demand-size lumpiness (CV²): {fmt(prof.get('cv2_demand_size'))} "
        f"(>= 0.49 = lumpy)"
    )


def _format_mase_table(results: dict) -> str:
    """One '- <label>: <mase>' line per model that ran, in MODEL_OPTIONS order."""
    lines = []
    for label in config.MODEL_OPTIONS:
        if label in results:
            m = results[label].get("mase")
            lines.append(f"- {label}: {'n/a' if m is None else format(float(m), '.2f')}")
    return "\n".join(lines) if lines else "- (no models scored)"


def _fit_block(state: AgentState) -> str:
    """The filled MODEL_FIT_BLOCK for ``state`` (demand profile + MASE table)."""
    prof = demand_profile(state["view"], state.get("cleaned_df"))
    return MODEL_FIT_BLOCK.format(
        profile=_format_profile(prof),
        mase_table=_format_mase_table(state.get("results", {})),
        best_model=state.get("best_model"),
        labels=", ".join(f'"{k}"' for k in config.MODEL_OPTIONS),
    )


def _match_model_label(raw):
    """Resolve the LLM's EXPECTED_MODEL text to an exact MODEL_OPTIONS label.

    Case/space-tolerant exact match first, then a substring fallback so a
    lightly-worded answer ("XGBoost model", "moving average") still lands on a
    known label. Returns None when nothing matches — the caller then publishes
    ``expected_best_model=None`` rather than a hallucinated label."""
    if not raw:
        return None
    norm = re.sub(r"\s+", " ", str(raw).strip().lower())
    labels = list(config.MODEL_OPTIONS)
    for label in labels:
        if re.sub(r"\s+", " ", label.lower()) == norm:
            return label
    for label in labels:
        lab = re.sub(r"\s+", " ", label.lower())
        if lab in norm or norm in lab:
            return label
    return None


def _parse_model_fit(text):
    """Split a three-section response into (narrative, expected_model, fit_note).

    Defensive, mirroring flag_anomalies' line parsing: if the labels are absent
    (a weak local model that ignored the format), the whole text becomes the
    narrative and the two new fields degrade to None — nothing regresses. An
    EXPECTED_MODEL that doesn't resolve to a known label also degrades to None.
    """
    if not text:
        return None, None, None
    sections: dict[str, list[str]] = {}
    current = None
    for line in str(text).splitlines():
        m = _FIT_LABEL_RE.match(line)
        if m:
            current = m.group(1).upper()
            sections[current] = [m.group(2)]
        elif current is not None:
            sections[current].append(line)

    if not sections:
        return str(text).strip(), None, None

    def joined(key):
        parts = sections.get(key)
        if not parts:
            return None
        val = "\n".join(parts).strip()
        return val or None

    narrative = joined("SUMMARY") or str(text).strip()
    expected = _match_model_label(joined("EXPECTED_MODEL"))
    note = joined("FIT_NOTE")
    return narrative, expected, note


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
    if _skip_llm():
        return {"anomalies": []}
    best = state["best_model"]
    summary = state["results"][best]["summary_df"]
    table = _anomaly_table(summary)
    text, err = safe_invoke(
        ANOMALY_PROMPT.format(view=state["view"], table=table, max_rows=MAX_ANOMALY_ROWS),
        view=state["view"],
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
        # produced a scoreable backtest (thin history). There's no MASE to
        # explain — don't call the LLM, just say so deterministically.
        return {
            "narrative": (
                f'No model produced a scoreable backtest for "{state["view"]}" — '
                "history is too short for a holdout. Review this view manually."
            )
        }
    if _skip_llm():
        return {"narrative": None, "expected_best_model": None, "model_fit_note": None}
    threshold = state.get("mase_confidence_threshold")
    if threshold is None:
        threshold = config.MASE_CONFIDENCE_THRESHOLD  # same fallback as select
    text, err = safe_invoke(
        LOW_CONFIDENCE_PROMPT.format(
            view=state["view"],
            best_model=best,
            mase=state["results"][best]["mase"],
            threshold=threshold,
            fit_block=_fit_block(state),
        ),
        view=state["view"],
    )
    if err:
        logger.warning("flag_low_confidence [%s]: %s", state.get("view", "?"), err)
        return {
            "narrative": None,
            "expected_best_model": None,
            "model_fit_note": None,
            "errors": state.get("errors", []) + [err],
        }
    narrative, expected, note = _parse_model_fit(text)
    return {"narrative": narrative, "expected_best_model": expected, "model_fit_note": note}


def summarize(state: AgentState) -> dict:
    if _skip_llm():
        return {"narrative": None, "expected_best_model": None, "model_fit_note": None}
    best = state["best_model"]
    text, err = safe_invoke(
        SUMMARY_PROMPT.format(
            view=state["view"],
            best_model=best,
            mase=state["results"][best]["mase"],
            anomalies="\n".join(state.get("anomalies", [])) or "None",
            fit_block=_fit_block(state),
        ),
        view=state["view"],
    )
    if err:
        logger.warning("summarize [%s]: %s", state.get("view", "?"), err)
        return {
            "narrative": None,
            "expected_best_model": None,
            "model_fit_note": None,
            "errors": state.get("errors", []) + [err],
        }
    narrative, expected, note = _parse_model_fit(text)
    return {"narrative": narrative, "expected_best_model": expected, "model_fit_note": note}
