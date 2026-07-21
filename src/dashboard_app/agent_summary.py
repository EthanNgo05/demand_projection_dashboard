"""Read + render the precomputed agent summary JSON for a view."""
import html
import time

import pandas as pd
import streamlit as st

from dashboard_app.config import model_display
from dashboard_app.compute import _load_agent_summary
from dashboard_app.summaries import _format_generated_at
from dashboard_app.refresh import start_agent_batch


# --------------------------------------------------------------------------- #
# Agent integration (Phase 5)                                                 #
# --------------------------------------------------------------------------- #
# The LangGraph agent runs out-of-band and writes its result to
# outputs/agent_summary_{view}.json (see agent/nodes/publish.py). The dashboard
# only reads that JSON back — it never threads LangGraph's execution model into
# Streamlit's rerun-on-every-interaction model. The agent is invoked strictly
# on an explicit button click (it calls an LLM and backtests every model), and
# the last result is shown from the cached JSON on subsequent reruns.

# Friendly sidebar label -> the LLM_PROVIDER value agent/llm.py resolves at call
# time. "anthropic" = Claude API (ANTHROPIC_API_KEY); "local" = the
# OpenAI-compatible server in LOCAL_LLM_* (see agent/config.py and .env.example).
LLM_PROVIDERS = {
    "Anthropic (Claude API)": "anthropic",
    "Local LLM": "local",
}


def _agent_scores(payload):
    """(scores_dict, is_mase) for an agent summary payload.

    Prefers the current ``mase_by_model`` key; falls back to the legacy
    ``mae_by_model`` written before the MASE migration so a stale JSON still
    renders (with the old MAE wording) until the nightly batch regenerates it.
    """
    if payload.get("mase_by_model") is not None:
        return payload["mase_by_model"], True
    return payload.get("mae_by_model") or {}, False


def _model_fit_callout(payload):
    """(kind, text) for the expected-vs-actual model-fit callout, or None.

    ``kind`` is "info" when the LLM's expected best model differs from the
    selected (MASE-winning) model — worth a prominent callout — and "caption"
    for the quieter agree/mismatch-less cases. Returns None when there's nothing
    to show (older JSONs written before these fields existed). Pure so the render
    branch is unit-testable without a Streamlit context.
    """
    expected = payload.get("expected_best_model")
    best = payload.get("best_model")
    note = payload.get("model_fit_note")
    if expected and best and expected != best:
        return "info", note or (
            f"Expected best fit: {model_display(expected)} — "
            f"{model_display(best)} won on backtest MASE."
        )
    if expected and note:
        return "caption", f"Expected best fit: {model_display(expected)} (matches the selected model). {note}"
    if note:
        return "caption", note
    return None


@st.dialog("Recommend a best model for every view?")
def _confirm_run_all_dialog(provider):
    """Confirmation modal for the all-views run (it can take up to an hour)."""
    st.write(
        "This backtests all models and recommends the most accurate one, plus an "
        "AI summary, for **every** view — the company total, each regional "
        "rollup, and every customer group (~60 views)."
    )
    st.warning(
        "It runs in the background and can take **up to 1 hour**. You can keep "
        "using the dashboard while it runs; each view's recommendation updates as "
        "it finishes, and the **Optimized Projections** view fills in as "
        "groups complete."
    )
    left, right = st.columns(2)
    if left.button("Cancel", key="confirm_batch_cancel", width="stretch"):
        st.rerun()
    if right.button("Recommend all views", key="confirm_batch_go",
                    type="primary", width="stretch"):
        ok, msg = start_agent_batch(provider)
        if ok:
            st.session_state["_batch_toast"] = (
                "Started — recommending the best model for every view. "
                "This can take up to an hour."
            )
        else:
            st.session_state["_batch_toast"] = f"⚠️ {msg}"
        st.rerun()


# Progress markers for the agent run. Keys are LangGraph node names (see
# agent/graph.py); each maps to (fraction_complete, user-facing label). graph
# .stream() yields one update per node as it finishes, so we bump the bar to the
# node's fraction when its update arrives. evaluate_models is the long pole (4
# models x 6 walk-forward re-fits), hence the big jump to 0.75.
_AGENT_NODE_PROGRESS = {
    "ingest": (0.15, "Loading & cleaning data…"),
    "run_all_models": (0.40, "Fitting the forecast models…"),
    "evaluate_models": (0.75, "Backtesting models (walk-forward) to compare accuracy…"),
    "select_best_model": (0.80, "Selecting the best model…"),
    "flag_anomalies": (0.88, "Flagging anomalies…"),
    "summarize": (0.95, "Writing the summary…"),
    "flag_low_confidence": (0.95, "Writing the low-confidence note…"),
    "publish": (1.0, "Publishing results…"),
}


def _run_agent_job(view, today_ts, shared):
    """Run the agent pipeline on a background thread, streaming progress.

    Runs OFF the main Streamlit script thread so the UI stays interactive. It
    must NOT touch any ``st.*`` API — it only mutates the plain ``shared`` dict
    (created on the main thread, polled by the progress fragment). LLM_PROVIDER
    is read from the env at call time by agent/llm.py, so the caller sets it
    before starting this thread.
    """
    # Per-model progress from inside the fit/backtest node loops. The nodes call
    # this via RunnableConfig (see agent/state.report_progress); it maps each
    # phase to its slice of the bar so the user sees e.g. "Fitting XGBoost (3/4)".
    def _cb(phase, model, done, total):
        total = max(int(total), 1)
        if phase == "fit":  # fit loop occupies 0.15 -> 0.40 of the bar
            shared["progress"] = 0.15 + 0.25 * (done / total)
            shared["step"] = f"Fitting {model} ({done}/{total})"
        elif phase == "backtest":  # backtest loop occupies 0.40 -> 0.75
            shared["progress"] = 0.40 + 0.35 * (done / total)
            shared["step"] = f"Backtesting {model} ({done}/{total})"

    try:
        # Import here, not at module top: keeps langgraph off the hot import
        # path for every rerun and matches the "only touched on click" rule.
        from agent.graph import build_graph

        graph = build_graph()
        # stream_mode="updates" (the default) yields {node_name: state_delta}
        # after each node finishes; accumulate the deltas so best_model/errors
        # are available when the run ends. The progress_cb (passed via config)
        # supplies finer per-model updates from inside the fit/backtest nodes.
        config = {"configurable": {"progress_cb": _cb}}
        for update in graph.stream({"view": view, "today_ts": today_ts}, config=config):
            for node_name, delta in update.items():
                frac, label = _AGENT_NODE_PROGRESS.get(
                    node_name, (shared.get("progress", 0.0), "Working…")
                )
                shared["progress"] = frac
                shared["step"] = label
                if isinstance(delta, dict):
                    shared["result"].update(delta)
        shared["status"] = "done"
    except Exception as exc:  # surface any failure to the UI instead of a dead spinner
        shared["error"] = f"{type(exc).__name__}: {exc}"
        shared["status"] = "error"


@st.fragment(run_every=0.5)
def _agent_progress_fragment():
    """Poll the background agent job and render its progress bar.

    Only THIS fragment reruns on the 0.5s timer — the rest of the page stays
    interactive while the pipeline runs on its background thread. When the job
    finishes, trigger one full app rerun so main() can finalize (toast, switch
    the model toggle, render the summary).
    """
    job = st.session_state.get("agent_job") or {}
    started = job.get("started_at")
    elapsed_txt = ""
    if started:
        secs = int(time.time() - started)
        elapsed_txt = f"  ·  {secs // 60}:{secs % 60:02d} elapsed"
    st.progress(
        min(float(job.get("progress", 0.0)), 1.0),
        text=f"Analyzing models — {job.get('step', 'Working…')}{elapsed_txt}",
    )
    if started:
        st.caption(f"Started at {time.strftime('%H:%M:%S', time.localtime(started))}")
    if job.get("status") in ("done", "error"):
        st.rerun(scope="app")


def _render_agent_summary(view):
    """Render the cached agent summary for `view` in the main body, if any."""
    payload = _load_agent_summary(view)
    if payload is None:
        return
    with st.expander("Model recommendation", expanded=True):
        gen = payload.get("generated_at")
        if gen:
            st.caption(f"Generated {_format_generated_at(gen)}  ·  view: {payload.get('view', view)}")

        if payload.get("errors"):
            st.error("\n".join(payload["errors"]))

        scores, is_mase = _agent_scores(payload)

        best = payload.get("best_model")
        if best:
            score = scores.get(best)
            label = f"Best model: {model_display(best)}"
            if score is not None:
                label += (
                    f"  (backtest MASE {score:.2f})" if is_mase
                    else f"  (backtest MAE {score:.1f})"
                )
            if payload.get("confidence_flag"):
                st.warning(label + "  —  ⚠️ low confidence")
            else:
                st.success(label)

        # Expected vs. actual best model: the LLM's a-priori pick from the view's
        # demand pattern, reconciled against the MASE winner. Guarded so older
        # summary JSONs (written before these fields existed) render as before.
        callout = _model_fit_callout(payload)
        if callout is not None:
            kind, text = callout
            (st.info if kind == "info" else st.caption)(text)

        # All models' backtest scores side by side, so the user can see how
        # close the call was. Scores come straight from publish.py; a model
        # whose backtest failed has None -> shown as "n/a", sorted last.
        if scores:
            col = "Backtest MASE (vs 8-wk avg)" if is_mase else "Backtest MAE"
            rows = [
                {
                    "Model": model_display(name),
                    col: (
                        "n/a" if score is None
                        else round(float(score), 2 if is_mase else 1)
                    ),
                    "Best": "✓" if name == best else "",
                    "_sort": (float("inf") if score is None else float(score)),
                }
                for name, score in scores.items()
            ]
            score_df = (
                pd.DataFrame(rows)
                .sort_values("_sort")
                .drop(columns="_sort")
                .reset_index(drop=True)
            )
            st.markdown("**Model comparison:**")
            st.dataframe(score_df, hide_index=True)
            if is_mase:
                st.caption(
                    "Backtest MASE from walk-forward (one-step-ahead) validation: "
                    "model error ÷ a plain 8-week moving average's error on the "
                    "same weeks. < 1 beats the 8-week average; lower = better; "
                    "winner chosen by lowest MASE."
                )
            else:
                st.caption(
                    "Backtest MAE from walk-forward (one-step-ahead) validation — "
                    "lower = closer fit; winner chosen by lowest MAE."
                )

        if payload.get("narrative"):
            st.write(payload["narrative"])

        anomalies = payload.get("anomalies") or []
        if anomalies:
            st.markdown("**Flagged anomalies:**")
            for a in anomalies:
                # publish stores bullets as-is; add a marker only if missing.
                st.markdown(a if a.lstrip().startswith(("-", "*", "•")) else f"- {a}")

        # Active SKUs the winning model leaves out because their demand predates
        # its history window (e.g. the 8-week moving average). Surfaced so a SKU
        # that an all-history model (Holt/XGBoost) would forecast isn't silently
        # dropped without explanation. Empty for an all-history winner.
        excluded = payload.get("window_excluded_skus") or []
        if excluded:
            best_lbl = model_display(payload.get("best_model")) or "this model"
            # Rendered as a native HTML <details> dropdown (collapsed by
            # default) so this list — often 15+ SKUs — doesn't dominate the
            # summary. A Streamlit st.expander can't be used here: it's illegal
            # to nest one inside the "Agent summary" expander this runs in.
            items = "".join(
                "<li>{}{}</li>".format(
                    html.escape(str(row.get("SKU", ""))),
                    " — " + html.escape(str(row.get("Description", "")))
                    if row.get("Description")
                    else "",
                )
                for row in excluded
            )
            st.markdown(
                "<details style='margin:0.25rem 0 0.5rem;'>"
                "<summary style='cursor:pointer;font-weight:600;'>"
                f"Active SKUs outside {html.escape(best_lbl)}'s history window "
                f"({len(excluded)})</summary>"
                "<div style='opacity:0.75;font-size:0.9em;margin:0.35rem 0;'>"
                "These have demand history but none inside the model's window, "
                "so they carry no projection here. Switch to an all-history "
                "model (Holt or XGBoost) to forecast them.</div>"
                f"<ul style='margin:0;padding-left:1.2rem;'>{items}</ul>"
                "</details>",
                unsafe_allow_html=True,
            )
