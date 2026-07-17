"""Phase 4: LLM reasoning nodes + provider switching (Claude API vs local LLM).

LLM output isn't deterministic even at temperature=0 across model versions, so
these tests mock the LLM (see conftest.fake_llm) and assert on structure and
inputs, not exact text. Two gated live smoke tests at the bottom hit real
providers and are skipped unless the relevant env is present.
"""

import os
import re

import pytest

from agent import llm as llm_mod
from agent.nodes.reasoning import (
    MAX_ANOMALY_ROWS,
    flag_anomalies,
    flag_low_confidence,
    summarize,
)

# ---------------------------------------------------------------------------
# Reasoning nodes (mocked LLM — no key, no network)
# ---------------------------------------------------------------------------


def test_flag_anomalies_returns_bullets(fake_llm, sample_state_with_summary):
    fake_llm(
        ["- SKU-001 jumped 300% week over week\n- SKU-002 flipped from growth to decline"]
    )
    out = flag_anomalies(sample_state_with_summary)
    assert len(out["anomalies"]) == 2
    assert "errors" not in out


def test_flag_anomalies_input_is_capped(fake_llm, large_summary_state):
    # large_summary_state has 400+ SKU rows for ALL CUSTOMERS.
    model = fake_llm(["no anomalies"])
    flag_anomalies(large_summary_state)
    prompt = model.prompts[0]
    assert prompt.count("\n") < 100, "table was not pre-filtered before prompting"
    # The cap keeps the biggest movers: SKU-001 (x3) must survive the filter.
    assert "SKU-001" in prompt


def test_flag_anomalies_sorts_by_swing(fake_llm, large_summary_state):
    model = fake_llm(["no anomalies"])
    flag_anomalies(large_summary_state)
    table_lines = [l for l in model.prompts[0].splitlines() if l.startswith("|")]
    assert len(table_lines) - 2 <= MAX_ANOMALY_ROWS  # header + separator
    # Top data row is the largest absolute % change (SKU-001, +200%).
    assert "SKU-001" in table_lines[2]


def extract_sku_like_tokens(text):
    """Matches the fixture/workbook SKU format, e.g. SKU-001."""
    return set(re.findall(r"\bSKU-\d{3}\b", text))


def test_narrative_does_not_invent_skus(fake_llm, sample_state_with_summary):
    fake_llm(["SKU-001 is trending up sharply."])  # SKU-001 IS in the fixture
    out = flag_anomalies(sample_state_with_summary)
    best = sample_state_with_summary["best_model"]
    real_skus = set(
        sample_state_with_summary["results"][best]["summary_df"]["SKU"].astype(str)
    )
    mentioned = extract_sku_like_tokens("\n".join(out["anomalies"]))
    assert mentioned <= real_skus, f"hallucinated SKUs not in input: {mentioned - real_skus}"


def test_llm_failure_lands_in_errors_not_raised(monkeypatch, sample_state_with_summary):
    # A failing LLM call must be caught by safe_invoke, not crash the graph.
    def _boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("agent.llm.get_llm", _boom)
    out = flag_anomalies(sample_state_with_summary)  # should not raise
    assert out["anomalies"] == []
    assert out["errors"] and "LLM call failed" in out["errors"][0]


def test_flag_low_confidence_handles_no_best_model(monkeypatch):
    # best_model=None (nothing scoreable) must not KeyError and must not call the LLM.
    def _boom(*a, **k):
        raise AssertionError("LLM must not be called when best_model is None")

    monkeypatch.setattr("agent.llm.get_llm", _boom)
    out = flag_low_confidence({"view": "Some Group", "best_model": None, "results": {}})
    assert out["narrative"] and "manually" in out["narrative"]


def test_both_branches_produce_a_narrative(fake_llm, sample_state_with_summary):
    fake_llm(["confident narrative"])
    confident_out = summarize(sample_state_with_summary)

    low_state = dict(sample_state_with_summary, confidence_flag=True)
    fake_llm(["low confidence explanation"])
    low_conf_out = flag_low_confidence(low_state)

    assert confident_out["narrative"] == "confident narrative"
    assert low_conf_out["narrative"] == "low confidence explanation"


def test_summarize_feeds_anomalies_into_prompt(fake_llm, sample_state_with_summary):
    model = fake_llm(["summary text"])
    state = dict(sample_state_with_summary, anomalies=["- SKU-001 tripled"])
    summarize(state)
    assert "- SKU-001 tripled" in model.prompts[0]
    assert "1.23" in model.prompts[0]  # the backtest MASE reaches the prompt


def test_low_confidence_prompt_falls_back_to_config_threshold(
    fake_llm, sample_state_with_summary
):
    # No per-run threshold in state -> the prompt must carry the config value
    # (guards against reintroducing a hardcoded default that drifts from
    # select's fallback).
    from agent import config

    model = fake_llm(["low confidence explanation"])
    state = dict(sample_state_with_summary, confidence_flag=True)
    assert "mase_confidence_threshold" not in state
    flag_low_confidence(state)
    assert str(config.MASE_CONFIDENCE_THRESHOLD) in model.prompts[0]


# ---------------------------------------------------------------------------
# Provider switching (construction only — no network)
# ---------------------------------------------------------------------------


def test_get_llm_defaults_to_anthropic(monkeypatch):
    from langchain_anthropic import ChatAnthropic

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm_mod.config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    model = llm_mod.get_llm()
    assert isinstance(model, ChatAnthropic)


def test_get_llm_local_uses_openai_compatible_endpoint(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://james-workstation:4000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4-31b")
    model = llm_mod.get_llm()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gemma4-31b"
    assert "james-workstation:4000/v1" in str(model.openai_api_base)


def test_anthropic_without_key_falls_back_to_local(monkeypatch):
    from langchain_openai import ChatOpenAI

    # Anthropic selected but no credentials => degrade to the local server
    # instead of failing every call with an auth error.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert llm_mod._resolve_provider() == "local"
    assert isinstance(llm_mod.get_llm(), ChatOpenAI)


def test_anthropic_with_key_stays_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert llm_mod._resolve_provider() == "anthropic"


def test_provider_argument_overrides_env(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    model = llm_mod.get_llm(provider="local")
    assert isinstance(model, ChatOpenAI)


def test_unknown_provider_degrades_via_safe_invoke(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "carrier-pigeon")
    text, err = llm_mod.safe_invoke("hello")
    assert text is None
    assert "carrier-pigeon" in err


# ---------------------------------------------------------------------------
# Gated live smoke tests — run manually before shipping, not on every commit.
#   pytest tests/test_phase4_reasoning.py -k live -s
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="requires a live Anthropic API key"
)
def test_live_anthropic_smoke():
    text, err = llm_mod.safe_invoke("Reply with the single word OK.", provider="anthropic")
    assert err is None and text


@pytest.mark.skipif(
    not os.environ.get("RUN_LOCAL_LLM_SMOKE"),
    reason="set RUN_LOCAL_LLM_SMOKE=1 with the local server reachable",
)
def test_live_local_smoke():
    text, err = llm_mod.safe_invoke("Reply with the single word OK.", provider="local")
    assert err is None and text
