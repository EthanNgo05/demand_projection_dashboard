"""Expected-vs-actual model-fit reasoning: the three-section parser, the two
reasoning nodes that populate the new fields, publish persistence, and the
dashboard callout branch.

LLM output is mocked (conftest.fake_llm) so these assert structure and parsing,
never exact prose.
"""

import json

import pytest

from agent.nodes.reasoning import (
    _match_model_label,
    _parse_model_fit,
    flag_low_confidence,
    summarize,
)

# A well-formed three-section response the LLM is asked to produce.
GOOD_RESPONSE = """EXPECTED_MODEL: TSB (intermittent demand)
FIT_NOTE: Demand here is intermittent, so I'd expect TSB, but XGBoost scored the
best MASE and is used instead.
SUMMARY: Demand is broadly flat with a few lumpy SKUs. Nothing alarming."""


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_parse_three_sections():
    narrative, expected, note = _parse_model_fit(GOOD_RESPONSE)
    assert expected == "TSB (intermittent demand)"
    assert "intermittent" in note and "XGBoost" in note
    assert narrative.startswith("Demand is broadly flat")
    # The label/note lines must not bleed into the narrative.
    assert "EXPECTED_MODEL" not in narrative
    assert "FIT_NOTE" not in narrative


def test_parse_unlabelled_text_degrades_to_narrative():
    """A weak model that ignores the format: whole text is the narrative, new
    fields None — nothing regresses vs. the old free-text behaviour."""
    narrative, expected, note = _parse_model_fit("Just a plain paragraph of prose.")
    assert narrative == "Just a plain paragraph of prose."
    assert expected is None
    assert note is None


def test_parse_invalid_expected_label_is_dropped():
    resp = "EXPECTED_MODEL: SomeModelThatDoesNotExist\nSUMMARY: fine."
    narrative, expected, note = _parse_model_fit(resp)
    assert expected is None
    assert narrative == "fine."


def test_parse_missing_summary_falls_back_to_full_text():
    """No SUMMARY section -> narrative falls back to the whole text so it's never
    lost, while the expected model still parses."""
    resp = "EXPECTED_MODEL: XGBoost\nFIT_NOTE: XGBoost, pooled across many SKUs."
    narrative, expected, note = _parse_model_fit(resp)
    assert expected == "XGBoost"
    assert note.startswith("XGBoost")
    assert narrative  # non-empty fallback


def test_parse_empty_text():
    assert _parse_model_fit("") == (None, None, None)


def test_match_model_label_tolerates_wording():
    assert _match_model_label("XGBoost") == "XGBoost"
    assert _match_model_label("  xgboost  ") == "XGBoost"
    assert _match_model_label("XGBoost model") == "XGBoost"  # substring fallback
    assert _match_model_label("TSB (intermittent demand)") == "TSB (intermittent demand)"
    assert _match_model_label("carrier pigeon") is None
    assert _match_model_label(None) is None


# --------------------------------------------------------------------------
# Reasoning nodes populate the new state fields
# --------------------------------------------------------------------------


def test_summarize_populates_expected_and_note(fake_llm, sample_state_with_summary):
    fake_llm([GOOD_RESPONSE])
    out = summarize(sample_state_with_summary)
    assert out["expected_best_model"] == "TSB (intermittent demand)"
    assert "XGBoost" in out["model_fit_note"]
    assert out["narrative"].startswith("Demand is broadly flat")


def test_low_confidence_populates_expected_and_note(fake_llm, sample_state_with_summary):
    state = dict(sample_state_with_summary, confidence_flag=True)
    fake_llm([GOOD_RESPONSE])
    out = flag_low_confidence(state)
    assert out["expected_best_model"] == "TSB (intermittent demand)"
    assert out["model_fit_note"]
    assert out["narrative"]


def test_fit_block_reaches_prompt(fake_llm, sample_state_with_summary):
    """The demand-pattern block + selectable labels must be in the prompt so the
    LLM's expectation is grounded."""
    model = fake_llm([GOOD_RESPONSE])
    summarize(sample_state_with_summary)
    prompt = model.prompts[0]
    assert "MODEL FIT" in prompt
    assert "EXPECTED_MODEL:" in prompt
    assert "avg demand interval" in prompt
    assert "TSB (intermittent demand)" in prompt  # a selectable label listed


def test_llm_failure_nulls_new_fields(monkeypatch, sample_state_with_summary):
    def _boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("agent.llm.get_llm", _boom)
    out = summarize(sample_state_with_summary)
    assert out["narrative"] is None
    assert out["expected_best_model"] is None
    assert out["model_fit_note"] is None
    assert out["errors"]


def test_skip_llm_nulls_new_fields(monkeypatch, sample_state_with_summary):
    monkeypatch.setenv("AGENT_SKIP_LLM", "1")
    out = summarize(sample_state_with_summary)
    assert out == {"narrative": None, "expected_best_model": None, "model_fit_note": None}


# --------------------------------------------------------------------------
# Publish persists the new fields
# --------------------------------------------------------------------------


def test_publish_writes_expected_and_note(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    from agent.nodes.publish import publish

    publish({
        "view": "Web Sales-CA",
        "best_model": "XGBoost",
        "results": {"XGBoost": {"mase": 0.72}, "TSB (intermittent demand)": {"mase": 0.95}},
        "expected_best_model": "TSB (intermittent demand)",
        "model_fit_note": "Intermittent demand — expected TSB, but XGBoost won on MASE.",
        "narrative": "Flat.",
        "anomalies": [],
        "confidence_flag": False,
        "errors": [],
    })
    payload = json.loads((tmp_path / "agent_summary_Web_Sales-CA.json").read_text())
    assert payload["expected_best_model"] == "TSB (intermittent demand)"
    assert "XGBoost" in payload["model_fit_note"]


def test_publish_defaults_new_fields_to_none(tmp_path, monkeypatch):
    """A --no-llm run leaves the fields unset; publish writes them as null, not a
    KeyError."""
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    from agent.nodes.publish import publish

    publish({"view": "V", "best_model": "XGBoost", "results": {"XGBoost": {"mase": 1.0}}})
    payload = json.loads((tmp_path / "agent_summary_V.json").read_text())
    assert payload["expected_best_model"] is None
    assert payload["model_fit_note"] is None


# --------------------------------------------------------------------------
# Dashboard callout branch (pure helper)
# --------------------------------------------------------------------------


def test_dashboard_callout_info_on_mismatch():
    pytest.importorskip("streamlit")
    import dashboard

    kind, text = dashboard._model_fit_callout({
        "best_model": "XGBoost",
        "expected_best_model": "TSB (intermittent demand)",
        "model_fit_note": "Expected TSB, XGBoost won on MASE.",
    })
    assert kind == "info"
    assert "XGBoost" in text


def test_dashboard_callout_caption_on_agreement():
    pytest.importorskip("streamlit")
    import dashboard

    kind, text = dashboard._model_fit_callout({
        "best_model": "TSB (intermittent demand)",
        "expected_best_model": "TSB (intermittent demand)",
        "model_fit_note": "Intermittent demand fits TSB.",
    })
    assert kind == "caption"
    assert "matches" in text


def test_dashboard_callout_none_for_legacy_payload():
    pytest.importorskip("streamlit")
    import dashboard

    # An older summary JSON without the fields -> nothing to render.
    assert dashboard._model_fit_callout({"best_model": "XGBoost"}) is None
