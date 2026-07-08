"""Provider-agnostic LLM access for the Phase 4 reasoning nodes.

Two providers, selected by the LLM_PROVIDER env var (or a per-call override):

- "anthropic" (default): Claude via langchain-anthropic. Needs ANTHROPIC_API_KEY.
- "local": any OpenAI-compatible server (LiteLLM / LM Studio / vLLM / Ollama),
  e.g. the gemma4-31b endpoint at http://james-workstation:4000/v1. Configured
  via LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL / LOCAL_LLM_API_KEY.

Nodes must only call safe_invoke() — it never raises, so a missing key, an
unreachable local server, or a bad provider name lands in state["errors"]
instead of crashing the graph after the (expensive) deterministic pipeline
already succeeded.
"""

import os
from typing import Optional

from agent import config


def _resolve_provider(provider: Optional[str] = None) -> str:
    # Read the env at call time (not import time) so run.py's --provider flag
    # and per-test monkeypatching take effect without re-importing config.
    return (provider or os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER).strip().lower()


def get_llm(temperature: float = 0, provider: Optional[str] = None):
    """Build a chat model for the configured provider.

    Both returned objects share the LangChain chat-model interface, so callers
    (safe_invoke) never care which provider is active.
    """
    resolved = _resolve_provider(provider)
    if resolved == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL, temperature=temperature, max_retries=2
        )
    if resolved == "local":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("LOCAL_LLM_MODEL", config.LOCAL_LLM_MODEL),
            base_url=os.environ.get("LOCAL_LLM_BASE_URL", config.LOCAL_LLM_BASE_URL),
            api_key=os.environ.get("LOCAL_LLM_API_KEY", config.LOCAL_LLM_API_KEY),
            temperature=temperature,
            max_retries=2,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER {resolved!r} — expected 'anthropic' or 'local'"
    )


def safe_invoke(prompt: str, temperature: float = 0, provider: Optional[str] = None):
    """Invoke the LLM; return (text, error). Never raises — a missing/invalid
    API key or a network failure must land in state["errors"], not crash the
    graph after the deterministic pipeline already succeeded."""
    try:
        return get_llm(temperature=temperature, provider=provider).invoke(prompt).content, None
    except Exception as e:  # noqa: BLE001 — anything here must degrade, not crash
        return None, f"LLM call failed (provider={_resolve_provider(provider)}): {e}"
