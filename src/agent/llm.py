"""Provider-agnostic LLM access for the Phase 4 reasoning nodes.

Two providers, selected by the LLM_PROVIDER env var (or a per-call override):

- "anthropic": Claude via langchain-anthropic. Needs ANTHROPIC_API_KEY.
- "local" (default): any OpenAI-compatible server (LiteLLM / LM Studio / vLLM / Ollama),
  e.g. the gemma4-31b endpoint at http://james-workstation:4000/v1. Configured
  via LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL / LOCAL_LLM_API_KEY.

Nodes must only call safe_invoke() — it never raises, so a missing key, an
unreachable local server, or a bad provider name lands in state["errors"]
instead of crashing the graph after the (expensive) deterministic pipeline
already succeeded.
"""

import os
from datetime import datetime
from typing import Optional

from agent import config
from log_config import dated_log_path

# Every prompt actually sent to an LLM is appended verbatim to the day's log,
# logs/<date>/llm_prompts.log, alongside the view it was generated for and the
# wall-clock time, so you can see exactly what left the machine. Kept in its own
# file — separate from app.log, the terse audit trail — because full prompts are
# multi-line and would drown the run log.
LLM_PROMPTS_FILENAME = "llm_prompts.log"


def _has_anthropic_credentials() -> bool:
    # ChatAnthropic resolves auth from any of these; without one it raises
    # "Could not resolve authentication method" the moment it's invoked.
    return any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    )


def _resolve_provider(provider: Optional[str] = None) -> str:
    # Read the env at call time (not import time) so run.py's --provider flag
    # and per-test monkeypatching take effect without re-importing config.
    resolved = (provider or os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER).strip().lower()
    # If Anthropic is selected but no key is configured, degrade to the local
    # server instead of failing every call with an auth error. This lets the
    # dashboard run out-of-the-box (e.g. the agent summary tab on startup)
    # without an ANTHROPIC_API_KEY in .env.
    if resolved == "anthropic" and not _has_anthropic_credentials():
        return "local"
    return resolved


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


def _resolve_model(resolved_provider: str) -> str:
    """The concrete model name for the resolved provider, matching get_llm."""
    if resolved_provider == "anthropic":
        return config.ANTHROPIC_MODEL
    return os.environ.get("LOCAL_LLM_MODEL", config.LOCAL_LLM_MODEL)


def _record_prompt(prompt: str, view: Optional[str], resolved_provider: str) -> None:
    """Append the exact prompt (with view + timestamp + target model) to
    the day's llm_prompts.log. Best-effort: a logging failure must never break a
    run, so a read-only filesystem or encoding error is swallowed silently."""
    entry = (
        f"{'=' * 80}\n"
        f"time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"view:     {view if view is not None else '(unspecified)'}\n"
        f"provider: {resolved_provider}\n"
        f"model:    {_resolve_model(resolved_provider)}\n"
        f"{'-' * 80}\n"
        f"{prompt}\n\n"
    )
    try:
        with open(dated_log_path(LLM_PROMPTS_FILENAME), "a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError:
        pass


def safe_invoke(
    prompt: str,
    temperature: float = 0,
    provider: Optional[str] = None,
    view: Optional[str] = None,
):
    """Invoke the LLM; return (text, error). Never raises — a missing/invalid
    API key or a network failure must land in state["errors"], not crash the
    graph after the deterministic pipeline already succeeded.

    Every prompt is recorded to the day's llm_prompts.log before the call, so
    the file reflects what was sent even if the call itself then fails."""
    resolved = _resolve_provider(provider)
    _record_prompt(prompt, view, resolved)
    try:
        return get_llm(temperature=temperature, provider=provider).invoke(prompt).content, None
    except Exception as e:  # noqa: BLE001 — anything here must degrade, not crash
        return None, f"LLM call failed (provider={resolved}): {e}"
