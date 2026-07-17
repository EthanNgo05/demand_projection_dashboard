"""LangGraph-orchestrated agent for the demand projection pipeline.

Phase 1 (foundations): state schema + config only. Later phases add nodes,
the compiled graph, and the CLI entry point. Nothing in this package may
import streamlit — the agent must run headless.
"""
