"""CLI entry point: python -m agent.run --view "ALL CUSTOMERS (combined)"

Placeholder in Phase 1 — wired up in Phase 2 once the deterministic
pipeline nodes exist.
"""

import sys


def main() -> int:
    print(
        "agent.run is not available yet: the pipeline nodes are built in "
        "Phase 2 (docs/agentic_workflow/02-deterministic-pipeline-nodes.md).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
