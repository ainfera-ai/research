"""eval_v2_ruler — the eval-v2 ruler harness for catalog model scoring.

Scores every catalog model on four admission gates (G1–G4) using a frozen
tool-use task set. The output is a CSV that feeds the 251→50 catalog cut.

Gates (vault: decisions/2026-07-06-eval-v2-ruler-scope.md):
  G1  tool-call validity ≥99%  — did the model emit a well-formed tool call?
  G2  context coherence        — did it call the RIGHT tool with valid args?
  G3  latency p95 < 3s         — first-token / first-tool-call latency
  G4  cost efficiency          — tokens per successful task (lower = better)

The harness calls models via the Ainfera gateway OpenAI-compat shim
(POST /v1/chat/completions) with tools[] — the router translates tools
through (AIN-347 retired the blanket 422; tools_dropped is always False).

No prod mutations. No routing_outcomes rows (pinned model slugs, not
ainfera-inference auto-route). Build phase only.
"""

from __future__ import annotations

__version__ = "0.1.0"
