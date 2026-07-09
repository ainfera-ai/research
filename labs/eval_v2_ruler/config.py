"""config.py — eval-v2 ruler configuration + guardrail constants.

Single place for ruler knobs. The harness is flag-gated OFF by default;
live scoring requires RULER_LIVE=true AND a gateway key.
"""

from __future__ import annotations

import os

# --- flag gates ------------------------------------------------------------

ENABLED_ENV = "RULER_ENABLED"
LIVE_ENV = "RULER_LIVE"


def ruler_enabled() -> bool:
    """True only when RULER_ENABLED is explicitly truthy."""
    return os.environ.get(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def live_enabled() -> bool:
    """True only when RULER_LIVE is explicitly truthy (live calls cost money)."""
    return os.environ.get(LIVE_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


# --- gateway ---------------------------------------------------------------

GATEWAY_BASE_URL = os.environ.get("AINFERA_BASE_URL", "https://api.ainfera.ai/v1")
GATEWAY_KEY_ENV = "AINFERA_FLEET_INFERENCE_KEY"
# The agent_id to use for scoring calls (pinned slugs write no routing_outcomes row).
AGENT_ID_ENV = "RULER_AGENT_ID"
DEFAULT_AGENT_ID = "5298a483-ce0f-479b-a6e9-f75054b08ad9"  # aule

CATALOG_URL = "https://api.ainfera.ai/v1/models"

# --- gates -----------------------------------------------------------------

# G1: tool-call validity. A model passes if ≥99% of tasks produce a
# well-formed tool call (valid JSON arguments, known tool name).
G1_THRESHOLD = float(os.environ.get("RULER_G1_THRESHOLD", "0.99"))

# G2: context coherence. A model passes if ≥95% of tasks call the RIGHT
# tool with semantically valid arguments (matching expected tool + args).
G2_THRESHOLD = float(os.environ.get("RULER_G2_THRESHOLD", "0.95"))

# G3: latency. p95 of first-tool-call latency must be < 8000ms.
# 8000ms accounts for Ainfera gateway routing overhead (~4-8s per call).
# Direct-to-provider calls may use 3000ms; gateway-routed calls need more.
G3_THRESHOLD_MS = float(os.environ.get("RULER_G3_THRESHOLD_MS", "8000"))

# G4: cost efficiency. Tokens per successful task. Lower is better.
# No hard pass/fail — ranked relative to the cohort. We set a soft ceiling
# at 2000 tokens/successful-task for a "pass" (most tool calls are <500 tokens).
G4_SOFT_CEILING = float(os.environ.get("RULER_G4_SOFT_CEILING", "2000"))

# --- cost guardrails -------------------------------------------------------

# Hard cost ceiling for one full ruler run, USD.
COST_CAP_USD = float(os.environ.get("RULER_COST_CAP_USD", "500"))

# Per-model-call token bound.
MAX_TOKENS = int(os.environ.get("RULER_MAX_TOKENS", "512"))

# Per-call timeout (seconds). Frontier models with tool-use can take 60s+.
CALL_TIMEOUT = int(os.environ.get("RULER_CALL_TIMEOUT", "120"))

# Retries on transient failures (502, timeout).
RETRIES = int(os.environ.get("RULER_RETRIES", "2"))
BACKOFF_BASE = float(os.environ.get("RULER_BACKOFF_BASE", "1.0"))

# Concurrency for parallel model scoring.
CONCURRENCY = int(os.environ.get("RULER_CONCURRENCY", "5"))

# --- output ----------------------------------------------------------------

OUTPUT_DIR = os.environ.get(
    "RULER_OUTPUT_DIR",
    "/Volumes/HFR WD_BLACK SN850X/code/ainfera-ai/research/labs/eval-v2-results",
)

DEFAULT_TASKSET = str(
    os.path.join(os.path.dirname(__file__), "fixtures", "tool_use_tasks.json")
)
