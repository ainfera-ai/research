"""config.py — eval-harness configuration + guardrail constants.

Single place for the proof-pipeline eval (AIN-458 · S1 scaffold) knobs and the
hard invariants the harness must never violate. Everything here is INERT in
Stage 1: the harness is flag-gated OFF and makes no live model calls.

Guardrails encoded here (canon: ainfera-vault
decisions/locks-2026-06-15-proof-pipeline.md):

  L1  Arm pins live ONLY in the harness — never in the production router. The
      concrete A/B/D model identifiers are env-injected on the Labs tenant and
      are NOT committed to this public repo (placeholder defaults below).
  L2  Eval runs on the Spark LABS tenant only; every eval call is excluded from
      `routing_outcomes` and from training. See EXCLUDE_* below.
  L3  Judge (Gemini 3.1 Pro) is held out of all four arms. assert_judge_held_out
      (arms.py) enforces it; JUDGE_MODEL must never appear as an arm model.
  L4  Names / counts / codes only. No secret values in code or artifacts.
"""

from __future__ import annotations

import os

# --- flag gate -------------------------------------------------------------

# Whole-harness kill switch. Default OFF (Stage 1 ships INERT). Stage 2 wiring
# (AIN-459) flips this on the Labs tenant only.
ENABLED_ENV = "LABS_EVAL_HARNESS_ENABLED"


def harness_enabled() -> bool:
    """True only when LABS_EVAL_HARNESS_ENABLED is explicitly truthy."""
    return os.environ.get(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


# --- tenant + exclusion invariants (L2) ------------------------------------

# Eval traffic lives on this tenant and nowhere else.
LABS_TENANT = os.environ.get("LABS_EVAL_TENANT", "labs")

# Every eval call is fenced out of the moat dataset + training corpus. These are
# contract constants, not toggles — the runner asserts them on every call.
EXCLUDE_FROM_ROUTING_OUTCOMES = True
EXCLUDE_FROM_TRAINING = True

# Tag stamped on every eval call so a downstream sink can filter it out even if
# it somehow reaches a shared pipe (defence in depth for L2).
EVAL_RUN_TAG = "proof_eval_labs"


# --- judge (L3) ------------------------------------------------------------

# Held-out judge. Named (methodology is public — judge_worker.py already names
# its labeler openly); the judge PROMPT stays closed (it is not in this repo).
JUDGE_MODEL = os.environ.get("LABS_EVAL_JUDGE_MODEL", "gemini-3-1-pro")

# Fusion panel (arm D) internal synthesizer. A DISTINCT role from the judge —
# it composes the panel's answer; it never scores outcomes.
FUSION_SYNTHESIZER = os.environ.get("LABS_EVAL_FUSION_SYNTH", "claude-opus-4-8")

# Fraction of judge labels queued for human spot-check (calibration).
HUMAN_SPOTCHECK_PCT = float(os.environ.get("LABS_EVAL_HUMAN_SPOTCHECK_PCT", "0.10"))


# --- pre-registered test parameters (eval spec) ----------------------------

# Success-rate tolerance band, in percentage points. C may trail A by at most
# this and still "not lose" on success.
EPSILON_PT = float(os.environ.get("LABS_EVAL_EPSILON_PT", "1.0"))

# Judge score (1-5) at/above which an outcome counts as a success.
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("LABS_EVAL_SUCCESS_SCORE", "3.0"))

# The arm whose per-type measured success rate sets the floor (premium pin).
FLOOR_ARM = "A"

# Held-out task-set sizing (Stage 2 enforcement; Stage 1 fixtures are smaller).
MIN_TASKS_PER_TYPE = int(os.environ.get("LABS_EVAL_MIN_TASKS_PER_TYPE", "200"))

# Deterministic seed for any sampling/spot-check selection (CRN, repo-wide idiom).
CRN_SEED = int(os.environ.get("LABS_CRN_SEED", "20260528"))


# === Stage 2 live wiring (AIN-459) — INERT until the founder enables ========
# A SEPARATE, stronger gate than ENABLED_ENV: live model calls cost money and
# need the Labs key. Default OFF. The cycle refuses a real run unless this is on
# AND the probe-agent exclusion is configured AND the key is present.
LIVE_ENV = "LABS_EVAL_LIVE"


def live_enabled() -> bool:
    """True only when LABS_EVAL_LIVE is explicitly truthy (default OFF)."""
    return os.environ.get(LIVE_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


# Gateway — names only; values are Doppler-rendered on the Labs tenant (L4).
GATEWAY_BASE_URL_ENV = "AINFERA_BASE_URL"   # default https://api.ainfera.ai/v1
GATEWAY_KEY_ENV = "AINFERA_API_KEY"          # Labs-tenant key (Doppler)

# Arm C uses model="ainfera-inference", which writes a routing_outcomes row. To
# keep it OUT of the moat (L2) it MUST run as a probe agent whose id is
# registered server-side in _PROBE_AGENT_IDS_DEFAULT (→ traffic_class
# 'internal_probe', excluded from training). The pinned arms (A/B/D members +
# synthesizer) write NO routing_outcomes row at all, so they are excluded by
# construction. Registering the probe-agent id is a founder tap.
PROBE_AGENT_ID_ENV = "LABS_EVAL_PROBE_AGENT_ID"
EXPECTED_TRAFFIC_CLASS = "internal_probe"
ROUTED_MODEL = "ainfera-inference"

# Hard cost ceiling for one cycle, USD (mirrors judge_worker's daily envelope).
COST_CAP_USD = float(os.environ.get("LABS_EVAL_COST_CAP_USD", "25"))

# Per-arm-call token bound (cost control).
MAX_TOKENS = int(os.environ.get("LABS_EVAL_MAX_TOKENS", "1024"))

# Where cycle artifacts are written — Labs-PRIVATE. NEVER the public repo / git:
# frozen task sets and rich results derive from real routing_outcomes prompts
# (the moat) and must not be committed. Default is an ephemeral dir.
ARTIFACT_DIR_ENV = "LABS_EVAL_ARTIFACT_DIR"   # default /tmp/labs-eval


def artifact_dir() -> str:
    return os.environ.get(ARTIFACT_DIR_ENV, "/tmp/labs-eval")
