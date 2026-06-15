"""Ainfera Labs — proof-pipeline eval harness (AIN-458 · S1 scaffold).

A 4-arm (A/B/C/D) cost-vs-outcome eval that measures whether the outcome-aware
router (arm C) wins on cost-per-successful-task without losing success rate,
against a premium pin (A), a cheap pin (B), and a fusion panel (D).

Stage 1 ships INTERFACES + STUBS only — flag-gated OFF (LABS_EVAL_HARNESS_ENABLED),
no live model calls. Stage 2 (AIN-459) wires it live on the Spark Labs tenant.

Modules:
  config   guardrail constants + the flag gate + pre-registered test params
  arms     the 4 arms as interfaces + Stage-1 stubs; L1/L3 assertions
  judge    held-out Gemini 3.1 Pro adapter (stub) + human spot-check queue
  metrics  success rate, cost_per_success, cost_per_call, floor breaches, n,
           95% CI (Wilson), and the pre-registered win check
  taskset  freeze + hash + version-pin a held-out task set
  runner   the orchestration entry (validate + dry-run; INERT in Stage 1)

Canon: ainfera-vault decisions/2026-06-15-proof-pipeline-end-to-end.md
Locks:  ainfera-vault decisions/locks-2026-06-15-proof-pipeline.md
Eval:   ainfera-vault decisions/2026-06-15-eval-spec-3arm-cost-outcome.md
"""

from __future__ import annotations

__version__ = "0.1.0"
__stage__ = "S1-scaffold-inert"
