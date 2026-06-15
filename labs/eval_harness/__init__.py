"""Ainfera Labs — proof-pipeline eval harness (AIN-458 · S1 scaffold).

A 4-arm (A/B/C/D) cost-vs-outcome eval that measures whether the outcome-aware
router (arm C) wins on cost-per-successful-task without losing success rate,
against a premium pin (A), a cheap pin (B), and a fusion panel (D).

Stage 1 (AIN-458) shipped INTERFACES + STUBS. Stage 2 (AIN-459) adds the LIVE
wiring — still INERT until the founder enables it on the Spark Labs tenant
(LABS_EVAL_LIVE, default OFF) after the Doppler key-fix + the 2026-06-19 gate.

Modules:
  config     guardrail constants + flag gates + pre-registered test params
  arms       the 4 arm specs + Stage-1 stubs; L1/L2/L3 assertions
  judge      held-out Gemini 3.1 Pro adapter (StubJudge + live GatewayJudge)
  metrics    success rate, cost_per_success, cost_per_call, floor breaches, n,
             95% CI (Wilson), and the pre-registered win check
  taskset    freeze + hash + version-pin a task set
  runner     Stage-1 validate + dry-run entry (INERT)
  gateway    the single Ainfera-gateway seam (lazy SDK import; mockable)
  live_arms  real arm runners (pinned A/B, router C, panel D)
  loader     freeze a held-out >=200/type sample from routing_outcomes (private)
  snapshot   sanitized web-shaped snapshot + private results artifacts
  cycle      one live eval cycle (INERT until LABS_EVAL_LIVE + probe + key set)

Canon: ainfera-vault decisions/2026-06-15-proof-pipeline-end-to-end.md
Locks:  ainfera-vault decisions/locks-2026-06-15-proof-pipeline.md
Eval:   ainfera-vault decisions/2026-06-15-eval-spec-3arm-cost-outcome.md
"""

from __future__ import annotations

__version__ = "0.2.0"
__stage__ = "S2-live-wiring-inert"
