# Proof-pipeline eval harness (Stage 1 scaffold)

A 4-arm cost-vs-outcome eval. It measures whether the outcome-aware router
(arm **C**) wins on cost per successful task without losing success rate, against
a premium pin (**A**), a cheap pin (**B**), and a fusion panel (**D**).

This is **Stage 1 (AIN-458)**: interfaces and stubs only. The harness is
flag-gated **OFF** and makes **no live model calls**. Stage 2 (AIN-459) wires it
live on the Spark Labs tenant.

## Arms

- `A` premium pin — a single premium model, pinned (eval-only)
- `B` cheap pin — a single cheap model, pinned (eval-only)
- `C` ainfera — the outcome-aware router (the thing under test)
- `D` fusion panel — a multi-model panel synthesized by a frontier model

Arm pins live ONLY in this harness, never in the production router. Concrete
model identifiers are env-injected on the Labs tenant and are not committed to
this public repo — see `config.py` and `arms.py` for the placeholder defaults.

## Guardrails (fail-closed)

- **L1** — arm pins never reach the production router.
- **L2** — eval runs on the Labs tenant only; every call is excluded from
  `routing_outcomes` and from training (`arms.assert_excluded`).
- **L3** — the judge (Gemini 3.1 Pro) is held out of all four arms
  (`arms.assert_judge_held_out`).

## Metrics

`metrics.py` computes, per arm: success rate, `cost_per_success`,
`cost_per_call`, floor breaches, `n`, and a 95% Wilson confidence interval. It
also implements the pre-registered win check for arm C.

## Modules

- `config` — guardrail constants, the flag gate, pre-registered test params
- `arms` — the 4 arms as interfaces and Stage-1 stubs
- `judge` — held-out judge adapter (stub) and human spot-check queue
- `metrics` — the metric math and the win check
- `taskset` — freeze, hash, and version-pin a held-out task set
- `runner` — the orchestration entry (validate and dry-run; inert in Stage 1)

## Run the tests

```bash
python3 -m pytest labs/tests/test_eval_harness.py -q
```

## Canon

- `decisions/2026-06-15-proof-pipeline-end-to-end.md`
- `decisions/locks-2026-06-15-proof-pipeline.md`
- `decisions/2026-06-15-eval-spec-3arm-cost-outcome.md`
