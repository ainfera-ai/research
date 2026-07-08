# eval_v2_ruler — Eval-v2 Ruler Harness

The evaluation harness that makes the 251→50 catalog cut real. Scores every
catalog model on four admission gates (G1–G4) using a frozen tool-use task set.

## Gates

| Gate | Metric | Threshold | Pass |
|------|--------|-----------|------|
| G1 | Tool-call validity | ≥99% well-formed tool calls | score ≥ 0.99 |
| G2 | Context coherence | ≥95% correct tool + valid args | score ≥ 0.95 |
| G3 | Latency | p95 first-tool-call latency | < 3000ms |
| G4 | Cost efficiency | Tokens per successful task | ≤ 2000 |

## Usage

```bash
# Dry run (INERT — no live calls)
python3 -m labs.eval_v2_ruler

# Live run (scores all catalog models)
RULER_ENABLED=1 RULER_LIVE=1 python3 -m labs.eval_v2_ruler

# Validation run (first N models only)
RULER_ENABLED=1 RULER_LIVE=1 python3 -m labs.eval_v2_ruler --max-models 10

# Custom output dir
RULER_ENABLED=1 RULER_LIVE=1 python3 -m labs.eval_v2_ruler --output-dir /tmp/eval-v2-test
```

## Output

CSV at `research/labs/eval-v2-results/ruler_scores.csv`:
```
model_id,g1_pass,g1_score,g2_pass,g2_score,g3_pass,g3_score,g4_pass,g4_score,overall_pass
```

Full detail JSON at `research/labs/eval-v2-results/ruler_scores.json` (per-task
scores, token counts, cost, errors — for audit).

## Task Set

Frozen 10-task tool-use set at `fixtures/tool_use_tasks.json`. SHA-256 hashed
and version-pinned. Each task defines a tool schema and a prompt that requires
a specific tool call. The hash is verified on load — a silent edit fails closed.

## Architecture

```
eval_v2_ruler/
  __init__.py      package metadata
  config.py        gate thresholds, cost cap, gateway config
  taskset.py       frozen task set loader + hash verification
  catalog.py       fetch live model catalog from gateway
  gateway.py       single seam to ainfera gateway (OpenAI-compat + tools)
  scorer.py        G1-G4 gate scoring logic
  runner.py        orchestrate full run, write CSV + JSON
  __main__.py      CLI entry point with flag gates
  fixtures/
    tool_use_tasks.json   frozen 10-task set
  tests/
    test_scorer.py        22 unit tests (all pass)
```

## Design Decisions

1. **Pinned slugs, not ainfera-inference.** Each model is called by its slug
   directly. Pinned slugs write NO routing_outcomes row — excluded from
   training by construction (L2).

2. **Pure stdlib.** No SDK dependency. Uses urllib for HTTP. Mockable for tests.

3. **Fail-soft on provider errors.** A 502/timeout records a 0 score for that
   task and moves on — one flaky provider doesn't crash the run.

4. **Cost guardrail.** Hard cap at $500 (configurable). Stops if cumulative
   spend exceeds the cap.

5. **Lenient args matching.** G2 args match is case-insensitive for strings,
   coerces int-from-string, and allows extra keys (some models add context).

## Canon

- Vault: `decisions/2026-07-06-eval-v2-ruler-scope.md`
- Vault: `decisions/2026-07-08-jul12-stack-d4-proposals.md` (item #5)
- DB migration: `api/alembic/versions/20260707_0090_labs_eval_results.py` (staged)

## Constraints

- No prod mutations. No routing_outcomes rows. Build phase only.
- Scores feed the founder decision on the 251→50 catalog cut, not an auto-flip.
