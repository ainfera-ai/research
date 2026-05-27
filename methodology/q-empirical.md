# q_empirical (public-safe)

`q_empirical` is the learned, per-model quality signal that overrides the static
prior `q_prior` as labeled outcomes accrue. This note describes the *method*. The
operated weights and the real corpus are closed.

## Objective — quality-floor-then-min-cost

For a request with task type `t` and candidate set `C` cleared by the policy veto
`M_allowed`:

1. Drop candidates whose predicted quality is below the `min_quality` bar.
2. Among those clearing the bar, pick the **cheapest** all-in (provider cost +
   Ainfera margin).
3. Ties -> higher predicted quality -> lower latency.

A constrained objective, **not** a weighted-sum scalarization.

## Signal sources

- **Implicit** — completion, retry, downstream acceptance, tool-call success
  (cheap, noisy, high-volume).
- **Judge labels** — 1-5 rubric on a 1-5% sample (lower-noise). See
  `judge-protocol.md`.

## Learning

A LinUCB contextual bandit over the candidate set balances exploitation of the
current estimate against exploration of under-sampled (model, task) cells. The
exploration floor is non-negotiable — see `exploration-floor.md`.

## Why it compounds

The signal lives at the *orchestration boundary* (did the task complete), not the
API layer (did the call return 200). That boundary is what gateway-layer rivals
cannot observe. More traffic -> more labeled cells -> tighter estimates -> better
routing -> more traffic.
