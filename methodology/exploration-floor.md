# Exploration floor (public-safe)

The policy reserves >= 5-10% of eligible traffic for exploration — routing to
candidates that are not the current argmax — to keep estimates honest across the
whole (model, task) grid.

## Why non-negotiable

Without a floor a bandit collapses onto an early local optimum: it stops sampling
alternatives, stale cells drift, and a deterministic argmax becomes a lookup
table an outsider can reverse-engineer. The floor keeps the learned policy a
moving target.

## Mechanics

- Floor applies only among candidates already cleared by veto + quality bar.
- Higher for cold cells, decaying with confidence, never below the global minimum.
- Exploration cost is bounded and reported as a line item — an investment in
  dataset breadth, which is harder to replicate than raw volume.
