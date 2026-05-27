# Judge protocol (public-safe)

The outcome-quality labeller scores a sampled slice of routed calls on a 1-5
rubric. Protocol + integrity rules are public; the exact prompt and operated
judge are closed (anti-gaming).

## Rubric

| Score | Meaning |
|---|---|
| 5 | Task fully completed; no correction needed |
| 4 | Completed; minor issues |
| 3 | Partially completed; usable with edits |
| 2 | Attempted; substantial gaps |
| 1 | Failed / off-task |

## Self-preference firewall (hard rule)

In any round judge `J` labels, `J` is **excluded from that round's routable
candidate set**. A judge never grades its own outputs.

## Sampling

1-5% of routed calls, stratified by task type so rare types are not starved of
labels. Async — never on the request path.

## Integrity

- Labels append-only; join to the hash-chained audit record.
- Judge identity + version recorded per label.
- Held-out human spot-check calibrates drift (closed).
