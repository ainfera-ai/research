# Results (synthetic)

Baseline = agent's own default (`model-b`). `save%` is cost cut vs that baseline; `done&cheaper%` is the share of tasks completed at <= baseline cost and ~>= baseline quality.

| policy | quality | done% | cost | save% | done&cheaper% |
|---|---|---|---|---|---|
| agent_baseline | 4.14 | 98.8 | 24.35 | +0.0 | 64.5 |
| oracle | 4.18 | 98.4 | 18.58 | +23.7 | 60.7 |
| ainfera_learned | 4.29 | 98.9 | 26.23 | -7.7 | 50.3 |
| ainfera_static | 4.05 | 94.1 | 22.00 | +9.6 | 46.5 |
| cheapest | 3.46 | 73.4 | 9.74 | +60.0 | 29.7 |
| round_robin | 4.03 | 92.8 | 32.06 | -31.7 | 28.0 |
| single_best | 4.50 | 99.8 | 73.04 | -200.0 | 0.0 |
