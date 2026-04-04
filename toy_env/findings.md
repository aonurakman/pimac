# Findings for `toy_env`

Archived source suite: `combined2` / `coop_line_world`

These notes summarize patterns from the archived top configs in `configs/`.
For this task, the main conclusion is that there is almost nothing useful to learn from hyperparameter rankings.

## Main takeaways
This task is a smoke test, not a discriminative benchmark.
On combined2, all algorithms saturated to the same score, so the task does not separate methods in any meaningful way.
Interpretation: keep it only for fast debugging, API checks, and quick smoke runs.

## Best study scores by algorithm
- `mappo`: `1.7239583730697632`
- `iql`: `1.7239583730697632`
- `ippo`: `1.7239583730697632`
- `qmix`: `1.7239583730697632`
- `vdn`: `1.7239583730697632`
- `pimac_v0`: `1.7239583730697632`
- `pimac_v1`: `1.7239583730697632`
- `pimac_v2`: `1.7239583730697632`
- `pimac_v3`: `1.7239583730697632`

## Parameter interpretation
- This task is too easy to support meaningful parameter conclusions. The archived top-five configs for each algorithm all reach the same score.
- Use the toy task only to check that an algorithm runs, learns something, writes outputs, and does not crash.
- Do not treat the archived toy configs as evidence that one hyperparameter zone is better than another. They are only convenient smoke-test seeds.
