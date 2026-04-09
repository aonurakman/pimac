# Findings for `robotic_warehouse_dynamic`

There are no archived RWARE sweeps in this repo yet.

This task is included as a supporting external dynamic-team benchmark:
- stronger local sensing and sparse coordination than `simple_spread_dynamic`,
- cleaner scaling pressure than the toy tasks,
- but less interpretable than `simple_spread_dynamic_hard` for the core local-context claim.

## Current benchmark role
- test whether the algorithms transfer to a different partial-observability structure,
- check scaling under variable team size on a compact external environment,
- and keep the implementation path inspectable enough to validate the full task loop directly.

## Known upstream integration notes
- RWARE registers these environments with individual reward by default. This task always overrides to global reward.
- The upstream `DictAgents` Gymnasium wrapper is not reset-safe under the current API, so this task uses a local adapter instead.
- The upstream `rgb_array` rendering path was unreliable in local validation, so this task uses a local PIL renderer for GIFs and Optuna videos.

## Interpretation policy
Until there are archived sweeps here, treat the configs in `configs/` as bootstrap defaults only.
Do not present this task as solved or tuned.
