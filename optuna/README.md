## Optuna sweep tools

This directory keeps the sweep workflow small and explicit.

- `study.py`: run one study manifest.
- `full_sweep.py`: run several manifests in sequence.
- `analyze.py`: read-only analysis helpers for finished suites.
- `search_spaces.py`: a generic sampler that reads typed parameter specs from JSON.
- `optuna_utils.py`: small shared helpers for manifests, suite paths, and task lookup.
- `study_library/`: the committed `base.json` template for future studies.
- `results/`: sweep outputs.

### Study library

Only one study manifest is committed:

- `base.json`
  - non-runnable template showing the maintained search surfaces and inheritance structure

Copy it first, rename it, remove `"template": true`, then trim it down to the study you actually want.

### Study manifests

One manifest file defines one study. The key pieces are:

- `task`: benchmark task id
- `seed`: study seed
- `task_overrides`: optional task-level overrides
- `algorithms`: ordered list of algorithm study blocks

Each algorithm block defines:

- `name`: algorithm id
- `trials`: number of Optuna trials
- `base_config`: optional JSON config to copy before sampling
- `search`: typed parameter specs
- `inherit`: optional chaining rules

### Parameter spec format

Every tunable parameter is a small JSON object with a `type` field.

- `constant`: fixed value
- `categorical`: choose one item from `values`
- `float`: sample between `low` and `high`
- `int`: sample between `low` and `high`

Optional helpers:

- `merge: true`
  - for categorical choices that are themselves dictionaries
- `targets: ["a", "b"]`
  - write one sampled list into several config keys
- `length_target: "mixing_num_hidden"`
  - write the sampled list length into one extra config key

### Execution semantics

- `parallel-jobs` means max concurrent algorithm studies inside one manifest DAG
- each algorithm study still runs Optuna trials sequentially with `n_jobs=1`
- roots can run together, and chained children wait for their parents
- if one algorithm study fails, newly unblocked dependents are not launched

The maintained inheritance shape in `base.json` is:

- `ippo -> mappo -> pimac_v0`
- `ippo -> mappo -> pimac_v1`
- `pimac_v1 -> pimac_v2`
- `pimac_v1 -> pimac_v3`
- `pimac_v1 -> pimac_v4`
- `iql -> vdn`
- `iql -> qmix`

Study-specific manifests can also add extra branches such as:

- `mappo -> pimac_v5`

### Active sweep protocol

Active sweep manifests should usually use the cheaper sweep path:

- `task_overrides.eval_every_episodes=0`
- no during-training checkpoint selection
- one held-out evaluation of the final checkpoint only
- `summary["test"]["objective_score"]` becomes that final held-out score

For these sweep runs, only `final_checkpoint.pt` is written.
The held-out sweep split still comes from the task `test_*` fields, so it is a selection split rather than an untouched final test split.
`level_based_foraging_dynamic` is the one maintained exception to mean-agent sweep scoring: its task hook reports team return instead.
`lbf_hard` uses the same team-return scoring but with a fixed-slot local entity observation wrapper that removes the native roster-size leak.

Value-based manifests can also use `temp_gap_fraction_at_budget`, which `search_spaces.py` converts into raw `temp_decay` with an approximate update-budget normalization.

### Step-by-step usage

1. Copy the template:

```bash
cp optuna/study_library/base.json optuna/study_library/my_study.json
```

2. Remove `"template": true` and edit the copied file.

3. Check the study without running it:

```bash
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/my_study.json \
  --suite-id study_01 \
  --dry-run
```

4. Run the study:

```bash
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/my_study.json \
  --suite-id study_01 \
  --parallel-jobs 4
```

5. Compare the selected checkpoints from one finished suite:

```bash
venv/bin/python optuna/analyze.py compare \
  --suite-id study_01 \
  --task simple_spread_dynamic_hard
```

For dynamic-team tasks this now also writes:
- `trial_return_by_count.csv`
- `trial_return_boxplots.png`

The boxplot figure uses completed sweep trials, one subplot per roster size, one box per algorithm, and split-aware panel backgrounds (`train` / `validation` / `test`).

6. Trace PIMAC coordination signals for the top ranked runs:

```bash
venv/bin/python optuna/analyze.py coordination \
  --suite-id study_01 \
  --task simple_spread_dynamic_hard
```

Current defaults are intentionally richer than the original smoke settings:
- `--top-k 2`
- `--rollouts-per-count 8`

The coordination analysis now writes:
- raw rows:
  - `student_rows.csv`
  - `token_rows.csv`
  - `step_metrics.csv`
- summaries:
  - `summary_by_trial_count.csv`
  - `summary_by_count.csv`
- PCA exports:
  - `token_pca_rows.csv`
  - `student_ctx_pca_rows.csv`
- plots:
  - `token_pca.png`
  - `student_ctx_pca.png`
  - `alignment_heatmap.png`
  - `gate_alignment_3d.png`
  - `gate_agentcount_heatmap.png`

`summary_by_count.csv` is built by first summarizing each run at each roster size, then averaging those per-run summaries. That avoids overweighting longer or denser traces.

For final multi-seed benchmark runs under `results/`, there is also a small standalone plotting helper:

```bash
venv/bin/python optuna/plot_learning_curves.py \
  --preset lbf_final_selected \
  --preset rware_final_selected \
  --preset spread_final_selected
```

It plots the maintained selected final curves with mean lines, CI bands, and optional curriculum-stage shading.
By default it also writes a secondary `*_by_stage.png` figure with one subplot per curriculum stage
and independent y-axes.
By default it also writes a `*_final_eval_boxplots.png` figure that groups the final per-seed
evaluation means by roster size and algorithm, using the persisted `eval_by_count.csv` files.
Pass `--save-legend-separately` to omit legends from the plot figures and write one extra
`*_legend.png` artifact per preset.
By default, rolling smoothing spans the whole run; enable stage-local smoothing with
`--reset-smoothing-at-stage-boundaries` if you want the rolling window to restart at each curriculum stage.
By default, consecutive curriculum stages with the same roster-count set are merged in the stage
visualization; disable that with `--no-merge-identical-adjacent-stages` if you want every stage shown separately.
Use `--list-presets` to see the available presets, and tweak smoothing / CI / labels from the CLI.
To switch which exported config is plotted for a preset, edit the `SELECTED_CONFIGS` block near the
top of [optuna/plot_learning_curves.py](/Users/akman/pimac/optuna/plot_learning_curves.py); the
rware preset currently points at the longer-budget suite under
`results/final_rware_long_01/robotic_warehouse_dynamic` and uses `pimac_v6/active_01` as the
selected `PC3D` config.
script derives the full run globs from those compact selections.

For coordination traces on one concrete final-results family (task + algorithm + config), use:

```bash
venv/bin/python optuna/plot_coordination_results.py \
  --task-results-dir results/simple_spread_dynamic_hard \
  --task simple_spread_dynamic_hard \
  --algorithm pimac_v6 \
  --config active_03
```

This reads all matching seeded runs under the requested task results directory and writes the same
PCA/alignment/gate artifacts as the suite-level coordination analysis, under
`<task-results-dir>/coordination_plots/<algorithm>/<config>/` by default.

There is also a checkpoint re-evaluation helper for final suites:

```bash
venv/bin/python optuna/backfill_final_eval_rollout_returns.py \
  --run-root results/final_rware_01/robotic_warehouse_dynamic \
  --extra-count 9 \
  --override-test-count 9 \
  --override-test-count 10 \
  --output-dir results/final_rware_01/_reanalysis_n9
```

In this mode the script does not retrain anything. It reads each run's existing `eval_by_count.csv`,
evaluates only the missing extra counts from the saved checkpoints, and writes supplementary
suite-level recomputed CSVs under the requested output directory.

7. Export the best configs from a finished suite back into the task config folders:

```bash
venv/bin/python optuna/analyze.py export-best \
  --suite-id study_01 \
  --task simple_spread_dynamic_hard
```

8. If you want a combined leaderboard from two suites:

```bash
venv/bin/python optuna/analyze.py merge \
  --suite-a suite_a \
  --suite-b suite_b \
  --merged-suite-id merged_suite
```
