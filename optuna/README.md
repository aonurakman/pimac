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

6. Export the best configs from a finished suite back into the task config folders:

```bash
venv/bin/python optuna/analyze.py export-best \
  --suite-id study_01 \
  --task simple_spread_dynamic_hard
```

7. If you want a combined leaderboard from two suites:

```bash
venv/bin/python optuna/analyze.py merge \
  --suite-a suite_a \
  --suite-b suite_b \
  --merged-suite-id merged_suite
```
