## Optuna sweep tools

This directory keeps the sweep workflow small and explicit.

- `study.py`: run one study manifest.
- `full_sweep.py`: run several manifests in sequence.
- `analyze.py`: read-only analysis helpers for finished suites.
- `search_spaces.py`: a generic sampler that reads typed parameter specs from JSON.
- `optuna_utils.py`: small shared helpers for manifests, suite paths, and task lookup.
- `study_library/`: ready-to-run study manifests and one rejected template file.
- `results/`: sweep outputs.

### Study library

These JSON files are the study definitions for the sweeps we actually ran or may want to rerun.

- `fixed_full.json`
  - full sweep for fixed-team `simple_spread`
- `dynamic_full.json`
  - full sweep for dynamic `simple_spread`
- `dynamic_focus.json`
  - focused dynamic study for `mappo` and the PIMAC chain
- `hard_full.json`
  - full sweep for hard dynamic `simple_spread`
- `hard_pimac_recovery.json`
  - the first hard-task PIMAC rescue sweep
- `hard_pimac_competitive.json`
  - the later hard-task PIMAC sweep aimed at stronger returns
- `toy_full.json`
  - full sweep for `toy_env`
- `core.json`
  - small library file that runs the four full benchmark studies above
- `base.json`
  - non-runnable template showing the largest supported manifest shape

So yes: these manifests correspond to the sweep setups we have been using, now rewritten into the
new simpler format.

### Study manifests

One manifest file defines one study. The key pieces are:

- `task`: benchmark task id.
- `seed`: study seed.
- `task_overrides`: optional task-level overrides.
- `algorithms`: ordered list of algorithm study blocks.

Each algorithm block defines:

- `name`: algorithm id.
- `trials`: number of Optuna trials.
- `base_config`: optional JSON config to copy before sampling.
  - if omitted, `study.py` uses `<task>/configs/<algorithm>/default.json`.
- `search`: typed parameter specs.
  - this can be an empty object if you want one fixed-config trial.
- `inherit`: optional chaining rules.
  - if omitted, the algorithm does not inherit from earlier studies.

### Parameter spec format

Every tunable parameter is a small JSON object with a `type` field.

- `constant`: fixed value.
- `categorical`: choose one item from `values`.
- `float`: sample between `low` and `high`.
- `int`: sample between `low` and `high`.

Optional helpers:

- `merge: true`
  - for categorical choices that are themselves dictionaries.
  - the sampled dictionary is merged into the config.
  - use this for grouped choices like actor architectures that set both `num_hidden` and `widths`.
- `targets: ["a", "b"]`
  - write one sampled list or tuple into several config keys.
  - useful for things like `ctx_logvar_bounds -> ctx_logvar_min/max`.
  - use this when the manifest name is a readable alias rather than a real config key.
- `length_target: "mixing_num_hidden"`
  - write the sampled list length into one extra config key.
  - useful for `mixing_widths`.
  - this runs after the main sampled value is written.

These helper fields are all optional. If you omit them, the parameter name itself is used as the
config key.

### PIMAC chaining

PIMAC chaining is handled in the manifest, not hidden in Python profiles.

Example:

```json
{
  "name": "pimac_v2",
  "trials": 8,
  "base_config": "simple_spread_dynamic_hard/configs/pimac_v2/default.json",
  "inherit": [
    {"from": "mappo", "keys": ["lr", "batch_size"], "required": true},
    {"from": "pimac_v1", "keys": ["critic_hidden_sizes", "num_tokens"], "required": true}
  ],
  "search": {
    "distill_weight": {"type": "float", "low": 0.0004, "high": 0.003, "log": true}
  }
}
```

When `study.py` reaches that block, it loads the best completed parent configs from the same suite
and copies the listed keys after local sampling.

Inside each inherit block:

- `from`: source algorithm in the same suite.
- `keys`: config keys to copy from that source.
- `required`: optional, defaults to `true`.
  - if `true`, the study fails when the parent result does not exist yet.
  - if `false`, the study continues without that inheritance source.

### Template file

`study_library/base.json` is a guide file with:

- all algorithms,
- all parameter shapes,
- example inheritance blocks,
- and a few notes.

It is intentionally rejected by `study.py`. Copy it first, then trim it down.

### Step-by-step usage

1. Pick a task manifest from `study_library/`, or copy `base.json` to a new file and edit it.
2. If you copied `base.json`, remove `"template": true`.
3. Check the study without running it:

```bash
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/hard_pimac_competitive.json \
  --suite-id hard_pimac_03 \
  --dry-run
```

4. Run the study:

```bash
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/hard_pimac_competitive.json \
  --suite-id hard_pimac_03 \
  --parallel-jobs 4
```

5. Run several manifests in sequence:

```bash
venv/bin/python optuna/full_sweep.py \
  --suite-id core_01 \
  --library optuna/study_library/core.json \
  --parallel-jobs 4
```

6. Compare the best checkpoints from one finished suite:

```bash
venv/bin/python optuna/analyze.py compare \
  --suite-id hard_pimac_03 \
  --task simple_spread_dynamic_hard
```

7. Export the best configs from a finished suite back into the task config folders:

```bash
venv/bin/python optuna/analyze.py export-best \
  --suite-id hard_pimac_03 \
  --task simple_spread_dynamic_hard
```

8. If you want a combined leaderboard from two suites:

```bash
venv/bin/python optuna/analyze.py merge \
  --suite-a suite_a \
  --suite-b suite_b \
  --merged-suite-id merged_suite
```

9. If you want PIMAC-specific analysis from a finished suite:

```bash
venv/bin/python optuna/analyze.py coordination \
  --suite-id hard_pimac_03 \
  --task simple_spread_dynamic_hard
```

The normal flow is:

- dry-run a manifest,
- run the study,
- compare the best checkpoints,
- then optionally export the best configs and run coordination or video analysis.
