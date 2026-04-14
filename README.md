# PIMAC

A small research repo for the PIMAC family and its benchmark tasks.

## Structure

- `algorithms/`: all benchmark learners in one place.
- `simple_spread/`, `simple_spread_dynamic/`, `simple_spread_dynamic_hard/`, `robotic_warehouse_dynamic/`, `level_based_foraging_dynamic/`, `toy_env/`: one task directory per environment.
  - `run.py`: the full task script.
  - `task.json`: default task settings.
  - `configs/`: runnable algorithm configs.
  - `findings.md`: archived takeaways from earlier sweeps.
- `optuna/`: compact sweep tooling.
  - `study.py`: run one study manifest.
  - `full_sweep.py`: run several manifests in sequence.
  - `analyze.py`: merge, compare, coordination, videos, export-best.
  - `study_library/`: the committed `base.json` template for future studies.
  - `README.md`: manifest format and chaining notes.
- `utils.py`: small shared helpers.
- `results/`: normal run outputs.
- `tests/`: focused checks for algorithms, task scripts, and sweep tools.

## Algorithms

The public benchmark algorithms are:

- `random`
- `iql`
- `ippo`
- `mappo`
- `qmix`
- `vdn`
- `pimac_v0`
- `pimac_v1`
- `pimac_v2`
- `pimac_v3`
- `pimac_v4`

All learned algorithms use the same benchmark interface:

- `Algorithm(env_spec, config, device)`
- `act_parallel(obs_dict)`
- `record_parallel_step(transition)`
- `maybe_update(global_step, episode_index)`
- `reset_episode()`
- `set_train_mode()`
- `set_eval_mode()`
- `save_checkpoint(path)`
- `load_checkpoint(path, env_spec, config, device)`
- `get_update_history()`

`ParallelTransition` uses `active_agent_mask_dict` and `next_active_agent_mask_dict` for joint learners. This repo does not use legal-action masks.
For `ippo`, `mappo`, and the `pimac_*` variants, evaluation mode keeps stochastic policy sampling and only switches modules to eval mode.
The simple-spread task family now uses one fully cooperative shared reward: landmark coverage plus a global collision penalty counted once per colliding pair.
In `pimac_v1` through `pimac_v4`, the centralized value critic consumes concatenated coordination tokens rather than mean-pooling them before the value head.

## Running one task

Every task uses the same CLI shape:

```bash
venv/bin/python simple_spread_dynamic_hard/run.py \
  --algorithm pimac_v2 \
  --alg-config simple_spread_dynamic_hard/configs/pimac_v2/default.json
```

Useful flags:

- `--task-config`: override the default `task.json`
- `--seed`: reproducibility seed
- `--results-root`: output root
- `--run-id`: custom run name
- `--device`: `cpu`, `cuda`, or `auto`
- `--skip-gif`: skip rollout animation

## Configs and findings

Each task directory keeps:

- `configs/manifest.csv`: the retained archived configs we currently want to keep
- `configs/<algorithm>/default.json`: the plain default starting point for studies
- `findings.md`: the current interpretation note for that task

Right now this repo has been reset to a fresh sweep state, so the tasks only keep `default.json` files
plus a minimal random baseline in `manifest.csv`. Add `best_*.json` files back only after fresh sweeps
are actually run and exported. The manifest should stay honest about what is actually ranked.

This repo keeps only the reusable configs and written findings, not the full bulk of sweep outputs.

## Optuna workflow

Create a study manifest by copying the template first:

```bash
cp optuna/study_library/base.json optuna/study_library/my_hard_study.json
```

Then remove `"template": true`, trim it down, and run it:

```bash
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/my_hard_study.json \
  --suite-id hard_study_01 \
  --parallel-jobs 3
```

`optuna/study_library/base.json` is a non-runnable template. Copy it first, then trim it down.

Active sweep semantics:

- `parallel-jobs` means max concurrent algorithm studies inside one manifest DAG.
- Trials inside each algorithm study run sequentially.
- Active sweep manifests set `eval_every_episodes=0`.
- In that mode, the sweep objective is one held-out final-checkpoint evaluation only.
- The held-out sweep split still uses the task `test_*` fields for simplicity.
- `level_based_foraging_dynamic` uses team return as its task KPI; other tasks keep mean-agent episode return.

Compare the selected checkpoints from a suite:

```bash
venv/bin/python optuna/analyze.py compare \
  --suite-id hard_study_01 \
  --task simple_spread_dynamic_hard
```

When a sweep disables validation, only `final_checkpoint.pt` is written and the analysis tools use that directly.

Other analysis subcommands are:

- `merge`
- `coordination`
- `videos`
- `export-best`

## Setup

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Tests

```bash
venv/bin/python -m pytest tests -q
```
