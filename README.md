# PIMAC

Private experimentation repo for PIMAC and the compact benchmark tasks around it. This is not a reusable framework or public package.

## Layout

- `algorithms/`: learners and the shared benchmark interface.
- task directories (`lbf_hard/`, `simple_spread_dynamic_hard/`, `robotic_warehouse_dynamic/`, ...): runnable task scripts, task defaults, archived configs, findings.
- `optuna/`: sweep runner, manifests, and suite analysis.
- `plotting/`: final-results plotting scripts and saved plot outputs.
- `server_scripts/`: small Slurm launchers for the current experiment families.
- `results/`: normal run outputs.
- `tests/`: focused checks only.

## Common commands

Run one task:

```bash
venv/bin/python simple_spread_dynamic_hard/run.py \
  --algorithm pimac_v6 \
  --alg-config simple_spread_dynamic_hard/configs/pimac_v6/active_03.json
```

Run one sweep:

```bash
cp optuna/study_library/base.json optuna/study_library/scratch.json
venv/bin/python optuna/study.py \
  --manifest optuna/study_library/scratch.json \
  --suite-id scratch_01
```

Plot maintained final curves:

```bash
venv/bin/python plotting/plot_learning_curves.py --list-presets
```

## Setup

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Tests

```bash
venv/bin/python -m pytest tests -q
```
