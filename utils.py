"""Small shared helpers for the benchmark repo.

This file intentionally stays simple. It only holds helpers that are genuinely identical across
tasks and sweep tooling: path handling, JSON/CSV IO, seed setup, a few plotting utilities, and
update-history export.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from algorithms.base import UpdateReport


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
OPTUNA_RESULTS_ROOT = ROOT / "optuna" / "results"

_MPLCONFIGDIR = ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DASHBOARD_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Policy optimization", ("policy_loss", "value_loss", "entropy")),
    ("PPO stability", ("approx_kl", "clip_frac", "explained_variance")),
    ("Q-learning", ("td_loss", "td_error_abs", "exploration_temperature")),
    ("Q scale", ("q_mean", "target_mean")),
    ("Teacher grounding", ("distill_loss", "distill_mse", "counterfactual_loss")),
    ("Context uncertainty", ("ctx_logvar_mean", "ctx_logvar_std", "gate_mean", "gate_std")),
    ("Policy adaptation", ("hyper_l2", "delta_w_norm_mean", "delta_b_norm_mean")),
)

DEFAULT_DASHBOARD_METRIC_KEY_ORDER: tuple[str, ...] = tuple(
    metric_key for _, metric_keys in DASHBOARD_METRIC_GROUPS for metric_key in metric_keys
)


def load_json(path: str | Path) -> dict:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return dict(data)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: str | Path, data) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        discovered: list[str] = []
        for row in rows:
            for key in row:
                if key not in discovered:
                    discovered.append(key)
        fieldnames = discovered
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_global_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> str:
    """Return the actual device string for one CLI request."""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(name)


def resolve_json_path(path: str, *, base_dir: str | Path, project_root: str | Path) -> Path:
    """Resolve a JSON path against the task directory first, then the repo root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    task_relative = (Path(base_dir) / candidate).resolve()
    if task_relative.exists():
        return task_relative

    project_relative = (Path(project_root) / candidate).resolve()
    if project_relative.exists():
        return project_relative
    return task_relative


def active_agent_mask(agent_ids: Sequence[object], active_ids: Sequence[object]) -> dict[object, float]:
    """Build the presence mask that joint learners need for padded team tensors."""
    active = set(active_ids)
    return {agent_id: 1.0 if agent_id in active else 0.0 for agent_id in agent_ids}


def learner_temperature(learner) -> float | None:
    """Read the learner temperature when the algorithm exposes one."""
    if hasattr(learner, "temperature"):
        return float(getattr(learner, "temperature"))
    return None


def make_run_dir(task_name: str, algorithm: str, *, results_root: str | Path | None = None, run_id: str | None = None) -> str:
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    resolved_run_id = str(run_id) if run_id is not None else str(int(time.time()))
    out_dir = root / task_name / algorithm / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def configured_checkpoint_episodes(task_config: dict) -> tuple[int, ...]:
    """Return extra training checkpoint episodes, excluding the final checkpoint.

    Task runners already save `final_checkpoint.pt` at the end of training. This helper computes
    additional trajectory snapshots requested through the task config so those can be written
    without changing evaluation semantics.
    """

    total_episodes = int(task_config["episodes"])
    checkpoints: set[int] = set()

    raw_explicit = task_config.get("save_checkpoint_episodes", [])
    if raw_explicit:
        for value in raw_explicit:
            episode = int(value)
            if 0 < episode < total_episodes:
                checkpoints.add(episode)

    every = int(task_config.get("save_checkpoint_every_episodes", 0) or 0)
    if every > 0:
        checkpoints.update(range(every, total_episodes, every))

    return tuple(sorted(checkpoints))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def flatten_update_history(update_history: Sequence[UpdateReport | dict]) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for entry in update_history:
        if isinstance(entry, UpdateReport):
            rows.append(entry.to_flat_dict())
        else:
            rows.append(dict(entry))
    return rows


def save_update_history_json(path: str | Path, update_history: Sequence[UpdateReport | dict]) -> None:
    rows = flatten_update_history(update_history)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(rows), handle, indent=2, allow_nan=False)


def save_update_history_csv(path: str | Path, update_history: Sequence[UpdateReport | dict]) -> None:
    write_csv(path, flatten_update_history(update_history))


def _coerce_metric_value(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def ordered_scalar_metric_keys(metric_history: Sequence[dict[str, float]] | None) -> list[str]:
    if not metric_history:
        return []
    scalar_keys: set[str] = set()
    for key in DEFAULT_DASHBOARD_METRIC_KEY_ORDER:
        values = np.asarray([_coerce_metric_value(entry.get(key, np.nan)) for entry in metric_history], dtype=np.float32)
        if np.isfinite(values).any():
            scalar_keys.add(key)
    return [key for key in DEFAULT_DASHBOARD_METRIC_KEY_ORDER if key in scalar_keys]


def grouped_dashboard_metric_keys(metric_history: Sequence[dict[str, float]] | None) -> list[tuple[str, list[str]]]:
    if not metric_history:
        return []
    scalar_keys = set(ordered_scalar_metric_keys(metric_history))
    grouped: list[tuple[str, list[str]]] = []
    for title, metric_keys in DASHBOARD_METRIC_GROUPS:
        present = [key for key in metric_keys if key in scalar_keys]
        if present:
            grouped.append((title, present))
    return grouped


def plot_dashboard_metric_group(
    axis,
    metric_history: Sequence[dict[str, float]],
    *,
    title: str,
    metric_keys: Sequence[str],
) -> None:
    for metric_key in metric_keys:
        values = np.asarray([_coerce_metric_value(row.get(metric_key, np.nan)) for row in metric_history], dtype=np.float32)
        if np.isfinite(values).any():
            axis.plot(np.arange(1, values.size + 1), values, linewidth=1.2, alpha=0.90, label=metric_key)
    axis.set_ylabel(title)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)


def plot_basic_curves(
    *,
    save_path: str | Path,
    title: str,
    rewards: Sequence[float],
    losses: Sequence[float],
    eval_x: Sequence[int] | None = None,
    eval_rewards: Sequence[float] | None = None,
    update_history: Sequence[dict[str, float]] | None = None,
    window: int = 100,
) -> None:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    losses_np = np.asarray(losses, dtype=np.float32)
    eval_x_np = np.asarray(eval_x or [], dtype=np.int64)
    eval_rewards_np = np.asarray(eval_rewards or [], dtype=np.float32)
    history_rows = list(update_history or [])
    metric_groups = grouped_dashboard_metric_keys(history_rows)

    fig_rows = 2 + len(metric_groups) if metric_groups else 2
    fig, axes = plt.subplots(fig_rows, 1, figsize=(12, 4 * fig_rows))
    if fig_rows == 1:
        axes = [axes]

    axes[0].plot(np.arange(1, rewards_np.size + 1), rewards_np, alpha=0.30, color="tab:blue", label="train return")
    if rewards_np.size:
        ma = moving_average(rewards_np, min(window, max(1, rewards_np.size)))
        ma_x = np.arange(rewards_np.size - ma.size + 1, rewards_np.size + 1)
        axes[0].plot(ma_x, ma, color="tab:orange", linewidth=2.0, label="moving avg")
    if eval_x_np.size and eval_rewards_np.size:
        axes[0].plot(eval_x_np, eval_rewards_np, marker="o", color="tab:green", label="validation")
    axes[0].set_title(title)
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(np.arange(1, losses_np.size + 1), losses_np, alpha=0.30, color="tab:red", label="episode loss")
    if losses_np.size:
        ma = moving_average(losses_np, min(window, max(1, losses_np.size)))
        ma_x = np.arange(losses_np.size - ma.size + 1, losses_np.size + 1)
        axes[1].plot(ma_x, ma, color="tab:purple", linewidth=2.0, label="moving avg")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if metric_groups:
        for axis, (title, metric_keys) in zip(axes[2:], metric_groups):
            plot_dashboard_metric_group(axis, history_rows, title=title, metric_keys=metric_keys)
        axes[-1].set_xlabel("Update")
    else:
        axes[1].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_gif(frames: list, path: str | Path, *, resize_to: Optional[tuple[int, int]] = (350, 350), duration_ms: int = 80) -> None:
    if not frames:
        return
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        image = Image.fromarray(frame)
        if resize_to is not None:
            image = image.resize(resize_to, Image.BILINEAR)
        images.append(image)
    images[0].save(
        resolved,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
