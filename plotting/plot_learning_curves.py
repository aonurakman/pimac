"""Plot mean training curves with confidence bands from final multi-seed run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PLOTTING_DIR / "plots"

# User-tunable defaults. These are the main knobs to edit when reusing the script.
DEFAULT_PRESETS = (
    "lbf_final_selected",
    "rware_final_selected",
    "spread_final_selected",
)
DEFAULT_X_KEY = "global_step"
DEFAULT_Y_KEY = "train_return_mean"
DEFAULT_SMOOTHING_WINDOW = 100
DEFAULT_RESET_SMOOTHING_AT_STAGE_BOUNDARIES = False
DEFAULT_CI_LEVEL = 0.95
DEFAULT_DPI = 300
DEFAULT_SHOW_STAGE_BANDS = True
DEFAULT_SHOW_STAGE_LABELS = True
DEFAULT_STAGE_ALPHA = 0.05
DEFAULT_SEPARATOR_ALPHA = 0.28
DEFAULT_SAVE_STAGE_PANELS = True
DEFAULT_STAGE_PANEL_COLUMNS = 4
DEFAULT_STAGE_PANEL_RELATIVE_X = True
DEFAULT_SAVE_LEGEND_SEPARATELY = False
DEFAULT_MERGE_IDENTICAL_ADJACENT_STAGES = True
DEFAULT_SAVE_FINAL_EVAL_BOXPLOTS = True
DEFAULT_FONT_FAMILY = "Charter"
PALETTE = ["#005d5d", "goldenrod", "#9f1853", "royalblue"]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    glob_pattern: str
    color: str


@dataclass(frozen=True)
class PlotPreset:
    title: str
    output_filename: str
    task_config_path: str
    stage_source: str
    series: tuple[SeriesSpec, ...]


@dataclass(frozen=True)
class StageSegment:
    stage_indices: tuple[int, ...]
    stage_name: str
    start_step: float
    end_step: float
    counts: tuple[int, ...]
    label: str


SELECTED_CONFIGS: dict[str, dict[str, str]] = {
    "lbf_final_selected": {
        "title": "LBF Hard Final Learning Curves",
        "output_filename": "lbf_hard_final_learning_curves.png",
        "task_config_path": "lbf_hard/task.json",
        "results_root": "results/lbf_hard",
        "run_prefix": "final_lbf_hard",
        "mappo": "best_01",
        "pimac_v0": "best_01",
        "pc3d": "active_01",
    },
    "rware_final_selected": {
        "title": "RWARE Final Learning Curves",
        "output_filename": "rware_final_learning_curves.png",
        "task_config_path": "robotic_warehouse_dynamic/task.json",
        "results_root": "results/final_rware_long_01/robotic_warehouse_dynamic",
        "run_prefix": "final_robotic_warehouse_dynamic",
        "mappo": "best_01",
        "pimac_v0": "best_01",
        "pc3d": "active_01",
    },
    "spread_final_selected": {
        "title": "Spread Hard Final Learning Curves",
        "output_filename": "spread_hard_final_learning_curves.png",
        "task_config_path": "simple_spread_dynamic_hard/task.json",
        "results_root": "results/simple_spread_dynamic_hard",
        "run_prefix": "final_simple_spread_dynamic_hard",
        "mappo": "best_01",
        "pimac_v0": "best_01",
        "pc3d": "active_03",
    },
}


def _build_selected_preset(selected: dict[str, str]) -> PlotPreset:
    results_root = str(selected["results_root"]).rstrip("/")
    run_prefix = str(selected["run_prefix"])
    pc3d_config = str(selected["pc3d"])
    return PlotPreset(
        title=str(selected["title"]),
        output_filename=str(selected["output_filename"]),
        task_config_path=str(selected["task_config_path"]),
        stage_source=f"{results_root}/pimac_v6/{run_prefix}_pimac_v6_{pc3d_config}_s42/train_history.csv",
        series=(
            SeriesSpec(
                label="MAPPO",
                glob_pattern=f"{results_root}/mappo/{run_prefix}_mappo_{selected['mappo']}_s*/train_history.csv",
                color=PALETTE[0],
            ),
            SeriesSpec(
                label="PIMAC v0",
                glob_pattern=f"{results_root}/pimac_v0/{run_prefix}_pimac_v0_{selected['pimac_v0']}_s*/train_history.csv",
                color=PALETTE[1],
            ),
            SeriesSpec(
                label="PC3D",
                glob_pattern=f"{results_root}/pimac_v6/{run_prefix}_pimac_v6_{pc3d_config}_s*/train_history.csv",
                color=PALETTE[2],
            ),
        ),
    )


PRESETS: dict[str, PlotPreset] = {
    preset_name: _build_selected_preset(selected)
    for preset_name, selected in SELECTED_CONFIGS.items()
}


STAGE_FACE_COLORS: tuple[str, ...] = (
    "#000000",
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#b279a2",
)


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rolling_mean expects a 1D sequence.")
    if window <= 1:
        return array.astype(np.float64, copy=True)

    result = np.empty_like(array, dtype=np.float64)
    cumulative = np.cumsum(array, dtype=np.float64)
    for index in range(array.shape[0]):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start > 0 else 0.0)
        result[index] = total / float(index - start + 1)
    return result


def rolling_mean_by_stage(values: Sequence[float], stage_indices: Sequence[int], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    stage_array = np.asarray(stage_indices)
    if array.shape[0] != stage_array.shape[0]:
        raise ValueError("Stage-aware rolling mean expects stage indices aligned with values.")
    if array.ndim != 1 or stage_array.ndim != 1:
        raise ValueError("Stage-aware rolling mean expects 1D inputs.")
    if window <= 1 or array.size == 0:
        return array.astype(np.float64, copy=True)

    result = np.empty_like(array, dtype=np.float64)
    start = 0
    while start < array.shape[0]:
        end = start + 1
        while end < array.shape[0] and stage_array[end] == stage_array[start]:
            end += 1
        result[start:end] = rolling_mean(array[start:end], window)
        start = end
    return result


def load_history(
    path: Path,
    *,
    x_key: str,
    y_key: str,
    smoothing_window: int,
    reset_smoothing_at_stage_boundaries: bool,
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"History file is empty: {path}")
    x_values = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y_values = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    if reset_smoothing_at_stage_boundaries and all("stage_index" in row and row["stage_index"] != "" for row in rows):
        stage_indices = [int(row["stage_index"]) for row in rows]
        smoothed = rolling_mean_by_stage(y_values, stage_indices, smoothing_window)
    else:
        smoothed = rolling_mean(y_values, smoothing_window)
    return x_values, smoothed


def load_stage_segments(path: Path, *, x_key: str) -> list[StageSegment]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    segments: list[StageSegment] = []
    current_index: int | None = None
    current_name: str | None = None
    segment_start: float | None = None
    previous_step: float | None = None

    for row in rows:
        stage_index = int(row["stage_index"])
        stage_name = str(row["stage_name"])
        step = float(row[x_key])
        if current_index is None:
            current_index = stage_index
            current_name = stage_name
            segment_start = step
            previous_step = step
            continue
        if stage_index != current_index:
            assert current_name is not None and segment_start is not None and previous_step is not None
            stage_indices, counts = parse_stage_descriptor(current_name)
            segments.append(
                StageSegment(
                    stage_indices=stage_indices or (current_index,),
                    stage_name=current_name,
                    start_step=segment_start,
                    end_step=step,
                    counts=counts,
                    label=format_stage_label_from_parts(stage_indices or (current_index,), counts),
                )
            )
            current_index = stage_index
            current_name = stage_name
            segment_start = step
        previous_step = step

    assert current_index is not None and current_name is not None and segment_start is not None
    final_step = float(rows[-1][x_key])
    stage_indices, counts = parse_stage_descriptor(current_name)
    segments.append(
        StageSegment(
            stage_indices=stage_indices or (current_index,),
            stage_name=current_name,
            start_step=segment_start,
            end_step=final_step,
            counts=counts,
            label=format_stage_label_from_parts(stage_indices or (current_index,), counts),
        )
    )
    return segments


def merge_adjacent_stage_segments(stage_segments: Sequence[StageSegment]) -> list[StageSegment]:
    merged: list[StageSegment] = []
    for segment in stage_segments:
        if (
            merged
            and segment.counts
            and merged[-1].counts == segment.counts
            and merged[-1].stage_indices
            and segment.stage_indices
            and segment.stage_indices[0] == merged[-1].stage_indices[-1] + 1
        ):
            previous = merged[-1]
            combined_indices = previous.stage_indices + segment.stage_indices
            merged[-1] = StageSegment(
                stage_indices=combined_indices,
                stage_name=f"{previous.stage_name}+{segment.stage_name}",
                start_step=previous.start_step,
                end_step=segment.end_step,
                counts=segment.counts,
                label=format_stage_label_from_parts(combined_indices, segment.counts),
            )
        else:
            merged.append(segment)
    return merged


def aggregate_runs(
    run_paths: Sequence[Path],
    *,
    x_key: str,
    y_key: str,
    smoothing_window: int,
    reset_smoothing_at_stage_boundaries: bool,
    x_min: float | None = None,
    x_max: float | None = None,
    relative_to: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = [
        load_history(
            path,
            x_key=x_key,
            y_key=y_key,
            smoothing_window=smoothing_window,
            reset_smoothing_at_stage_boundaries=reset_smoothing_at_stage_boundaries,
        )
        for path in run_paths
    ]
    reference_index = min(range(len(runs)), key=lambda idx: len(runs[idx][0]))
    reference_x = runs[reference_index][0]
    min_shared_x = max(float(x_values[0]) for x_values, _ in runs)
    max_shared_x = min(float(x_values[-1]) for x_values, _ in runs)
    if x_min is not None:
        min_shared_x = max(min_shared_x, float(x_min))
    if x_max is not None:
        max_shared_x = min(max_shared_x, float(x_max))
    if min_shared_x > max_shared_x:
        raise ValueError("Requested x-range does not overlap across runs.")
    reference_x = reference_x[(reference_x >= min_shared_x) & (reference_x <= max_shared_x)]
    if reference_x.size == 0:
        raise ValueError("Run histories do not share any common x-axis support.")
    if reference_x[0] > min_shared_x:
        reference_x = np.concatenate([np.asarray([min_shared_x], dtype=np.float64), reference_x])
    if reference_x[-1] < max_shared_x:
        reference_x = np.concatenate([reference_x, np.asarray([max_shared_x], dtype=np.float64)])
    reference_x = np.unique(reference_x)

    resampled = []
    for run_path, (x_values, y_values) in zip(run_paths, runs):
        if np.any(np.diff(x_values) < 0):
            raise ValueError(f"History x-axis is not monotonic for {run_path}")
        resampled.append(np.interp(reference_x, x_values, y_values))

    stacked = np.vstack(resampled)
    mean = np.mean(stacked, axis=0)
    if stacked.shape[0] == 1:
        band = np.zeros_like(mean)
    else:
        stderr = np.std(stacked, axis=0, ddof=1) / math.sqrt(stacked.shape[0])
        band = stderr
    if relative_to is not None:
        reference_x = reference_x - float(relative_to)
    return reference_x, mean, band


def confidence_scale(level: float) -> float:
    if not (0.0 < level < 1.0):
        raise ValueError("CI level must be between 0 and 1.")
    return float(NormalDist().inv_cdf(0.5 + level / 2.0))


def parse_stage_descriptor(raw_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if "__" not in raw_name:
        stage_text = raw_name.replace("stage_", "")
        return ((int(stage_text),) if stage_text.isdigit() else tuple(), tuple())

    stage_prefix, counts_suffix = raw_name.split("__", maxsplit=1)
    stage_text = stage_prefix.replace("stage_", "")
    stage_indices = ((int(stage_text),) if stage_text.isdigit() else tuple())
    counts = [chunk for chunk in counts_suffix.split("_") if chunk]
    return stage_indices, tuple(int(chunk) for chunk in counts)


def format_stage_indices(stage_indices: Sequence[int]) -> str:
    if not stage_indices:
        return "S?"
    if len(stage_indices) == 1:
        return f"S{stage_indices[0]}"
    return f"S{stage_indices[0]}-{stage_indices[-1]}"


def format_stage_label(raw_name: str) -> str:
    stage_indices, counts = parse_stage_descriptor(raw_name)
    return format_stage_label_from_parts(stage_indices, counts)


def format_stage_label_from_parts(stage_indices: Sequence[int], counts: Sequence[int]) -> str:
    prefix = format_stage_indices(tuple(stage_indices))
    if counts:
        counts_text = ",".join(str(count) for count in counts)
        return rf"{prefix}: $n \in \{{{counts_text}\}}$"
    return prefix


def stage_panel_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_by_stage{output_path.suffix}")


def eval_boxplot_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_final_eval_boxplots{output_path.suffix}")


def legend_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_legend{output_path.suffix}")


def build_legend_handles(series_specs: Sequence[SeriesSpec]) -> list[Line2D]:
    return [Line2D([0], [0], color=series.color, linewidth=2.0, label=series.label) for series in series_specs]


def save_standalone_legend(output_path: Path, series_specs: Sequence[SeriesSpec], *, dpi: int) -> Path:
    legend_path = legend_output_path(output_path)
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    handles = build_legend_handles(series_specs)
    fig = plt.figure(figsize=(max(2.8, 1.8 * len(handles)), 0.9), dpi=dpi)
    fig.legend(handles=handles, frameon=False, ncol=max(1, len(handles)), loc="center")
    fig.savefig(legend_path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return legend_path


def uses_step_axis(x_key: str) -> bool:
    return "step" in x_key.lower()


def scaled_step_xlabel(base_label: str) -> str:
    return f"{base_label} (×1e5)"


def apply_step_axis_format(axis, *, x_key: str) -> None:
    if not uses_step_axis(x_key):
        return
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1e5:g}"))


def display_x_label(x_key: str) -> str:
    if x_key == "global_step":
        return "Environment Steps"
    return x_key.replace("_", " ").title()


def simplified_task_title(raw_title: str) -> str:
    return raw_title.replace(" Final Learning Curves", "")


def load_task_count_split_labels(task_config_path: Path) -> dict[int, str]:
    task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for count in task_config.get("train_counts", []):
        labels[int(count)] = "seen"
    for count in task_config.get("validation_counts", []):
        labels[int(count)] = "unseen/val"
    for count in task_config.get("test_counts", []):
        labels[int(count)] = "unseen/test"
    return labels


def load_final_eval_means(eval_path: Path) -> dict[int, float]:
    with eval_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, float] = {}
    for row in rows:
        if row.get("phase") != "final_checkpoint_test":
            continue
        result[int(row["n_agents"])] = float(row["return_mean"])
    if not result:
        raise ValueError(f"No final_checkpoint_test rows found in {eval_path}")
    return result


def coarse_split_label(label: str) -> str:
    return "seen" if label == "seen" else "unseen"


def plot_final_eval_boxplots(
    preset: PlotPreset,
    *,
    output_path: Path,
    dpi: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_labels = load_task_count_split_labels(REPO_ROOT / preset.task_config_path)

    values_by_count_by_series: dict[int, list[list[float]]] = {}
    for series_index, series in enumerate(preset.series):
        run_paths = sorted(REPO_ROOT.glob(series.glob_pattern))
        if not run_paths:
            raise FileNotFoundError(f"No run histories matched pattern: {series.glob_pattern}")
        series_eval_means = [load_final_eval_means(path.with_name("eval_by_count.csv")) for path in run_paths]
        series_counts = set.intersection(*(set(eval_means) for eval_means in series_eval_means))
        for count in sorted(series_counts):
            values_by_count_by_series.setdefault(count, [[] for _ in preset.series])
            values_by_count_by_series[count][series_index] = [eval_means[count] for eval_means in series_eval_means]

    counts = sorted(
        count for count, grouped_values in values_by_count_by_series.items() if all(len(values) > 0 for values in grouped_values)
    )
    if not counts:
        raise ValueError(f"No final evaluation counts found for preset {preset.title}")

    ncols = len(counts)
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.75 * ncols, 3.05),
        dpi=dpi,
        sharey=False,
        squeeze=False,
    )
    box_face_alpha = 0.55

    for axis in axes.flat[len(counts) :]:
        axis.set_visible(False)

    for axis, count in zip(axes.flat, counts):
        split_label = coarse_split_label(split_labels.get(count, "unseen"))
        grouped_values = values_by_count_by_series[count]
        boxplot = axis.boxplot(
            grouped_values,
            patch_artist=True,
            widths=0.65,
            medianprops={"color": "#222222", "linewidth": 1.4},
            whiskerprops={"color": "#444444", "linewidth": 1.0},
            capprops={"color": "#444444", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "color": "#444444"},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#666666", "markeredgewidth": 0.0, "alpha": 0.5},
        )
        for patch, series in zip(boxplot["boxes"], preset.series):
            patch.set_facecolor(series.color)
            patch.set_alpha(box_face_alpha)

        axis.set_title(f"n={count} ({split_label})", fontsize=10, fontweight="bold")
        axis.set_xticks(range(1, len(preset.series) + 1))
        axis.set_xticklabels([series.label for series in preset.series], rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.18)

    fig.suptitle(f"{simplified_task_title(preset.title)}: final evaluation", fontsize=12, fontweight="bold", y=0.975)
    fig.supylabel("Mean evaluation returns")
    fig.supxlabel("Algorithm")
    fig.tight_layout(rect=(0.01, 0.045, 0.997, 0.93), pad=0.45, w_pad=0.3, h_pad=0.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_preset(
    preset_name: str,
    preset: PlotPreset,
    *,
    output_dir: Path,
    x_key: str,
    y_key: str,
    smoothing_window: int,
    ci_level: float,
    dpi: int,
    show_stage_bands: bool,
    show_stage_labels: bool,
    stage_alpha: float,
    separator_alpha: float,
    save_stage_panels: bool,
    stage_panel_columns: int,
    stage_panel_relative_x: bool,
    reset_smoothing_at_stage_boundaries: bool,
    save_legend_separately: bool,
    merge_identical_adjacent_stages: bool,
    save_final_eval_boxplots: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / preset.output_filename

    fig, axis = plt.subplots(figsize=(9.0, 5.5), dpi=dpi)
    stage_segments = load_stage_segments(REPO_ROOT / preset.stage_source, x_key=x_key)
    if merge_identical_adjacent_stages:
        stage_segments = merge_adjacent_stage_segments(stage_segments)

    if show_stage_bands:
        for segment_index, segment in enumerate(stage_segments):
            face_color = STAGE_FACE_COLORS[segment.stage_indices[0] % len(STAGE_FACE_COLORS)]
            axis.axvspan(segment.start_step, segment.end_step, color=face_color, alpha=stage_alpha, linewidth=0)
            if segment_index > 0:
                axis.axvline(segment.start_step, color="#666666", linewidth=0.8, alpha=separator_alpha, linestyle="--")
            if show_stage_labels:
                axis.text(
                    (segment.start_step + segment.end_step) / 2.0,
                    0.98,
                    segment.label,
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#444444",
                    alpha=0.80,
                )

    z_value = confidence_scale(ci_level)
    for series in preset.series:
        run_paths = sorted(REPO_ROOT.glob(series.glob_pattern))
        if not run_paths:
            raise FileNotFoundError(f"No run histories matched pattern: {series.glob_pattern}")
        x_values, mean, stderr = aggregate_runs(
            run_paths,
            x_key=x_key,
            y_key=y_key,
            smoothing_window=smoothing_window,
            reset_smoothing_at_stage_boundaries=reset_smoothing_at_stage_boundaries,
        )
        band = z_value * stderr
        axis.plot(x_values, mean, label=series.label, color=series.color, linewidth=2.0)
        axis.fill_between(x_values, mean - band, mean + band, color=series.color, alpha=0.18)

    axis.set_title(simplified_task_title(preset.title))
    x_label = display_x_label(x_key)
    axis.set_xlabel(scaled_step_xlabel(x_label) if uses_step_axis(x_key) else x_label)
    axis.set_ylabel(y_key.replace("_", " ").title())
    apply_step_axis_format(axis, x_key=x_key)
    axis.grid(True, alpha=0.18)
    if not save_legend_separately:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    if save_stage_panels and stage_segments:
        plot_stage_panels(
            preset,
            output_path=stage_panel_output_path(output_path),
            stage_segments=stage_segments,
            x_key=x_key,
            y_key=y_key,
            smoothing_window=smoothing_window,
            ci_level=ci_level,
            dpi=dpi,
            columns=stage_panel_columns,
            relative_x=stage_panel_relative_x,
            reset_smoothing_at_stage_boundaries=reset_smoothing_at_stage_boundaries,
            save_legend_separately=save_legend_separately,
        )
    if save_final_eval_boxplots:
        plot_final_eval_boxplots(
            preset,
            output_path=eval_boxplot_output_path(output_path),
            dpi=dpi,
        )
    if save_legend_separately:
        save_standalone_legend(output_path, preset.series, dpi=dpi)
    return output_path


def plot_stage_panels(
    preset: PlotPreset,
    *,
    output_path: Path,
    stage_segments: Sequence[StageSegment],
    x_key: str,
    y_key: str,
    smoothing_window: int,
    ci_level: float,
    dpi: int,
    columns: int,
    relative_x: bool,
    reset_smoothing_at_stage_boundaries: bool,
    save_legend_separately: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = max(1, min(int(columns), len(stage_segments)))
    nrows = int(math.ceil(len(stage_segments) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.15 * nrows),
        dpi=dpi,
        sharey=False,
        squeeze=False,
    )
    z_value = confidence_scale(ci_level)

    for axis in axes.flat[len(stage_segments) :]:
        axis.set_visible(False)

    for axis, segment in zip(axes.flat, stage_segments):
        for series in preset.series:
            run_paths = sorted(REPO_ROOT.glob(series.glob_pattern))
            if not run_paths:
                raise FileNotFoundError(f"No run histories matched pattern: {series.glob_pattern}")
            x_values, mean, stderr = aggregate_runs(
                run_paths,
                x_key=x_key,
                y_key=y_key,
                smoothing_window=smoothing_window,
                reset_smoothing_at_stage_boundaries=reset_smoothing_at_stage_boundaries,
                x_min=segment.start_step,
                x_max=segment.end_step,
                relative_to=segment.start_step if relative_x else None,
            )
            band = z_value * stderr
            axis.plot(x_values, mean, color=series.color, linewidth=2.0, label=series.label)
            axis.fill_between(x_values, mean - band, mean + band, color=series.color, alpha=0.18)

        axis.set_title(segment.label, fontsize=10, fontweight="bold")
        axis.grid(True, alpha=0.18)
        axis.set_ylabel(y_key.replace("_", " ").title())
        apply_step_axis_format(axis, x_key=x_key)

    fig.suptitle(f"{simplified_task_title(preset.title)}: by curriculum stage", fontsize=12, fontweight="bold", y=0.975)
    if not save_legend_separately:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=max(1, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 0.945))
        fig.tight_layout(rect=(0.01, 0.05, 0.995, 0.84), pad=0.55, w_pad=0.45, h_pad=0.75)
    else:
        fig.tight_layout(rect=(0.01, 0.05, 0.995, 0.90), pad=0.55, w_pad=0.45, h_pad=0.75)
    stage_x_label = "Environment Steps Within Stage" if relative_x else display_x_label(x_key)
    fig.supxlabel(scaled_step_xlabel(stage_x_label) if uses_step_axis(x_key) else stage_x_label)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot final multi-seed learning curves with CI bands.")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="Preset(s) to render. Defaults to all maintained presets.",
    )
    parser.add_argument("--list-presets", action="store_true", help="Print the available preset names and exit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for written plots.")
    parser.add_argument("--x-key", default=DEFAULT_X_KEY, help="CSV column used for the x-axis.")
    parser.add_argument("--y-key", default=DEFAULT_Y_KEY, help="CSV column used for the y-axis.")
    parser.add_argument("--window", type=int, default=DEFAULT_SMOOTHING_WINDOW, help="Rolling-mean window size.")
    parser.add_argument(
        "--reset-smoothing-at-stage-boundaries",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESET_SMOOTHING_AT_STAGE_BOUNDARIES,
        help="Reset the rolling-mean window at curriculum stage boundaries instead of smoothing across them.",
    )
    parser.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL, help="Confidence interval level, e.g. 0.95.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure DPI.")
    parser.add_argument("--show-stage-bands", action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW_STAGE_BANDS)
    parser.add_argument("--show-stage-labels", action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW_STAGE_LABELS)
    parser.add_argument("--stage-alpha", type=float, default=DEFAULT_STAGE_ALPHA, help="Background alpha for curriculum bands.")
    parser.add_argument("--separator-alpha", type=float, default=DEFAULT_SEPARATOR_ALPHA, help="Alpha for stage separator lines.")
    parser.add_argument(
        "--save-stage-panels",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_STAGE_PANELS,
        help="Also save a secondary figure with one subplot per curriculum stage.",
    )
    parser.add_argument("--stage-panel-columns", type=int, default=DEFAULT_STAGE_PANEL_COLUMNS, help="Maximum subplot columns for the stage-panel figure.")
    parser.add_argument(
        "--stage-panel-relative-x",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STAGE_PANEL_RELATIVE_X,
        help="Plot stage-panel x-axis relative to each stage start instead of global step.",
    )
    parser.add_argument(
        "--save-legend-separately",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_LEGEND_SEPARATELY,
        help="Omit legends inside the plot figures and write a separate *_legend.png image instead.",
    )
    parser.add_argument(
        "--merge-identical-adjacent-stages",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MERGE_IDENTICAL_ADJACENT_STAGES,
        help="Merge consecutive curriculum stages when they use the same roster-count set.",
    )
    parser.add_argument(
        "--save-final-eval-boxplots",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_FINAL_EVAL_BOXPLOTS,
        help="Also save a final-evaluation boxplot figure using one final eval mean per seed and roster size.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = DEFAULT_FONT_FAMILY
    plt.rcParams["mathtext.it"] = f"{DEFAULT_FONT_FAMILY}:italic"
    plt.rcParams["mathtext.bf"] = f"{DEFAULT_FONT_FAMILY}:bold"

    if args.list_presets:
        for preset_name in sorted(PRESETS):
            print(preset_name)
        return 0

    selected_presets = args.preset or list(DEFAULT_PRESETS)
    for preset_name in selected_presets:
        output_path = plot_preset(
            preset_name,
            PRESETS[preset_name],
            output_dir=args.output_dir,
            x_key=args.x_key,
            y_key=args.y_key,
            smoothing_window=args.window,
            reset_smoothing_at_stage_boundaries=bool(args.reset_smoothing_at_stage_boundaries),
            ci_level=args.ci_level,
            dpi=args.dpi,
            show_stage_bands=bool(args.show_stage_bands),
            show_stage_labels=bool(args.show_stage_labels),
            stage_alpha=float(args.stage_alpha),
            separator_alpha=float(args.separator_alpha),
            save_stage_panels=bool(args.save_stage_panels),
            stage_panel_columns=int(args.stage_panel_columns),
            stage_panel_relative_x=bool(args.stage_panel_relative_x),
            save_legend_separately=bool(args.save_legend_separately),
            merge_identical_adjacent_stages=bool(args.merge_identical_adjacent_stages),
            save_final_eval_boxplots=bool(args.save_final_eval_boxplots),
        )
        print(output_path)
        if bool(args.save_stage_panels):
            print(stage_panel_output_path(output_path))
        if bool(args.save_final_eval_boxplots):
            print(eval_boxplot_output_path(output_path))
        if bool(args.save_legend_separately):
            print(legend_output_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
