"""Reusable plotting helpers shared by Optuna analysis and final-results plotters."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SPLIT_FACE_COLORS: dict[str, str] = {
    "train": "#eef6ff",
    "validation": "#fff6df",
    "test": "#fdecec",
    "other": "#f4f4f4",
}
ALGORITHM_DISPLAY_NAMES: dict[str, str] = {
    "mappo": "MAPPO",
    "pimac_v0": "PIC-MAPPO",
    "pimac_v6": "PC3D",
    "ippo": "IPPO",
}


def display_algorithm_name(algorithm: str) -> str:
    return ALGORITHM_DISPLAY_NAMES.get(str(algorithm), str(algorithm))


def _ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _algorithm_palette(algorithms: Sequence[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    return {algorithm: cmap(index % 10) for index, algorithm in enumerate(algorithms)}


def _count_color_values(rows: Sequence[dict[str, Any]]) -> list[int]:
    values = sorted({int(row["n_agents"]) for row in rows})
    if not values:
        return [0]
    return values


def _build_count_norm(counts: Sequence[int]) -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    ordered = sorted(int(value) for value in counts)
    cmap = plt.get_cmap("viridis", len(ordered))
    boundaries = np.arange(len(ordered) + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _n_agents_color_index_map(counts: Sequence[int]) -> dict[int, int]:
    return {int(count): index for index, count in enumerate(sorted(int(value) for value in counts))}


def _project_pca(matrix: np.ndarray, *, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("PCA projection expects a 2D matrix.")
    n_components = int(n_components)
    if n_components <= 0:
        raise ValueError("PCA projection expects a positive component count.")
    if array.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float32), np.zeros(n_components, dtype=np.float32)
    centered = array - np.mean(array, axis=0, keepdims=True)
    if centered.shape[0] == 1 or centered.shape[1] == 0:
        return np.zeros((centered.shape[0], n_components), dtype=np.float32), np.zeros(n_components, dtype=np.float32)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    if projected.shape[1] < n_components:
        projected = np.pad(projected, ((0, 0), (0, n_components - projected.shape[1])), mode="constant")
    explained_variance = (singular_values ** 2) / max(1, centered.shape[0] - 1)
    explained_total = float(np.sum(explained_variance))
    if explained_total <= 1e-12:
        explained_ratio = np.zeros(n_components, dtype=np.float32)
    else:
        explained_ratio = np.zeros(n_components, dtype=np.float32)
        limit = min(n_components, explained_variance.shape[0])
        explained_ratio[:limit] = explained_variance[:limit] / explained_total
    return projected.astype(np.float32), explained_ratio.astype(np.float32)


def build_grouped_pca_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str = "run_label",
    dim_prefix: str = "dim_",
    max_points_per_group: int = 4000,
    sample_seed: int = 0,
    n_components: int = 2,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    rng = np.random.default_rng(int(sample_seed))
    projected_rows: list[dict[str, Any]] = []
    for group_name, group_rows in grouped.items():
        if not group_rows:
            continue
        dim_keys = sorted(key for key in group_rows[0] if key.startswith(dim_prefix))
        if not dim_keys:
            continue
        if len(group_rows) > int(max_points_per_group):
            sampled_indices = np.sort(rng.choice(len(group_rows), size=int(max_points_per_group), replace=False))
            sampled_rows = [group_rows[int(index)] for index in sampled_indices]
        else:
            sampled_rows = list(group_rows)
        matrix = np.asarray(
            [[float(row[key]) for key in dim_keys] for row in sampled_rows],
            dtype=np.float32,
        )
        projected, explained_ratio = _project_pca(matrix, n_components=n_components)
        for row, coords in zip(sampled_rows, projected):
            projected_row = {
                "run_label": group_name,
                "algorithm": row.get("algorithm"),
                "rank": row.get("rank"),
                "trial_number": row.get("trial_number"),
                "n_agents": int(row["n_agents"]),
            }
            for component_index in range(int(n_components)):
                projected_row[f"pc{component_index + 1}"] = float(coords[component_index])
                projected_row[f"pca_explained_pc{component_index + 1}"] = float(explained_ratio[component_index])
            projected_rows.append(projected_row)
    return projected_rows


def plot_dynamic_trial_return_boxplots(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    algorithms: Sequence[str],
    task_title: str,
) -> Path:
    if not rows:
        raise ValueError("Dynamic trial boxplot requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    ordered_counts = sorted({int(row["n_agents"]) for row in rows})
    ordered_algorithms = [algorithm for algorithm in algorithms if any(row["algorithm"] == algorithm for row in rows)]
    if not ordered_algorithms:
        ordered_algorithms = sorted({str(row["algorithm"]) for row in rows})

    palette = _algorithm_palette(ordered_algorithms)
    all_returns = [float(row["return_mean"]) for row in rows]
    y_min = min(all_returns)
    y_max = max(all_returns)
    y_pad = max(1e-6, 0.05 * (y_max - y_min if y_max > y_min else max(abs(y_min), 1.0)))

    ncols = min(4, max(1, len(ordered_counts)))
    nrows = int(math.ceil(len(ordered_counts) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 3.2 * nrows), sharey=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes_array.flat[len(ordered_counts) :]:
        axis.set_visible(False)

    split_patches: dict[str, Patch] = {}
    positions = np.arange(1, len(ordered_algorithms) + 1)
    for axis, n_agents in zip(axes_array.flat, ordered_counts):
        count_rows = [row for row in rows if int(row["n_agents"]) == n_agents]
        split_name = str(count_rows[0].get("split", "other"))
        axis.set_facecolor(SPLIT_FACE_COLORS.get(split_name, SPLIT_FACE_COLORS["other"]))
        split_patches.setdefault(
            split_name,
            Patch(facecolor=SPLIT_FACE_COLORS.get(split_name, SPLIT_FACE_COLORS["other"]), edgecolor="none", label=split_name),
        )

        data: list[np.ndarray] = []
        nonempty_positions: list[int] = []
        for position, algorithm in zip(positions, ordered_algorithms):
            values = np.asarray(
                [float(row["return_mean"]) for row in count_rows if row["algorithm"] == algorithm],
                dtype=np.float32,
            )
            if values.size == 0:
                continue
            data.append(values)
            nonempty_positions.append(int(position))
        if data:
            boxplot = axis.boxplot(
                data,
                positions=nonempty_positions,
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "#111111", "markeredgecolor": "#111111", "markersize": 3.5},
                medianprops={"color": "#222222", "linewidth": 1.2},
                whiskerprops={"color": "#555555", "linewidth": 1.0},
                capprops={"color": "#555555", "linewidth": 1.0},
            )
            for patch, position in zip(boxplot["boxes"], nonempty_positions):
                algorithm = ordered_algorithms[position - 1]
                patch.set_facecolor(palette[algorithm])
                patch.set_alpha(0.88)
                patch.set_edgecolor("#444444")

        axis.set_title(f"N={n_agents} ({split_name})", fontsize=10, fontweight="bold")
        axis.set_xticks(positions)
        axis.set_xticklabels([display_algorithm_name(algorithm) for algorithm in ordered_algorithms], rotation=30, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.tick_params(axis="y", labelsize=8)

    for axis in axes_array[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("Final eval return", fontsize=9)
    for axis in axes_array[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Algorithm", fontsize=9)

    fig.suptitle(f"{task_title}: final-checkpoint return distributions across completed sweep trials", fontsize=12, fontweight="bold")
    fig.legend(
        handles=[split_patches[key] for key in ("train", "validation", "test", "other") if key in split_patches],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, len(split_patches)),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
    fig.savefig(resolved_output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path


def plot_pca_projection_grid(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    dpi: int = 300,
) -> Path:
    if not rows:
        raise ValueError("PCA projection plot requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    run_labels = list(dict.fromkeys(str(row["run_label"]) for row in rows))
    count_values = _count_color_values(rows)
    color_index = _n_agents_color_index_map(count_values)
    cmap, norm = _build_count_norm(count_values)

    ncols = min(3, max(1, len(run_labels)))
    nrows = int(math.ceil(len(run_labels) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for axis in axes.flat[len(run_labels) :]:
        axis.set_visible(False)

    scatter_artist = None
    for axis, run_label in zip(axes.flat, run_labels):
        run_rows = [row for row in rows if str(row["run_label"]) == run_label]
        pc1 = np.asarray([float(row["pc1"]) for row in run_rows], dtype=np.float32)
        pc2 = np.asarray([float(row["pc2"]) for row in run_rows], dtype=np.float32)
        colors = np.asarray([color_index[int(row["n_agents"])] for row in run_rows], dtype=np.float32)
        scatter_artist = axis.scatter(
            pc1,
            pc2,
            c=colors,
            cmap=cmap,
            norm=norm,
            alpha=0.55,
            s=12,
            edgecolors="none",
            rasterized=True,
        )
        explained_1 = 100.0 * float(run_rows[0]["pca_explained_pc1"])
        explained_2 = 100.0 * float(run_rows[0]["pca_explained_pc2"])
        axis.set_title(f"{run_label}\nPC1 {explained_1:.1f}% | PC2 {explained_2:.1f}%", fontsize=9, fontweight="bold")
        axis.set_xlabel("PC1", fontsize=9)
        axis.set_ylabel("PC2", fontsize=9)
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.07, top=0.84, wspace=0.28, hspace=0.50)
    if scatter_artist is not None:
        colorbar = fig.colorbar(scatter_artist, ax=[axis for axis in axes.flat if axis.get_visible()], fraction=0.02, pad=0.02)
        colorbar.set_ticks(np.arange(len(count_values)))
        colorbar.set_ticklabels([str(value) for value in count_values])
        colorbar.set_label("Number of agents", fontsize=9)
    fig.savefig(resolved_output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path


def plot_task_count_alignment_gate_heatmap(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 300,
    font_family: str = "Charter",
) -> Path:
    if not rows:
        raise ValueError("Task-count alignment heatmap requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    task_labels = list(dict.fromkeys(str(row["task_label"]) for row in rows))
    counts = sorted({int(row["n_agents"]) for row in rows})
    alignment_mean = np.full((len(task_labels), len(counts)), np.nan, dtype=np.float32)
    alignment_std = np.full_like(alignment_mean, np.nan)
    gate_mean = np.full_like(alignment_mean, np.nan)
    gate_std = np.full_like(alignment_mean, np.nan)

    for row in rows:
        task_index = task_labels.index(str(row["task_label"]))
        count_index = counts.index(int(row["n_agents"]))
        alignment_mean[task_index, count_index] = float(row["alignment_mean"])
        alignment_std[task_index, count_index] = float(row["alignment_std"])
        gate_value = row.get("gate_mean")
        if gate_value is not None:
            gate_mean[task_index, count_index] = float(gate_value)
        gate_std_value = row.get("gate_std")
        if gate_std_value is not None:
            gate_std[task_index, count_index] = float(gate_std_value)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": [font_family, "XCharter", "Bitstream Charter", "DejaVu Serif"],
        }
    ):
        masked_alignment = np.ma.masked_invalid(alignment_mean)
        fig_height = max(2.2, 0.62 * len(task_labels) + 1.55)
        fig, axis = plt.subplots(figsize=(1.18 * len(counts) + 2.6, fig_height))
        image = axis.imshow(masked_alignment, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(counts)))
        axis.set_xticklabels([str(value) for value in counts], fontsize=10)
        axis.set_yticks([])
        axis.set_xlabel("Number of agents", fontsize=12)
        if title:
            axis.set_title(title, fontsize=12, fontweight="bold")

        for row_index in range(alignment_mean.shape[0]):
            for col_index in range(alignment_mean.shape[1]):
                if not np.isfinite(alignment_mean[row_index, col_index]):
                    continue
                if not np.isfinite(gate_mean[row_index, col_index]):
                    cell_text = "NA"
                else:
                    cell_text = f"{gate_mean[row_index, col_index]:.2f}\n±{gate_std[row_index, col_index]:.2f}"
                text_color = "white" if alignment_mean[row_index, col_index] < 0.55 else "#111111"
                axis.text(
                    col_index,
                    row_index,
                    cell_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        colorbar.set_label("Cosine alignment", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(resolved_output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return resolved_output_path


def plot_alignment_heatmap(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    dpi: int = 300,
) -> Path:
    if not rows:
        raise ValueError("Alignment heatmap requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    run_labels = list(dict.fromkeys(str(row["run_label"]) for row in rows))
    counts = sorted({int(row["n_agents"]) for row in rows})
    matrix = np.full((len(run_labels), len(counts)), np.nan, dtype=np.float32)
    for row in rows:
        run_index = run_labels.index(str(row["run_label"]))
        count_index = counts.index(int(row["n_agents"]))
        matrix[run_index, count_index] = float(row["teacher_alignment_cosine_mean"])

    masked = np.ma.masked_invalid(matrix)
    fig_height = max(3.0, 0.55 * len(run_labels) + 1.8)
    fig, axis = plt.subplots(figsize=(1.25 * len(counts) + 2.6, fig_height))
    image = axis.imshow(masked, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(counts)))
    axis.set_xticklabels([str(value) for value in counts], fontsize=9)
    axis.set_yticks(np.arange(len(run_labels)))
    axis.set_yticklabels(run_labels, fontsize=9)
    axis.set_xlabel("Number of agents", fontsize=10)
    axis.set_ylabel("Run", fontsize=10)
    axis.set_title(title, fontsize=12, fontweight="bold")

    if len(run_labels) <= 12 and len(counts) <= 12:
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                if np.isfinite(matrix[row_index, col_index]):
                    axis.text(
                        col_index,
                        row_index,
                        f"{matrix[row_index, col_index]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[row_index, col_index] < 0.45 else "#111111",
                        fontsize=8,
                    )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("Teacher-student cosine alignment", fontsize=9)
    fig.tight_layout()
    fig.savefig(resolved_output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path


def plot_gate_alignment_3d(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    if not rows:
        raise ValueError("Gate/alignment 3D plot requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    count_values = _count_color_values(rows)
    color_index = _n_agents_color_index_map(count_values)
    cmap, norm = _build_count_norm(count_values)

    x = np.asarray([float(row["gate_mean"]) for row in rows], dtype=np.float32)
    y = np.asarray([float(row["gate_std"]) for row in rows], dtype=np.float32)
    z = np.asarray([float(row["teacher_alignment_cosine_mean"]) for row in rows], dtype=np.float32)
    c = np.asarray([color_index[int(row["n_agents"])] for row in rows], dtype=np.float32)

    fig = plt.figure(figsize=(8.4, 6.6))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(x, y, z, c=c, cmap=cmap, norm=norm, s=42, alpha=0.85, edgecolors="#333333", linewidths=0.3)
    axis.set_xlabel("Gate mean", labelpad=8)
    axis.set_ylabel("Gate std", labelpad=8)
    axis.set_zlabel("Teacher-student cosine", labelpad=8)
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.view_init(elev=24, azim=-58)
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.03, pad=0.08)
    colorbar.set_ticks(np.arange(len(count_values)))
    colorbar.set_ticklabels([str(value) for value in count_values])
    colorbar.set_label("Number of agents", fontsize=9)
    fig.tight_layout()
    fig.savefig(resolved_output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path


def plot_gate_agent_heatmap(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    bins: int = 24,
) -> Path:
    if not rows:
        raise ValueError("Gate/agent heatmap requires at least one row.")
    resolved_output_path = _ensure_parent(output_path)

    gate_mean = np.asarray([float(row["gate_mean"]) for row in rows], dtype=np.float32)
    gate_std = np.asarray([float(row["gate_std"]) for row in rows], dtype=np.float32)
    n_agents = np.asarray([int(row["n_agents"]) for row in rows], dtype=np.int32)

    x_min, x_max = float(np.min(gate_mean)), float(np.max(gate_mean))
    y_min, y_max = float(np.min(gate_std)), float(np.max(gate_std))
    if math.isclose(x_min, x_max):
        x_min -= 0.05
        x_max += 0.05
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05

    x_edges = np.linspace(x_min, x_max, int(bins) + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, int(bins) + 1, dtype=np.float32)
    cell_agents: dict[tuple[int, int], list[int]] = defaultdict(list)
    for x_value, y_value, count in zip(gate_mean, gate_std, n_agents):
        x_bin = min(int(np.digitize(x_value, x_edges, right=False)) - 1, len(x_edges) - 2)
        y_bin = min(int(np.digitize(y_value, y_edges, right=False)) - 1, len(y_edges) - 2)
        x_bin = max(0, x_bin)
        y_bin = max(0, y_bin)
        cell_agents[(y_bin, x_bin)].append(int(count))

    present_counts = sorted({int(value) for value in n_agents})
    count_to_color_index = _n_agents_color_index_map(present_counts)
    heatmap = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=np.float32)
    for (y_bin, x_bin), values in cell_agents.items():
        modal_count = Counter(values).most_common(1)[0][0]
        heatmap[y_bin, x_bin] = float(count_to_color_index[int(modal_count)])

    cmap, norm = _build_count_norm(present_counts)
    masked = np.ma.masked_invalid(heatmap)
    fig, axis = plt.subplots(figsize=(7.4, 5.8))
    mesh = axis.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm, shading="auto")
    axis.scatter(gate_mean, gate_std, c="#111111", s=10, alpha=0.18, linewidths=0.0)
    axis.set_xlabel("Gate mean", fontsize=10)
    axis.set_ylabel("Gate std", fontsize=10)
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.grid(True, alpha=0.15)
    colorbar = fig.colorbar(mesh, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_ticks(np.arange(len(present_counts)))
    colorbar.set_ticklabels([str(value) for value in present_counts])
    colorbar.set_label("Modal number of agents per bin", fontsize=9)
    fig.tight_layout()
    fig.savefig(resolved_output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path
