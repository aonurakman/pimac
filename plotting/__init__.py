"""Shared plotting helpers and entry points for final-results visualization."""

from .shared import (
    build_grouped_pca_rows,
    plot_alignment_heatmap,
    plot_dynamic_trial_return_boxplots,
    plot_gate_agent_heatmap,
    plot_gate_alignment_3d,
    plot_pca_projection_grid,
)

__all__ = [
    "build_grouped_pca_rows",
    "plot_alignment_heatmap",
    "plot_dynamic_trial_return_boxplots",
    "plot_gate_agent_heatmap",
    "plot_gate_alignment_3d",
    "plot_pca_projection_grid",
]

