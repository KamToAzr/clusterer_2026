from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    LabelSet,
)
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, save


def _format_metrics_html(metrics: Dict[str, Any]) -> str:
    def fmt(x):
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    keys = [
        "n_total",
        "n_clustered",
        "n_noise",
        "noise_fraction",
        "silhouette_filtered",
        "dbcv_filtered",
        "exclude_noise_for_metrics",
        "hdbscan_metric",
    ]
    rows = []
    for k in keys:
        if k in metrics:
            rows.append(f"<tr><td><b>{k}</b></td><td>{fmt(metrics[k])}</td></tr>")

    return (
        "<div style='max-width:900px;'>"
        "<h3 style='margin:6px 0 8px 0;'>Run metrics</h3>"
        "<table style='border-collapse:collapse;'>"
        + "".join(rows)
        + "</table></div>"
    )


def render_bokeh(
    cfg: Dict[str, Any],
    assignments: pd.DataFrame,
    out_path: str | Path,
    metrics: Optional[Dict[str, Any]] = None,
):
    # columns for coordinates
    coord_cols = [c for c in ["u1", "u2", "u3"] if c in assignments.columns]
    if "u1" not in assignments.columns or "u2" not in assignments.columns:
        raise ValueError("assignments must contain at least u1 and u2 columns.")

    # split noise vs clustered for easy toggling
    noise_df = assignments[assignments["cluster_label"] == -1].copy()
    clus_df = assignments[assignments["cluster_label"] != -1].copy()

    # assign colors to clusters (noise gets grey)
    palette = Category20[20]
    unique_clusters = sorted(clus_df["cluster_label"].unique().tolist())

    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_clusters)}
    clus_df["color"] = clus_df["cluster_label"].map(color_map)
    noise_df["color"] = "#cccccc"

    # sources
    src_clustered = ColumnDataSource(clus_df)
    src_noise = ColumnDataSource(noise_df)

    # centroids (for labels)
    centroids = (
        clus_df.groupby("cluster_label", as_index=False)[["u1", "u2"]].mean()
        if len(clus_df) > 0
        else pd.DataFrame(columns=["cluster_label", "u1", "u2"])
    )
    centroids["label"] = centroids["cluster_label"].astype(str)
    src_centroids = ColumnDataSource(centroids)

    p = figure(
        title="UMAP + HDBSCAN clustering",
        width=900,
        height=700,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("file", "@filename"),
                ("cluster", "@cluster_label"),
                ("prob", "@probability"),
            ]
        )
    )

    r_clustered = p.scatter(
        x="u1",
        y="u2",
        source=src_clustered,
        color="color",
        size=6,
        alpha=0.75,
        legend_label="clustered",
    )

    r_noise = p.scatter(
        x="u1",
        y="u2",
        source=src_noise,
        color="color",
        size=6,
        alpha=0.35,
        legend_label="noise (-1)",
    )

    # centroid labels
    labels = LabelSet(
        x="u1",
        y="u2",
        text="label",
        source=src_centroids,
        text_font_size="10pt",
        x_offset=6,
        y_offset=6,
    )
    p.add_layout(labels)

    p.legend.location = "top_left"

    # noise toggle (simple + robust): toggle renderer visibility
    checkbox = CheckboxGroup(labels=["Show noise"], active=[0 if cfg.get("viz", {}).get("bokeh", {}).get("show_noise_default", True) else -1])
    # set initial visibility
    r_noise.visible = cfg.get("viz", {}).get("bokeh", {}).get("show_noise_default", True)

    checkbox.js_on_change(
        "active",
        CustomJS(
            args=dict(r_noise=r_noise),
            code="""
            // active contains indices of checked boxes
            r_noise.visible = (cb_obj.active.length > 0);
        """,
        ),
    )

    metrics_div = Div(
        text=_format_metrics_html(metrics or {}),
        sizing_mode="stretch_width",
    )

    layout = column(p, checkbox, metrics_div)

    out_path = Path(out_path)
    output_file(str(out_path), mode="inline")
    save(layout)