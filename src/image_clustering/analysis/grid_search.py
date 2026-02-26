from __future__ import annotations

from typing import Dict, Any
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from image_clustering.reduce.umap_reduce import run_umap
from image_clustering.cluster.hdbscan_cluster import run_hdbscan
from image_clustering.cluster.evaluate import evaluate


def run_grid_search(
    cfg: Dict[str, Any],
    X: np.ndarray,
    meta: pd.DataFrame,
    output_root: Path,
) -> None:
    """
    Runs a grid search over UMAP + HDBSCAN parameters and writes a single CSV:
      outputs/grid_search_<date_stamp>.csv

    Notes:
    - Clustering space is taken from cfg["clustering"]["space"] ("umap" or "embedding").
    - UMAP is always computed because it is needed for "umap" clustering and for potential
      future diagnostics; if clustering_space == "embedding" then UMAP params do not
      affect clustering, but are still logged.
    - Embeddings are assumed already computed and cached upstream.
    """

    grid_cfg = cfg["grid_search"]
    date_stamp = cfg.get("export", {}).get("date_stamp", "grid")
    clustering_space = cfg["clustering"]["space"].lower()

    if clustering_space not in {"umap", "embedding"}:
        raise ValueError("clustering.space must be 'umap' or 'embedding'.")

    results = []

    for n_neighbors, min_dist in product(
        grid_cfg["umap"]["n_neighbors"],
        grid_cfg["umap"]["min_dist"],
    ):
        # temporarily update config for UMAP
        cfg["umap"]["n_neighbors"] = int(n_neighbors)
        cfg["umap"]["min_dist"] = float(min_dist)

        coords = run_umap(cfg, X, meta)

        for min_cluster_size, min_samples in product(
            grid_cfg["hdbscan"]["min_cluster_size"],
            grid_cfg["hdbscan"]["min_samples"],
        ):
            cfg["hdbscan"]["min_cluster_size"] = int(min_cluster_size)
            cfg["hdbscan"]["min_samples"] = int(min_samples)

            if clustering_space == "umap":
                data = coords[["u1", "u2"]].to_numpy()
            else:
                data = X

            labels, probs, _ = run_hdbscan(cfg, data)

            metrics = evaluate(
                cfg,
                labels_full=labels,
                data_for_metrics=data,
            )

            row = {
                "clustering_space": clustering_space,
                "umap_n_neighbors": int(n_neighbors),
                "umap_min_dist": float(min_dist),
                "hdbscan_min_cluster_size": int(min_cluster_size),
                "hdbscan_min_samples": int(min_samples),
            }
            row.update(metrics)
            results.append(row)

            print(
                f"nn={n_neighbors} md={min_dist} "
                f"mcs={min_cluster_size} ms={min_samples} "
                f"dbcv={metrics.get('dbcv_filtered')}"
            )

    df = pd.DataFrame(results)

    # ---- ranking score (higher is better) ----
    # Conservative handling of missing values:
    # - missing DBCV or silhouette -> 0
    # - missing noise_fraction -> 1 (worst penalty)
    dbcv = df["dbcv_filtered"].astype(float, errors="ignore").fillna(0.0)
    sil = df["silhouette_filtered"].astype(float, errors="ignore").fillna(0.0)
    noise = df["noise_fraction"].astype(float, errors="ignore").fillna(1.0)

    df["rank_score"] = 0.5 * dbcv + 0.3 * sil - 0.2 * noise

    # Sort by rank_score (desc), then DBCV (desc), then silhouette (desc), then noise (asc)
    df = df.sort_values(
        by=["rank_score", "dbcv_filtered", "silhouette_filtered", "noise_fraction"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    out_path = output_root / f"grid_search_{date_stamp}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nGrid search saved to: {out_path}")
    print("Top 5 by rank_score:")
    print(df.head(5)[
        [
            "rank",
            "rank_score",
            "dbcv_filtered",
            "silhouette_filtered",
            "noise_fraction",
            "n_clusters",
            "largest_cluster_share",
            "cluster_entropy",
            "umap_n_neighbors",
            "umap_min_dist",
            "hdbscan_min_cluster_size",
            "hdbscan_min_samples",
        ]
    ])