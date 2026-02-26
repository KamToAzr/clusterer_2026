from __future__ import annotations

from typing import Dict, Any
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np

from image_clustering.reduce.umap_reduce import run_umap
from image_clustering.cluster.hdbscan_cluster import run_hdbscan
from image_clustering.cluster.evaluate import evaluate


def run_grid_search(
    cfg: Dict[str, Any],
    X: np.ndarray,
    meta: pd.DataFrame,
    output_root: Path,
):

    grid_cfg = cfg["grid_search"]
    date_stamp = cfg.get("export", {}).get("date_stamp", "grid")

    clustering_space = cfg["clustering"]["space"]

    results = []

    for n_neighbors, min_dist in product(
        grid_cfg["umap"]["n_neighbors"],
        grid_cfg["umap"]["min_dist"],
    ):

        # temporarily update config
        cfg["umap"]["n_neighbors"] = n_neighbors
        cfg["umap"]["min_dist"] = min_dist

        coords = run_umap(cfg, X, meta)

        for min_cluster_size, min_samples in product(
            grid_cfg["hdbscan"]["min_cluster_size"],
            grid_cfg["hdbscan"]["min_samples"],
        ):

            cfg["hdbscan"]["min_cluster_size"] = min_cluster_size
            cfg["hdbscan"]["min_samples"] = min_samples

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
                "umap_n_neighbors": n_neighbors,
                "umap_min_dist": min_dist,
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_samples,
            }

            row.update(metrics)
            results.append(row)

            print(
                f"nn={n_neighbors} md={min_dist} "
                f"mcs={min_cluster_size} ms={min_samples} "
                f"dbcv={metrics.get('dbcv_filtered')}"
            )

    df = pd.DataFrame(results)

    out_path = output_root / f"grid_search_{date_stamp}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nGrid search saved to: {out_path}")

