from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from hdbscan.validity import validity_index


def evaluate(
    cfg: Dict[str, Any],
    labels_full: np.ndarray,
    data_for_metrics: np.ndarray,
) -> dict:
    """
    Evaluates clustering on the provided data matrix (embedding or UMAP space),
    optionally excluding noise (-1).
    """
    exclude_noise = bool(cfg["evaluation"]["exclude_noise"])
    metric = str(cfg["hdbscan"]["metric"])

    n_total = int(len(labels_full))
    n_noise = int((labels_full == -1).sum())
    n_clustered = int(n_total - n_noise)

    metrics = {
        "n_total": n_total,
        "n_noise": n_noise,
        "n_clustered": n_clustered,
        "noise_fraction": (n_noise / n_total) if n_total else None,
        "exclude_noise_for_metrics": exclude_noise,
        "hdbscan_metric": metric,
    }

    if not exclude_noise:
        Y = data_for_metrics
        labels = labels_full
    else:
        mask = labels_full != -1
        Y = data_for_metrics[mask]
        labels = labels_full[mask]

        # remap labels to 0..k-1 for DBCV stability
        uniq = np.unique(labels)
        remap = {old: i for i, old in enumerate(uniq)}
        labels = np.array([remap[x] for x in labels], dtype=int)

    if len(Y) >= 2 and len(np.unique(labels)) >= 2:
        metrics["silhouette_filtered"] = float(silhouette_score(Y, labels))
        metrics["dbcv_filtered"] = float(validity_index(Y.astype(np.float64, copy=False), labels, metric=metric))
    else:
        metrics["silhouette_filtered"] = None
        metrics["dbcv_filtered"] = None

    return metrics