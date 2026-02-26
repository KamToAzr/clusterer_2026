from __future__ import annotations

from typing import Dict, Any
import numpy as np
import warnings

from sklearn.metrics import silhouette_score
from hdbscan.validity import validity_index


def _cluster_distribution_metrics(labels_full: np.ndarray) -> dict:
    """
    Computes structural diagnostics excluding noise (-1).
    """

    labels = labels_full[labels_full != -1]

    if len(labels) == 0:
        return {
            "n_clusters": 0,
            "largest_cluster_share": None,
            "cluster_entropy": None,
        }

    unique, counts = np.unique(labels, return_counts=True)
    shares = counts / counts.sum()

    entropy = -np.sum(shares * np.log(shares + 1e-12))
    largest_share = float(np.max(shares))

    return {
        "n_clusters": int(len(unique)),
        "largest_cluster_share": largest_share,
        "cluster_entropy": float(entropy),
    }


def evaluate(
    cfg: Dict[str, Any],
    labels_full: np.ndarray,
    data_for_metrics: np.ndarray,
) -> dict:
    """
    Evaluates clustering in the same space used for clustering.

    - Computes silhouette (optional noise exclusion)
    - Computes DBCV (robust to overflow warnings)
    - Computes distribution diagnostics
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

    # ----------------------------
    # Prepare data for evaluation
    # ----------------------------

    if not exclude_noise:
        Y = data_for_metrics
        labels = labels_full
    else:
        mask = labels_full != -1
        Y = data_for_metrics[mask]
        labels = labels_full[mask]

        if len(labels) > 0:
            uniq = np.unique(labels)
            remap = {old: i for i, old in enumerate(uniq)}
            labels = np.array([remap[x] for x in labels], dtype=int)

    # ----------------------------
    # Silhouette
    # ----------------------------

    if len(Y) >= 2 and len(np.unique(labels)) >= 2:
        try:
            metrics["silhouette_filtered"] = float(
                silhouette_score(Y, labels)
            )
        except Exception:
            metrics["silhouette_filtered"] = None
    else:
        metrics["silhouette_filtered"] = None

    # ----------------------------
    # DBCV (robust handling)
    # ----------------------------

    if len(Y) >= 2 and len(np.unique(labels)) >= 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    val = validity_index(
                        Y.astype(np.float64, copy=False),
                        labels,
                        metric=metric,
                    )

            if val is None or not np.isfinite(val):
                metrics["dbcv_filtered"] = None
            else:
                metrics["dbcv_filtered"] = float(val)

        except Exception:
            metrics["dbcv_filtered"] = None
    else:
        metrics["dbcv_filtered"] = None

    # ----------------------------
    # Distribution diagnostics
    # ----------------------------

    metrics.update(_cluster_distribution_metrics(labels_full))

    return metrics