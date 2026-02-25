from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import hdbscan


def run_hdbscan(cfg: Dict[str, Any], data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """
    Runs HDBSCAN on an explicit numeric matrix.
    Returns: labels, probabilities, fitted clusterer
    """
    hcfg = cfg["hdbscan"]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hcfg["min_cluster_size"]),
        min_samples=int(hcfg["min_samples"]),
        cluster_selection_epsilon=float(hcfg["cluster_selection_epsilon"]),
        metric=str(hcfg["metric"]),
    )

    labels = clusterer.fit_predict(data)
    probs = clusterer.probabilities_

    return labels, probs, clusterer