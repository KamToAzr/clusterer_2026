from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import umap


def run_umap(cfg: Dict[str, Any], X, meta: pd.DataFrame) -> pd.DataFrame:
    ucfg = cfg["umap"]

    reducer = umap.UMAP(
        n_components=int(ucfg["n_components"]),
        n_neighbors=int(ucfg["n_neighbors"]),
        min_dist=float(ucfg["min_dist"]),
        metric=str(ucfg["metric"]),
        random_state=int(ucfg["random_state"]),
    )

    coords = reducer.fit_transform(X)

    out = meta[["image_id", "filename", "relpath"]].copy()
    out["u1"] = coords[:, 0]
    out["u2"] = coords[:, 1]
    if coords.shape[1] == 3:
        out["u3"] = coords[:, 2]

    return out