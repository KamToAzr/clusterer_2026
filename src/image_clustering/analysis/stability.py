from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from image_clustering.reduce.umap_reduce import run_umap
from image_clustering.cluster.hdbscan_cluster import run_hdbscan


def _ari_excluding_noise(a: np.ndarray, b: np.ndarray) -> float | None:
    """
    ARI computed on items that are non-noise in BOTH labelings.
    If <2 labels after filtering, returns None.
    """
    mask = (a != -1) & (b != -1)
    if mask.sum() < 2:
        return None
    return float(adjusted_rand_score(a[mask], b[mask]))


def _pairwise_ari_matrix(labels_by_seed: Dict[int, np.ndarray], exclude_noise: bool) -> pd.DataFrame:
    seeds = sorted(labels_by_seed.keys())
    M = np.zeros((len(seeds), len(seeds)), dtype=float)

    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            a = labels_by_seed[si]
            b = labels_by_seed[sj]
            if exclude_noise:
                val = _ari_excluding_noise(a, b)
                M[i, j] = np.nan if val is None else val
            else:
                M[i, j] = float(adjusted_rand_score(a, b))

    return pd.DataFrame(M, index=seeds, columns=seeds)


def run_stability(
    cfg: Dict[str, Any],
    X: np.ndarray,
    meta: pd.DataFrame,
    output_root: Path,
    embed_id: str,
) -> Path:
    """
    Re-runs the chosen configuration across multiple seeds and computes ARI stability.

    Outputs:
      outputs/stability_<date_stamp>_<embed_id>/
        labels_by_seed.csv
        ari_matrix.csv
        stability_summary.json
    """
    stab_cfg = cfg.get("stability", {})
    seeds: List[int] = [int(s) for s in stab_cfg.get("seeds", [1, 2, 3, 4, 5])]
    exclude_noise = bool(stab_cfg.get("exclude_noise_for_ari", True))

    clustering_space = cfg["clustering"]["space"].lower()
    if clustering_space not in {"umap", "embedding"}:
        raise ValueError("clustering.space must be 'umap' or 'embedding'.")

    date_stamp = cfg.get("export", {}).get("date_stamp", "undated")
    out_dir = output_root / f"stability_{date_stamp}_{embed_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_seed: Dict[int, np.ndarray] = {}

    # If clustering is in embedding space, UMAP randomness does not affect clustering.
    # We still run multiple seeds, but you should expect identical labels across seeds.
    for seed in seeds:
        # set UMAP seed for this run (only relevant if clustering_space == "umap")
        cfg["umap"]["random_state"] = int(seed)

        coords = run_umap(cfg, X, meta)

        if clustering_space == "umap":
            data = coords[["u1", "u2"]].to_numpy()
        else:
            data = X

        labels, probs, _ = run_hdbscan(cfg, data)
        labels_by_seed[int(seed)] = labels

    # Save labels (wide format: one column per seed)
    labels_df = pd.DataFrame({"image_id": meta["image_id"].values})
    for seed in sorted(labels_by_seed.keys()):
        labels_df[f"label_seed_{seed}"] = labels_by_seed[seed]
    labels_df.to_csv(out_dir / "labels_by_seed.csv", index=False)

    # ARI matrix
    ari_df = _pairwise_ari_matrix(labels_by_seed, exclude_noise=exclude_noise)
    ari_df.to_csv(out_dir / "ari_matrix.csv", index=True)

    # Summary stats (off-diagonal only)
    vals = ari_df.values
    off_diag = vals[~np.eye(vals.shape[0], dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]

    summary = {
        "embed_id": embed_id,
        "clustering_space": clustering_space,
        "seeds": seeds,
        "exclude_noise_for_ari": exclude_noise,
        "ari_mean": float(np.mean(off_diag)) if len(off_diag) else None,
        "ari_median": float(np.median(off_diag)) if len(off_diag) else None,
        "ari_min": float(np.min(off_diag)) if len(off_diag) else None,
        "ari_max": float(np.max(off_diag)) if len(off_diag) else None,
        "n_pairs": int(len(off_diag)),
    }
    (out_dir / "stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nStability outputs saved to: {out_dir}")
    print(f"ARI mean={summary['ari_mean']}, median={summary['ari_median']}, min={summary['ari_min']}, max={summary['ari_max']}")

    return out_dir