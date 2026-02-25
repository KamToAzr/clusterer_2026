from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from image_clustering.io.manifest import build_manifest
from image_clustering.embeddings.dinov2 import compute_dinov2_embeddings
from image_clustering.reduce.umap_reduce import run_umap
from image_clustering.cluster.hdbscan_cluster import run_hdbscan
from image_clustering.cluster.evaluate import evaluate
from image_clustering.utils.hashing import stable_hash_dict, stable_hash_str
from image_clustering.viz.bokeh_scatter import render_bokeh
from image_clustering.viz.montage import build_montages


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_run_log(output_root: Path, row: Dict[str, Any]) -> None:
    log_path = output_root / "run_log.csv"
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def _cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["u1", "u2", "u3"] if c in assignments.columns]
    g = assignments.groupby("cluster_label", dropna=False)
    summary = g.size().reset_index(name="n_members")

    for c in cols:
        cent = g[c].mean().reset_index(name=f"{c}_centroid")
        summary = summary.merge(cent, on="cluster_label", how="left")

    summary["share"] = summary["n_members"] / summary["n_members"].sum()
    return summary.sort_values(["cluster_label"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_root = Path(cfg["paths"]["output_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Manifest
    # --------------------
    manifest = build_manifest(cfg)
    manifest_hash = stable_hash_str("|".join(manifest["image_id"].tolist()), n_chars=10)

    # --------------------
    # Embeddings cache key (independent of UMAP/HDBSCAN)
    # --------------------
    emb_cfg = {
        "method": cfg["embedding"]["method"],
        "preprocess": cfg["embedding"]["preprocess"],
        "store": cfg["embedding"]["store"],
        "dinov2": cfg["embedding"].get("dinov2", None),
        "clip": cfg["embedding"].get("clip", None),
        "inception": cfg["embedding"].get("inception", None),
    }
    embed_id = stable_hash_dict({"manifest": manifest_hash, "embedding": emb_cfg}, n_chars=16)
    emb_cache_dir = cache_root / "embeddings" / embed_id
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_cache_dir / "embeddings.npy"
    meta_path = emb_cache_dir / "embeddings_meta.csv"

    # --------------------
    # Run ID (depends on embed_id + downstream params incl clustering space)
    # --------------------
    downstream_cfg = {
        "clustering": cfg.get("clustering", {"space": "umap"}),
        "umap": cfg.get("umap", {}),
        "hdbscan": cfg.get("hdbscan", {}),
        "evaluation": cfg.get("evaluation", {}),
        "viz": cfg.get("viz", {}),
    }
    run_id = stable_hash_dict({"embed_id": embed_id, "downstream": downstream_cfg}, n_chars=20)

    date_stamp = cfg.get("export", {}).get("date_stamp", "undated")
    method = cfg["embedding"]["method"].upper()
    out_dir = output_root / f"{method}_{date_stamp}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot + manifest
    (out_dir / "config.used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    manifest.to_csv(out_dir / "run_manifest.csv", index=False)

    # --------------------
    # Embeddings
    # --------------------
    if emb_path.exists() and meta_path.exists():
        X = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        print(f"Loaded cached embeddings: {X.shape} from {emb_cache_dir}")
    else:
        if cfg["embedding"]["method"] != "dinov2":
            raise NotImplementedError("Only dinov2 wired for now.")
        X, meta = compute_dinov2_embeddings(cfg, manifest)
        np.save(emb_path, X)
        meta.to_csv(meta_path, index=False)
        print(f"Saved embeddings: {X.shape} to {emb_cache_dir}")

    # Keep meta copy in run dir (small)
    run_meta_path = out_dir / "embeddings_meta.csv"
    if not run_meta_path.exists():
        meta.to_csv(run_meta_path, index=False)

    # --------------------
    # UMAP (for visualization always)
    # --------------------
    umap_path = out_dir / "umap_coords.csv"
    if umap_path.exists():
        coords = pd.read_csv(umap_path)
        print("Loaded cached UMAP coords.")
    else:
        coords = run_umap(cfg, X, meta)
        coords.to_csv(umap_path, index=False)
        print("Saved UMAP coords.")

    # --------------------
    # Choose clustering space
    # --------------------
    clustering_space = cfg.get("clustering", {}).get("space", "umap").lower()
    if clustering_space not in {"umap", "embedding"}:
        raise ValueError("clustering.space must be 'umap' or 'embedding'.")

    if clustering_space == "umap":
        cols = [c for c in ["u1", "u2", "u3"] if c in coords.columns]
        cluster_data = coords[cols].to_numpy()
    else:
        cluster_data = X

    # --------------------
    # HDBSCAN labels/probabilities (cached via assignments)
    # --------------------
    assign_path = out_dir / "assignments.csv"
    if assign_path.exists():
        assignments = pd.read_csv(assign_path)
        print("Loaded cached assignments.")
    else:
        labels, probs, _clusterer = run_hdbscan(cfg, cluster_data)

        # Build assignments anchored on UMAP coords for downstream viz
        assignments = coords.copy()
        assignments["cluster_label"] = labels
        assignments["probability"] = probs

        assignments.to_csv(assign_path, index=False)
        print("Saved assignments.")

    # --------------------
    # Metrics + cluster summary
    # --------------------
    metrics_path = out_dir / "run_metrics.json"
    summary_path = out_dir / "cluster_summary.csv"

    if metrics_path.exists() and summary_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("Loaded cached metrics + cluster summary.")
    else:
        labels_full = assignments["cluster_label"].to_numpy()

        # evaluate on the *actual clustering space*
        metrics = evaluate(cfg, labels_full=labels_full, data_for_metrics=cluster_data)

        # add embedding + run metadata
        metrics.update({
            "embed_id": embed_id,
            "embedding_shape": [int(X.shape[0]), int(X.shape[1])],
            "embedding_method": cfg["embedding"]["method"],
            "dinov2_model_id": cfg.get("embedding", {}).get("dinov2", {}).get("model_id", None),
            "clustering_space": clustering_space,
            "umap_n_components": cfg["umap"]["n_components"],
            "umap_n_neighbors": cfg["umap"]["n_neighbors"],
            "hdbscan_min_cluster_size": cfg["hdbscan"]["min_cluster_size"],
            "hdbscan_min_samples": cfg["hdbscan"]["min_samples"],
        })

        summary = _cluster_summary(assignments)

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.to_csv(summary_path, index=False)
        print("Saved metrics + cluster summary.")

    # --------------------
    # Bokeh scatter (inline; includes metrics table)
    # --------------------
    scatter_path = out_dir / "viz_scatter.html"
    if not scatter_path.exists():
        render_bokeh(cfg, assignments, scatter_path, metrics=metrics)
        print("Saved Bokeh scatter.")

    # --------------------
    # Montages (needs abspath)
    # --------------------
    assignments_for_montage = assignments.merge(meta[["image_id", "abspath"]], on="image_id", how="left")
    build_montages(cfg, assignments_for_montage, out_dir)
    print("Saved montages.")

    # --------------------
    # Append run log (one line per run)
    # --------------------
    # Count clusters excluding noise
    n_clusters = int(len(set(assignments["cluster_label"])) - (1 if (-1 in set(assignments["cluster_label"])) else 0))

    log_row = {
        "date_stamp": date_stamp,
        "run_id": run_id,
        "embed_id": embed_id,
        "n_images": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "embedding_method": cfg["embedding"]["method"],
        "dinov2_model_id": cfg.get("embedding", {}).get("dinov2", {}).get("model_id", None),
        "clustering_space": clustering_space,
        "umap_n_components": cfg["umap"]["n_components"],
        "umap_n_neighbors": cfg["umap"]["n_neighbors"],
        "umap_min_dist": cfg["umap"]["min_dist"],
        "hdbscan_min_cluster_size": cfg["hdbscan"]["min_cluster_size"],
        "hdbscan_min_samples": cfg["hdbscan"]["min_samples"],
        "hdbscan_epsilon": cfg["hdbscan"]["cluster_selection_epsilon"],
        "silhouette_filtered": metrics.get("silhouette_filtered", None),
        "dbcv_filtered": metrics.get("dbcv_filtered", None),
        "noise_fraction": metrics.get("noise_fraction", None),
        "n_clusters": n_clusters,
        "output_dir": str(out_dir),
    }
    _append_run_log(output_root, log_row)

    print(f"\nDone. Outputs in: {out_dir}")
    print(f"Embeddings cache: {emb_cache_dir}")
    print(f"Run log: {output_root / 'run_log.csv'}")


if __name__ == "__main__":
    main()