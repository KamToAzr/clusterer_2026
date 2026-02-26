from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from image_clustering.io.cluster_folders import export_cluster_folders


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        default="run",
        choices=["run", "grid", "stability"]
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_root = Path(cfg["paths"]["output_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Manifest
    # ------------------------
    manifest = build_manifest(cfg)
    manifest_hash = stable_hash_str("|".join(manifest["image_id"].tolist()), 10)

    # ------------------------
    # Embedding cache key
    # ------------------------
    emb_cfg = {
        "method": cfg["embedding"]["method"],
        "preprocess": cfg["embedding"]["preprocess"],
        "store": cfg["embedding"]["store"],
        "dinov2": cfg["embedding"].get("dinov2"),
    }

    embed_id = stable_hash_dict(
        {"manifest": manifest_hash, "embedding": emb_cfg},
        16,
    )

    emb_cache_dir = cache_root / "embeddings" / embed_id
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_cache_dir / "embeddings.npy"
    meta_path = emb_cache_dir / "embeddings_meta.csv"

    if emb_path.exists() and meta_path.exists():
        X = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        print(f"Loaded cached embeddings: {X.shape}")
    else:
        X, meta = compute_dinov2_embeddings(cfg, manifest)
        np.save(emb_path, X)
        meta.to_csv(meta_path, index=False)
        print(f"Saved embeddings: {X.shape}")

    # ------------------------
    # GRID MODE
    # ------------------------
    if args.mode == "grid":
        from image_clustering.analysis.grid_search import run_grid_search
        run_grid_search(cfg, X, meta, output_root)
        return

    # ------------------------
    # STABILITY MODE
    # ------------------------
    if args.mode == "stability":
        from image_clustering.analysis.stability import run_stability
        run_stability(cfg, X, meta, output_root, embed_id=embed_id)
        return

    # ------------------------
    # NORMAL RUN
    # ------------------------

    clustering_space = cfg["clustering"]["space"].lower()

    coords = run_umap(cfg, X, meta)

    if clustering_space == "embedding":
        cluster_data = X
    else:
        # If you later support n_components=3, update to use ["u1","u2","u3"] when needed.
        cluster_data = coords[["u1", "u2"]].to_numpy()

    labels, probs, _ = run_hdbscan(cfg, cluster_data)

    assignments = coords.copy()
    assignments["cluster_label"] = labels
    assignments["probability"] = probs

    metrics = evaluate(
        cfg,
        labels_full=labels,
        data_for_metrics=cluster_data,
    )

    date_stamp = cfg["export"]["date_stamp"]
    run_id = stable_hash_dict(
        {"embed": embed_id, "params": cfg["hdbscan"], "umap": cfg["umap"]},
        20,
    )

    out_dir = output_root / f"DINOV2_{date_stamp}_{run_id}"
    out_dir.mkdir(exist_ok=True)

    # Save core outputs
    assignments.to_csv(out_dir / "assignments.csv", index=False)
    coords.to_csv(out_dir / "umap_coords.csv", index=False)
    meta.to_csv(out_dir / "embeddings_meta.csv", index=False)

    (out_dir / "run_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    # Merge absolute paths for montage + per-cluster folders
    assignments_for_export = assignments.merge(
        meta[["image_id", "abspath"]],
        on="image_id",
        how="left",
    )

    # ------------------------
    # Cluster summary (explicit cluster sizes)
    # ------------------------
    cluster_summary = (
        assignments_for_export
        .groupby("cluster_label", dropna=False)
        .agg(
            n_images=("image_id", "count"),
            mean_probability=("probability", "mean"),
        )
        .reset_index()
    )
    cluster_summary["share"] = cluster_summary["n_images"] / len(assignments_for_export)
    cluster_summary = cluster_summary.sort_values("n_images", ascending=False)
    cluster_summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    # ------------------------
    # Visualisation
    # ------------------------
    render_bokeh(
        cfg,
        assignments,
        out_dir / "viz_scatter.html",
        metrics=metrics,
    )

    # ------------------------
    # Montage + cluster folders
    # ------------------------
    build_montages(cfg, assignments_for_export, out_dir)
    export_cluster_folders(cfg, assignments_for_export, out_dir)

    print(f"\nDone. Outputs in: {out_dir}")
    print(f"Embeddings cache: {emb_cache_dir}")


if __name__ == "__main__":
    main()