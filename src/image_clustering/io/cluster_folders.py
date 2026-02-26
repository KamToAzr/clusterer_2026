from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import os
import shutil

import pandas as pd


def _safe_mkdir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            return
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        # hardlink fails across different drives/volumes
        os.link(str(src), str(dst))
        return

    if mode == "symlink":
        os.symlink(str(src), str(dst))
        return

    raise ValueError(f"Unknown export_clusters.mode: {mode}")


def export_cluster_folders(
    cfg: Dict[str, Any],
    assignments_with_paths: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Create one folder per cluster label under:
        <out_dir>/clusters/<cluster_prefix><label>/

    Also optionally:
        <out_dir>/clusters/<noise_folder>/

    Expects assignments_with_paths to have:
      - image_id
      - cluster_label (int, noise = -1)
      - abspath (absolute path to file)
    """

    exp = cfg.get("export_clusters", {})
    if not exp.get("enabled", False):
        return

    mode = str(exp.get("mode", "hardlink")).lower()
    include_noise = bool(exp.get("include_noise", True))
    prefix = str(exp.get("folder_prefix", "cluster_"))
    noise_name = str(exp.get("noise_folder", "noise"))
    zero_pad = int(exp.get("zero_pad", 4))
    overwrite = bool(exp.get("overwrite", False))

    base = out_dir / "clusters"
    _safe_mkdir(base, overwrite=overwrite)

    required = {"cluster_label", "abspath"}
    missing = required - set(assignments_with_paths.columns)
    if missing:
        raise KeyError(f"export_cluster_folders missing columns: {missing}")

    df = assignments_with_paths.copy()
    df["abspath"] = df["abspath"].astype(str)

    # iterate by cluster
    for label, sub in df.groupby("cluster_label"):
        if int(label) == -1 and not include_noise:
            continue

        if int(label) == -1:
            folder = base / noise_name
        else:
            folder = base / f"{prefix}{int(label):0{zero_pad}d}"

        folder.mkdir(parents=True, exist_ok=True)

        for p in sub["abspath"]:
            src = Path(p)
            if not src.exists():
                continue
            dst = folder / src.name
            _link_or_copy(src, dst, mode=mode)

    print(f"Saved per-cluster image folders to: {base} (mode={mode})")