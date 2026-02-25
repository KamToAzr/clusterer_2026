from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import random
import pandas as pd
from PIL import Image


def build_montages(cfg: Dict[str, Any], assignments: pd.DataFrame, out_dir: Path):
    n_per_cluster = cfg["viz"]["montage"]["n_per_cluster"]
    seed = cfg["project"]["seed"]

    random.seed(seed)

    montage_dir = out_dir / "montages"
    montage_dir.mkdir(exist_ok=True)

    clusters = sorted(assignments["cluster_label"].unique())

    for label in clusters:
        if label == -1:
            continue

        subset = assignments[assignments["cluster_label"] == label]
        sample = subset.sample(
            n=min(n_per_cluster, len(subset)),
            random_state=seed
        )

        images = []
        for p in sample["abspath"]:
            img = Image.open(p).convert("RGB")
            img = img.resize((200, 200))
            images.append(img)

        if not images:
            continue

        cols = 5
        rows = (len(images) + cols - 1) // cols

        montage = Image.new("RGB", (cols * 200, rows * 200), color=(255, 255, 255))

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            montage.paste(img, (c * 200, r * 200))

        montage.save(montage_dir / f"cluster_{label}.jpg")