from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image

from image_clustering.utils.hashing import stable_hash_str


def build_manifest(cfg):
    input_dir = Path(cfg["paths"]["input_dir"])
    exts = set(x.lower() for x in cfg["ingest"]["extensions"])
    max_images = cfg["ingest"]["max_images"]

    paths = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
            if max_images is not None and len(paths) >= int(max_images):
                break

    rows = []
    for p in paths:
        st = p.stat()
        rel = p.relative_to(input_dir).as_posix()

        # stable ID based on relpath + size + mtime
        image_id = stable_hash_str(f"{rel}|{st.st_size}|{int(st.st_mtime)}", n_chars=16)

        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode

        rows.append({
            "image_id": image_id,
            "filename": p.name,
            "relpath": rel,
            "abspath": str(p.resolve()),
            "width": w,
            "height": h,
            "mode": mode,
            "filesize_bytes": int(st.st_size),
            "mtime_epoch": int(st.st_mtime),
        })

    return pd.DataFrame(rows)