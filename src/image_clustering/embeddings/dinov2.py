from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model


def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg["project"].get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _center_crop_transform(input_size: int, mean, std) -> transforms.Compose:
    # Resize shorter side -> center crop -> tensor -> normalize
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def compute_dinov2_embeddings(cfg: Dict[str, Any], manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Proper DINOv2 embeddings using Hugging Face Dinov2Model.

    Embedding = CLS token from last_hidden_state: shape (N, hidden_size)
    """
    dinocfg = cfg["embedding"].get("dinov2", {})
    model_id = dinocfg.get("model_id", "facebook/dinov2-base")

    device = _device(cfg)
    batch_size = int(cfg["embedding"].get("batch_size", 32))
    input_size = int(cfg["embedding"]["preprocess"].get("input_size", 224))

    # Processor provides canonical mean/std for the checkpoint
    processor = AutoImageProcessor.from_pretrained(model_id)
    mean = processor.image_mean
    std = processor.image_std

    tfm = _center_crop_transform(input_size=input_size, mean=mean, std=std)

    model = Dinov2Model.from_pretrained(model_id)
    model.eval().to(device)

    paths = manifest["abspath"].tolist()
    X_chunks = []

    for i in tqdm(range(0, len(paths), batch_size), desc=f"DINOv2 embeddings ({model_id})"):
        batch_paths = paths[i:i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(tfm(img))
        pixel_values = torch.stack(imgs).to(device)

        with torch.no_grad():
            out = model(pixel_values=pixel_values)
            # CLS token
            feats = out.last_hidden_state[:, 0, :]  # (B, hidden_size)

        X_chunks.append(feats.detach().cpu().numpy().astype(np.float32))

    X = np.vstack(X_chunks)

    if cfg["embedding"]["store"].get("l2_normalize", False):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms

    meta = manifest[[
        "image_id", "filename", "relpath", "abspath", "width", "height", "mode",
        "filesize_bytes", "mtime_epoch"
    ]].copy()

    return X, meta