from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg["project"].get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _center_crop_transform(input_size: int, mean, std) -> transforms.Compose:
    # Resize -> center crop -> tensor -> normalize
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def compute_dinov3_embeddings(cfg: Dict[str, Any], manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    DINOv3 embeddings using Hugging Face AutoModel (transformers >= 4.56.0).

    Embedding = pooled CLS representation: shape (N, hidden_size).
    """
    dinocfg = cfg["embedding"].get("dinov3", {})
    model_id = dinocfg.get("model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m")

    device = _device(cfg)
    batch_size = int(cfg["embedding"].get("batch_size", 32))
    input_size = int(cfg["embedding"]["preprocess"].get("input_size", 224))

    # Processor provides canonical mean/std for the checkpoint
    processor = AutoImageProcessor.from_pretrained(model_id)
    mean = processor.image_mean
    std = processor.image_std

    tfm = _center_crop_transform(input_size=input_size, mean=mean, std=std)

    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)

    paths = manifest["abspath"].tolist()
    X_chunks = []

    for i in tqdm(range(0, len(paths), batch_size), desc=f"DINOv3 embeddings ({model_id})"):
        batch_paths = paths[i:i + batch_size]

        imgs = [tfm(Image.open(p).convert("RGB")) for p in batch_paths]
        pixel_values = torch.stack(imgs).to(device)

        with torch.inference_mode():
            out = model(pixel_values=pixel_values)
            if getattr(out, "pooler_output", None) is not None:
                feats = out.pooler_output              # (B, hidden_size)
            else:
                feats = out.last_hidden_state[:, 0, :]  # CLS fallback

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

def compute_dinov2_embeddings(cfg, manifest):
    """Legacy DINOv2 backend."""
    dinocfg = cfg["embedding"].get("dinov2", {})
    model_id = dinocfg.get("model_id", "facebook/dinov2-base")

    device = _device(cfg)
    batch_size = int(cfg["embedding"].get("batch_size", 32))
    input_size = int(cfg["embedding"]["preprocess"].get("input_size", 224))

    processor = AutoImageProcessor.from_pretrained(model_id)
    tfm = _center_crop_transform(input_size, processor.image_mean, processor.image_std)

    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(model_id).eval().to(device)

    paths = manifest["abspath"].tolist()
    X_chunks = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"DINOv2 ({model_id})"):
        imgs = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]
        pixel_values = torch.stack(imgs).to(device)
        with torch.inference_mode():
            out = model(pixel_values=pixel_values)
            feats = out.last_hidden_state[:, 0, :]   # CLS token
        X_chunks.append(feats.cpu().numpy().astype(np.float32))

    X = np.vstack(X_chunks)
    if cfg["embedding"]["store"].get("l2_normalize", False):
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    meta = manifest[["image_id","filename","relpath","abspath",
                     "width","height","mode","filesize_bytes","mtime_epoch"]].copy()
    return X, meta
def compute_embeddings(cfg, manifest):
    method = cfg["embedding"].get("method", "dinov3").lower()
    if method == "dinov3":
        return compute_dinov3_embeddings(cfg, manifest)
    elif method == "dinov2":
        return compute_dinov2_embeddings(cfg, manifest)
    raise ValueError(f"Unknown embedding method: {method}")