from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceImageDataset, build_eval_transform


@torch.no_grad()
def embed_paths(
    model,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    dataset = FaceImageDataset(paths, transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    embeddings: dict[str, np.ndarray] = {}
    for images, batch_paths in tqdm(loader, desc="Embedding", leave=False):
        images = images.to(device, non_blocking=True)
        feats = model(images)
        if tta:
            flipped_feats = model(torch.flip(images, dims=[3]))
            feats = F.normalize(feats + flipped_feats)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        for path, feat in zip(batch_paths, feats):
            embeddings[path] = feat
    return embeddings


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.0
    best_acc = -1.0
    for threshold in np.linspace(-0.5, 0.9, 1401):
        preds = (scores >= threshold).astype(np.int64)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


@torch.no_grad()
def evaluate_verification(
    model,
    pairs: list[tuple[Path, Path]],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta: bool = False,
) -> tuple[float, float]:
    unique_paths = sorted({path for pair in pairs for path in pair})
    embeddings = embed_paths(model, unique_paths, device, batch_size, num_workers, tta=tta)
    scores = np.array(
        [np.dot(embeddings[str(left)], embeddings[str(right)]) for left, right in pairs],
        dtype=np.float32,
    )
    return choose_threshold(scores, labels)
