import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def competition_path(data_root: Path, split: str, rel_path: str) -> Path:
    return data_root / split / Path(rel_path.replace("\\", "/"))


def load_train_items(data_root: Path) -> list[tuple[str, int]]:
    with (data_root / "train_list.pkl").open("rb") as f:
        return pickle.load(f)


def load_test_pairs(data_root: Path) -> list[tuple[str, str]]:
    with (data_root / "test_list.pkl").open("rb") as f:
        return pickle.load(f)


def split_identities(items: list[tuple[str, int]], val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    labels = sorted({label for _, label in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    val_count = max(1, int(len(labels) * val_ratio))
    val_labels = set(labels[:val_count])
    train_labels = set(labels[val_count:])
    return train_labels, val_labels


def remap_labels(items: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], dict[int, int]]:
    old_labels = sorted({label for _, label in items})
    label_map = {old: new for new, old in enumerate(old_labels)}
    return [(path, label_map[label]) for path, label in items], label_map


def build_train_transform(augment: str = "v1") -> transforms.Compose:
    """训练数据增强。

    v1 是原始保守增强；v2 增加轻微仿射变换；v3 在 v2 基础上增加轻微模糊和自动对比度。
    """
    if augment == "none":
        before_tensor = []
        erase_p = 0.0
    elif augment == "v1":
        before_tensor = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
        ]
        erase_p = 0.1
    elif augment == "v2":
        before_tensor = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.12),
            transforms.RandomGrayscale(p=0.05),
        ]
        erase_p = 0.1
    elif augment == "v3":
        before_tensor = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomGrayscale(p=0.06),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.08),
            transforms.RandomAutocontrast(p=0.08),
        ]
        erase_p = 0.12
    else:
        raise ValueError(f"Unknown augment policy: {augment}")

    return transforms.Compose(
        [
            *before_tensor,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=erase_p, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class FaceClassificationDataset(Dataset):
    def __init__(self, data_root: Path, items: list[tuple[str, int]], transform=None):
        self.data_root = data_root
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        rel_path, label = self.items[index]
        image = Image.open(competition_path(self.data_root, "train", rel_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class FaceImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


def make_verification_pairs(
    items: list[tuple[str, int]],
    data_root: Path,
    positives: int,
    negatives: int,
    seed: int,
) -> tuple[list[tuple[Path, Path]], np.ndarray]:
    by_label: dict[int, list[str]] = defaultdict(list)
    for rel_path, label in items:
        by_label[label].append(rel_path)

    rng = np.random.default_rng(seed)
    eligible = [label for label, paths in by_label.items() if len(paths) >= 2]
    labels = []
    pairs = []

    for label in rng.choice(eligible, size=positives, replace=True):
        rel_paths = by_label[int(label)]
        left_idx, right_idx = rng.choice(len(rel_paths), size=2, replace=False)
        pairs.append(
            (
                competition_path(data_root, "train", rel_paths[left_idx]),
                competition_path(data_root, "train", rel_paths[right_idx]),
            )
        )
        labels.append(1)

    class_ids = np.array(sorted(by_label.keys()))
    while len(labels) < positives + negatives:
        left_label, right_label = rng.choice(class_ids, size=2, replace=False)
        left = rng.choice(by_label[int(left_label)])
        right = rng.choice(by_label[int(right_label)])
        pairs.append((competition_path(data_root, "train", left), competition_path(data_root, "train", right)))
        labels.append(0)

    return pairs, np.array(labels, dtype=np.int64)
