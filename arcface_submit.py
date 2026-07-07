import argparse
import csv
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_NAME = "w600k_r50.onnx"


def ensure_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME
    if model_path.exists():
        return model_path

    zip_path = model_dir / "buffalo_l.zip"
    if not zip_path.exists():
        print(f"Downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(MODEL_NAME)]
        if not matches:
            raise RuntimeError(f"{MODEL_NAME} not found in {zip_path}")
        with zf.open(matches[0]) as src, model_path.open("wb") as dst:
            dst.write(src.read())

    return model_path


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((112, 112))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def embed_images(session: ort.InferenceSession, paths: list[Path], batch_size: int) -> dict[str, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    embeddings: dict[str, np.ndarray] = {}

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = np.stack([load_image(path) for path in batch_paths], axis=0)
        feats = session.run([output_name], {input_name: batch})[0].astype(np.float32)
        feats = l2_normalize(feats)
        for path, feat in zip(batch_paths, feats):
            embeddings[str(path)] = feat
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"Embedded {min(start + batch_size, len(paths))}/{len(paths)} images", flush=True)

    return embeddings


def make_train_val_pairs(
    train_items: list[tuple[str, int]],
    data_root: Path,
    positive_count: int,
    negative_count: int,
) -> tuple[list[tuple[Path, Path]], np.ndarray]:
    by_label: dict[int, list[str]] = {}
    for rel_path, label in train_items:
        by_label.setdefault(label, []).append(rel_path)

    rng = np.random.default_rng(2026)
    eligible_labels = [label for label, rel_paths in by_label.items() if len(rel_paths) >= 2]

    positive_pairs: list[tuple[Path, Path]] = []
    labels = []
    for label in rng.choice(eligible_labels, size=positive_count, replace=True):
        rel_paths = by_label[int(label)]
        left_idx, right_idx = rng.choice(len(rel_paths), size=2, replace=False)
        positive_pairs.append((data_root / "train" / rel_paths[left_idx], data_root / "train" / rel_paths[right_idx]))
        labels.append(1)

    class_ids = np.array(list(by_label.keys()))
    negative_pairs: list[tuple[Path, Path]] = []
    while len(negative_pairs) < negative_count:
        left_label, right_label = rng.choice(class_ids, size=2, replace=False)
        left = rng.choice(by_label[int(left_label)])
        right = rng.choice(by_label[int(right_label)])
        negative_pairs.append((data_root / "train" / left, data_root / "train" / right))
        labels.append(0)

    return positive_pairs + negative_pairs, np.array(labels, dtype=np.int64)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.0
    best_acc = -1.0
    for threshold in np.linspace(-0.2, 0.8, 1001):
        preds = (scores >= threshold).astype(np.int64)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("submission_arcface.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Debug only: use first N test pairs")
    parser.add_argument("--calibrate", action="store_true", help="Tune threshold on sampled train pairs")
    parser.add_argument("--val-negatives", type=int, default=20000)
    parser.add_argument("--val-positives", type=int, default=20000)
    args = parser.parse_args()

    model_path = ensure_model(args.data_root / "models")
    print(f"Using model: {model_path}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)

    if args.calibrate:
        with (args.data_root / "train_list.pkl").open("rb") as f:
            train_items = pickle.load(f)
        val_pairs, val_labels = make_train_val_pairs(
            train_items,
            args.data_root,
            positive_count=args.val_positives,
            negative_count=args.val_negatives,
        )
        val_unique_paths = sorted({path for pair in val_pairs for path in pair})
        print(f"Validation pairs: {len(val_pairs)}; unique images: {len(val_unique_paths)}", flush=True)
        val_embeddings = embed_images(session, val_unique_paths, args.batch_size)
        scores = np.array(
            [np.dot(val_embeddings[str(left)], val_embeddings[str(right)]) for left, right in val_pairs],
            dtype=np.float32,
        )
        args.threshold, val_acc = choose_threshold(scores, val_labels)
        print(f"Calibrated threshold={args.threshold:.4f}; sampled train accuracy={val_acc:.5f}", flush=True)

    with (args.data_root / "test_list.pkl").open("rb") as f:
        pairs = pickle.load(f)
    if args.limit:
        pairs = pairs[: args.limit]

    resolved_pairs = [
        (args.data_root / "test" / left, args.data_root / "test" / right)
        for left, right in pairs
    ]
    unique_paths = sorted({path for pair in resolved_pairs for path in pair})
    print(f"Pairs: {len(resolved_pairs)}; unique images: {len(unique_paths)}", flush=True)

    embeddings = embed_images(session, unique_paths, args.batch_size)

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for idx, (left, right) in enumerate(resolved_pairs):
            sim = float(np.dot(embeddings[str(left)], embeddings[str(right)]))
            pred = 1 if sim >= args.threshold else 0
            writer.writerow([idx, pred])

    print(f"Wrote {args.out} with threshold={args.threshold}", flush=True)


if __name__ == "__main__":
    main()
