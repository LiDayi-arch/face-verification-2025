import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dataset import competition_path, load_test_pairs
from evaluate import embed_paths
from models import build_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_from_scratch.csv"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    threshold = checkpoint.get("threshold", 0.0) if args.threshold is None else args.threshold

    model = build_backbone(checkpoint["backbone"], embedding_size=checkpoint["embedding_size"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    pairs = load_test_pairs(args.data_root)
    resolved_pairs = [
        (competition_path(args.data_root, "test", left), competition_path(args.data_root, "test", right))
        for left, right in pairs
    ]
    unique_paths = sorted({path for pair in resolved_pairs for path in pair})
    embeddings = embed_paths(model, unique_paths, device, args.batch_size, args.num_workers)

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for idx, (left, right) in enumerate(resolved_pairs):
            score = float(np.dot(embeddings[str(left)], embeddings[str(right)]))
            writer.writerow([idx, 1 if score >= threshold else 0])

    print(f"Wrote {args.out} with threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
