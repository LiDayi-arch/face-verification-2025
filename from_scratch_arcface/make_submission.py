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
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_from_scratch.csv"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--scores-out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="Average original and horizontal-flip embeddings")
    return parser.parse_args()


def checkpoint_scores(checkpoint_path: Path, paths, pairs, device, batch_size: int, num_workers: int, tta: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_backbone(checkpoint["backbone"], embedding_size=checkpoint["embedding_size"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Scoring {checkpoint_path} threshold={checkpoint.get('threshold', 0.0):.4f}")
    embeddings = embed_paths(model, paths, device, batch_size, num_workers, tta=tta)
    scores = np.array(
        [np.dot(embeddings[str(left)], embeddings[str(right)]) for left, right in pairs],
        dtype=np.float32,
    )
    return scores, float(checkpoint.get("threshold", 0.0))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_test_pairs(args.data_root)
    resolved_pairs = [
        (competition_path(args.data_root, "test", left), competition_path(args.data_root, "test", right))
        for left, right in pairs
    ]
    unique_paths = sorted({path for pair in resolved_pairs for path in pair})

    all_scores = []
    thresholds = []
    for checkpoint_path in args.checkpoint:
        scores, checkpoint_threshold = checkpoint_scores(
            checkpoint_path,
            unique_paths,
            resolved_pairs,
            device,
            args.batch_size,
            args.num_workers,
            args.tta,
        )
        all_scores.append(scores)
        thresholds.append(checkpoint_threshold)

    scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    threshold = float(np.mean(thresholds)) if args.threshold is None else args.threshold

    if args.scores_out is not None:
        with args.scores_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "score", "subset"])
            for idx, ((left, _), score) in enumerate(zip(pairs, scores)):
                subset = Path(left.replace("\\", "/")).parts[1]
                writer.writerow([idx, f"{float(score):.8f}", subset])
        print(f"Wrote scores to {args.scores_out}")

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, 1 if score >= threshold else 0])

    print(f"Wrote {args.out} with threshold={threshold:.4f}, tta={args.tta}, checkpoints={len(args.checkpoint)}")


if __name__ == "__main__":
    main()
