import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset import load_train_items, make_verification_pairs, split_identities
from evaluate import choose_threshold, embed_paths
from models import build_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".."))
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("error_analysis"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-positives", type=int, default=3000)
    parser.add_argument("--val-negatives", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--top-k", type=int, default=200, help="HTML 中每类最多展示多少个错例")
    parser.add_argument("--copy-images", action="store_true", help="复制错例图片，方便下载到本地查看")
    return parser.parse_args()


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in "-_." else "_")
    return "".join(keep).strip("_") or "run"


def infer_run_name(checkpoints: list[Path]) -> str:
    parents = [path.parent.parent if path.parent.name == "snapshots" else path.parent for path in checkpoints]
    names = sorted({path.name for path in parents})
    if len(names) == 1:
        return names[0]
    return "ensemble_" + "_".join(names)


@torch.no_grad()
def score_checkpoint(checkpoint_path: Path, unique_paths: list[Path], pairs: list[tuple[Path, Path]], args, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_backbone(checkpoint["backbone"], embedding_size=checkpoint["embedding_size"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    embeddings = embed_paths(model, unique_paths, device, args.batch_size, args.num_workers, tta=args.tta)
    return np.array(
        [np.dot(embeddings[str(left)], embeddings[str(right)]) for left, right in pairs],
        dtype=np.float32,
    )


def copy_pair_images(out_dir: Path, error_type: str, rank: int, left: Path, right: Path) -> tuple[str, str]:
    pair_dir = out_dir / "images" / error_type / f"{rank:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    left_out = pair_dir / f"left_{left.name}"
    right_out = pair_dir / f"right_{right.name}"
    shutil.copy2(left, left_out)
    shutil.copy2(right, right_out)
    return str(left_out.relative_to(out_dir)).replace("\\", "/"), str(right_out.relative_to(out_dir)).replace("\\", "/")


def write_html(out_dir: Path, rows: list[dict], top_k: int, copy_images: bool) -> None:
    false_positive = [row for row in rows if row["error_type"] == "false_positive"][:top_k]
    false_negative = [row for row in rows if row["error_type"] == "false_negative"][:top_k]

    def render_section(title: str, items: list[dict]) -> str:
        blocks = [f"<h2>{html.escape(title)} ({len(items)})</h2>"]
        for rank, row in enumerate(items, start=1):
            if copy_images:
                left_src, right_src = copy_pair_images(
                    out_dir,
                    row["error_type"],
                    rank,
                    Path(row["left"]),
                    Path(row["right"]),
                )
            else:
                left_src = Path(row["left"]).as_posix()
                right_src = Path(row["right"]).as_posix()
            blocks.append(
                f"""
                <div class="pair">
                  <div class="meta">
                    #{rank} score={row['score']} threshold={row['threshold']} label={row['label']} pred={row['pred']}
                  </div>
                  <div class="imgs">
                    <div><img src="{html.escape(left_src)}"><p>{html.escape(row['left'])}</p></div>
                    <div><img src="{html.escape(right_src)}"><p>{html.escape(row['right'])}</p></div>
                  </div>
                </div>
                """
            )
        return "\n".join(blocks)

    page = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Validation Error Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .pair {{ border: 1px solid #ddd; padding: 12px; margin: 12px 0; }}
    .meta {{ font-weight: 700; margin-bottom: 8px; }}
    .imgs {{ display: flex; gap: 16px; }}
    img {{ width: 112px; height: 112px; object-fit: contain; border: 1px solid #ccc; }}
    p {{ max-width: 520px; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Validation Error Analysis</h1>
  {render_section("False Positive: 真实不同人，模型预测同人", false_positive)}
  {render_section("False Negative: 真实同人，模型预测不同人", false_negative)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = safe_name(args.run_name or infer_run_name(args.checkpoint))
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = load_train_items(args.data_root)
    _, val_labels = split_identities(all_items, args.val_ratio, args.seed)
    val_items = [(path, label) for path, label in all_items if label in val_labels]
    pairs, labels = make_verification_pairs(
        val_items,
        args.data_root,
        positives=args.val_positives,
        negatives=args.val_negatives,
        seed=args.seed,
    )
    unique_paths = sorted({path for pair in pairs for path in pair})

    all_scores = [
        score_checkpoint(checkpoint_path, unique_paths, pairs, args, device)
        for checkpoint_path in args.checkpoint
    ]
    scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    threshold, acc = choose_threshold(scores, labels) if args.threshold is None else (args.threshold, float(((scores >= args.threshold).astype(np.int64) == labels).mean()))
    preds = (scores >= threshold).astype(np.int64)

    rows = []
    for idx, ((left, right), label, pred, score) in enumerate(zip(pairs, labels, preds, scores)):
        if int(label) == int(pred):
            continue
        error_type = "false_positive" if int(pred) == 1 else "false_negative"
        rows.append(
            {
                "index": idx,
                "error_type": error_type,
                "left": str(left),
                "right": str(right),
                "label": int(label),
                "pred": int(pred),
                "score": f"{float(score):.8f}",
                "threshold": f"{threshold:.4f}",
                "margin_to_threshold": f"{abs(float(score) - threshold):.8f}",
            }
        )

    rows.sort(key=lambda row: float(row["margin_to_threshold"]), reverse=True)
    csv_path = out_dir / "val_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "error_type", "left", "right", "label", "pred", "score", "threshold", "margin_to_threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)

    false_positive_count = sum(row["error_type"] == "false_positive" for row in rows)
    false_negative_count = sum(row["error_type"] == "false_negative" for row in rows)
    summary = {
        "run_name": run_name,
        "checkpoints": [str(path) for path in args.checkpoint],
        "threshold": threshold,
        "val_accuracy": acc,
        "total_pairs": len(labels),
        "total_errors": len(rows),
        "false_positive": false_positive_count,
        "false_negative": false_negative_count,
        "tta": args.tta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(out_dir, rows, args.top_k, args.copy_images)

    print(f"Validation acc={acc:.5f}, threshold={threshold:.4f}")
    print(f"Errors: total={len(rows)}, false_positive={false_positive_count}, false_negative={false_negative_count}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
