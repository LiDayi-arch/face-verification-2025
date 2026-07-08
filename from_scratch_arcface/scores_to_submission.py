import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-name", type=str, default=None, help="阈值扫描结果名；不填则根据 threshold 自动生成")
    return parser.parse_args()


def threshold_name(threshold: float) -> str:
    """把阈值转换成文件名安全的字符串，例如 0.194 -> t0p1940。"""
    return f"t{threshold:.4f}".replace(".", "p").replace("-", "m")


def resolve_out_path(scores_path: Path, threshold: float, eval_name: str | None) -> Path:
    """默认把阈值扫描结果放到 scores.csv 同级目录，用文件名区分。"""
    if eval_name:
        name = eval_name
    else:
        base = scores_path.stem
        if base.startswith("scores_"):
            base = base[len("scores_") :]
        name = f"{base}_{threshold_name(threshold)}"
    out_path = scores_path.parent / f"submission_{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def main():
    args = parse_args()
    with args.scores.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_path = args.out if args.out is not None else resolve_out_path(args.scores, args.threshold, args.eval_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for row in rows:
            writer.writerow([row["Index"], 1 if float(row["score"]) >= args.threshold else 0])

    metadata = {
        "scores": str(args.scores),
        "submission": str(out_path),
        "threshold": args.threshold,
        "num_pairs": len(rows),
    }
    (out_path.parent / f"{out_path.stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} from {args.scores} with threshold={args.threshold:.4f}")


if __name__ == "__main__":
    main()
