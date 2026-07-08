import argparse
import csv
import json
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--scores-out", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("submissions"), help="默认提交结果和 scores 的根目录")
    parser.add_argument("--eval-name", type=str, default=None, help="本次推理/提交实验名，例如 baseline、tta、snap_tta")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="启用水平翻转 TTA：平均原图和翻转图的 embedding")
    return parser.parse_args()


def safe_name(value: str) -> str:
    """把实验名转换成适合作为目录名的字符串。"""
    keep = []
    for char in value:
        if char.isalnum() or char in "-_.":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def infer_run_name(checkpoints: list[Path]) -> str:
    """从 checkpoint 路径推断训练实验名。"""
    parents = [path.parent for path in checkpoints]
    if len(checkpoints) == 1:
        parent = parents[0]
        if parent.name == "snapshots":
            return parent.parent.name
        return parent.name

    normalized = [parent.parent if parent.name == "snapshots" else parent for parent in parents]
    names = {path.name for path in normalized}
    if len(names) == 1:
        return next(iter(names))
    return "ensemble_" + "_".join(sorted(safe_name(name) for name in names))


def infer_eval_name(checkpoints: list[Path], tta: bool, threshold: float | None) -> str:
    """根据 checkpoint 组合自动生成推理实验名。"""
    if len(checkpoints) == 1:
        name = checkpoints[0].stem
        if checkpoints[0].parent.name == "snapshots":
            name = f"snapshot_{name}"
    else:
        stems = [path.stem for path in checkpoints]
        name = f"ensemble{len(checkpoints)}_" + "_".join(stems)

    if tta:
        name += "_tta"
    if threshold is not None:
        name += f"_t{threshold:.4f}".replace(".", "p").replace("-", "m")
    return safe_name(name)


def write_output_readme(output_root: Path) -> None:
    """写一个简短说明，避免后面文件多了看不懂。"""
    readme = output_root / "README.md"
    if readme.exists():
        return
    output_root.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        """# Submission Outputs

这个文件夹保存 Kaggle 提交文件和对应的原始 similarity 分数。

推荐结构：

```text
submissions/
  <run-name>/
    submission_<eval-name>.csv
    scores_<eval-name>.csv
    metadata_<eval-name>.json
```

命名含义：

- `<run-name>`：训练实验名，建议和 `histories/<run-name>/`、`models/<run-name>/` 保持一致。
- `submission_*.csv`：可以直接上传 Kaggle 的 0/1 预测文件。
- `scores_*.csv`：每个测试 pair 的原始 cosine similarity 分数，用来后续快速扫 threshold。
- `metadata_*.json`：记录本次提交用了哪些 checkpoint、threshold、是否 TTA 等信息。
- `baseline`：单个 `best.pt`，不使用 TTA，不使用 ensemble。
- `tta`：单个 `best.pt`，使用水平翻转 TTA。
- `snapshot_...`：使用若干 snapshot checkpoint 做分数平均。
- `ensemble...`：使用多个 checkpoint 或多个模型做分数平均。
- `t0p1900`：手动 threshold=0.1900 生成的提交文件，`.` 会写成 `p`，方便作为文件名。
""",
        encoding="utf-8",
    )


def resolve_outputs(args) -> tuple[Path, Path | None, Path, str]:
    """统一管理输出路径；未显式指定时写入 submissions/run/。"""
    run_name = safe_name(infer_run_name(args.checkpoint))
    eval_name = safe_name(args.eval_name) if args.eval_name else infer_eval_name(args.checkpoint, args.tta, args.threshold)
    write_output_readme(args.output_root)
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out if args.out is not None else output_dir / f"submission_{eval_name}.csv"
    scores_path = args.scores_out if args.scores_out is not None else output_dir / f"scores_{eval_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path is not None:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path, scores_path, output_dir, eval_name


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
    out_path, scores_out_path, output_dir, eval_name = resolve_outputs(args)

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

    if scores_out_path is not None:
        with scores_out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "score", "subset"])
            for idx, ((left, _), score) in enumerate(zip(pairs, scores)):
                subset = Path(left.replace("\\", "/")).parts[1]
                writer.writerow([idx, f"{float(score):.8f}", subset])
        print(f"Wrote scores to {scores_out_path}")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "ID"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, 1 if score >= threshold else 0])

    metadata = {
        "output_dir": str(output_dir),
        "eval_name": eval_name,
        "submission": str(out_path),
        "scores": str(scores_out_path) if scores_out_path is not None else None,
        "checkpoints": [str(path) for path in args.checkpoint],
        "checkpoint_thresholds": thresholds,
        "threshold": threshold,
        "threshold_source": "manual" if args.threshold is not None else "checkpoint_average",
        "tta": args.tta,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_pairs": len(pairs),
    }
    metadata_path = output_dir / f"metadata_{eval_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} with threshold={threshold:.4f}, tta={args.tta}, checkpoints={len(args.checkpoint)}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
