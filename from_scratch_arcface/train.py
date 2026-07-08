import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    FaceClassificationDataset,
    build_train_transform,
    load_train_items,
    make_verification_pairs,
    remap_labels,
    split_identities,
)
from evaluate import evaluate_verification
from losses import ArcMarginProduct
from models import build_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".."))
    parser.add_argument("--backbone", choices=["mobilefacenet", "iresnet18", "iresnet34"], default="mobilefacenet")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-positives", type=int, default=3000)
    parser.add_argument("--val-negatives", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"), help="Deprecated alias for --model-root")
    parser.add_argument("--model-root", type=Path, default=None, help="Root directory for .pt model files")
    parser.add_argument("--history-root", type=Path, default=Path("histories"), help="Root directory for logs/configs")
    parser.add_argument("--run-name", type=str, default=None, help="If set, use model_root/run_name and history_root/run_name")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint_epoch_XXX.pt every N epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    history_dir = args.checkpoint_dir
    if args.run_name:
        model_root = args.model_root if args.model_root is not None else args.experiment_root
        args.checkpoint_dir = model_root / args.run_name
        history_dir = args.history_root / args.run_name
    torch.manual_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.checkpoint_dir / "snapshots"
    if args.save_every > 0:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["resolved_model_dir"] = str(args.checkpoint_dir)
    config["resolved_history_dir"] = str(history_dir)
    (history_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_items = load_train_items(args.data_root)
    train_labels, val_labels = split_identities(all_items, args.val_ratio, args.seed)
    train_items_raw = [(path, label) for path, label in all_items if label in train_labels]
    val_items = [(path, label) for path, label in all_items if label in val_labels]
    train_items, label_map = remap_labels(train_items_raw)
    num_classes = len(label_map)

    print(f"Train images: {len(train_items)}; train identities: {num_classes}")
    print(f"Val images: {len(val_items)}; val identities: {len(val_labels)}")

    train_dataset = FaceClassificationDataset(args.data_root, train_items, transform=build_train_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_pairs, val_pair_labels = make_verification_pairs(
        val_items,
        args.data_root,
        positives=args.val_positives,
        negatives=args.val_negatives,
        seed=args.seed,
    )

    model = build_backbone(args.backbone, embedding_size=args.embedding_size).to(device)
    head = ArcMarginProduct(args.embedding_size, num_classes, scale=args.scale, margin=args.margin).to(device)
    optimizer = AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    best_acc = 0.0
    best_threshold = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * images.size(0)
            preds = logits.detach().argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += images.size(0)
            pbar.set_postfix(
                loss=running_loss / running_total,
                cls_acc=running_correct / running_total,
                lr=optimizer.param_groups[0]["lr"],
            )

        scheduler.step()

        threshold, val_acc = evaluate_verification(
            model,
            val_pairs,
            val_pair_labels,
            device=device,
            batch_size=max(args.batch_size, 256),
            num_workers=args.num_workers,
        )
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} train_cls_acc={train_acc:.5f} "
            f"val_ver_acc={val_acc:.5f} threshold={threshold:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_acc": train_acc,
            "val_ver_acc": val_acc,
            "threshold": threshold,
        }
        history.append(record)
        (history_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        best_record = max(history, key=lambda item: item["val_ver_acc"])
        (history_dir / "best_summary.json").write_text(json.dumps(best_record, indent=2), encoding="utf-8")

        state = {
            "epoch": epoch,
            "backbone": args.backbone,
            "embedding_size": args.embedding_size,
            "num_classes": num_classes,
            "threshold": threshold,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        torch.save(state, args.checkpoint_dir / "last.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(state, snapshots_dir / f"epoch_{epoch:03d}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            best_threshold = threshold
            torch.save(state, args.checkpoint_dir / "best.pt")
            print(f"Saved best checkpoint: acc={best_acc:.5f}, threshold={best_threshold:.4f}")


if __name__ == "__main__":
    main()
