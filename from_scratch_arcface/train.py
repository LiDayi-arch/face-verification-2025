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
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

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
        if val_acc > best_acc:
            best_acc = val_acc
            best_threshold = threshold
            torch.save(state, args.checkpoint_dir / "best.pt")
            print(f"Saved best checkpoint: acc={best_acc:.5f}, threshold={best_threshold:.4f}")


if __name__ == "__main__":
    main()
