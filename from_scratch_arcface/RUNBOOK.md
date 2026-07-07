# Runbook

## Recommended Order

1. Start with MobileFaceNet.
2. Submit the best checkpoint result.
3. Train IR-ResNet18.
4. If IR-ResNet18 improves validation and public score, train IR-ResNet34.

## MobileFaceNet

```bash
python train.py --data-root .. --backbone mobilefacenet --epochs 40 --batch-size 512 --num-workers 8 --amp
python make_submission.py --data-root .. --checkpoint checkpoints/best.pt --out submission_mobilefacenet.csv --batch-size 512
```

If GPU memory is below 16 GB:

```bash
python train.py --data-root .. --backbone mobilefacenet --epochs 40 --batch-size 256 --num-workers 8 --amp
```

## IR-ResNet18

```bash
python train.py --data-root .. --backbone iresnet18 --epochs 45 --batch-size 256 --num-workers 8 --amp --checkpoint-dir checkpoints_iresnet18
python make_submission.py --data-root .. --checkpoint checkpoints_iresnet18/best.pt --out submission_iresnet18.csv --batch-size 512
```

## IR-ResNet34

```bash
python train.py --data-root .. --backbone iresnet34 --epochs 50 --batch-size 192 --num-workers 8 --amp --checkpoint-dir checkpoints_iresnet34
python make_submission.py --data-root .. --checkpoint checkpoints_iresnet34/best.pt --out submission_iresnet34.csv --batch-size 512
```

## What To Watch

- `train_cls_acc`: identity classification accuracy. It can be low early because there are many classes.
- `val_ver_acc`: verification accuracy on held-out identities. This is the main signal.
- `threshold`: cosine threshold selected on validation pairs.

Prefer checkpoints with better `val_ver_acc`, not necessarily lower training loss.

For a more stable validation estimate after the first successful run, add:

```bash
--val-positives 10000 --val-negatives 10000
```
