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

## TTA And Score Sweep

Generate an iresnet18 submission with horizontal-flip TTA and save raw similarity scores:

```bash
python make_submission.py --data-root .. --checkpoint checkpoints_iresnet18_m035/best.pt --out submission_iresnet18_tta.csv --scores-out scores_iresnet18_tta.csv --batch-size 512 --tta
```

Generate new submissions from the saved scores without recomputing embeddings:

```bash
python scores_to_submission.py --scores scores_iresnet18_tta.csv --threshold 0.17 --out submission_iresnet18_tta_t017.csv
python scores_to_submission.py --scores scores_iresnet18_tta.csv --threshold 0.19 --out submission_iresnet18_tta_t019.csv
python scores_to_submission.py --scores scores_iresnet18_tta.csv --threshold 0.21 --out submission_iresnet18_tta_t021.csv
```

Average multiple checkpoints or models by passing more than one checkpoint:

```bash
python make_submission.py --data-root .. --checkpoint checkpoints_iresnet18_m035/best.pt checkpoints_iresnet18_m035/last.pt --out submission_iresnet18_best_last_tta.csv --scores-out scores_iresnet18_best_last_tta.csv --batch-size 512 --tta
```

## Snapshot Checkpoints

Recommended experiment layout:

```text
models/
  iresnet18_m035_lr1e3_bs256/
    best.pt
    last.pt
    snapshots/
      epoch_005.pt
      epoch_010.pt
      epoch_015.pt
      ...
histories/
  iresnet18_m035_lr1e3_bs256/
    config.json
    history.json
    best_summary.json
```

Keep `models/` on the remote server. Download/sync only `histories/`, submissions, and score CSVs.

Save one snapshot every 5 epochs during training:

```bash
python train.py --data-root .. \
  --backbone iresnet18 \
  --epochs 30 \
  --batch-size 256 \
  --num-workers 8 \
  --amp \
  --margin 0.35 \
  --lr 1e-3 \
  --model-root models \
  --history-root histories \
  --run-name iresnet18_m035_lr1e3_bs256 \
  --save-every 5
```

Use selected snapshots for ensemble. Do not blindly average very early bad snapshots; prefer snapshots around and after the best validation epoch.

```bash
python make_submission.py --data-root .. \
  --checkpoint models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_010.pt models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_015.pt models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_020.pt \
  --out submission_iresnet18_snap_tta.csv \
  --scores-out scores_iresnet18_snap_tta.csv \
  --batch-size 512 \
  --tta
```
