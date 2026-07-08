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
- `margin`: current ArcFace margin. It changes only when dynamic margin is enabled.
- `center_loss`: Center Loss value. It is useful only when `--center-loss-weight > 0`.

Prefer checkpoints with better `val_ver_acc`, not necessarily lower training loss.

For a more stable validation estimate after the first successful run, add:

```bash
--val-positives 10000 --val-negatives 10000
```

## Dynamic Margin, Augmentation, And Center Loss

The default training behavior stays the same unless these arguments are provided.

Available augmentation policies:

```text
none  no random training augmentation except tensor conversion and normalization
v1    original conservative augmentation
v2    v1 + light RandomAffine(degrees=5, translate=3%, scale=0.95~1.05)
v3    v2 + light blur/autocontrast, slightly stronger color jitter
```

First recommended IR-ResNet18 experiment:

```bash
python train.py --data-root .. \
  --backbone iresnet18 \
  --epochs 30 \
  --batch-size 256 \
  --num-workers 8 \
  --amp \
  --lr 1e-3 \
  --margin 0.40 \
  --margin-start 0.10 \
  --margin-end 0.40 \
  --margin-warmup-epochs 10 \
  --augment v2 \
  --center-loss-weight 0.001 \
  --center-lr 1e-2 \
  --model-root models \
  --history-root histories \
  --run-name iresnet18_dynm010to040_augv2_center0001_bs256_snap5 \
  --save-every 5
```

If this improves over the previous `iresnet18_m035_lr1e3_bs256_snap5`, move the same settings to IR-ResNet34.

## Validation Error Analysis

Use this to inspect false positives and false negatives on held-out training identities. This is allowed because it uses validation data split from `train`, not Kaggle test labels.

Single checkpoint:

```bash
python analyze_val_errors.py --data-root .. \
  --checkpoint models/iresnet18_m035_lr1e3_bs256_snap5/best.pt \
  --run-name iresnet18_m035_lr1e3_bs256_snap5_best \
  --batch-size 512 \
  --top-k 200 \
  --copy-images
```

Snapshot + TTA error analysis:

```bash
python analyze_val_errors.py --data-root .. \
  --checkpoint \
  models/iresnet18_m035_lr1e3_bs256_snap5/snapshots/epoch_010.pt \
  models/iresnet18_m035_lr1e3_bs256_snap5/snapshots/epoch_015.pt \
  models/iresnet18_m035_lr1e3_bs256_snap5/snapshots/epoch_020.pt \
  --run-name iresnet18_snapshot_10_15_20_tta \
  --batch-size 512 \
  --tta \
  --top-k 200 \
  --copy-images
```

Outputs:

```text
error_analysis/<run-name>/
  val_errors.csv
  summary.json
  index.html
  images/
```

## TTA And Score Sweep

Submission outputs are now organized automatically. If `--out` and `--scores-out` are not specified,
`make_submission.py` writes files like this:

```text
submissions/
  README.md
  iresnet18_m035_lr1e3_bs256_snap5/
    submission_baseline.csv
    scores_baseline.csv
    metadata_baseline.json
    submission_tta.csv
    scores_tta.csv
    metadata_tta.json
    submission_snapshot_10_15_20_tta.csv
    scores_snapshot_10_15_20_tta.csv
    metadata_snapshot_10_15_20_tta.json
    submission_tta_t0p1900.csv
```

Generate the plain baseline submission:

```bash
python make_submission.py --data-root .. \
  --checkpoint models/iresnet18_m035_lr1e3_bs256_snap5/best.pt \
  --eval-name baseline \
  --batch-size 512
```

Generate a submission with horizontal-flip TTA:

```bash
python make_submission.py --data-root .. \
  --checkpoint models/iresnet18_m035_lr1e3_bs256_snap5/best.pt \
  --eval-name tta \
  --batch-size 512 \
  --tta
```

Generate new submissions from the saved scores without recomputing embeddings:

```bash
python scores_to_submission.py \
  --scores submissions/iresnet18_m035_lr1e3_bs256_snap5/scores_tta.csv \
  --threshold 0.17

python scores_to_submission.py \
  --scores submissions/iresnet18_m035_lr1e3_bs256_snap5/scores_tta.csv \
  --threshold 0.19

python scores_to_submission.py \
  --scores submissions/iresnet18_m035_lr1e3_bs256_snap5/scores_tta.csv \
  --threshold 0.21
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
  --checkpoint \
  models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_010.pt \
  models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_015.pt \
  models/iresnet18_m035_lr1e3_bs256/snapshots/epoch_020.pt \
  --eval-name snapshot_10_15_20_tta \
  --batch-size 512 \
  --tta
```
