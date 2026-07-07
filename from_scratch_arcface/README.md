# Face Verification From Scratch

This project trains a face embedding model from random initialization using only the competition training data.

It does not load pretrained weights.

## Data Layout

Expected parent directory:

```text
机器学习竞赛/
  train_list.pkl
  test_list.pkl
  predict.csv
  train/train/<identity>/<image>.jpg
  test/test/<subset>/imgs/<image>.jpg
  from_scratch_arcface/
```

Run commands from `from_scratch_arcface`.

## First GPU Run

Install dependencies on the rented server:

```bash
pip install -r requirements.txt
```

Train MobileFaceNet from scratch:

```bash
python train.py --data-root .. --backbone mobilefacenet --epochs 30 --batch-size 256 --num-workers 8 --amp
```

If GPU memory allows, use a larger batch:

```bash
python train.py --data-root .. --backbone mobilefacenet --epochs 40 --batch-size 512 --num-workers 8 --amp
```

Train IR-ResNet18 after MobileFaceNet baseline:

```bash
python train.py --data-root .. --backbone iresnet18 --epochs 40 --batch-size 256 --num-workers 8 --amp
```

Generate a submission from a checkpoint:

```bash
python make_submission.py --data-root .. --checkpoint checkpoints/best.pt --out submission_from_scratch.csv --batch-size 512
```

## Notes

- The model is trained as identity classification with ArcFace loss.
- Validation is face verification on identities held out from training.
- The checkpoint stores the best validation verification accuracy and threshold.
- For Kaggle submission, only the backbone embedding model is used.
