# Submission Outputs

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
