# Face Verification Project - V1 现状记录

## 项目概述

**竞赛名称**: [Face Verification 2025 Postgraduate](https://www.kaggle.com/competitions/face-verification-2025-postgraduate)

**任务类型**: 人脸验证

**数据规模**: ~20 万张训练图片

**目标**: 判断两张人脸图片是否为同一人

---

## 模型架构

### 1. MobileFaceNet（轻量级）
**架构特点**：
- 使用 Depthwise Separable Convolution
- 17 个 MobileBottleneck
- 参数量小，推理速度快

**配置**：
- Embedding size: 512
- ArcFace margin: 0.35
- ArcFace scale: 64.0

### 2. iResNet18（中等规模）
**架构特点**：
- 8 个 IRBlock（Inverted Residual Block）
- 4 个阶段：[2, 2, 2, 2]
- 平衡性能和速度

**配置**：
- Embedding size: 512
- Dropout: 0.4
- ArcFace margin: 0.35
- ArcFace scale: 64.0

### 3. iResNet34（大规模）
**架构特点**：
- 16 个 IRBlock
- 4 个阶段：[3, 4, 6, 3]
- 表达能力强，但参数量大

**配置**：
- Embedding size: 512
- Dropout: 0.4
- ArcFace margin: 0.35
- ArcFace scale: 64.0

---

## Baseline 结果

### 模型性能对比

| 模型 | 最佳 Epoch | 训练 Loss | 训练准确率 | 验证准确率 | Threshold | 训练时长 |
|------|-----------|---------|-----------|-----------|-----------|---------|
| **MobileFaceNet** | 37 | 5.33 | 53.4% | **90.70%** | 0.169 | 7.5h (40 epochs) |
| **iResNet18** | 15 | 3.80 | 58.8% | **91.07%** | 0.189 | 7.5h (19 epochs) |
| **iResNet34** | 训练中 | - | - | ~92% (Epoch 13) | - | 15min/epoch |

**关键发现**：
- iResNet18 比 MobileFaceNet 高 0.37%
- iResNet18 在第 15 轮达到峰值，之后开始过拟合
- MobileFaceNet 训练更稳定，40 轮后仍在 90.6-90.7% 波动
- iResNet34 在训练中表现更好（Epoch 13 达到 ~92%）

---

## 当前 Baseline: iResNet18

**选择原因**：
- 验证准确率最高（91.07%）
- 训练时间适中（7.5h）
- 过拟合风险可控
- 与 iResNet34 差异大，适合集成

**模型信息**：
- Checkpoint: checkpoints_iresnet18_m035/best.pt
- 验证准确率: 91.07%
- Threshold: 0.189
- 最佳 Epoch: 15

---

## 待改进方向

### 优先级 1（立即可做，低成本）

#### 1. TTA（测试时增强）
- **方法**: 测试时对图像进行水平翻转，然后平均 embedding
- **预期收益**: 1-2%
- **实现时间**: 15 分钟
- **实现难度**: 简单

#### 2. Snapshot Ensemble
- **方法**: 在训练过程中保存多个 checkpoint，预测时平均
- **预期收益**: 1-2%
- **实现时间**: 30 分钟代码 + 0 额外训练时间
- **实现难度**: 简单

#### 3. iResNet18 + iResNet34 集成
- **方法**: 两个模型提取 embedding 后取平均
- **预期收益**: 2-3%
- **实现时间**: 30 分钟代码
- **实现难度**: 简单

### 优先级 2（需要重新训练，中等成本）

#### 4. Dynamic Margin
- **方法**: 根据训练阶段动态调整 ArcFace margin
- **预期收益**: 1-2%
- **实现时间**: 7.5h (iresnet18)
- **实现难度**: 简单

#### 5. 更强的数据增强
- **方法**: 添加 RandomResizedCrop, RandomRotation
- **预期收益**: 1-2%
- **实现时间**: 7.5h (iresnet18)
- **实现难度**: 简单

### 优先级 3（高级方法，高成本）

#### 6. Softmax Teacher + ArcFace Student
- **方法**: 先用 Softmax 训练 teacher，再用 soft labels 训练 ArcFace
- **预期收益**: 3-5%
- **实现时间**: 15h (两阶段训练)
- **实现难度**: 中等

#### 7. Mixed Loss（ArcFace + Center Loss）
- **方法**: 结合 ArcFace 和 Center Loss
- **预期收益**: 1-2%
- **实现时间**: 7.5h (iresnet18)
- **实现难度**: 中等

---

## 实验策略（推荐）

### 阶段 1: 快速验证（使用 iResNet18）
1. **TTA**（15 分钟）→ 预期 92.5-93%
2. **Snapshot Ensemble**（0 成本）→ 预期 93-94%
3. **Dynamic Margin**（7.5h）→ 预期 94-95%
4. **更强的数据增强**（7.5h）→ 验证是否有效

**预计总收益**: 91.07% → 93.5-94.5%

### 阶段 2: 模型集成
5. **iResNet18 + iResNet34 集成**（30 分钟代码）→ 预期 94-95%
6. **TTA + Ensemble** → 预期 95-95.5%

**预计总收益**: 91.07% → 95-95.5%

### 阶段 3: 高级方法（如果需要达到 96%）
7. **Softmax Teacher**（15h）→ 预期 96-96.5%
8. **Mixed Loss**（7.5h）→ 额外 0.5-1%

**预计总收益**: 91.07% → 96-96.5%

---

## 下一步行动

### 立即执行
- [x] 初始化 Git 仓库
- [x] 创建 .gitignore
- [x] 提交到 GitHub
- [ ] 实现 TTA（15 分钟）
- [ ] 等待 iResNet34 训练完成

### 短期目标（1-2 天）
- [ ] 实现并测试 TTA
- [ ] 实现 Snapshot Ensemble
- [ ] iResNet18 + iResNet34 集成
- [ ] 达到 94-95%

### 中期目标（3-5 天）
- [ ] 用 iResNet18 验证 Dynamic Margin
- [ ] 用 iResNet18 验证更强的数据增强
- [ ] 选择有效的方法应用到 iResNet34
- [ ] 达到 95-96%

---

## 版本历史

| 版本 | 日期 | 说明 | 验证准确率 |
|------|------|------|-----------|
| v1 | 2026-07-07 | Baseline: iResNet18 (margin=0.35) | 91.07% |

---

## 备注

1. **训练时间成本**: iResNet34 一个 epoch 约 15 分钟，30 轮约 7.5 小时
2. **过拟合风险**: iResNet18 在第 15 轮后开始过拟合，iResNet34 可能更严重
3. **验证集大小**: 6000 对（3000 正样本 + 3000 负样本）
4. **Git 仓库**: 已推送到 GitHub

---

**最后更新**: 2026-07-07
**当前状态**: iResNet34 训练中（Epoch 13/30，~92%）
