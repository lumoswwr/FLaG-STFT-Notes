# Sprint：数据集与训练协议

将 STSB 中选定的 **E12 结构**（STFT 16/16，无重叠）用于 SprintDuplicateQuestions，验证它在另一种文本任务中的表现。Sprint 的模型**重新训练 pooling**，不直接使用 STSB 的 checkpoint。

## 1. 数据集与配置

Sprint 任务：输入两个英文问题，判断是否语义重复（二分类）。本实验使用的适配训练集来自官方 validation 的固定分层 90:10 划分；官方 test 用于最终评估。

| 数据 | 数量 | 用途 |
| --- | ---: | --- |
| Adaptation train | 90,900 对 | 训练 |
| Adaptation validation | 10,100 对 | 选择 checkpoint 和分类阈值 |
| Official test | 101,000 对 | 最终评估 |

划分随机种子为 **42**，所有模型共享同一划分。各实验的**模型随机种子数**另在 S1–S5 标注。

| 设置 | Sprint |
| --- | --- |
| Backbone | **冻结 RoBERTa-base** |
| Pooling | Mean、global FLaG 或 E12，重新初始化并训练 |
| E12 固定结构 | `win=16, hop=16, rect, center=False`；无帧位置编码 |
| Global/E12 核心设置 | 8 latent queries、4 attention heads、residual gate、masked max pooling |
| Max length | 128 |
| Loss | BCEWithLogitsLoss |
| Epochs / Batch | 10 / 32 |
| 优化器 / 学习率 | AdamW；pooling 和分类 head LR=`1e-3` |
| Checkpoint | 按 validation 上**校准阈值后的 Accuracy** 选择 |

与 STSB 不同：这里 **backbone 冻结，任务变为二分类，输出增加可训练分类 head**。

## 2. 预测与评价指标

两句分别经过 backbone 和 pooling，得到向量 `z1`、`z2`，使用 cosine 和可训练的正尺度、偏置得到分类 logit：

$$
\ell=\operatorname{softplus}(w)\operatorname{cosine}(z_1,z_2)+b.
$$

| 指标 | 含义 |
| --- | --- |
| **AP（Average Precision）** | 按预测分数排序，汇总不同召回率处的 precision；**无需分类阈值** |
| Accuracy | 预测正确的样本占比 |
| Precision | 预测为重复的问题中，实际重复的比例 |
| Recall | 所有重复问题中，被找出的比例 |
| F1 | Precision 与 Recall 的调和平均 |

本实验每轮在 **adaptation validation** 上分别选取 **Accuracy 阈值**和 **F1 阈值**。最终 test Accuracy 用前者，test F1 / Precision / Recall 用后者；AP 不使用阈值。由于正类数量较少，**主要比较 AP**，不能只看很高的 Accuracy。

## 3. 结果口径

- **S1/S2**：早期 official-test 结果。FLaG 为 `dropout=0.1, norm=False`，E12 为 `dropout=0, norm=True`，**配置不同**。
- **S3**：固定模型，在 adaptation validation 上进行 global/local gate 和重建干预。
- **S4**：FLaG 和 E12 都使用 `dropout=0, norm=True`，比较 **10-seed validation AP**。
- **S5**：只针对 **global FLaG** 检查 dropout × norm 的四种配置，同样报告 **10-seed validation AP**。

**注意：** S1/S2 的结果可以比较两套完整配置，但不能单独说明 local STFT 的作用；S4 才匹配了 dropout 与 norm。另外，Sprint S5 的 AP 与 STSB E14 的 Spearman **不属于同一个任务/指标**。
