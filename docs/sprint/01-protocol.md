# Sprint：数据集与训练设置

SprintDuplicateQuestions 是**判断两句话是否为重复问题**的二分类任务。沿用 STSB 选定的 **E12 结构（STFT 16/16）**，但在 Sprint 上**重新初始化并训练 pooling**，RoBERTa-base 则冻结。

## 数据与训练

| 项目 | 设置 |
| --- | --- |
| 数据 | `mteb/sprintduplicatequestions-pairclassification` |
| Adaptation train | 90,900 对，来自官方 validation 的分层 90% |
| Adaptation validation | 10,100 对，剩余分层 10% |
| Official test | 101,000 对，用于最终评估 |
| 划分随机种子 | 42，所有模型共享 |
| Backbone | **冻结 RoBERTa-base** |
| Max length | 128 |
| Epochs / Batch | 10 / 32 |
| 优化器 / 学习率 | AdamW；pooling/head 1e-3 |
| Loss | BCEWithLogitsLoss |
| Checkpoint | validation 上校准阈值后的 Accuracy |

## 指标

模型输出是两句 cosine 经可训练的正尺度和 bias 变换后的 logit：

$$
\ell=\operatorname{softplus}(w)\operatorname{cosine}(z_1,z_2)+b.
$$

主要指标 **AP（Average Precision）** 衡量重复问题的排序质量，**不需要分类阈值**。Accuracy 是总体正确率，F1 综合 Precision（查准率）与 Recall（查全率）。

在 **adaptation validation** 上分别选择 Accuracy 阈值、F1 阈值；最终 **test Accuracy 用前者**，**test F1/Precision/Recall 用后者**。两个指标使用的分类阈值不同。

## 配置说明

早期 Sprint 实验 **S1/S2**：FLaG 为 `dropout=0.1、norm=False`，E12 为 `dropout=0、norm=True`。两种配置不同，不能将它们的 AP 差全部归因于 STFT。

**S3/S4/S5** 只报告 **adaptation validation** 指标；S4 将 FLaG/E12 对齐为 `0/True`；S5 只对 **global FLaG** 做 dropout × norm 2×2。
