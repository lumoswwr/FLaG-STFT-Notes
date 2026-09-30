# Sprint 迁移：数据、训练协议与指标计算

本节说明 **SprintDuplicateQuestions** 如何复用 STSB 已选定的 E12（`win=16,hop=16,rect,center=False`），以及为什么**不能直接比较 STSB 的 Spearman 与 Sprint 的 AP**。S1–S5 的完整实验与每组 seeds 见[下一页](02-experiments.md)，全项目约定见[统一实验协议](../methods.md)。

## 1. 要回答的跨任务问题

STSB 是连续语义相似度回归；FLaG 原论文在 STSB 中低于 Mean，我们观察到 E12 早期小幅正向信号，但 10 seeds 未证实稳定优势。为考察机制结果是否只存在于 STSB，将**已确定结构的 E12**移到**重复问题检测** Sprint。这里不因 Sprint 结果重新调局部 `win/hop`，避免直接在新的测试集上继续筛结构。

**这是一种跨任务检验**，不是把 STSB 已训练 checkpoint 直接迁移到 Sprint：Sprint 的 pooling 从头初始化并训练，backbone 冻结，标签、loss 与评估也不同。

## 2. 数据集和划分

本项目采用 `mteb/sprintduplicatequestions-pairclassification` 的相应数据配置，输入字段为 `sentence1`、`sentence2`、`label`，目标是识别是否重复。所采用的任务配置没有用于本实验的独立官方训练划分，因此固定从官方 validation 内分出 adaptation train/validation，并保留官方 test 做最终评估：

| 数据 | 来源与用途 | 句对数 |
| --- | --- | ---: |
| Adaptation train | 官方 validation 的分层 90%，仅此部分参与训练 loss | **90,900** |
| Adaptation validation | 官方 validation 的其余分层 10%，用于 checkpoint 与分类阈值 | **10,100** |
| Official test | 不参与训练 loss、不在其上选择 epoch/阈值，最终指标 | **101,000** |

adaptation split 固定分层、`random_state=42`，不同 pooling 方法和模型训练 seeds 使用**同一组样本划分**。由于本项目在方法探索中曾查看过早期 test 调试结果，后续 test 结果应该视为已受探索过程影响的评估，不应宣称“从未查看 test 的新确认”。

## 3. STSB 与 Sprint 的协议差别

| 项目 | STSB | Sprint |
| --- | --- | --- |
| 任务 | 句对语义相似度回归 | 句对是否重复（二分类） |
| Backbone | RoBERTa-base **端到端微调** | RoBERTa-base **全部冻结**并设为 eval |
| Pooling | 重新训练各 FLaG/STFT 变体 | 各变体重新初始化并训练；E12 **只固定结构，不继承 STSB 权重** |
| 输出 | 两个句向量的 cosine | 两句 cosine 经**单调正尺度 + bias**变成 logit |
| Loss | `MSE(cosine, label/5)` | `BCEWithLogitsLoss(logit,label)` |
| Epoch | 3 | 10 |
| 主要学习率 | backbone `1e-5`，pool `1e-3` | pooling/head `1e-3`，backbone 冻结 |
| Batch | 4 个句对，累积 4 次 | 32 个句对 |
| Max length | 128 | 128 |
| 验证选 checkpoint | validation Spearman | validation **阈值校准后 Accuracy** |
| 主指标 | official-test Spearman、Pearson | official-test AP、Accuracy、F1 等 |

Sprint 的具体优化器、精确 batch、固定 stratified split 和 head 参数化是**我们的复现代码设置**，不应在没有额外佐证时表述为原论文作者公布的全部实现参数。

## 4. Sprint 的打分器与评估公式

每条输入句子使用**相同冻结 RoBERTa**产生 token hidden，经对应 pooling 得到句向量 `z1,z2`。训练的轻量打分器使用：

$$
c=\operatorname{cosine}(z_1,z_2),\qquad
\ell=\operatorname{softplus}(w)c+b .
$$

\(\operatorname{softplus}(w)>0\)，因此在数学上，\(\ell\) 保持余弦分数的排序。模型优化 **BCEWithLogitsLoss**，\(\ell\) 是 logit，\(\sigma(\ell)\) 为正类概率式分数。head 主要负责缩放与阈值校准；它本身不能通过严格单调变换提升固定 embedding 的排序 AP。

### 为什么主要报告 AP，而不仅看 Accuracy？

**Average Precision（AP）** 不是一个阈值下的分类正确率。按预测分数从高到低扫描，记第 `k` 个召回率层级的 precision 为 \(P_k\)、recall 为 \(R_k\)，其常用离散定义为：

$$
\operatorname{AP}=\sum_k(R_k-R_{k-1})P_k .
$$

它刻画正类靠前排序的质量，**越大越好**。在该任务中正类较稀少，即使大部分样本一律预测为非重复，Accuracy 也可能非常高，因此 Accuracy 必须和 AP/F1 一起解释。

其余分类指标定义：

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}.
$$

### 阈值与模型选择，必须说清楚

项目代码每个 epoch 都只在 **adaptation validation** 上计算候选 logits/cosine 与标签，再**分别**求两个阈值：

1. **Accuracy 阈值**：在 validation 上搜索 Accuracy 的最优阈值，并以该 validation Accuracy 选择 checkpoint（严格优于才替换，因此同分保留更早 epoch）。
2. **F1 阈值**：在 validation 上另外搜索 F1 的阈值，连同所选 checkpoint 一起保存，用于后面的 F1/Precision/Recall。

最终对 official test **冻结**这两个阈值：test **Accuracy 使用 Accuracy 阈值**；test **F1/Precision/Recall 使用 F1 阈值**；AP 则不使用分类阈值。**所以单行 Accuracy 与 F1 不对应同一张混淆矩阵**，不能依据这两个报告数反推出统一 TP/FP/TN/FN。

## 5. 核心结构设置与后续控制

- Mean：对有效 token hidden 取平均，pool 本身无可训练参数，只有 Sprint 的轻量 head 参与适配。
- Global FLaG：8 latents、4 heads、residual gate、masked max pooling、global FFT。
- E12 STFT-FLaG：继承 STSB 选定的 `win=16,hop=16,rect,center=False`、无 overlap、无 frame positional encoding，其余结构与 global FLaG 对齐。

!!! warning "Sprint 早期性能比较存在配置混杂"
    S1/S2 的**原始训练记录**为 FLaG `dropout=0.1,post_pool_norm=False`，E12 `dropout=0,post_pool_norm=True`。因此早期 E12 对 FLaG 的 AP 差异同时包含**算子、dropout 和 normalization**。其数字是真实的整套系统比较，但**不能**当成“STFT 算子造成的 AP 提升”。S4 补充匹配 global，S5 专门测试 global 上的 dropout × norm 交互。

**论文方法文本**对 language 的描述是 `dropout=0.1` 且 **post-pool LayerNorm=True**。公开代码路径与文字存在需区分的实现细节；我们的 S1 早期版本不能代称“严格论文定义下的 Sprint FLaG”。Sprint 目前没有重新训练一套 **`0.1/True` 的 E12** 与该 global 配置做完整匹配对比；只有 **STSB E14** 执行了这一种配置下的 10-seed global/local 匹配实验。

## 6. 结果报告规则

- **S1**：早期 3-seed official-test 性能，只供探索观察；**S2**：扩为 10 seeds，但**并未消除早期配置不匹配**。
- **S3**：使用固定源 checkpoint 的**validation 推理干预**（GG/LG/GL/LL），比较同权重下的 operator 路线，不代表四组重新训练的测试性能。
- **S4**：`dropout=0/norm=True` 匹配非算子设置，global vs E12，**10-seed adaptation validation AP**；与 S2 的 official-test AP 分开报告。
- **S5**：仅针对 **global FLaG** 的 2×2 dropout/norm，**10-seed adaptation validation AP**，不是 global/local 的 2×2×2，更不是新的 official-test 结论。

**代码索引：** [Sprint 训练与阈值](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)，[Sprint global/local 2×2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_sprint_global_local_2x2.py)，[STFT/FLaG pooling](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/factory/pooling/flag_pooling.py)。
