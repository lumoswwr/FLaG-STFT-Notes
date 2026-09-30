# 实验协议、配置与评价指标

本页是 E1–E14、P1/P2 和 Sprint S1–S5 的统一说明。



## 1. 两个数据集与任务

| 项目 | STSBenchmark（STSB） | SprintDuplicateQuestions（Sprint） |
| --- | --- | --- |
| 输入 | 两个英文句子 | 两个英文问题 |
| 任务 | 预测连续语义相似度 | 二分类：是否为重复问题 |
| 标签 | 原始分数 0–5，训练时归一化到 0–1 | 0 / 1 |
| Backbone | RoBERTa-base，**端到端微调** | RoBERTa-base，**冻结** |
| 训练目标 | 两句余弦相似度与归一化标签的 MSE | 正尺度余弦仿射 logit 的 BCEWithLogitsLoss |
| 主比较指标 | test **Spearman**、test **Pearson** | **Average Precision（AP）**，并报告 Accuracy、F1 等 |
| 选择模型 | validation Spearman | validation Accuracy（采用 validation 校准的 Accuracy 阈值） |
| 划分 | 官方 train 训练、development/validation 选 checkpoint、官方 test 评估 | 使用所采用配置的官方 validation，固定分层划分 90%/10% 作为 adaptation train/validation；官方 test 最后评估 |
| 随机性 | 训练 seeds；同一组比较应配对相同 seeds | 训练 seeds；adaptation split 固定 `random_state=42`，所有模型一致 |



## 2. 统一模型流程与可变因素

`RoBERTa hidden states [B,T,D] → 频域分析 → latent attention → 共享的通道 gate → gated spectrum → 时域重建 → masked max pooling → 可选 post-pool LayerNorm → 输出 projection`

- **Global FLaG**：对 attention mask 处理后的整条 batch-padded 序列做 global `rFFT`，之后用相同长度 `irFFT`。默认 FFT 长度跟当前 batch 的 padded 长度 `T` 变化。
- **Local STFT-FLaG**：按固定 `win_length` / `hop_length` 切分有效序列，每帧做局部 FFT，将有效帧频率 token 汇总用于生成**一个句级 gate**，再对各帧分别做逆变换并拼回 token 序列。**不是每帧生成一个 gate**。
- **E3**：局部 `win=8, hop=4, rect, center=False`，相邻帧重叠 4 个位置。
- **E12**：局部 `win=16, hop=16, rect, center=False`，无重叠，无额外 frame position encoding。选作后续研究的**固定 local 候选**，而非已证实的通用最优模型。

共同核心结构：**8 latent queries、4 attention heads、residual channel gate（`1+sigmoid`）、masked max pooling**。gate 是一个长为 `2D` 的样本级向量，分别作用于实部和虚部**通道**，在当前实现中各频率 bin 共享同一 gate；不要将其描述成“为各频段分别学习一套 gate”。

### 必须区分的 dropout / normalization 配置

| 使用场景 | FFT 算子 | pooling dropout | post-pool LayerNorm |
| --- | --- | ---: | :---: |
| STSB 最初 FLaG、E1–E13 机制探索及原 E12 | 按实验分别为 global / local | 0 | 有 |
| STSB E14 论文文本配置补充（FLaG / E12） | 匹配比较 | 0.1 | 有 |
| Sprint 早期 S1/S2 FLaG | Global | 0.1 | **无** |
| Sprint 早期 S1/S2 E12 | Local 16/16 | 0 | **有** |
| Sprint S4 matched global 与 E12 | 匹配比较 | 0 | 有 |
| Sprint S5（仅 global FLaG） | Global | 0 或 0.1 | 有或无 |

**论文正文定义**的文本配置是 **dropout=0.1，post-pool LayerNorm=True**。原公开实现中定义了 `norm3`，但部分 released forward 路径没有使用它；我们早期 Sprint 复现沿用了 `0.1/无 norm`。 S1/S2 的早期 FLaG 与 E12 同时改变了算子、dropout、norm，S4 才是匹配这些非算子因素后的比较。

## 3. 训练参数

| 项目 | STSB | Sprint |
| --- | --- | --- |
| Backbone | RoBERTa-base 可训练 | RoBERTa-base 冻结并保持 eval |
| Max sequence length | 128 | 128 |
| Optimizer | AdamW | AdamW |
| Backbone LR | `1e-5` | 冻结、不更新 |
| Pooling / head LR | `1e-3`（STSB 的 head 为余弦计算，无单独可学习分类 head） | `1e-3` |
| Epoch | 3 | 10 |
| Batch | 4 个句对，累积 4 次（等效 16） | 32 个句对 |
| 目标 | MSE | BCEWithLogitsLoss |
| 模型选择 | validation Spearman | validation 上校准 Accuracy 阈值后，以 Accuracy 选 epoch |
| 原始主实验 seed | STSB FLaG / Mean 后续统计使用 0–9；早期 E1–E13 主要使用 0–2；E12 后扩展到 0–9 | 早期 S1 为 0–2；S2 及 S4/S5 控制为 0–9 |



## 4. 指标究竟在测什么

### 4.1 STSB：Spearman、Pearson

设共有 `n` 个句对，模型预测的两句余弦相似度为 `s_i`，归一化后的真实标签（STSB 原始分数除以 5）为 `y_i`：

- **Spearman**：对 `s_i` 和 `y_i` 分别排序，计算两组秩之间的相关系数。衡量句对相似程度的**排序**是否一致；越接近 1 越好。
- **Pearson**：衡量预测与标签之间的**线性相关性**，也越接近 1 越好。
- **test 指标**：使用 validation Spearman 选出 checkpoint 后，在官方 test 上计算，不能当成 validation 分数。
- 表中的 `mean ± std`：先逐 seed 得到完整指标，再对 seeds 的指标算均值及**样本标准差**（`ddof=1`）。比较 FLaG 和 E12 时，应对**同 seed**做差后再汇总，不能用两种模型的边际标准差推出配对差的标准差。

### 4.2 Sprint：AP、Accuracy、F1、Precision、Recall

分类模型生成 `logit = softplus(raw_scale) × cosine(z_1,z_2) + bias`，正尺度保证 logit 与 cosine 在数学上保持相同排序（浮点精度下可能有极小同分误差）。

- **AP（Average Precision）**：沿预测分数从高到低排序，汇总各召回率提升位置的 precision。用于评价**不固定阈值时**检索重复问题的质量；这里不是 Accuracy，也不等于 ROC-AUC。
- **Accuracy**：`(TP+TN)/(TP+TN+FP+FN)`；在重复问题正类稀少的数据上，Accuracy 容易受负类占比主导。
- **Precision**：`TP/(TP+FP)`；**Recall**：`TP/(TP+FN)`；**F1**：`2PR/(P+R)`。
- 本项目在 **adaptation validation** 上分别选择 **Accuracy 阈值**与 **F1 阈值**；在最终 test 上冻结阈值：test Accuracy 使用前者，test F1/Precision/Recall 使用后者。因此这些指标**不对应同一套分类阈值或同一张混淆矩阵**。AP 为 threshold-free。
- **validation AP** 是从适配 validation 上得到的排序指标，**test AP** 来自官方 test；即使数值接近也不能互换。Sprint S4/S5 的 matched/2×2 控制是 validation-only。

### 4.3 机制探针的统计量，不能与性能指标混用

| 指标 | 用途 | 定义 |
| --- | --- | --- |
| 频带 knockout 的 Spearman 降幅 | E6 | 在符合 8 频带条件的测试子集上，`Spearman_before - Spearman_after_band_removed` |
| 单 token 预测响应 | E7 | 删掉一个内容 token 的 hidden 后，两句 cosine 的绝对变化 |
| 单 token 平方误差响应 | E7 | `|(s_after-y)^2 - (s_before-y)^2|`；先对相同位置跨 seed 对齐，再统计句内位置响应差异 |
| DC attention 偏好比 | E8 | **平均每个 DC 频率 token 的 attention / 平均每个非 DC 频率 token 的 attention**；它不是 DC 能量占比，也不是 knockout 重要性 |
| 平均绝对预测漂移 | E9/P1/P2/S3 | 相同输入/同 checkpoint，在两个推理处理方式下的 `mean(|prediction_A - prediction_B|)`；不是 Spearman/AP 提升 |

## 5. 实验结果使用规则

**3-seed 探索结果**（特别是 E1–E13 的候选搜索）只描述观察到的方向；**10-seed 配对结果**可用于考察其是否跨 seed 稳定，但显著性不足时不能称为稳定优势。STSB 早期探索 FLaG/E12 均为 `dropout=0/norm=True`：10-seed test Spearman 分别为 `0.841559±0.003538` 与 `0.842739±0.003214`，配对增量 `+0.001180±0.002107`，7/10 正向，区间跨 0。STSB E14 统一论文文本配置 `0.1/norm=True` 时，10-seed 分别为 `0.839065±0.004154` 与 `0.838944±0.001882`，配对增量约 `-0.000121±0.004039`，5/10 正向。

**研究主张的范围**：P1/P2 反映当前实现和被测 checkpoint 的结构、预测扰动；不意味着 local reconstruction 总能改善下游性能，也不能把 Sprint 早期整套配置增益全部归因于 STFT。
