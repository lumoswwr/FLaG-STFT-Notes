# 实验设置与指标

## 先看懂我们比较了哪些东西

本项目最开始比较 **Global FLaG（对整句话做 FFT）** 与 **STFT-FLaG（把句子分成小窗口做 FFT）**。随后为了解释预测差异，分析了 Padding、重建、Gate、DC 频带和可训练输出线性层，并尝试把 Mean 与 FLaG 结合起来。

不同阶段的实验不一定采用同一套模型参数。阅读结果时，尤其需要看清三件事：

1. **是否重新训练**：改变训练结构，与在已训练模型上临时关闭某个模块，是两个不同实验。
2. **报告的是 Validation 还是 Test**：Validation 用来选择模型或配置，Test 用来报告选定模型在测试数据上的效果。
3. **FLaG 最后采用 Max 还是 Mean，以及是否开启后池化 LayerNorm**：这些改变可能明显影响性能，不能把不同配置的数字直接混在一起比较。

各篇报告开头的“本页实验概览”先用普通语言给出每项实验的目的、思路、结果和结论；下面的详细记录用于核对完整设置。

## 1. 模型

原始 FLaG：`RoBERTa hidden → Global rFFT → latent attention → gate → irFFT → masked max pooling`。

STFT-FLaG：将 global FFT 换成**逐帧局部 FFT**。全部有效帧的频率 token 共同生成**一个句级 gate**，然后分别门控局部频谱、逐帧逆变换并拼接。

两者共同配置：**8 latent queries、4 attention heads、residual gate（1+sigmoid）、masked max pooling**。gate 按通道区分实部和虚部，在频率位置之间共享。

常用局部模型：**E3 = win8/hop4**；**E12 = win16/hop16**。除实验另有说明，均为 rectangular window、`center=False`，不使用帧位置编码。

## 2. 早期 STFT 结构对照：数据与训练协议

| 设置 | STSBenchmark（STSB） | SprintDuplicateQuestions（Sprint） |
| --- | --- | --- |
| 任务 | 句对语义相似度（分数 0–5） | 判断句对是否重复（0/1） |
| 数据 | 官方 train / validation / test | 官方 validation 按固定 90:10 分为训练 90,900 对、验证 10,100 对；official test 101,000 对 |
| Backbone | RoBERTa-base，**微调** | RoBERTa-base，**冻结** |
| 输出与 loss | `cosine(z1,z2)` 对比 `label/5`；MSE | `softplus(w)×cosine+b`；BCEWithLogitsLoss |
| 训练 | 3 epochs；batch=4、累积 4 次 | 10 epochs；batch=32 |
| 学习率 | backbone 1e-5，pooling 1e-3 | pooling/head 1e-3 |
| Max length | 128 | 128 |
| 选 checkpoint | validation Spearman | validation 上校准阈值后的 Accuracy |
| 主要指标 | **test Spearman、Pearson** | **AP**，并报告 Accuracy、F1 等 |

使用 AdamW。Sprint 的训练/验证划分固定 `random_state=42`，不同模型共享同一划分。**STSB 与 Sprint 的评价指标不能直接比较。**

## 3. 指标说明

- **Spearman**：模型预测与真实标签的**排序相关系数**，越高越好。
- **Pearson**：预测与真实标签的**线性相关系数**，越高越好。
- **AP（Average Precision）**：按预测分数排序，汇总各召回率位置的 precision，**不需要分类阈值**；Sprint 主要看这个指标。
- **Accuracy**：预测正确的样本比例；**Precision**：预测为正的样本中有多少是真的；**Recall**：正类中找出了多少；**F1**：Precision 和 Recall 的调和平均。

Sprint 分别在 validation 选择 **Accuracy 阈值**与 **F1 阈值**：最终 Accuracy 用前者，F1/Precision/Recall 用后者；AP 不使用阈值。

表中的 **mean ± std** 为各 seed 指标的均值 ± **样本标准差**。比较两个模型时，同一 seed 的差值应先配对，再汇总。

## 4. 必须区分的配置

| 实验 | FLaG dropout / post-pool Norm | E12 dropout / post-pool Norm |
| --- | --- | --- |
| STSB 初期探索、P1/P2 | 0 / 有 | 0 / 有 |
| STSB E14（论文 text 配置） | 0.1 / 有 | 0.1 / 有 |
| Sprint 早期 S1/S2 | **0.1 / 无** | **0 / 有** |
| Sprint 匹配实验 S4 | 0 / 有 | 0 / 有 |
| Sprint S5 | **仅 global FLaG**：dropout 0/0.1 × norm 有/无 | 未执行 |

**Sprint S1/S2 的两种模型配置并不匹配**，不能把整套配置的性能差全部归因于 STFT。**S4/S5 目前只报告 adaptation validation AP**，不与 S1/S2 的 official-test AP 直接比较。

## 5. 后续研究采用的不同协议

| 研究阶段 | 使用什么模型与训练设置 | 主要关注什么 |
| --- | --- | --- |
| [早期 STFT 结构与 Padding](stsb/01-experiments.md) | Global FLaG / E3 / E12；STSB 微调 RoBERTa，通常使用 masked Max 和 post-pool LayerNorm | Global vs. Local、窗口和 Padding |
| [Sprint 早期 S1–S5](sprint/02-experiments.md) | 冻结 RoBERTa；S1/S2 的 Global/Local dropout 和 norm 不匹配；S4 匹配到 dropout=0、norm=True | 结构差异与训练配置效应 |
| [DC 和频带机制](dc-frequency.md) | STSB、Sprint 都采用 **冻结 RoBERTa** 的控制实验；后续 FLaG 仍可训练 | 删除、只保留和从训练开始限定频率 |
| [Mean 与 FLaG 融合](mean-compatible-flag.md) | 五种子版本使用 masked Max、启用后池化 LayerNorm；STSB/IMDB 微调 RoBERTa，Sprint 冻结 | 在不同任务里组合 Mean 与 FLaG |
| [Gate 和 Projection 机制](flag-gate-projection.md) | Global FFT、**time pooling=masked Mean、关闭 post-pool LayerNorm、dropout=0**；STSB 微调，Sprint 冻结 | Gate 在哪里改变表示、是否可以使用简单线性模型替代 |

### 最新 Sprint Projection 对照的 checkpoint 选择

早期 Sprint S1/S2 和一部分机制实验按照 **validation Accuracy** 选择 checkpoint；后期专门比较 FLaG 与 Mean + Projection 的多种子实验，则统一按 **validation AP** 选择 checkpoint。前者和后者不能作为同一组对照直接汇总。

最新严格配对实验还使两个模型具有**完全相同的初始 Projection 权重和偏置**，以便判断加入完整频域 Gate 后 Test AP 如何变化。详见 [Gate 和 Projection 的作用](flag-gate-projection.md)。

### B0 与 exact DC 不是同一个概念

[STSB E6](stsb/01-experiments.md) 与频带分析中会把 DCT 频谱分为 B0–B7，**B0 是最低频带**；后面的 [exact DC 实验](dc-frequency.md) 则只保留 rFFT 的 $k=0$，即**严格的零频率分量**。两类频域操作和实验问题不同，报告分别写明。
