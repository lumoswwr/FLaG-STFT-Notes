# 实验设置与指标

## 1. 模型

原始 FLaG：`RoBERTa hidden → Global rFFT → latent attention → gate → irFFT → masked max pooling`。

STFT-FLaG：将 global FFT 换成**逐帧局部 FFT**。全部有效帧的频率 token 共同生成**一个句级 gate**，然后分别门控局部频谱、逐帧逆变换并拼接。

两者共同配置：**8 latent queries、4 attention heads、residual gate（1+sigmoid）、masked max pooling**。gate 按通道区分实部和虚部，在频率位置之间共享。

常用局部模型：**E3 = win8/hop4**；**E12 = win16/hop16**。除实验另有说明，均为 rectangular window、`center=False`，不使用帧位置编码。

## 2. 数据与训练协议

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
