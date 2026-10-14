# 实验协议与评价指标

## 模型

原始 FLaG：`RoBERTa hidden → Global rFFT → latent attention → gate → irFFT → masked max pooling`。

STFT-FLaG：逐帧使用局部 FFT；各有效帧的频率 token 共同生成一个句级 Gate，再分别门控局部频谱、逐帧逆变换并拼接。

两者常见设置为 8 个 latent queries、4 个 attention heads、residual Gate（1 + sigmoid）和 masked max pooling。Gate 按通道区分实部和虚部，且在频率位置间共享。E3 使用 `win8/hop4`，E12 使用 `win16/hop16`；除专门对照外，使用矩形窗与 `center=False`。

## STSB 与 Sprint 协议

| 设置 | STSBenchmark（STSB） | SprintDuplicateQuestions（Sprint） |
| --- | --- | --- |
| 任务 | 句对语义相似度，分数 0–5 | 句对是否重复，二分类 |
| 数据划分 | 官方 train / validation / test | 官方 validation 分层拆分为 90,900 对 train、10,100 对 validation；official test 为 101,000 对 |
| Backbone | 早期 STSB STFT 实验微调 RoBERTa-base | 冻结 RoBERTa-base |
| Loss | cosine 与 label/5 的 MSE | BCEWithLogitsLoss，logit 为 `softplus(w) × cosine + b` |
| 训练 | 3 epochs，batch=4，梯度累积 4 | 10 epochs，batch=32 |
| 学习率 | Backbone 1e-5，pooling 1e-3 | pooling/head 1e-3 |
| Max length | 128 | 128 |
| Checkpoint | validation Spearman | validation 经阈值校准后的 Accuracy |
| 主要指标 | test Spearman / Pearson | AP、Accuracy、F1 等 |

训练采用 AdamW；Sprint 训练/验证划分固定随机种子 42。后续 frozen-STSB、Mean 融合与 Gate/Projection 等实验采用各自的协议和 seed 数，不直接等同于上述早期配置。

## 评价指标

**Spearman** 衡量排序相关，**Pearson** 衡量线性相关。Sprint 的 **AP** 是不依赖分类阈值的 Average Precision；Accuracy、Precision、Recall 与 F1 需指定阈值。Sprint 在验证集分别选择 Accuracy 阈值与 F1 阈值；test Accuracy 使用前者，F1/Precision/Recall 使用后者。

表中 mean ± std 是多个随机种子的均值与样本标准差；配对差值先按相同 seed 相减再汇总。

## 必须区分的训练配置

| 实验阶段 | Global FLaG：dropout / post-pool Norm | E12：dropout / post-pool Norm |
| --- | --- | --- |
| STSB 早期与 P1/P2 | 0 / 有 | 0 / 有 |
| STSB E14 | 0.1 / 有 | 0.1 / 有 |
| Sprint S1/S2 | **0.1 / 无** | **0 / 有** |
| Sprint S4 | 0 / 有 | 0 / 有 |
| Sprint S5 | 仅 Global FLaG：dropout {0, 0.1} × Norm {无, 有} | 未执行 |

Sprint S1/S2 的 Global 与 Local **未匹配** dropout 和 LayerNorm；它们的整体性能差不能单独归因于频谱算子。Sprint S4/S5 报告 adaptation validation AP，与 S1/S2 的 official test AP 分开比较。
