# Sprint：换一个任务重新比较 Global 与 STFT

> **先看实验概览**：逐项说明实验目的、方法、结果与结论；之后保留详细实验记录、原始指标、公式和数据分析。这里的概览使用你确认的 38 项汇报稿，不与其他实验协议的结果混合比较。

## 本页实验概览（5 项）

**对应文件：** `docs/sprint/02-experiments.md`

把 STFT 应用到判断两句话是否重复的 Sprint 任务，先看性能，再统一训练配置，避免把 dropout 或 LayerNorm 的作用误认成局部 STFT 的优势。

### 实验十八（S1）：把 STFT-FLaG 用到 Sprint，效果怎样？

- **实验目的：** 检查 STFT 在 STSB 以外的任务中是否仍然有效。
- **实验思路：** 冻结 RoBERTa，在 Sprint 上分别训练直接 Mean、Global FLaG 和 E12，比较三个种子的测试集性能。
- **实验结果：** Mean、FLaG、E12 的 Test AP 分别为 0.428907、0.721611、0.765655。
- **实验结论：** 这组 Sprint 实验中，E12 所使用的整套配置得到最高 Test AP。由于 Global 与 E12 的 dropout/LayerNorm 同时不同，下一步需要统一配置比较。

### 实验十九（S2）：STFT 在 Sprint 上的优势是否稳定？

- **实验目的：** 将 S1 的 Sprint 对照扩展至十个种子，观察两套配置的差距是否仍存在。
- **实验思路：** 保持 S1 的两套配置不变，将 FLaG 和 E12 的实验扩展至十个种子。
- **实验结果：** FLaG 的平均 Test AP 为 0.713121，E12 为 0.761344。E12 在十个种子中全部领先，平均高 0.048222。
- **实验结论：** E12 的原有配置在十个种子中均高于 FLaG 的原有配置，但这项结果比较的是两套完整配置，而不是只改变 FFT 与 STFT。

### 实验二十（S3）：在 Sprint 上，Global 和 Local 的预测差异来自哪里？

- **实验目的：** 检查 STSB 上 P2 的机制发现，在 Sprint 上是否仍然成立。
- **实验思路：** 固定训练好的模型，分别交换 Gate 来源和频谱重建方式，观察预测变化。
- **实验结果：** 只更换 Gate，预测几乎不变；更换频谱重建路径后，平均 cosine 变化约为 0.032 至 0.037。
- **实验结论：** Sprint 与 STSB 的 P2 结果一致：更换 Gate 来源几乎无影响，切换完整的频谱重建路线才会明显改变预测。

### 实验二十一（S4）：统一训练配置后，STFT 还比 Global FFT 好吗？

- **实验目的：** 排除 S1/S2 中 dropout 和 LayerNorm 不一致带来的影响。
- **实验思路：** 将 FLaG 和 E12 都设置为 dropout=0、启用后池化 LayerNorm，重新进行十个种子比较。
- **实验结果：** FLaG 的平均 Validation AP 为 0.795830，E12 为 0.789136，E12 只在四个种子中领先。
- **实验结论：** 将 dropout 与 LayerNorm 统一后，E12 的平均 Validation AP 反而低于 FLaG；S1/S2 中观察到的优势不能由 STFT 操作单独解释。

### 实验二十二（S5）：Dropout 和 LayerNorm 到底对 FLaG 有多大影响？

- **实验目的：** 进一步解释为什么同一个 FLaG，在不同训练配置下性能差异明显。
- **实验思路：** 固定 Global FLaG，只改变 dropout 是否为 0.1，以及是否启用后池化 LayerNorm，测试四种组合。
- **实验结果：** 在十种子 Validation AP 上，dropout=0.1 且启用 LayerNorm 的组合最高，为 0.819156；dropout=0.1 且关闭 LayerNorm 时仅为 0.750936。
- **实验结论：** Global FLaG 在 dropout=0.1、启用 LayerNorm 时取得本次四种配置中最高的 Validation AP；两项设置的作用需要结合考虑。

**本篇小结：** E12 的早期配置确实有更高 AP；但统一 dropout 和 LayerNorm 后，E12 的优势消失，说明配置差异对性能影响很大。

---

## 详细实验记录与数据

数据集、冻结 RoBERTa、分类 head 和评价指标见[Sprint 设置](01-protocol.md)。**S1/S2 报 official test**，**S3–S5 报 adaptation validation**，二者不能直接比较。

### 实验对照表

| 实验 | 对比内容 | Seeds | 指标 |
| --- | --- | --- | --- |
| S1 | Mean、早期 FLaG、E12 | 0–2（3） | test AP/Accuracy/F1 |
| S2 | 将 S1 扩至 10 seeds | 0–9（10） | test AP 等 |
| S3 | 固定 checkpoint，做 GG/LG/GL/LL | 0–2（3） | validation cosine 漂移、AP |
| S4 | FLaG 与 E12 均设 dropout=0、Norm=True | 0–9（10） | validation AP |
| S5 | **只对 global FLaG** 做 dropout × Norm 2×2 | 0–9（10） | validation AP |

### S1：初步性能对比

**具体问题：** E12 结构用于 Sprint 后，与 Mean、早期 FLaG 相比效果如何？

**具体做法与参数：** 冻结 RoBERTa，分别重新训练 pooling/head，**3 seeds（0–2）**。当时的实际模型配置：

| 模型 | Pooling | Dropout | Post-pool Norm |
| --- | --- | ---: | :---: |
| Mean | 有效 token 均值 | 不适用 | 不适用 |
| FLaG | Global FFT | **0.1** | **无** |
| E12 | STFT win16/hop16 | **0** | **有** |

**详细结果：** **3-seed official-test（mean ± std）**：

| 模型 | AP ↑ | Accuracy ↑ | F1 ↑ |
| --- | ---: | ---: | ---: |
| Mean | 0.428907 ± 0.000003 | 0.992040 ± 0 | 0.460390 ± 0 |
| FLaG | 0.721611 ± 0.009747 | 0.993980 ± 0.000298 | 0.667938 ± 0.007364 |
| E12 | 0.765655 ± 0.023788 | 0.994495 ± 0.000423 | 0.690782 ± 0.017933 |

**结果说明：** E12 早期配置的 AP 更高，但 FLaG/E12 的 dropout 和 norm **不匹配**，还不能判断是否由 STFT 算子造成。

### S2：扩大到 10 Seeds

**具体问题：** S1 的结果是否在更多随机种子下保持？

**具体做法与参数：** 延续 S1 两种实际配置，扩至 **10 seeds（0–9）**，其他协议不变。

**详细结果：** **10-seed official-test（mean ± std）**：

| 指标 ↑ | FLaG（0.1/无 Norm） | E12（0/有 Norm） |
| --- | ---: | ---: |
| AP | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 |
| Accuracy | 0.993857 ± 0.000275 | 0.994485 ± 0.000268 |
| F1 | 0.650671 ± 0.016939 | 0.685348 ± 0.020150 |
| Precision | 0.745202 ± 0.033959 | 0.814274 ± 0.032556 |
| Recall | 0.579800 ± 0.037446 | 0.594100 ± 0.041906 |

配对 E12−FLaG 的 AP 差 **+0.048222 ± 0.017641**，**10/10 seeds 正向**。

**结果说明：** 两套**整体配置**之间的 AP 差在 10 seeds 中保持，但仍不能单独归因于局部 STFT。

### S3：Sprint 上复现 P2

**具体问题：** 在 Sprint 上，global/local 的输出差异主要来自 gate，还是重建路线？

**具体做法与参数：** 固定早期 FLaG 和 E12 各 **3 个 checkpoint（0–2）**，在 **adaptation validation** 上交叉使用两种 gate 来源和重建路线：

| 配置 | Gate 来源 | Gate 作用的频谱及重建 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 中 global 频谱只用于生成 gate，实际被门控、逆变换的仍是 **local 频谱**。同一训练来源的一行实验共享同一个 checkpoint，不重新训练。

**详细结果：** **3-seed validation 平均绝对 cosine 漂移**：

| 权重来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG-trained | 约 0 | **0.032432 ± 0.003114** |
| E12-trained | 约 0 | **0.037477 ± 0.003556** |

**3-seed validation AP 均值**：

| 权重来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| E12-trained | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

**结果说明：** 与 STSB 的 P2 一致：只换 gate 几乎不改变输出，差异主要来自**重建路线**；直接换成 local 重建并未提高当前模型的 validation AP。

### S4：对齐配置后比较 Global 与 Local

**具体问题：** 去掉 dropout/norm 的配置差异后，E12 还有稳定的 AP 优势吗？

**具体做法与参数：** FLaG、E12 都使用 **dropout=0、post-pool Norm=True**，其余训练条件一致，比较 **10 seeds（0–9）**。两者主要区别是 global FFT 与 E12 的 STFT 16/16。

**详细结果：** **10-seed adaptation-validation AP**：

| 模型 | mean ± std |
| --- | ---: |
| FLaG | **0.795830 ± 0.017156** |
| E12 | **0.789136 ± 0.014094** |
| 配对 E12−FLaG | **−0.006693 ± 0.017279** |

**4/10 seeds** 的差值为正。

**结果说明：** 在匹配配置下没有观察到稳定的 E12 优势，S1/S2 的差异不能直接归因于 STFT 算子。

### S5：Global FLaG 的 Dropout × Norm

**具体问题：** dropout 与 post-pool Norm 是否影响 Sprint 的 global FLaG？

**具体做法与参数：** **固定 global FFT**，交叉测试 dropout `{0,0.1}` 和 Norm `{无,有}` 四种设置；每组 **10 seeds（0–9）**，其余训练条件一致。

**详细结果：** **10-seed adaptation-validation AP**：

| Dropout | Norm | AP（mean ± std） |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 ± 0.025820 |
| 0.1 | **有** | **0.819156 ± 0.015250** |
| 0 | 无 | 0.798974 ± 0.007324 |
| 0 | 有 | 0.795830 ± 0.017156 |

相同 dropout 下**增加 Norm**：dropout=0.1 时 AP 差为 **+0.068220**；dropout=0 时为 **−0.003145**。

**结果说明：** dropout 与 Norm 存在交互，本实验四种 **global FLaG** 配置中 `0.1 + 有 Norm` 的 validation AP 最高。**本实验没有测试 `0.1 + 有 Norm` 的 Sprint E12**，不能据此比较这项配置下的 global 与 local。
