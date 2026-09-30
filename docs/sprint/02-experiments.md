# Sprint：S1–S5 实验

统一任务配置、数据划分和指标见 [Sprint 协议](01-protocol.md)。**S1/S2 报 official-test 指标；S3–S5 报 adaptation-validation 指标**，二者不直接混合比较。

## 实验配置总表

| 实验 | 对比内容 | 模型 seeds | 指标 / 数据 |
| --- | --- | --- | --- |
| S1 | Mean、早期 global FLaG 与 E12 | 0–2（3 个） | official-test AP、Accuracy、F1 |
| S2 | 将 S1 扩展到 10 seeds | 0–9（10 个） | official-test AP 等 |
| S3 | 固定 checkpoint，复现 P2 的 GG/LG/GL/LL | 0–2（3 个） | validation cosine 漂移、AP |
| S4 | **统一 dropout=0、norm=True** 后比较 global FLaG 与 E12 | 0–9（10 个） | validation AP |
| S5 | **仅 global FLaG**：dropout × norm 2×2 | 0–9（10 个） | validation AP |

## S1：Sprint 上的初步性能

**Q：** 将 E12 结构用于 Sprint，与 Mean 和早期 global FLaG 相比表现如何？

**M：** 冻结 RoBERTa-base，分别重新训练 pooling 和分类 head；训练 **3 seeds（0–2）**。当时的实际配置为：

| 模型 | Pooling | Dropout | Post-pool norm |
| --- | --- | ---: | :---: |
| Mean | 有效 token 平均 | 不适用 | 不适用 |
| FLaG | Global FFT | 0.1 | 无 |
| E12 | STFT 16/16 | 0 | 有 |

**R：** 3-seed **official test**，均值 ± 标准差：

| 模型 | AP ↑ | Accuracy ↑ | F1 ↑ |
| --- | ---: | ---: | ---: |
| Mean | 0.428907 ± 0.000003 | 0.992040 ± 0 | 0.460390 ± 0 |
| FLaG | 0.721611 ± 0.009747 | 0.993980 ± 0.000298 | 0.667938 ± 0.007364 |
| E12 | 0.765655 ± 0.023788 | 0.994495 ± 0.000423 | 0.690782 ± 0.017933 |

**C：** E12 早期整体配置的 AP 高于当时的 FLaG，因此扩展 seeds。但两者**dropout、norm 不一致**，此处不能把差异全部归因于 STFT。

## S2：扩展到 10 seeds

**Q：** S1 观察到的结果在更多随机种子下是否保持？

**M：** 延续 S1 的实际配置，扩大为 **seeds 0–9**，仍使用相同数据划分和训练协议。**两种模型的非算子配置仍未对齐。**

**R：** 10-seed **official test**，均值 ± 标准差：

| 指标 ↑ | FLaG（0.1 / 无 norm） | E12（0 / 有 norm） |
| --- | ---: | ---: |
| AP | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 |
| Accuracy | 0.993857 ± 0.000275 | 0.994485 ± 0.000268 |
| F1 | 0.650671 ± 0.016939 | 0.685348 ± 0.020150 |
| Precision | 0.745202 ± 0.033959 | 0.814274 ± 0.032556 |
| Recall | 0.579800 ± 0.037446 | 0.594100 ± 0.041906 |

配对 E12 − FLaG：AP 为 **+0.048222 ± 0.017641（10/10 seeds 正向）**，F1 为 **+0.034676 ± 0.024246（8/10 正向）**。Accuracy 使用 validation 选定的 Accuracy 阈值；F1、Precision 和 Recall 使用另一个 validation F1 阈值。

**C：** 早期 E12 **整体配置**的 AP 差异较稳定，但算子和 dropout/norm 同时变化，尚不能确定是哪一部分带来的差异。S4 将补做匹配对照。

## S3：在 Sprint 复现 P2

**Q：** 与 STSB P2 相同，Sprint 的 global/local 预测差异主要来自 gate 还是重建？

**M：** 分别读取早期 FLaG 与 E12 的 **3 个 checkpoint（seeds 0–2）**，固定权重，在 **adaptation validation** 上进行 GG/LG/GL/LL 干预：

| 配置 | Gate 的频谱来源 | 实际应用 gate 并重建 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 是用 global 频谱生成 gate，随后将其作用于 **local 频谱**，再做局部逆变换与帧拼接；并非把 global 频谱直接交给 local iFFT。每个训练来源内部共享同一 checkpoint，**不重新训练**。

**R1：** 3-seed validation **平均绝对 cosine 漂移**：

| 训练来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG | 约 0 | **0.032432 ± 0.003114** |
| E12 | 约 \(8.30\times10^{-13}\) | **0.037477 ± 0.003556** |

**R2：** 3-seed validation **AP 均值**：

| 训练来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| E12 | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

**C：** 与 STSB 一致，只换 gate 几乎没有影响，global/local 输出差异主要来自**重建路线**；但固定模型、仅改变重建路线并未提高 validation AP。

## S4：统一配置后的 global/local 对比

**Q：** 控制 dropout 和 norm 后，E12 是否仍比 global FLaG 有稳定的 AP 优势？

**M：** global FLaG 和 E12 **均设置 dropout=0、post-pool norm=True**，训练 **10 seeds（0–9）**；其余 Sprint 协议一致。两者的主要区别是 global FFT 与 E12 的 STFT 16/16。

**R：** **10-seed adaptation-validation AP**：

| 模型 | AP（均值 ± 标准差） |
| --- | ---: |
| Global FLaG | **0.795830 ± 0.017156** |
| E12 | **0.789136 ± 0.014094** |
| 配对 E12 − Global | **−0.006693 ± 0.017279** |

只有 **4/10 seeds** 为正。

**C：** 配置对齐后，没有观察到稳定的 E12 优势。因此 S1/S2 的较大 AP 差异不能直接归因于 STFT 算子。

## S5：Global FLaG 的 dropout × norm 控制实验

**Q：** Sprint 的 global FLaG 对 dropout 和 post-pool normalization 有多敏感？两者是否存在交互影响？

**M：** **固定 global FFT**，交叉设置 `dropout∈{0,0.1}` 与 `post-pool norm∈{False,True}`，得到四种配置；每种训练 **10 seeds（0–9）**，数据及其他训练条件相同。

**R：** **10-seed adaptation-validation AP**：

| Dropout | Post-pool norm | AP（均值 ± 标准差） |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 ± 0.025820 |
| 0.1 | 有 | **0.819156 ± 0.015250** |
| 0 | 无 | 0.798974 ± 0.007324 |
| 0 | 有 | 0.795830 ± 0.017156 |

具体配对变化：

| 固定条件 | 修改 | ΔAP（均值 ± 标准差） |
| --- | --- | ---: |
| 无 norm | dropout 0.1 → 0 | +0.048038 ± 0.024758 |
| 有 norm | dropout 0.1 → 0 | −0.023326 ± 0.017170 |
| dropout=0.1 | 增加 norm | +0.068220 ± 0.029151 |
| dropout=0 | 增加 norm | −0.003145 ± 0.014885 |

两因素交互项约为 **−0.071364 ± 0.021526**（10/10 seeds 为负）。

**C：** dropout 与 norm 的影响取决于另一项的设置。本次 **global FLaG validation** 对照中，`dropout=0.1 + norm=True` 的平均 AP 最高。

这里的 `0.819156` 是 **Sprint global validation AP**，而 [STSB E14](../stsb/02-padding.md) 的 `0.838944` 是 **E12 test Spearman**，不能比较。目前也**没有** Sprint 下 `dropout=0.1,norm=True` 的 E12 10-seed 匹配结果。
