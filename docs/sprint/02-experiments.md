# Sprint：S1–S5

数据集、冻结 RoBERTa、分类 head 和评价指标见[Sprint 设置](01-protocol.md)。**S1/S2 报 official test**，**S3–S5 报 adaptation validation**，二者不能直接比较。

## 实验对照表

| 实验 | 对比内容 | Seeds | 指标 |
| --- | --- | --- | --- |
| S1 | Mean、早期 FLaG、E12 | 0–2（3） | test AP/Accuracy/F1 |
| S2 | 将 S1 扩至 10 seeds | 0–9（10） | test AP 等 |
| S3 | 固定 checkpoint，做 GG/LG/GL/LL | 0–2（3） | validation cosine 漂移、AP |
| S4 | FLaG 与 E12 均设 dropout=0、Norm=True | 0–9（10） | validation AP |
| S5 | **只对 global FLaG** 做 dropout × Norm 2×2 | 0–9（10） | validation AP |

## S1：初步性能对比

**Q：** E12 结构用于 Sprint 后，与 Mean、早期 FLaG 相比效果如何？

**M：** 冻结 RoBERTa，分别重新训练 pooling/head，**3 seeds（0–2）**。当时的实际模型配置：

| 模型 | Pooling | Dropout | Post-pool Norm |
| --- | --- | ---: | :---: |
| Mean | 有效 token 均值 | 不适用 | 不适用 |
| FLaG | Global FFT | **0.1** | **无** |
| E12 | STFT win16/hop16 | **0** | **有** |

**R：** **3-seed official-test（mean ± std）**：

| 模型 | AP ↑ | Accuracy ↑ | F1 ↑ |
| --- | ---: | ---: | ---: |
| Mean | 0.428907 ± 0.000003 | 0.992040 ± 0 | 0.460390 ± 0 |
| FLaG | 0.721611 ± 0.009747 | 0.993980 ± 0.000298 | 0.667938 ± 0.007364 |
| E12 | 0.765655 ± 0.023788 | 0.994495 ± 0.000423 | 0.690782 ± 0.017933 |

**C：** E12 早期配置的 AP 更高，但 FLaG/E12 的 dropout 和 norm **不匹配**，还不能判断是否由 STFT 算子造成。

## S2：扩大到 10 Seeds

**Q：** S1 的结果是否在更多随机种子下保持？

**M：** 延续 S1 两种实际配置，扩至 **10 seeds（0–9）**，其他协议不变。

**R：** **10-seed official-test（mean ± std）**：

| 指标 ↑ | FLaG（0.1/无 Norm） | E12（0/有 Norm） |
| --- | ---: | ---: |
| AP | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 |
| Accuracy | 0.993857 ± 0.000275 | 0.994485 ± 0.000268 |
| F1 | 0.650671 ± 0.016939 | 0.685348 ± 0.020150 |
| Precision | 0.745202 ± 0.033959 | 0.814274 ± 0.032556 |
| Recall | 0.579800 ± 0.037446 | 0.594100 ± 0.041906 |

配对 E12−FLaG 的 AP 差 **+0.048222 ± 0.017641**，**10/10 seeds 正向**。

**C：** 两套**整体配置**之间的 AP 差在 10 seeds 中保持，但仍不能单独归因于局部 STFT。

## S3：Sprint 上复现 P2

**Q：** 在 Sprint 上，global/local 的输出差异主要来自 gate，还是重建路线？

**M：** 固定早期 FLaG 和 E12 各 **3 个 checkpoint（0–2）**，在 **adaptation validation** 上交叉使用两种 gate 来源和重建路线：

| 配置 | Gate 来源 | Gate 作用的频谱及重建 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 中 global 频谱只用于生成 gate，实际被门控、逆变换的仍是 **local 频谱**。同一训练来源的一行实验共享同一个 checkpoint，不重新训练。

**R：** **3-seed validation 平均绝对 cosine 漂移**：

| 权重来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG-trained | 约 0 | **0.032432 ± 0.003114** |
| E12-trained | 约 0 | **0.037477 ± 0.003556** |

**3-seed validation AP 均值**：

| 权重来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| E12-trained | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

**C：** 与 STSB 的 P2 一致：只换 gate 几乎不改变输出，差异主要来自**重建路线**；直接换成 local 重建并未提高当前模型的 validation AP。

## S4：对齐配置后比较 Global 与 Local

**Q：** 去掉 dropout/norm 的配置差异后，E12 还有稳定的 AP 优势吗？

**M：** FLaG、E12 都使用 **dropout=0、post-pool Norm=True**，其余训练条件一致，比较 **10 seeds（0–9）**。两者主要区别是 global FFT 与 E12 的 STFT 16/16。

**R：** **10-seed adaptation-validation AP**：

| 模型 | mean ± std |
| --- | ---: |
| FLaG | **0.795830 ± 0.017156** |
| E12 | **0.789136 ± 0.014094** |
| 配对 E12−FLaG | **−0.006693 ± 0.017279** |

**4/10 seeds** 的差值为正。

**C：** 在匹配配置下没有观察到稳定的 E12 优势，S1/S2 的差异不能直接归因于 STFT 算子。

## S5：Global FLaG 的 Dropout × Norm

**Q：** dropout 与 post-pool Norm 是否影响 Sprint 的 global FLaG？

**M：** **固定 global FFT**，交叉测试 dropout `{0,0.1}` 和 Norm `{无,有}` 四种设置；每组 **10 seeds（0–9）**，其余训练条件一致。

**R：** **10-seed adaptation-validation AP**：

| Dropout | Norm | AP（mean ± std） |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 ± 0.025820 |
| 0.1 | **有** | **0.819156 ± 0.015250** |
| 0 | 无 | 0.798974 ± 0.007324 |
| 0 | 有 | 0.795830 ± 0.017156 |

相同 dropout 下**增加 Norm**：dropout=0.1 时 AP 差为 **+0.068220**；dropout=0 时为 **−0.003145**。

**C：** dropout 与 Norm 存在交互，本实验四种 **global FLaG** 配置中 `0.1 + 有 Norm` 的 validation AP 最高。**本实验没有测试 `0.1 + 有 Norm` 的 Sprint E12**，不能据此比较这项配置下的 global 与 local。
