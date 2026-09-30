# STSB：E9–E14，Padding、局部窗口与配置复核

本页承接 [E1–E8](01-experiments.md)。E9/E10 检查**固定 checkpoint 在推理时**对外部补零的敏感性；E11/E12/E13 是训练新模型；E14 是后续查阅原论文后补做的**匹配配置与 10-seed** 实验。它们回答的问题和指标不同，见[统一实验协议](../methods.md)。

## 实验配置总览

| 编号 | 研究对象 | 是否重新训练 | 数据与 seeds | 改变的因素 | 核心观测 |
| --- | --- | :---: | --- | --- | --- |
| E9 | 已训练 FLaG / E3 | 否 | STSB；探针脚本默认 seed=0、最多 200 对短句；表中历史执行数量仍待日志复核 | 同样的有效句子 hidden，额外右 padding 到 16/24/32/64/128 | 句对 cosine 绝对变化 |
| E10 | 已训练 FLaG / E3 | 否 | STSB test，3 seeds（0–2） | normal / pair_exact / sentence_exact / fixed_128 四种推理 padding 约定 | 全 test Spearman |
| E11 | STFT 8/8 | **是** | STSB train→validation→test；3 seeds（0–2） | rect、win=8、hop=8、无 overlap | test Spearman / Pearson |
| E12 | STFT 16/16 | **是** | 同上，先 3 seeds，后扩展到 **10 seeds（0–9）** | rect、win=16、hop=16、无 overlap | test Spearman / Pearson |
| E13 | Global FLaG fixed FFT 128 | **是** | STSB，3 seeds（0–2） | **训练与推理**均固定 global FFT/iFFT 长度 128 | test Spearman / Pearson |
| E14 | Global FLaG 与 E12 | **是** | STSB，**10 seeds（0–9）** | 在同一 **论文 text 非算子配置**下比较：dropout=0.1，post-pool norm=True | **test** Spearman / Pearson 及配对差 |

除 E14 外，重训练候选遵循早期 STSB 统一非算子设置：**pooling dropout=0，post-pool LayerNorm=True**，8 latent、4 heads、masked max pooling、3 epochs、RoBERTa-base 端到端微调。E9/E10 只改变**已算出的 encoder hidden 进入 pooling 时的补零**，不重训 backbone；因此不等同于重新 tokenization、重新编码或完整系统对任意 batch 的位级不变性测试。

---

## E9：增加外部右 padding 会怎样？

**Q：** E8 没有提供稳定的性能原因；local STFT 的固定帧长度是否使其比 global FFT 对额外的 batch 右 padding 更稳定？

**M：** 使用已训练 **FLaG 与 E3** checkpoint。选有效 token 长度不超过 16 的 STSB 短句，将**相同的有效 hidden** 按长度 16、24、32、64、128 补零，仅改变传给 pooling 的右 padding 总长度与 mask，不改变内容 token、encoder 输出或模型参数。使用按每句自身有效长度处理时的预测作为参照，比较句对预测 cosine 的**绝对漂移**。global FFT 的变换长度随外部 padding 变化；E3 则对同样有效 token 按固定 8/4 帧构造局部频谱，批内额外的无效帧被 mask。

**R：** 在补到 128 的该短句诊断里（**代码的默认运行设置是 seed=0、最多 200 对短句**）：

| 同一输入相对自身有效长度 | Global FLaG | Local E3 |
| --- | ---: | ---: |
| 平均绝对 cosine 漂移 | 0.006003 | 数值精度范围内约 0 |
| 最大绝对 cosine 漂移 | 0.035915 | 数值精度范围内约 0 |

**C：** 当前 global FLaG 输出对额外右 padding 长度敏感；当前 E3 实现对这类**encoder hidden 不变时的额外外部零 padding**近似不变。

## E10：Padding 敏感性能解释整体 test 性能差异吗？

**Q：** E9 只挑短句并测预测漂移；把 padding 约定改到整个 STSB test，最终 Spearman 是否有系统性变化？是否足以解释 E3 和 global FLaG 的差异？

**M：** 读取 **3 seeds（0–2）** 的 FLaG 和 E3 已训练 checkpoint，在**完整 STSB test** 上每个句子只计算一次 RoBERTa hidden；然后保持有效 hidden 不变，用四种方式组织**送入 pooling 的长度**：

| 推理方式 | 实际操作 |
| --- | --- |
| `normal` | 与原始训练/评估一致，当前 batch 动态 padding |
| `pair_exact` | 一对句子的左右两侧分别使用相同的长度，取该**句对两句有效长度的较大者** |
| `sentence_exact` | 每句话独立使用自己的实际有效 token 长度 |
| `fixed_128` | 两句均在 pooling 输入端右 padding 到 128 |

每种方式重新计算整套 test 预测，分别计算每 seed 的 test Spearman，再在三个 seeds 上求均值。**这是固定模型的 inference-time intervention，不是四套分别训练的模型。**

**R：**

| 推理方式 | Global FLaG：3-seed mean test Spearman ↑ | E3：3-seed mean test Spearman ↑ |
| --- | ---: | ---: |
| normal | 0.841128 | 0.843000 |
| pair_exact | 0.841022 | 0.843000 |
| sentence_exact | 0.841288 | 0.843000 |
| fixed_128 | 0.841458 | 0.843000 |

**C：** global FLaG 的 test Spearman 会随 padding 约定小幅变化，E3 在这组固定 hidden 的探针中几乎不变；但切换 padding 约定**没有稳定抹平**两种模型的性能差。**结构上的 padding 不变性与下游性能收益是两个不同命题**。

## E11/E12：局部窗口大小 × overlap 2×2

**Q：** E2/E3 都采用半窗口 overlap。局部 STFT 的潜在效果是否必须依赖 frame 重叠？去掉 overlap 是否简化实现并保持结果？

**M：** 在 STSB 上按原训练协议**重新训练**四种 rect、`center=False`、无额外 frame positional encoding 的 local 配置。两种窗长 `win∈{8,16}` 与两种 hop（半窗或整窗）交叉，非算子因素全部统一为 `dropout=0,norm=True`，每种用 **seeds 0–2**。其中 E2/E3 是已经执行过的配置，补充不重叠的 E11/E12。重叠位置按重建实现进行 window-envelope 归一化。

**R：** **STSB official test，3-seed Spearman（mean ± sample std）**：

| 窗长 | 半窗 overlap | 不 overlap |
| ---: | ---: | ---: |
| 8 | **E3**：win=8, hop=4；**0.843000 ± 0.001047** | **E11**：win=8, hop=8；**0.842166 ± 0.001423** |
| 16 | **E2**：win=16, hop=8；**0.842204 ± 0.001824** | **E12**：win=16, hop=16；**0.843419 ± 0.000531** |

同窗长情况下，overlap 与不 overlap 的 Spearman 均值差：`win=8` 时 `+0.000834`，`win=16` 时 `−0.001215`。三个 seeds 的早期 E12 Pearson 为 `0.839528±0.002141`；E12 相对同 seed global FLaG 的 Spearman 差为 `+0.001259,+0.000591,+0.005023`。

**C：** overlap 不呈现一致的有利方向。因此后续选择**结构更简单**的 E12（rect、16/16、无重叠、非居中、无帧位置编码）用于机制研究和 Sprint 迁移。

### E12 扩展到 10 seeds 后怎样？

原来的 FLaG 和 E12 均保持 `dropout=0,norm=True`、相同训练预算，不改超参数，扩展到 **seeds 0–9**：

| 模型 | 10-seed **test Spearman** ↑ | 10-seed **test Pearson** ↑ |
| --- | ---: | ---: |
| Global FLaG | 0.841559 ± 0.003538 | 0.838504 ± 0.003444 |
| E12（16/16） | 0.842739 ± 0.003214 | 0.839393 ± 0.003467 |
| 配对 E12 − FLaG | **+0.001180 ± 0.002107** | **+0.000889 ± 0.003033** |

Spearman 正向 seeds **7/10**，配对 t 95% CI 约 `[−0.000327,+0.002687]`，双侧 p≈0.110。区间跨 0；早期 3/3 的正向信号**没有形成稳定的 10-seed 性能证明**。此外，STSB Mean pooling 10-seed Spearman `0.851201±0.002000`，高于当前这两组 FLaG 系列。

## E13：从训练开始固定 global FFT 长度

**Q：** E10 只在**推理**时尝试 fixed_128，但 FLaG 在**训练**中仍受到动态 padding 影响。如果从训练开始固定 global FFT 长度，是否能重现 E12 早期观察到的性能变化？

**M：** **重新训练** global FLaG，在训练、validation 和 test 阶段都使用 `fixed_fft_length=128`（global rFFT 和 irFFT 都按这个长度工作）；不引入局部 frame，保持 STSB 早期 `dropout=0,norm=True`、3 epochs、**seeds 0–2**。与这三个 seeds 的原始动态 global FLaG 配对比较。

**R：** 3-seed **test** Spearman `0.841856±0.001000`；Pearson `0.837795±0.001660`。配对 Spearman E13 − 原 global 为 `−0.000445,−0.001592,+0.004222`，均值 `+0.000728±0.003079`，**1/3 正向**。

**C：** 固定 global FFT 长度消除了这一路径的 batch-dependent 变换长度变化，但**没有稳定提高 STSB 性能**，也没有在这三个 seeds 上稳定重现 E12 的观察值。不能把“padding 敏感性存在”直接升级为“padding 是性能差距主因”。

## E14：在原论文 text 配置下重新比较 FLaG 和 E12

**Q：** 上面的 STSB 探索都用了 `dropout=0,norm=True`。后来查阅论文正文发现其 text 主配置为 **`dropout=0.1, post-pool LayerNorm=True`**；此前观察到的 E12 微小正向趋势，对 dropout 是否敏感？

**M：** 不改 STSB 任务协议、backbone、训练预算、E12 16/16 结构等，仅把 FLaG 和 E12 都统一改为**`dropout=0.1,norm=True`**，**重新训练 seeds 0–9（10 seeds）**。validation Spearman 选 checkpoint；下表为 official **test** 指标，FLaG/E12 按 seed 配对比较。

**R：**

| 模型 | 10-seed **test Spearman** ↑ | 10-seed **test Pearson** ↑ |
| --- | ---: | ---: |
| Global FLaG，0.1/True | **0.839065 ± 0.004154** | **0.834980 ± 0.003942** |
| E12，0.1/True | **0.838944 ± 0.001882** | **0.834015 ± 0.002119** |
| 配对 E12 − FLaG | **−0.000121 ± 0.004039** | **−0.000965 ± 0.003878** |

Spearman 正向 **5/10 seeds**，均值差近 0。论文原文报告的 STSB FLaG Spearman 约 `0.8368±0.004`；本重建的 `0.839065` 在数值上接近。

**C：** `0/True` 时的 E12 10-seed 小幅正向均值，在论文文本配置 `0.1/True` 下基本消失。**两套匹配配置都不支持“局部 STFT 稳定优于 global FLaG”的强结论**；也不能反向声称 global 在所有配置上一定更强。

---

**接下来的机制问题：** E9/E10 显示 global 对外部 padding 敏感，但 E13 没有建立性能原因。为了理解这一点，下一页推导**实虚门控产生的循环反射**，再用 P1 检查变化来自 gate 还是 reconstruction；P2 则在另一维度上拆分 global/local。
