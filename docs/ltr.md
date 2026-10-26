# 第四阶段：STSB 上的经典 LTR 排序目标对照

## 本页实验概览（1 项）

<a id="exp-18-summary"></a>

### 实验十八：修正后的 Pairwise / Listwise 目标下，FLaG 的排序偏转是否有效？

- **实验目的：** 回到最初的问题：FLaG 相对 Mean 引入了额外的表示变化和参数自由度，为何没有转化为更好的 STSB 排序？检验这种现象在不同 LTR 训练目标下是否仍然存在。
- **实验思路：** 修正旧版 LTR 的评分尺度、Margin 和并列标签处理问题后，固定 RoBERTa-base、数据划分、训练预算与 checkpoint 选择规则。在 Mean、FLaG-Zero 和原始 FLaG 上测试 7 种纯排序目标，分别运行 5 个 seed，并以同协议 MSE 为基线。
- **实验结果：** 共 **3 模型 × 8 训练目标 × 5 seeds = 120 个条件**（包含 MSE 和已完成的 Pure Soft Spearman）。在 **7/7 种纯 LTR** 下，验证集均为 **Mean > FLaG-Zero > FLaG**；FLaG 与 FLaG-Zero 在同一目标下对 Mean 的 **5/5 个配对 seed 全部落后**。
- **实验结论：** 当前结果不支持“只要换成排序损失，FLaG 相对 Mean 的额外表示偏转就能改善排序”的解释。部分目标下 FLaG 的训练 Spearman 甚至更高，但验证 Spearman 更低。**本页只证明性能现象，不直接证明 Gate、Projection 或转角本身是因果原因。**

**[↓ 查看详细实验与数据](#exp-18-detail)**

---

## 详细实验记录与数据

<a id="exp-18-detail"></a>

### 1. 研究问题、模型与排序目标

最初要解释的不是“LTR 能否优于 MSE”，而是：

> **FLaG 相比 Mean 学到的额外表示方向（转角／偏转）为什么没有起作用？如果训练目标直接要求正确排序，FLaG 能否利用这些额外自由度超过 Mean？**

此前 LTR-v2 的训练设计存在分数尺度与排序目标实现上的不足，不能拿旧版 3-epoch / 未修正损失的结果与新实验直接比较。本页**整体采用修正后的端到端协议**，旧结果不再用于本页结论。

**模型设置**

- **Mean：** Masked Mean Pooling。
- **FLaG-Zero：** `FLaG_B2`，Mean 等价初始化（频域 Gate 的初始残差为零、时域 Projection 初始为恒等、time pooling 为 mean、不使用 post-pool norm）；训练前检查初始 pooling / cosine 与 Mean 的差异，容差为 `1e-5`。
- **原始 FLaG：** FFT latent attention + frequency gate + iFFT + max pooling + projection，使用原协议的 post-pool norm。

三个模型均从 RoBERTa-base 预训练权重出发，**Encoder 不冻结**；这里的 FLaG-Zero 是“从与 Mean 函数等价的池化初始化开始训练”，**并非从已训练的高性能 Mean checkpoint 做 warm-start**。因此本实验属于模型架构与优化轨迹对照，而不是冻结 Encoder 条件下的 Gate 因果实验。

**排序目标（全部为纯排序训练，不含 MSE 反向传播）**

1. **RankNet（Pairwise）：** 在非并列标签对上计算 logistic loss；固定评分温度，修正旧版未经缩放的 cosine 差值。
2. **Hinge Fixed（Pairwise）：** 评分 logits 的 margin 为 1，对应 raw cosine 的 margin 0.05，不再沿用旧版在 raw score 上取 margin 1 的设置。
3. **Hinge Label-aware（Pairwise）：** margin 随标签差距变化，范围为 logits 尺度上的 0.25–1。
4. **LambdaRank-style（Pairwise / Listwise 加权）：** 用随机 minibatch 内的 NDCG 变化对 pairwise logistic loss 加权；它是 **batch-local 近似**，不是标准 query-grouped LambdaRank。
5. **ListNet KL（Listwise）：** 对归一化标签分布和 cosine 分数的 softmax 分布计算 KL；并列标签自然得到相同的目标概率。
6. **Grouped ListMLE（Listwise）：** 先按标签值分组，再对有序 relevance 组计算 Plackett–Luce 式代理损失，避免为相同标签强行指定排列；并非精确枚举全部 tie-consistent 排列。
7. **Pure Soft Spearman（Listwise 排序相关性代理）：** 用可微 SoftRank 近似预测名次，以真实标签的平均名次处理 ties，最小化 `1 − rank correlation`。

另外保留 **MSE** 作为非 LTR 参照，**MSE + 0.05 Soft Spearman** 是另一组混合目标实验，不计入本页的 7 种纯 LTR。

### 2. 训练与评估设置

| 设置 | 修正后的统一配置 |
| --- | --- |
| 数据集 | STS Benchmark（STSB）官方 train / validation |
| 模型 | Mean、FLaG-Zero、原始 FLaG |
| 随机种子 | 0、1、2、3、4 |
| 新增训练 | 6 种经典 LTR × 3 模型 × 5 seeds = 90 次 |
| 合并对照 | 已有 MSE 与 Pure Soft Spearman，共 120 个条件 |
| Encoder | RoBERTa-base，端到端训练（不冻结） |
| 训练 / 评估 batch size | 32 / 64 |
| 最大序列长度 | 128 |
| 优化器 | Adam，weight decay = 0 |
| Encoder 学习率 | \(1\times10^{-5}\) |
| Pooling 学习率 | \(1\times10^{-3}\) |
| 最大 / 最小训练轮数 | 最多 20 epochs，至少 12 epochs |
| Early stopping | Patience = 5 |
| 纯 LTR 评分 | Raw cosine；固定 score temperature = 0.05 |
| ListNet 标签温度 | 0.20（标签先除以 5） |
| Score affine head | 在纯 LTR 下固定 scale / bias，不参与训练 |
| Checkpoint 选择 | 只按官方 Validation Spearman 选择（包含 epoch 0） |
| 本页主要指标 | **Validation Spearman**，5-seed mean ± sample std |
| Test | **本轮没有读取或评估 Test** |

STSB 是独立句对相似度评分任务，**没有自然的 query/document 分组**。这里的 Pairwise 和 Listwise 在随机 minibatch 内构造排序关系，不应与带 query ID 的搜索 LTR 协议混为一谈。

### 3. 修正后全部结果：Mean、FLaG-Zero 与 FLaG

以下为 **Validation Spearman（5 seeds mean ± sample std）**。每一行加粗同一训练目标下的最佳模型，每一列的最佳纯 LTR 在表后单独说明。

| 训练目标 | Mean | FLaG-Zero | 原始 FLaG |
| --- | ---: | ---: | ---: |
| MSE（基线） | **0.887813 ± 0.002137** | 0.880521 ± 0.002249 | 0.874077 ± 0.002288 |
| Pure Soft Spearman | **0.890544 ± 0.001242** | 0.882284 ± 0.001639 | 0.872682 ± 0.002478 |
| RankNet | **0.891177 ± 0.000950** | 0.881677 ± 0.001521 | 0.871636 ± 0.000625 |
| Hinge Fixed | **0.891492 ± 0.000930** | 0.881495 ± 0.001188 | 0.873738 ± 0.001429 |
| Hinge Label-aware | **0.889612 ± 0.001938** | 0.882708 ± 0.000766 | 0.877168 ± 0.003000 |
| LambdaRank-style | **0.878351 ± 0.001351** | 0.868304 ± 0.001971 | 0.860642 ± 0.002489 |
| ListNet KL | **0.879715 ± 0.002021** | 0.871968 ± 0.003235 | 0.867436 ± 0.001452 |
| Grouped ListMLE | **0.890560 ± 0.001272** | 0.874172 ± 0.002612 | 0.866297 ± 0.003806 |

**各模型在纯 LTR 下的最佳验证结果**

| 模型 | 最佳纯排序目标 | Validation Spearman |
| --- | --- | ---: |
| Mean | Hinge Fixed | **0.891492 ± 0.000930** |
| FLaG-Zero | Hinge Label-aware | **0.882708 ± 0.000766** |
| 原始 FLaG | Hinge Label-aware | **0.877168 ± 0.003000** |

这里最重要的现象并非哪个 Loss 的绝对分数最高，而是：**每一个修正后的纯排序目标都保持 Mean > FLaG-Zero > FLaG 的排序。**

### 4. 同一种 LTR 下的架构差距

为了排除“对不同模型选择不同 Loss 导致的比较偏差”，下面**固定同一个目标**，计算配对 seed 的模型差：

\[
\Delta_{\mathrm{Zero}}=\rho(\mathrm{FLaG\!-\!Zero})-\rho(\mathrm{Mean}),
\qquad
\Delta_{\mathrm{FLaG}}=\rho(\mathrm{FLaG})-\rho(\mathrm{Mean}).
\]

| 训练目标 | FLaG-Zero − Mean | FLaG − Mean |
| --- | ---: | ---: |
| MSE（参照） | −0.007293 | −0.013736 |
| Pure Soft Spearman | −0.008260 | −0.017863 |
| RankNet | −0.009500 | −0.019541 |
| Hinge Fixed | −0.009996 | −0.017754 |
| Hinge Label-aware | **−0.006905** | **−0.012445** |
| LambdaRank-style | −0.010048 | −0.017709 |
| ListNet KL | −0.007747 | −0.012279 |
| Grouped ListMLE | −0.016388 | −0.024263 |

上表为五种子**同 seed 配对差值的均值**，不是三个模型分别挑选最佳 Loss 后计算的差。

- 对 **7/7 种纯排序目标**，FLaG-Zero 与 FLaG 的均值都低于相同 Loss 的 Mean。
- 不仅是均值：两种 FLaG 变体在**每个目标的 5/5 个配对 seed** 上都没有超过 Mean。
- Hinge Label-aware 使两种 FLaG 的差距相对较小，但**没有改变架构排序**。

因此现象具有一定的跨损失一致性：**在当前训练协议下，只把 MSE 换成更直接的排序监督，不能使 FLaG 的额外参数自由度在 STSB 上超过 Mean。**

### 5. 训练集排序更好，却没有带来验证集优势

Hinge Label-aware 是最值得用于连接原始机制问题的案例：FLaG 在该目标下有更高的**完整训练集** Spearman，但验证集 Spearman 更低。

下表取各 seed **由 Validation Spearman 选出的最佳 checkpoint**，再评估完整训练集；数值均为 5 seeds 的均值。

| 模型 | Train Spearman | Validation Spearman | Train − Validation |
| --- | ---: | ---: | ---: |
| Mean | 0.978632 | **0.889612** | 0.089020 |
| FLaG-Zero | 0.980794 | 0.882708 | 0.098087 |
| 原始 FLaG | **0.982147** | 0.877168 | **0.104980** |

与 Mean 相比：

- 原始 FLaG 的 **Train Spearman 高约 0.003515**。
- 但原始 FLaG 的 **Validation Spearman 低约 0.012445**。
- 其训练—验证差距比 Mean 大约 **0.015960**。

这与最初的“**FLaG 的额外能力改善训练拟合，却没有改善排序泛化**”高度相似，现在该现象也出现在**纯 LTR 目标**下。

不过不同 Loss 的训练 Spearman 并不都呈现相同排序。例如 RankNet 下，Mean 的完整训练集 Spearman 仍高于 FLaG。因而**不能概括为 FLaG 在所有 LTR 下都更擅长拟合训练排序**，只能说 Hinge Label-aware 提供了一个明确、可复现的切入点。

### 6. 结果观察与机制含义

**（1）最初的问题仍然存在。**

旧版 LTR 的负面结果有损失尺度、Margin 与并列标签处理上的设计问题。修正后，多种排序目标确实能在 Mean 上正常优化，并且验证 Spearman 达到较高水平。但是**不管哪一种排序监督，同协议下 FLaG 仍落后于 Mean**。

因此不能再把 FLaG 的排序劣势简单归因于“原来使用 MSE，而不是 LTR”。

**（2）FLaG-Zero 从 Mean 等价起点出发，也没有保住 Mean 的泛化水平。**

FLaG-Zero 在初始化时能够近似复现 Mean，但随后 Gate / Projection 等可训练参数会使优化轨迹发生变化。最终排序较差，表明**新增自由度没有在这一协议下自然转化为有用的排序增益**。

这**并不意味着**“Mean 等价初始化的数学性质有问题”，也**不能单独证明** Gate 或 Projection 是罪魁祸首。当前 Encoder 同样在更新，架构参数、随机优化轨迹与表示变化耦合在一起。

**（3）“转角存在”与“转角有用”是两个不同问题。**

我们真正想知道的是：相对 Mean，FLaG 的额外表示变化是否沿着有利于 STSB 排序的方向。

本轮只测到了**排名相关性的训练／验证结果**，**没有直接测量句向量转角、投影残差与排序增量的对应关系**。因此当前能够支持的结论是：

> **在七种纯排序监督下，FLaG 相对 Mean 的额外表示和优化自由度没有转化为更好的验证集 Spearman；它们为何无效，仍需独立的几何和梯度诊断。**

### 7. 下一步：回到 Gate / Projection 偏转机制

本阶段经典 LTR 对照可以收束，不再以“LTR 比 MSE 高多少”为主要研究问题。后续优先回到**冻结 Encoder、从已训练高性能 Mean checkpoint 严格函数等价初始化的 FLaG-Zero** 机制实验，避免把 Encoder 的联合更新与 Gate 的偏转效应混在一起。

建议围绕以下可检验问题推进：

1. **转角是否真的变大？** 在固定样本和 checkpoint 上比较 Mean、FLaG-Zero、FLaG 的归一化表示方向，量化夹角与正交偏转的大小。
2. **偏转是否具有排序效益？** 将偏转带来的 cosine 分数变化，与标签顺序、pairwise inversion 的改善／恶化逐一对应，判断它是否只是改变表示而没有修正错排。
3. **究竟谁在驱动偏转？** 在严格 Mean 等价起点，分离 Gate、Projection 与 Encoder 的梯度和参数更新；使用 Gate-only、Projection-only 或冻结模块对照，区分有用更新与额外自由度造成的偏离。

上述均为**下一阶段待验证的机制假设**，不是本页实验已经证明的结果。

### 8. 代码、原始记录与结果来源

- 修正版经典 Pairwise / Listwise 训练：[`run_stsb_classic_ltr_suite.py`](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_stsb_classic_ltr_suite.py)
- 六种经典 Loss：[`stsb_classic_ltr_losses.py`](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/stsb_classic_ltr_losses.py)
- Pure Soft Spearman 训练：[`run_stsb_pure_softspearman.py`](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_stsb_pure_softspearman.py)
- 正式实验协议：[`STSB_CLASSIC_LTR_SUITE_PROTOCOL.md`](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STSB_CLASSIC_LTR_SUITE_PROTOCOL.md)
- 五种子逐轮日志及指标：[`research_audit_exports/stsb_classic_ltr_final`](https://github.com/lumoswwr/AMPCliff/tree/FLaG-STFT-mechanism/research_audit_exports/stsb_classic_ltr_final)

**数据口径：** 本页所有正式结果均使用 STSB **Validation Spearman**；统一报告 5-seed mean ± sample std。同 seed 的架构差距采用配对差值。**本轮经典 LTR 没有 Test 指标**，不能将历史 LTR-v2 的 Test 结果填入修正后实验表格。

**[↑ 返回实验概览](#exp-18-summary)**
