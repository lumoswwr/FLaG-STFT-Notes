# Mean + Projection 的简化结构对照

依次比较从头训练的简化模型、输出层初始化与严格共享初始 Projection 的配对结果。

## 实验总结

<a id="summary-36"></a>

  ### 实验三十六：如果去掉 FFT 和 Attention，简单模型能否达到相近性能？

  - **实验目的：** 判断完整的频域门控是否真的必要，还是只要在 Mean 向量之后加入一个可训练的线性层就够了。
  - **实验思路：** 从零训练三个简化模型：固定实部/虚部通道权重、固定逐通道权重，以及直接 Mean + 可训练 Projection，并与完整 B2 比较。
  - **实验结果：** Seed 0 的 Sprint Test AP 中，B2 为 0.834217，直接 Mean + Projection 为 0.833765，仅差 0.000452。
  - **实验结论：** 在本次 Sprint 的 seed-0 对照中，Mean + 可训练 Projection 几乎恢复了 B2 的 AP，说明复杂的 FFT/Attention/Gate 不是达到这一结果所必需的训练结构。

[查看本实验详细论述](#detail-36)

<a id="summary-37"></a>

  ### 实验三十七：Projection 的初始化方式会影响结果吗？

  - **实验目的：** 进一步理解为什么简单的 Mean + Projection 表现不差，并排除不同初始化方式造成的比较偏差。
  - **实验思路：** 对比随机初始化和单位矩阵初始化的 Mean + Projection，并与 FLaG、B2 一起进行三个seed的训练。统一使用验证集 AP 选择模型。
  - **实验结果：** 随机初始化 Mean + Projection 的平均 Sprint Test AP 为 0.840923，单位矩阵初始化为 0.818806。前者在三个seed中都更高。
  - **实验结论：** 随机初始化的 Mean + Projection 比单位矩阵初始化高 0.022117 AP，表明输出线性层的初始化会显著影响本次训练结果。

[查看本实验详细论述](#detail-37)

<a id="summary-38"></a>

  ### 实验三十八：在相同初始 Projection 下，完整 FLaG 还有额外优势吗？

  - **实验目的：** 令 FLaG 与 Mean + Projection 使用完全相同的初始线性层，直接比较加入完整频域门控前后的 Sprint 预测性能。
  - **实验思路：** 强制两个模型使用完全相同的初始 Projection 权重与偏置，保持相同的数据划分和训练协议，运行三个种子。
  - **实验结果：** Sprint Test AP 中，FLaG 为 0.830539 ± 0.016711，Mean + Projection 为 0.841386 ± 0.008833。三个种子中简单模型全部领先。不过 Validation AP 的平均值是 FLaG 略高。
  - **实验结论：** 在严格统一初始 Projection 后，Mean + 可训练线性层的 Sprint Test AP 为 0.841386，高于完整 FLaG 的 0.830539，三个seed结果方向一致:完整频域门控没有在这次对照中带来额外测试集收益。

[查看本实验详细论述](#detail-38)

---

## 实验详细论述

<a id="detail-36"></a>

### 实验三十六：如果去掉 FFT 和 Attention，简单模型能否达到相近性能？

**实验设置与评价：** Sprint · test · seed 0 · 简化结构从头训练。

Sprint seed-0 Test AP：B2 为 0.834217；Mean + 可训练 Projection 为 0.833765，相差 0.000452。

#### 为什么首先想到 Mean + 一个可训练线性层？

直接 Mean 与 FLaG 不仅差在频域门控，还差在 **Projection**：

$$
m=\operatorname{Mean}(H),\qquad z=Wm+b.
$$

其中 $W$ 是训练过程中会更新的矩阵，不是固定随机矩阵。它可以组合原有的 768 个特征通道，使句对 cosine 更适合当前任务。

这和简单地“算出 Mean 就直接取 cosine”不是同一个模型。

从数学上，如果 Gate 是一套不随句子变化的逐通道缩放系数，把它写成对角矩阵 $D$，后面再接完整可训练线性层 $W$，就有：

$$
z=W(Dm)+b=(WD)m+b=W'm+b.
$$

所以：**静态对角 Gate + 完整可训练线性层，不增加 Mean + 线性层的函数表达范围**。不过参数初始化、优化过程和学习到的结果仍可能不同。这也不适用于任意输入相关的动态 Gate。

#### 第一轮简化模型分别是什么？

| 代码代号 | 通俗解释 | 是否有可训练 Projection |
| --- | --- | --- |
| StaticReIm | 学习所有句子共享的一套**实部和虚部**通道缩放系数，使用已知的时域等价公式完成处理，不运行 FFT / latent attention | 有，单位矩阵初始化 |
| StaticDiag | 在直接 Mean 的 768 个通道上分别乘以一套可训练、但不随句子变化的系数 | 有，单位矩阵初始化 |
| MeanProjection | 没有 Gate，直接 Mean 后接可训练线性层 | 有，单位矩阵初始化 |
| FLaG-B2 | 保留频域门控，但 Gate 与 Projection 初始化时严格输出 Mean | 有，单位矩阵初始化 |

这里的 **Static（静态）只意味着同一套 Gate 用于不同句子**，不是 Gate 永远不学习。它的参数仍在训练中更新。

StaticReIm 依据的是网站 [P1](03-padding-reconstruction.md#detail-16) 已经证明的时域等价公式：

$$
y_t=\frac{g_R+g_I}{2}x_t+\frac{g_R-g_I}{2}x_{(-t)\bmod N}.
$$

它解释了为什么实部与虚部门控不同时会出现**循环反转/反射项**，以及为什么 padding 长度可能影响输出。**这条数学发现来自此前 P1 实验，不是本轮重新提出的创新点**。

#### 只训练一个随机种子的初步比较

Seed 0，official test：

| 模型 | STSB Spearman | Sprint AP |
| --- | ---: | ---: |
| FLaG-B2 | **0.845490** | **0.834217** |
| StaticReIm | 0.841483 | 0.820715 |
| StaticDiag | 0.840198 | 0.824215 |
| Mean + 可训练 Projection（单位矩阵初始化） | 0.841056 | 0.833765 |

最引人注意的是 Sprint：Mean + Projection 与复杂 B2 只差 $0.000452$ AP。

**这个结果使我们提出新问题**：过去 Sprint 直接 Mean 的 AP 约为 0.4289，是不是不能说明 DC 中缺乏任务信息，而主要说明“直接 Mean + cosine”缺少合适的可训练变换？

这一判断也与此前 [DC-only 机制实验](05-dc-bands.md#detail-26) 相呼应：即使只保留 DC，只要训练合适的后续处理，仍然可以得到很强的 Sprint 结果。

但是单个种子不足以给出稳定结论，于是我们把对照扩大到三个种子。


B2 与 Mean+Projection 的 AP 接近，仅限于 seed-0 训练比较，不代表全部 seed 或任务均等价。

[返回实验 36 的总结](#summary-36)

---

<a id="detail-37"></a>

### 实验三十七：Projection 的初始化方式会影响结果吗？

**实验设置与评价：** Sprint · test · 3 seeds · 随机/单位初始化对照。

Sprint 3-seed Test AP：随机初始化 Mean+Projection 为 0.840923 ± 0.008811，单位初始化为 0.818806 ± 0.016929，差 0.022117。

#### 先比较随机初始化和单位矩阵初始化

原始 FLaG 的 Projection 是随机初始化，B2 与上面的 MeanProjection 是单位矩阵初始化。为了公平比较，我们增加另一种简单模型：

**MeanProjRand = 直接 Mean + 随机初始化、随后正常训练的线性 Projection。**

“随机初始化”不等于“随机矩阵固定不训练”。矩阵训练时仍然通过梯度更新。

采用 Sprint seeds 0、1、2，并统一使用 **validation AP** 选择最佳 epoch，第一轮得到：

| 模型 | 官方 test AP，3 seeds（mean ± sample SD） |
| --- | ---: |
| 原始 FLaG | 0.829813 ± 0.016874 |
| B2 | 0.824462 ± 0.011258 |
| Mean + 单位矩阵初始化 Projection | 0.818806 ± 0.016929 |
| **Mean + 随机初始化 Projection** | **0.840923 ± 0.008811** |

两种 Mean + Projection 只差在**初始矩阵**，随机初始化的 test AP 平均高 $0.022117$，三个种子都更高。这说明初始化与训练结果有关，不能忽略。

而随机初始化的 Mean + Projection 平均比原始 FLaG 高 $0.011110$ AP，三个种子都高。

不过，相同 seed 在两个不同结构中，并不保证一开始的 Projection 矩阵逐元素相同：初始化 Gate、latent attention 等其他参数会消耗随机数。因此我们又增加最后一个对照。


种子相同不代表不同结构初始化时所有参数逐元素相同，严格匹配 Projection 的对照另见实验 38。

[返回实验 37 的总结](#summary-37)

---

<a id="detail-38"></a>

### 实验三十八：在相同初始 Projection 下，完整 FLaG 还有额外优势吗？

**实验设置与评价：** Sprint · test / validation AP · 3 seeds · 严格匹配初始 Projection。

严格匹配初始 Projection 后，Sprint 3-seed Test AP：FLaG 0.830539 ± 0.016711，Mean+Projection 0.841386 ± 0.008833；Validation AP：FLaG 0.853535 ± 0.019891，Mean+Projection 0.851664 ± 0.017371。

#### 最后一次严格控制：两个模型从相同 Projection 权重开始

这一次比较只保留：

- **FLaG**：完整频域 Gate + 可训练随机 Projection；
- **MeanProjRand**：直接 Mean + 可训练随机 Projection。

对于每一个 seed，程序强制把两者的初始 Projection 权重 $W$ 和偏置 $b$ 设为**完全相同**，并记录 SHA-256 指纹用于复查。RoBERTa 冻结状态、数据划分、训练 epoch 和 validation AP 选择规则均保持匹配。

**Sprint official test，三个种子：**

| 模型 | Test AP（mean ± sample SD） | Validation AP（mean ± sample SD） | 选中 epoch |
| --- | ---: | ---: | --- |
| FLaG | 0.830539 ± 0.016711 | **0.853535 ± 0.019891** | 10、9、8 |
| **Mean + 可训练随机 Projection** | **0.841386 ± 0.008833** | 0.851664 ± 0.017371 | 10、10、8 |

严格配对的差值（FLaG 的 test AP 减去 Mean + Projection 的 test AP）：

| Seed | 差值 |
| --- | ---: |
| 0 | -0.001096 |
| 1 | -0.028025 |
| 2 | -0.003420 |
| **平均 ± sample SD** | **-0.010847 ± 0.014922** |

我们现在能说：

1. 三个种子中，**Mean + Projection 的 test AP 均高于 FLaG**；
2. 这个现象在排除了**初始 Projection 权重不一致**的影响后仍然存在；
3. 但平均 **validation AP 反而是 FLaG 略高**（0.853535 vs. 0.851664），所以不能说简单模型在所有划分上都更强；

!!! note "什么是“严格对齐 Projection”？"
    我们只把**初始输出线性层**的矩阵和偏置设为一致，并不是让两个网络所有参数都一致：FLaG 额外包含 FFT 后的 latent attention 和 Gate，Mean + Projection 没有这些模块。实验目的正是让两者在这个关键的共同部件上公平起跑。


该对照中 Mean+Projection 的平均 test AP 较高，但 FLaG 的平均 validation AP 略高，应同时报告两者。

[返回实验 38 的总结](#summary-38)

