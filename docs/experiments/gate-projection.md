### 实验 PROJECTION：从头训练 Mean + Projection 与严格同初始化对照

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 5. 第四步：不使用 FFT 和 latent attention，重新训练简单模型

#### 5.1 为什么首先想到 Mean + 一个可训练线性层？

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

#### 5.2 第一轮简化模型分别是什么？

| 代码代号 | 通俗解释 | 是否有可训练 Projection |
| --- | --- | --- |
| StaticReIm | 学习所有句子共享的一套**实部和虚部**通道缩放系数，使用已知的时域等价公式完成处理，不运行 FFT / latent attention | 有，单位矩阵初始化 |
| StaticDiag | 在直接 Mean 的 768 个通道上分别乘以一套可训练、但不随句子变化的系数 | 有，单位矩阵初始化 |
| MeanProjection | 没有 Gate，直接 Mean 后接可训练线性层 | 有，单位矩阵初始化 |
| FLaG-B2 | 保留频域门控，但 Gate 与 Projection 初始化时严格输出 Mean | 有，单位矩阵初始化 |

这里的 **Static（静态）只意味着同一套 Gate 用于不同句子**，不是 Gate 永远不学习。它的参数仍在训练中更新。

StaticReIm 依据的是网站 [P1](../stsb/03-mechanism.md) 已经证明的时域等价公式：

$$
y_t=\frac{g_R+g_I}{2}x_t+\frac{g_R-g_I}{2}x_{(-t)\bmod N}.
$$

它解释了为什么实部与虚部门控不同时会出现**循环反转/反射项**，以及为什么 padding 长度可能影响输出。**这条数学发现来自此前 P1 实验，不是本轮重新提出的创新点**。

#### 5.3 只训练一个随机种子的初步比较

Seed 0，official test：

| 模型 | STSB Spearman | Sprint AP |
| --- | ---: | ---: |
| FLaG-B2 | **0.845490** | **0.834217** |
| StaticReIm | 0.841483 | 0.820715 |
| StaticDiag | 0.840198 | 0.824215 |
| Mean + 可训练 Projection（单位矩阵初始化） | 0.841056 | 0.833765 |

最引人注意的是 Sprint：Mean + Projection 与复杂 B2 只差 $0.000452$ AP。

**这个结果使我们提出新问题**：过去 Sprint 直接 Mean 的 AP 约为 0.4289，是不是不能说明 DC 中缺乏任务信息，而主要说明“直接 Mean + cosine”缺少合适的可训练变换？

这一判断也与此前 [DC-only 机制实验](../dc-frequency.md) 相呼应：即使只保留 DC，只要训练合适的后续处理，仍然可以得到很强的 Sprint 结果。

但是单个种子不足以给出稳定结论，于是我们把对照扩大到三个种子。

### 6. 第五步：排除 Projection 初始化差异后再比较

#### 6.1 先比较随机初始化和单位矩阵初始化

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

#### 6.2 最后一次严格控制：两个模型从相同 Projection 权重开始

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
    

    新训练程序保存初始 Projection 指纹，并在配对汇总时检查匹配；归档时仍应保存相关日志、配置和指标文件，便于复查。

### 证据解释的范围

相同初始化只涉及共同的输出 Projection 权重/偏置，不意味着两个网络所有权重相同；此轮 3-seed test 对照未观察到复杂 FLaG 超过 Mean+Projection，但 validation 均值方向相反。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/flag-gate-projection.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/FLAG_PAIRED_AP_MECHANISM_PLAN.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_flag_identical_projection_control.sh)

[返回实验导航](index.md)
