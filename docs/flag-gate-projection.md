# FLaG 的门控与输出线性变换：从表示夹角到简化模型对照

!!! info "研究范围与结果性质"
    本页是原有 [Mean-compatible FLaG](mean-compatible-flag.md)、[循环反射和 P1/P2](stsb/03-mechanism.md)、[DC / 频带机制](dc-frequency.md) 的**后续机制研究**，不是对原有实验的替代。核心研究对象是 **Global FLaG 在 time pooling = masked Mean、关闭 post-pool LayerNorm、dropout = 0 的专门对照配置**。原论文常见的 Max pooling / LayerNorm 配置，以及局部 STFT-FLaG，不应直接套用本页结论。
    
    本页既有只看一个训练完成模型的推理时修改实验，也有从头训练不同结构的实验。前者检验已经训练好的模型依赖什么；后者检验简单结构能否在相同协议下训练出相近性能。

## 1. 为什么开始这组实验？

之前有两个现象需要解释：

- STSB 上，直接 Mean pooling 往往比原始 FLaG 的结果更好；但 Sprint 上，原始 FLaG 又明显好于直接 Mean。
- 原有 P1、P2 和 Sprint S3 发现，**换 Gate 的生成来源几乎不改变预测，改变频谱重建路径却会改变输出**。之前已解释实部与虚部门控不同产生循环反射，并不是本页首次推导。

于是本阶段不再只比较 Global FFT 和 Local STFT，而是逐层追问：

1. FLaG 的哪个环节使输出不再与 Mean 同方向？
2. 训练后的 Gate 是否真的随着输入句子显著改变，它调整了哪些通道？
3. 如果不使用 FFT 和 latent attention，只保留可学习的通道缩放或线性变换，模型还能达到怎样的性能？

本页继续沿用原报告的 **Q（问题）/ M（方法）/ R（结果）/ C（结论）** 格式。

## 2. 实验协议与必须区分的变量

| 设置 | STSB | SprintDuplicateQuestions |
| --- | --- | --- |
| 任务 | 句对语义相似度 | 重复问题判别 |
| Backbone | RoBERTa-base，参与微调 | RoBERTa-base，冻结 |
| 训练 | 3 epochs | 10 epochs |
| Batch | 32 | 32 |
| 主要结果 | 官方 test Spearman | 官方 test AP（Average Precision） |
| 选择 checkpoint | validation Spearman | 早期机制实验：validation Accuracy；最新多种子结构对照：**validation AP** |
| FLaG 时间池化 | masked Mean | masked Mean |
| Post-pool LayerNorm | 关闭 | 关闭 |
| Pooling dropout | 0 | 0 |

各组实验使用相同任务内部的数据集划分方法、冻结状态和学习率，但**早期结果与最新结果的 checkpoint 选择方式不同**。不能把不同选择规则下的模型混在一起求均值。

尤其需要记住：在 Sprint 中，原始直接 Mean 通常把 RoBERTa 冻结后生成的 Mean 向量直接用于 cosine；新的 Mean + Projection 则额外**训练**了一个线性层。这不是相同容量的模型。

## 3. Q1：FLaG 在哪一步使 Mean 向量发生方向变化？

**Q：** FFT 的零频率分量已经包含均值信息，为什么最终的 FLaG 与直接 Mean 方向不一致？

**M：** 以一个句子的 RoBERTa 输出 $H\in\mathbb R^{T\times D}$ 为起点。直接 Mean 向量为 $m$，频域 DC（第 0 个频率位置）为：

$$
X_0=\sum_{t\in\mathrm{valid}}h_t=Lm .
$$

因此理论上 DC 和 Mean 同向。我们在相同输入上记录 DC、Gate 处理后的 DC、iFFT 后 masked Mean 的输出和最终 Projection 输出，并逐阶段计算角度。为了做出严格的 Mean 起点，还比较了以下配置：

| 代号 | Gate | 最后线性 Projection 初始化 |
| --- | --- | --- |
| FLaG | $g=1+\sigma(\ell)$ | 随机 |
| A1 | $g=1+\sigma(\ell)$ | 单位矩阵 |
| B1 | $g=2\sigma(\ell)$ | 单位矩阵 |
| B2 | $g=2\sigma(\ell)$，最后 Gate 输出层参数初始化为 0 | 单位矩阵 |

B2 初始化时 $g=2\sigma(0)=1$，因此在本页的 Mean-pooling / no-LN 配置下，初始输出**严格等于直接 Mean**。

**R：** 在 seed 0 的 B2 第 1 个 epoch 后：

| 阶段角度 | STSB | Sprint |
| --- | ---: | ---: |
| 直接 Mean 与原始 DC | 约 0° | 约 0° |
| 原始 DC 与经过 Gate 的 DC | 64.715° | 77.867° |
| 原始 Mean 与 iFFT 后 masked Mean | 64.523° | 77.655° |
| Projection 输入与输出 | 34.657° | 83.668° |

**C：** FFT 的 DC 仍与 Mean 同向；主要的向量方向变化发生在 **Gate 通道缩放**和**最后的可学习 Projection**。B2 即使严格从 Mean 开始，也会在训练后改变方向。

但是，这里测的是同一句话的 Mean 与 FLaG **两个表示向量之间的角度**，并不是不同句子之间的相似度。角度大不能直接证明语义信息受损或任务表现变差。

另一个严格的数学事实：原始 Gate 的缩放值在 $(1,2)$，对固定 DC 向量的逐通道正数缩放造成的角度理论上不超过约 $19.47^\circ$；B2 的门控值位于 $(0,2)$，允许部分通道接近零，因此可产生更大的转角。两个模型角度差异，不能直接解释成一个学得更好或更坏。

## 4. Q2：训练后的 Gate 是否随句子明显变化？

**Q：** FLaG 利用频谱和 latent attention，为每个句子计算 Gate。训练后这种输入相关性在输出上是否明显？

**M：** 固定已经训练好的 seed-0 checkpoint，不重新训练；分别使用：

| 名称 | 实际操作 |
| --- | --- |
| Full | 每句使用由自身频谱和 latent attention 计算出的 Gate |
| Training-set mean Gate | 从训练集句子估算一套平均 Gate，评估时所有句子共用 |
| Swap Gate between sentences | 在同一批中交换不同句子的 Gate，保持句子内容和模型其他参数不变 |

三个操作使用相同的 Backbone、输出 Projection、句对打分头和验证/测试样本。

**R：** test 指标：

| Task / Model | Full | 训练集平均 Gate | 交换句子 Gate |
| --- | ---: | ---: | ---: |
| STSB FLaG，Spearman | 0.843379 | 0.843379 | 0.843379 |
| STSB B2，Spearman | 0.845488 | 0.845488 | 0.845488 |
| Sprint FLaG，AP | 0.848503 | 0.848503 | 0.848503 |
| Sprint B2，AP | 0.833946 | 0.833946 | 0.833946 |

对应的 Gate 与训练集均值的差异，在原有输出中打印至五位小数为 0.00000。Gate sigmoid 输出也高度接近两端，Sprint B2 中约 95.96% 高于 0.95、约 4.04% 低于 0.05。

**C：** 在**这些训练好的 checkpoint 的推理行为**中，用训练集固定 Gate 或别的句子的 Gate 替换自身 Gate，并未使最终评价指标发生六位小数可见的变化。这支持“Gate 在训练后近乎固定”的解释，且比仅仅比较 Global/Local Gate 来源更直接。

需要特别注意：

- “指标显示相同”不代表每个 Gate 元素逐位严格相同；还需要最大绝对误差、Sigmoid 前后通道方差等高精度数值。
- “训练后近乎固定”不代表 FFT/latent attention 在**训练过程中**一定没有作用。
- 早期 P1/P2 已经发现 Gate 来源不敏感，本次新增的是直接交换**不同句子生成的 Gate**。

## 5. Q3：Gate 调整了 DC 的哪些特征通道？

**Q：** 如果只有少数通道的 Gate 值明显偏低，为什么它仍能造成很大的 DC 转角？

**M：** 对一个句子的原始 DC 实部向量 $x=(x_1,\ldots,x_D)$，取 $D=768$。Gate 的输出是 $2D$ 维，前 $D$ 项缩放实部、后 $D$ 项缩放虚部。DC 的虚部为零，因此这一分析只看**实部 Gate**。

设 $S$ 为实部 Gate 较低的**通道编号集合**，而不是 token 位置或频率位置。本实验低值标准是原始 Sigmoid 输出小于 $0.05$：

- 原始 FLaG：$g_i=1+\sigma(\ell_i)$，低值对应 $g_i<1.05$；
- B2：$g_i=2\sigma(\ell_i)$，低值对应 $g_i<0.1$。

计算这些位置在**Gate 作用前**占全部 DC 平方能量的比例：

$$
E_S=\frac{\sum_{i\in S}x_i^2}{\sum_{i=1}^{D}x_i^2}.
$$

**R：** Sprint official test，seed 0：

| 统计 | FLaG | B2 |
| --- | ---: | ---: |
| 低 Gate 值实部通道占比 | 2.214% | 3.776% |
| 这些通道占原始 DC 平方能量的比例 | 93.720% | 95.488% |
| 平均 DC Gate 转角 | 12.811° | 77.773° |

STSB test 对应的低值实部通道能量占比为 75.416%（FLaG）与 80.208%（B2）。

**C：** 在当前 RoBERTa 输出坐标下，DC 平方能量高度集中于少数通道，而训练后的 Gate 正好对这些高能通道赋予相对较低的缩放权重。

这解释了为什么很少的通道被重新加权，就可能造成大的向量方向变化。但 **FLaG 的低值是接近 1 而非 0**，应称为相对降低权重，不能说它删除了这些通道。更不能直接推断高能通道都是无用噪声，或断言它们对应某个已确定的语义成分。

## 6. Q4：关闭 Gate 的哪些部分，预测会变化？

**Q：** Gate 改变向量方向，是否真的对已训练模型的预测有作用？

**M：** 固定训练好的 checkpoint，在推理阶段分别关闭 DC 门控、关闭 DC 以外的门控、完全关闭 Gate，以及绕过最后 Projection。其他部分仍使用训练好的参数。这属于**训练后干预**，不是从零训练结构消融。

**R：** Sprint official test，seed 0：

| 推理方式 | FLaG AP | B2 AP |
| --- | ---: | ---: |
| Full | 0.848503 | 0.833946 |
| DC 的 Gate 改成 1 | 0.838674 | **0.678850** |
| 非 DC 频率的 Gate 改成 1 | 0.849550 | 0.835823 |
| 全部 Gate 改成 1 | 0.840069 | 0.784842 |
| 保留 Gate，但跳过最后 Projection | **0.464409** | **0.494782** |

**C：** 已训练的 B2 尤其依赖其 DC Gate；完整 FLaG 和 B2 在 Sprint 上绕过 Projection 后，性能都显著下降。另一方面，关闭非 DC 部分并没有明显伤害 Sprint AP。

但一个模块被训练好的模型依赖，并不意味着该模块对**从零训练一个同等性能的模型**不可替代；参数可能已经与它共同适配。这正是下一部分需要独立训练简化模型的原因。

注意上述 Sprint checkpoint 来自**早期按 validation Accuracy 选择的 seed-0 机制实验**，不能与后面采用 validation AP 选模的三种子结果混合求均值。

## 7. Q5：如果不使用 FFT 和 latent attention，从零训练能否得到相近性能？

**Q：** 如果已训练 Gate 的作用接近固定通道缩放，我们是否需要完整频域模型才能达到类似性能？

**M：** 从头训练三个结构简单的模型，且均使用可训练、初始为单位矩阵的输出 Projection：

| 名称 | 输入 Mean 之后的操作 |
| --- | --- |
| StaticReIm | 学习一套对所有句子共享的实部/虚部通道系数，用原有 FFT 调制 + iFFT 的**等价时域公式**计算，再做 Projection |
| StaticDiag | 学习一套对所有句子共享的逐通道系数，直接乘到 Mean 向量，再做 Projection |
| MeanProjection | 不使用 Gate，直接对 Mean 向量做可训练线性 Projection |

这里 Static 表示**相同权重用于不同句子**，不代表权重在训练中不更新。

原有 P1 已推导：当实部与虚部使用不等的共享通道门控系数时，固定门控后的 FFT → Gate → iFFT 等价于原序列和循环反转序列的加权混合。因此 StaticReIm 不必真的做 FFT；这条公式在原有 [P1 报告](stsb/03-mechanism.md) 已记录。

**R：** 第一轮 seed-0 official test：

| 方法 | STSB Spearman | Sprint AP |
| --- | ---: | ---: |
| FLaG-B2 | 0.845490 | 0.834217 |
| StaticReIm | 0.841483 | 0.820715 |
| StaticDiag | 0.840198 | 0.824215 |
| MeanProjection（单位矩阵初始） | 0.841056 | 0.833765 |

对于 Sprint，MeanProjection 比 B2 只低 0.000452 AP。

**C：** 在这个 seed 上，**仅对 Mean 向量学习一个线性变换**已经能得到接近 B2 的 AP。

从数学上，对于静态对角门控 $D$ 与任意可训练线性投影 $W$：

$$
z=W(Dm)+b=(WD)m+b=W'm+b.
$$

因此静态逐通道门控再接一个不受约束的线性 Projection，并没有增加模型能表达的函数范围；但不同参数化仍可能影响初始化与训练优化。这个代数关系不意味着原始**输入相关**的 Gate 和具有实虚不对称反转项的完整 FLaG 也等价于 Mean + Linear。

## 8. Q6：Projection 初始化和真正的结构差异，谁影响更大？

**Q：** 原始 FLaG 的 Projection 从随机参数开始，而 B2/前述 MeanProjection 从单位矩阵开始。是否应当把原始 FLaG 直接同**随机初始化的 Mean + Projection**比较？

**M：** 使用 Sprint seeds 0、1、2，统一在 validation 上根据 **AP** 选 checkpoint，并在相同冻结 RoBERTa、训练/验证划分及 Mean-pooling/no-LN 协议下从零训练：

- 原始 FLaG：原始 residual-sigmoid Gate，随机 Projection；
- FLaG-B2：从严格 Mean 输出开始的 centered-sigmoid Gate、单位 Projection；
- MeanProj-B2：直接 Mean + 单位初始化的可训练 Projection；
- MeanProjRand：直接 Mean + **随机初始化、训练时仍会学习**的 Projection。

**R：** 第一轮配对三种子 official-test AP：

| 模型 | Test AP（mean ± sample SD） |
| --- | ---: |
| FLaG | 0.829813 ± 0.016874 |
| FLaG-B2 | 0.824462 ± 0.011258 |
| MeanProj-B2 | 0.818806 ± 0.016929 |
| MeanProjRand | **0.840923 ± 0.008811** |

同 seed 的三个配对差值（FLaG − MeanProjRand）全为负，平均为 −0.011110 ± 0.013482。MeanProjRand 与 MeanProj-B2 的平均差值为 +0.022117，三个种子均为正。

不过：**相同 seed 不保证不同结构里的随机 Projection 矩阵完全相同**。这是因为创建其他网络层会消耗随机数。为控制这个混杂因素，我们追加了更严格的对照。

### 8.1 严格匹配最初 Projection 矩阵的三种子实验（最后补充）

**M：** 对每个 seed，强制让 FLaG 与 MeanProjRand 使用**完全相同的初始 Projection 权重和偏置**。相同任务、相同划分和 validation AP 选模。实现记录 Projection 初始参数的 SHA-256 指纹，用来核对两组是否一致。

**R：** 最新 Sprint official test 结果：

| 方法 | Test AP（3 seeds） | Validation AP（3 seeds） | 选中 epoch |
| --- | ---: | ---: | --- |
| FLaG | 0.830539 ± 0.016711 | **0.853535 ± 0.019891** | 10、9、8 |
| MeanProjRand | **0.841386 ± 0.008833** | 0.851664 ± 0.017371 | 10、10、8 |

严格配对的 test AP 差值：

| Seed | FLaG − MeanProjRand |
| --- | ---: |
| 0 | −0.001096 |
| 1 | −0.028025 |
| 2 | −0.003420 |
| **均值 ± sample SD** | **−0.010847 ± 0.014922** |

三个 seed 中 MeanProjRand 都高于 FLaG 的 test AP。

还有一个应该如实记录的现象：FLaG 的**平均 validation AP**（0.853535）反而稍高于 MeanProjRand（0.851664）；到了 official test，顺序相反。因此不能说 MeanProjRand 在所有拆分上都领先，也不能由三种子结果推断确定的泛化优势。

**C：** 即使控制初始 Projection 参数完全相同，当前 Sprint Mean-pooling/no-LN 协议的三种子结果中，原始 FLaG 也**没有表现出相对于 Mean + 可训练随机 Projection 的 test AP 优势**。更简单的模型在三个种子上都略高或明显更高。

这不等于数学上证明原始 FLaG 的 FFT 或 latent attention 无用，也不代表适用于原论文的 Max pooling / post-pool LayerNorm 或局部 STFT 模型。它只限定于当前控制实验，并且样本量是三个随机种子。

!!! note "严格配对的复现检查"
    最新运行脚本会记录两组模型初始 Projection 的 SHA-256 指纹，并在汇总时进行一致性核对。最终归档时应一并保存训练日志或对应的模型指标文件，确保能复查指纹一致性。本页数值来自已完成的三种子汇总。

## 9. 目前最清楚的研究结论与尚未解决的问题

### 已有证据支持的结论

1. **DC 与直接 Mean 同向**；在本配置下使表示偏离 Mean 方向的主要操作是 Gate 和 Projection。向量角度本身不能代替下游任务性能。
2. 已训练 Gate 的 sigmoid 输出高度饱和；交换句子 Gate 或使用训练集平均 Gate，指标在六位小数精度下与 Full 相同。说明**这些 checkpoint 在推理时几乎不依赖明显的逐句 Gate 变化**。
3. 低 Gate 值的**实部通道**集中承载大量原始 DC 平方能量。当前 Gate 的一个显著作用是对不同通道重新赋权，但目前不能判断这些通道是否语义冗余。
4. 对冻结 RoBERTa 的 Sprint，**Mean + 可训练 Projection** 是一个比“直接 Mean + cosine”强得多的必要对照。原始 FLaG 优于直接 Mean，不能把增益全部归功于 FFT 或 latent attention。
5. 三种子、相同初始随机 Projection 权重的结果中，完整 FLaG 没有优于 Mean + Linear。此处结论应写为**未观察到额外优势**，不是“证明它必然没有贡献”。

### 尚不能从现有结果断言

- FFT / latent attention 在训练全过程中完全不重要；
- 被压低的高能通道一定是噪声或与语义无关；
- 任意文本任务和原始 Max-pooling 版本的 FLaG 都可无损替换为 Mean + Linear；
- 三个随机种子已足以证明一个稳定、普适的性能排序。

### 阶段性判断

本阶段最值得继续追问的科学问题，已经从“怎样把 FLaG 的角度拉回 Mean”变为：

> **在冻结预训练编码器的句对任务里，FLaG 相比直接 Mean 的增益，有多少来自频域门控本身，有多少来自对 Mean 向量施加的可训练线性变换？**

在向学姐做第一次阶段汇报时，以上实验已经能组成完整证据链。后续是否扩大到更多 seeds、其他任务或恢复原始 Max/LayerNorm 配置，可以在汇报后再确定。

!!! warning "实验范围与探索性分析"
    本研究是在多次查看同一 Sprint test 结果后逐步设计实验的，因此应将本阶段 test 比较视为**探索性**证据。用于正式论文的最终结论，建议预先确定实验方案，并采用新的独立测试任务或评估数据进行确认。原报告 STFT 章节中某些 seed、数据划分和 checkpoint 选择规则与本页不同，必须分别记录。
