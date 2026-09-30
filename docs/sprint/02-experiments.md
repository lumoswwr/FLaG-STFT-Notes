# Sprint：S1–S5，跨任务性能、重建干预和配置控制

先阅读[Sprint 数据与评估协议](01-protocol.md)：这里 AP、Accuracy、F1 的含义及阈值不同，**S1/S2 属于 official-test 性能记录，S3/S4/S5 属于 adaptation-validation 上的干预/控制**。为了保留研究历程，同时避免不正确的因果比较，以下将**早期整套配置**和**后续 matched controls**明确分开。

## A. S1–S5 实验矩阵

| 编号 | 要回答的问题 | 数据/评估位置 | Seed | 重训练还是固定 checkpoint | 非算子配置 |
| --- | --- | --- | --- | --- | --- |
| S1 | 早期 Mean、global FLaG、冻结结构 E12 的跨任务表现怎样？ | Sprint **official test** | **0–2（3）** | 新任务上重新训练/适配 | 早期 FLaG **0.1/False**；E12 **0/True** |
| S2 | S1 的早期整套方案扩至 10 seeds 后表现怎样？ | Sprint **official test** | **0–9（10）** | 扩展重新训练 | **延续 S1，仍不匹配** |
| S3 | 在 Sprint 也能观察到 P2 的 gate/reconstruction 分工吗？ | Sprint adaptation **validation** | **0–2（3）** | **固定 S1 所训练 checkpoint** 做 GG/LG/GL/LL | 对每个源 checkpoint 内拷贝同一套非算子设置；两种训练来源本身仍不同 |
| S4 | 控制 dropout/norm 后，local E12 是否稳定优于 global？ | Sprint adaptation **validation** | **0–9（10）** | 匹配的 global **重新训练**；比较 E12 | 两者均 **0/True**；算子不同 |
| S5 | Sprint 的结果是否对 dropout 与 post-pool norm 有交互敏感性？ | Sprint adaptation **validation** | **0–9（10）** | **重新训练**四种 **global FLaG** | dropout `{0,0.1}` × norm `{False,True}` |

**记号：** `0/True` 表示 pooling dropout=0、post-pool LayerNorm=开启；`0.1/False` 表示 dropout=0.1、无 post-pool LayerNorm。论文方法正文写明**文本**为 `0.1/True`。S1 的 `0.1/False` 是本项目早期公开代码风格的复现实现，**不是**论文方法段所描述的文本归一化设置。

## B. 共同 Sprint 任务设置

| 项目 | 配置 |
| --- | --- |
| 数据 | SprintDuplicateQuestions，输入英文句对，二分类标签 0/1 |
| 样本划分 | 官方 validation → 固定分层 90:10：adaptation train **90,900** 对、validation **10,100** 对；official test **101,000** 对 |
| Split 随机性 | `random_state=42`；训练随机 seeds 按各实验单独列出；各模型共享同一划分 |
| Encoder | **冻结** RoBERTa-base；max length=128 |
| Head | 单调正尺度 cosine-logit（`softplus(raw_scale)×cosine+bias`） |
| Loss | BCEWithLogitsLoss |
| 训练预算 | 10 epochs；batch=32；pool/head LR=`1e-3` |
| 选择 | **validation 校准 Accuracy 阈值后，以 validation Accuracy 选 checkpoint** |
| 最终指标 | AP 不依赖阈值；test Accuracy 用 validation 的 Accuracy 阈值；test F1/Precision/Recall **另用** validation 的 F1 阈值 |
| 模型核心 | Global/E12 都是 8 latents、4 heads、residual gate、masked max pooling；E12 固定 rect、win16/hop16、无 overlap、无新帧位置编码 |

---

## S1：早期跨任务迁移和 3-seed 对比

**Q：** 将 STSB 选定的 E12 **架构**迁至另一种任务，在冻结 RoBERTa 的重复问题检测中，整体表现如何？与 Mean 和早期 global FLaG 对比是什么样？

**M：** 在 Sprint 上分别初始化并训练/适配 Mean、global FLaG 与 E12；**并非沿用 STSB checkpoint**。三个模型共享上面的数据划分、冻结 backbone、loss、训练预算及模型选择方式，使用 **seeds 0、1、2**，在 official test 计算 AP/Accuracy/F1。本阶段**原始实际配置**如下：

| 模型 | Pooling | Dropout | post-pool LayerNorm | 额外说明 |
| --- | --- | ---: | :---: | --- |
| Mean | 有效 token mean | 不适用 | 不适用 | Mean pool 本身没有可训练参数，仅 head 学习校准 |
| 早期 FLaG | Global FFT | **0.1** | **否** | 沿用当时训练脚本的 global 路径 |
| E12 | STFT，rect 16/16 | **0** | **是** | STSB 探索线选定的 local 结构 |

**R：** **3-seed official-test 结果（mean±sample std）**：

| 模型 | test AP ↑ | test Accuracy ↑ | test F1 ↑ |
| --- | ---: | ---: | ---: |
| Mean | 0.428907 ± 0.000003 | 0.992040 ± 0.000000 | 0.460390 ± 0.000000 |
| 早期 FLaG | 0.721611 ± 0.009747 | 0.993980 ± 0.000298 | 0.667938 ± 0.007364 |
| E12（早期配置） | 0.765655 ± 0.023788 | 0.994495 ± 0.000423 | 0.690782 ± 0.017933 |

这里的 F1 与 Accuracy 采用的是**两套分别在 validation 选出的阈值**。Mean frozen pooling 与具有大量可训练参数的 FLaG/E12 **模型容量不同**，因此不能仅凭 Mean 的 AP 较低，认定 FFT/latent 是性能提升的唯一原因。

**C：** 早期 E12 **整套训练配置**出现正向跨任务信号，促使 S2 扩 seeds；但这里同时改变了**算子、dropout 和 LayerNorm**，所以**不能**把分数差写成“STFT 本身带来的增益”。

## S2：扩展到 10 seeds 后的早期整套配置表现

**Q：** S1 的 3-seed 差异扩展到 10 seeds 后是否仍能观察到？它是否能证明 local operator 的作用？

**M：** **保留 S1 的实际配置不变**，global FLaG 为 `0.1/False`，E12 为 `0/True`，用 **seeds 0–9** 各重新进行 Sprint 训练/适配；仍在 validation 选 checkpoint/阈值后只对 official test 报指标。这个扩 seed 不会自动使两种配置匹配。

**R：** **10-seed official-test（均值±样本标准差）**：

| 指标 ↑ | 早期 FLaG，0.1/False | E12，0/True | 平均配对 E12 − FLaG |
| --- | ---: | ---: | ---: |
| AP（无阈值） | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 | **+0.048222 ± 0.017641**，10/10 正向 |
| Accuracy（val Accuracy 阈值） | 0.993857 ± 0.000275 | 0.994485 ± 0.000268 | **+0.000628 ± 0.000432**，10/10 正向 |
| F1（val F1 阈值） | 0.650671 ± 0.016939 | 0.685348 ± 0.020150 | **+0.034676 ± 0.024246**，8/10 正向 |
| Precision（val F1 阈值） | 0.745202 ± 0.033959 | 0.814274 ± 0.032556 | +0.069072，9/10 正向 |
| Recall（val F1 阈值） | 0.579800 ± 0.037446 | 0.594100 ± 0.041906 | +0.014300，6/10 正向 |

AP 配对差的 95% t 区间约 `[+0.0356,+0.0608]`。它表明**这两套实际训练配置**的 AP 差在本实验中跨 seed 一致；**不表示**局部 STFT operator 的因果收益也一致。

**C：** E12 的**整体配置**与早期 global 方案相比确实有较大的 official-test AP 差，但两个因素不匹配导致不能归因。下面 S3 从**推理路径**拆机制，S4 再从**重新训练的控制配置**隔离非算子因素。

---

## S3：Sprint 上重放 GG/LG/GL/LL

**Q：** STSB 的 P2 发现固定模型下 global/local 的预测变化几乎全部发生在重建路线。换到 Sprint 的冻结-backbone/分类任务，是否有类似现象？这种预测变化是否等于 AP 改善？

**M：** 使用原 S1/S2 配置下已训练的 **FLaG-trained（0.1/False）**和 **E12-trained（0/True）**，每种各 **seeds 0–2**。固定每个 seed 的参数，eval 模式，在同一 **Sprint adaptation validation** 上生成相同的 RoBERTa hidden，并让这份权重分别生成 global/local gate，再交叉组合以下两条重建路线：

| 模式 | Gate 来源 | Gate 作用的频谱与重建 |
| --- | --- | --- |
| GG | Global | global 频谱 × gate → global irFFT |
| LG | Local | global 频谱 × gate → global irFFT |
| GL | Global | **local** 频谱 × global gate → local irFFT + frame stitching |
| LL | Local | local 频谱 × gate → local irFFT + frame stitching |

**GL 不是把 global 频谱交给局部 iFFT。** Global 频谱只用来**算 gate**，实际被重建的仍是 local 频谱。对每个源 checkpoint，global/local operator 加载**相同 state_dict**、保留该 checkpoint 自己的 dropout/norm 属性；手动 GG、LL 重放原生 forward 并检查预测/AP。该实验**没有**为 GL/LG 训练新模型，故只用于推理反事实解释。

**R1：** **3-seed validation 句对 cosine 的平均绝对预测变化**：

| 训练来源 | 只换 gate，\(\operatorname{mean}|LG-GG|\) | 保持 global gate，只换完整重建路线，\(\operatorname{mean}|GL-GG|\) | 整体变为 local，\(\operatorname{mean}|LL-GG|\) |
| --- | ---: | ---: | ---: |
| 早期 FLaG-trained | 数值精度内约 0 | **0.032432 ± 0.003114** | 约 0.032432 |
| E12-trained | 约 \(8.30\times10^{-13}\) | **0.037477 ± 0.003556** | 约 0.037477 |

**R2：** **3-seed adaptation validation AP 均值**，同一行只能比较该训练来源的干预效果：

| 训练来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| 早期 FLaG-trained | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| E12-trained | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

**C：** 在这组 Sprint checkpoint 中，改变 gate 的 global/local 频谱来源几乎不改变输出；完整 global/local reconstruction 路线对余弦分数的作用很明显，**与 STSB 的 P2 观察一致**。但在此 **3-seed validation、固定权重**干预中，local 重建**没有**产生更高 AP。跨两种**不同训练配置**的原生 GG 与 LL 的 AP 数字仍然不能拿来做单因素 operator 的因果分析。

---

## S4：匹配非算子设置后的 global vs local

**Q：** S1/S2 的 AP 差同时改变了算子、dropout、norm。如果把两者的非算子配置统一，仅保留 global/local FFT 路线的差别，还存在稳定的 E12 AP 优势吗？

**M：** 新训练 **GlobalMatch（global FLaG）**：把其设置匹配到 E12 的非算子配置 **dropout=0、post-pool LayerNorm=True**，使用与 E12 相同的冻结 RoBERTa、8 latents、4 heads、residual gate、max pooling、head、adaptation 划分、10 epochs、模型选择规则，global 只保留**整句 FFT**，E12 只保留**16/16 local STFT**。**seeds 0–9**。为避免把 validation-only 控制说成新的 held-out test 结论，本节所有数均明确为**adaptation validation AP**。

**R：** **10-seed validation AP**：

| 模型/差值 | mean ± std |
| --- | ---: |
| matched Global FLaG，0/True | **0.795830 ± 0.017156** |
| E12，0/True | **0.789136 ± 0.014094** |
| 同 seed 的 E12 − GlobalMatch | **−0.006693 ± 0.017279** |

配对差 **4/10 seeds 正向**，95% t 区间约 `[−0.0191,+0.0057]`，双侧配对 t 检验约 `p=0.252`。区间跨 0；这是“**未观察到稳定 local 优势**”，不能说成“证明 global 永远比 local 更强”或已经证明两者等效。

**C：** 匹配的 global 方案无需引入 local STFT，也能在这组 validation 控制中达到接近 E12 的 AP。**S1/S2 的整套配置差不能归因于局部算子单独造成**。这个结论与 S3 的“结构差异主要进入重建路线”不矛盾：**模型输出会变，并不代表该改变会稳定提高 AP**。

---

## S5：global FLaG 的 dropout × post-pool norm 2×2

**Q：** 重新核对原 FLaG 论文方法后确认：其文本主设置为 **dropout=0.1 且 post-pool LayerNorm=True**。Sprint early global FLaG 的 **0.1/False** 与这一方法文字不同。到底 dropout、post-pool norm 及两者**交互**对本项目 Sprint global 模型有多大影响？

**M：** **固定 global FFT**，对**global FLaG 一种 operator**做两因素交叉训练：`dropout ∈ {0,0.1}` 与 `post_pool_norm ∈ {False,True}`，四组分别使用相同的冻结 RoBERTa、head、adaptation train/validation 划分、10 epochs 和 checkpoint 选择方式，**每组 seeds 0–9（10 seeds）**。此处**没有同时训练四种 local STFT**，所以不是 global/local × dropout × norm 的 2×2×2 证明。

**R1：** **10-seed adaptation validation AP（mean ± std）**：

| Global FLaG dropout | Post-pool LayerNorm | Validation AP ↑ |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 ± 0.025820 |
| **0.1** | **有** | **0.819156 ± 0.015250** |
| 0 | 无 | 0.798974 ± 0.007324 |
| 0 | 有 | 0.795830 ± 0.017156 |

**R2：** 固定另一因素后比较**同 seed 的条件效应**。正数表示新配置的 validation AP 较高：

| 配对操作 | mean ΔAP ± std | 正向 seeds |
| --- | ---: | ---: |
| 无 norm 时，dropout `0.1 → 0` | **+0.048038 ± 0.024758** | 10/10 |
| 有 norm 时，dropout `0.1 → 0` | **−0.023326 ± 0.017170** | 0/10 |
| dropout=0.1 时，norm `False → True` | **+0.068220 ± 0.029151** | 10/10 |
| dropout=0 时，norm `False → True` | **−0.003145 ± 0.014885** | 5/10 |

按上述差分定义计算的 dropout × norm **交互项**约为 `−0.071364±0.021526`，10/10 seeds 为负。这说明两个因素**不能分别用“去掉 dropout 总是更好”或“加 norm 总是更好”来概括**。

**C：** 在这四组**global FLaG adaptation validation** 控制中，`dropout=0.1 + post-pool LayerNorm=True` 的平均 AP 为 `0.819156`，是本次四种 global 配置中数值最高的一组，且与原早期 `0.1/False` 的配对提升 10/10 seeds 为正。它也吻合论文方法文字描述的**text 配置**；但同一 validation 用于模型选择及后续实验比较，不能将此增益冒称已由独立 official test 证实。

!!! important "尚未进行的比较"
    Sprint **尚未**在相同 `dropout=0.1,norm=True` 下对**global FLaG 与 E12**重新训练完整的 10-seed matched 对照。因此这里不能写“论文配置下 Sprint global 一定优于 E12”；它需要单独的 matched 实验。已完成的这组论文 text 配置 global/local 10-seed 配对检验发生在 **STSB E14**，指标是 **test Spearman/Pearson**，与 Sprint 的 validation AP 不可横向比较。

---

## 这一部分支持什么，不支持什么？

**已经观察到的结构事实**：Sprint S3 与 STSB P2 均显示，在所测 checkpoint 和输入上，global/local **gate 观测路径变化很小**，完整 reconstruction/temporal-support 路线的切换能明显改变句对 cosine。

**性能层面的限制**：Sprint S1/S2 的大幅 AP 差是**两套原始配置**之间的结果；在 S4 的 `0/True` 匹配控制中，不存在稳定的 local 优势；S5 说明 dropout×norm 本身足以大幅改变 global 的 validation AP。**不能**把“reconstruction 是表示差异主要来源”换写成“local reconstruction 是性能提升的已证明原因”。

**下一步跨任务配置线索**：STSB 的 [E14](../stsb/02-padding.md) 在论文文本配置 `0.1/True` 下，10-seed FLaG/E12 **test Spearman** 分别为 `0.839065±0.004154` 与 `0.838944±0.001882`（配对 E12−FLaG `−0.000121±0.004039`，5/10 正向）。这是另一数据集的独立配置敏感性观察，**不是** Sprint 的 S5 结果。

**源代码：** [Sprint train](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py) · [Sprint P2 probe](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_sprint_global_local_2x2.py) · [机制研究过程摘要](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)。
