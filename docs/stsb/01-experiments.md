# STSB：E1–E8，频域变换探索及早期机制探针

这部分研究从“global FFT 是否受到序列边界影响”出发，逐步尝试 local STFT、局部帧位置编码和窗函数，随后使用 frequency/token knockout 与 DC-attention 诊断解释模型行为。**E1–E5 是重新训练的候选模型；E6–E8 是已训练模型的推理诊断，不能与前者混作一张性能排行榜。**

## A. STSB 统一协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | STSBenchmark（STSB），英文句对语义相似度，原始标签 0–5 |
| 划分与用途 | 官方 train 训练；官方 development/validation 按 Spearman 选 epoch；官方 test 报最终性能 |
| Backbone | RoBERTa-base，端到端微调；两句各产生一个句向量 |
| 主性能指标 | **test Spearman**（预测/标签的秩相关）和 **test Pearson**（线性相关），均越高越好 |
| 训练目标 | `MSE(cosine(z1,z2), label/5)` |
| Epoch / Batch | 3 epochs；batch=4 句对；梯度累积 4（等效 batch=16）；max length=128 |
| 优化 | AdamW；backbone LR=`1e-5`，pool LR=`1e-3`，weight decay=`0.01`，warmup ratio=`0.1`，线性 LR schedule，clip grad=1 |
| FLaG/STFT 共享设置 | 8 个 latent queries；4 attention heads；residual gate；masked max time pooling；输出 projection |
| **E1–E5 统一非算子设置** | **pooling dropout=0；post-pool LayerNorm=True**；窗口/位置编码差异见下表 |
| Seeds | **E1–E5：3 seeds（0、1、2）**。E6–E8：读取相同三个 seed 的 FLaG/E3 checkpoint，不重新训练 |
| 统计 | 逐 seed 计算 test 指标，再报 **3-seed 均值 ± 样本标准差**；对比优先使用同 seed 配对差 |

!!! note "配置与论文的区别"
    本页使用 **0 / True** 是原 STSB 探索线的统一控制条件。原论文文本主配置为 **0.1 / True**，后续补充实验 E14 专门检查这一配置，不应把两条线混作同一实验。

## B. E1–E8 配置对照表

| 编号 | 训练/分析对象 | 频域与关键配置（相对基线的变化） | Seeds | 结果属于什么指标 |
| --- | --- | --- | --- | --- |
| Baseline | Global FLaG | 整句 `rFFT/irFFT`；无窗口；`FFT length=batch T` | 0、1、2 | test Spearman / Pearson |
| **E1** | 新训练：Global Hann | 在 global rFFT 前给各样本**有效 token**加 symmetric Hann；FFT 仍为 global | 0、1、2 | test Spearman / Pearson |
| **E2** | 新训练：STFT | rect；`win=16,hop=8,center=False`，8-token overlap | 0、1、2 | test Spearman / Pearson |
| **E3** | 新训练：STFT | rect；`win=8,hop=4,center=False`，4-token overlap | 0、1、2 | test Spearman / Pearson |
| **E4** | 新训练：E3 + 帧位置编码 | E3 上为 attention 的局部频谱 token 加 learned frame position | 0、1、2 | test Spearman / Pearson |
| **E5** | 新训练：E3 + Hann/center | E3 改为 `periodic Hann,center=True`，仍 `win=8,hop=4` | 0、1、2 | test Spearman / Pearson |
| **E6** | **探针**：FLaG vs E3 | 对最终层 content-token hidden 做 DCT，划为 8 频带，每次置零一个后逆变换 | 0、1、2 | eligible test 子集的 Spearman 降幅及预测扰动 |
| **E7** | **探针**：FLaG vs E3 | 逐个将 content-token hidden 置零，不重新运行/训练 backbone | 0、1、2 | 平方误差响应、预测响应、位置异质性 |
| **E8** | **探针**：FLaG vs E3 | 读取频谱 latent-attention，分别统计 DC 与非 DC 的**每 token 平均权重** | 0、1、2 | test 句子的 DC attention 偏好比 |

E2–E5 的 local STFT 都将所有**有效帧的频率 token**送给同一组 latent queries，汇总生成一个**句级**通道 gate；门控所有频率 token，之后才逐帧逆变换，并非每个 frame 生成独立 gate。

## C. E1–E5 性能汇总

下表均为 **STSB official test，seeds=0/1/2**；E3/E12 后续的 10-seed 数据另有表述。

| 模型 | 主要修改 | test Spearman ↑ | test Pearson ↑ | 与同 seed FLaG 的平均 Spearman 差 |
| --- | --- | ---: | ---: | ---: |
| FLaG baseline | 整句 global FFT | 0.841128 ± 0.002177 | 0.836765 ± 0.001782 | 0 |
| E1 | global Hann | 0.837019 ± 0.002763 | 0.836682 ± 0.000282 | −0.004109 |
| E2 | STFT 16/8 rect | 0.842204 ± 0.001824 | 0.837783 ± 0.003514 | +0.001076，2/3 正向 |
| E3 | STFT 8/4 rect | 0.843000 ± 0.001047 | 0.838685 ± 0.001421 | +0.001873，3/3 正向 |
| E4 | E3 + learned frame pos | 0.842282 ± 0.001834 |  | 参见 E4；相对 E3 −0.000718 |
| E5 | E3 + periodic Hann + center | 0.843027 ± 0.000593 | 0.838083 ± 0.002720 | 相对 E3 +0.000027 |



---

## E1：global FFT 前加入 Hann

**Q（问题）：** global FFT 默认将有限长度序列周期性延拓，序列首尾不连续可能引起频谱泄漏；在输入前加 Hann 是否改善 STSB 表现？

**M（方法）：** 以相同 seed 的 FLaG 为对照，仅在 FFT 前加窗。根据每个样本的 attention mask 得到有效长度 `L`，在这些位置使用 symmetric Hann（`periodic=False`）；padding 仍为零。对于 `L>2`：

$$
w[n]=\frac{1}{2}\left(1-\cos\frac{2\pi n}{L-1}\right),\quad n=0,\ldots,L-1.
$$

窗函数逐位置乘在 RoBERTa hidden 上（**有效特殊 token 也参与加窗**），随后仍按 batch padded 长度做**整句** global rFFT。原实现对 `L≤2` 使用全 1 窗口。除加窗外，gate、逆变换、池化、训练和 seed 全部不变，**没有固定 FFT 长度**。

**R（结果）：** 3-seed test Spearman `0.837019±0.002763`，低于配对 FLaG 的 `0.841128±0.002177`（平均 −0.004109）；Pearson `0.836682±0.000282`，与 FLaG `0.836765±0.001782` 很接近。

**C（结论）：** 当前加 Hann 的实现没有提升 STSB 性能，但**不能推断频谱泄漏不存在**。加窗同时改变了靠近句子首尾的 token hidden，泄漏变化与输入信息减弱并未分离。

## E2：第一次引入 local STFT（16/8）

**Q：** 如果整句的频域表示不够适合语义信息，改为相互重叠的局部频谱是否带来变化？

**M：** 不加 Hann，采用 rectangular frame，`win_length=16`、`hop_length=8`、`center=False`，相邻局部帧重叠 8 token；有效句子尾部不足一帧时在局部完成零填充。每帧做 16 点 rFFT，展平有效帧的频率 token 供共享 latent attention 产生**一个**句级 gate；对各帧频谱施加 gate 后逐帧逆变换，并按实现的重叠处理恢复 token 序列。其余配置与 FLaG 一致（0/True，3 seeds）。

**R：** 3-seed **test** Spearman `0.842204±0.001824`、Pearson `0.837783±0.003514`。相对同 seed FLaG 的平均 Spearman 差为 `+0.001076`，2/3 seeds 正向。

**C：** 看到小幅探索性正向信号

## E3：缩短局部窗口（8/4）

**Q：** `win=16` 时，部分短句只生成一个有效帧；缩小窗口能否让短句也有更细的局部频谱？

**M：** 在 E2 的 rect、非居中、单个句级 gate 基础上，改为 `win=8,hop=4`，相邻帧重叠 4 token。其余协议保持不变，仍为 **3 seeds（0–2）**。

**R：** 3-seed test Spearman `0.843000±0.001047`，Pearson `0.838685±0.001421`；配对 FLaG 的 Spearman 差为 `+0.002003,+0.000591,+0.003024`（均值 `+0.001873`，3/3 正向）。相对 E2 的平均 Spearman 为 `+0.000796`。

**C：** 在这三个 seeds 上有小幅提高，因此后续暂用 E3 开展机制探针。该结果不是 10-seed 显著性结论，也不能仅凭分数认定模型一定利用了更多局部结构。

## E4：局部帧的位置编码

**Q：** E3 将 frame 频率 token 展平后送入 attention，没有额外的帧位置信息；加入 learned frame position 是否有帮助？

**M：** 在 E3 上启用 `use_frame_positional_encoding=True`（实现中 `max_frame_positions=32`），按 frame id 将**可学习帧位置向量加在进入 attention 的频率 token 上**；局部窗、hop、重建和其他设置不变。与 E3 同样的 seeds 0–2 重新训练。位置编码改变了参数化，因此这是新的训练实验，不是给 E3 checkpoint 直接附加位置编码。

**R：** 3-seed test Spearman `0.842282±0.001834`，相对 E3 的均值为 `−0.000718`；

**C：** **这种**可学习帧位置编码未观察到收益，后续候选不使用它；

## E5：周期 Hann + centered STFT

**Q：** 即使局部 STFT 已引入局部边界，各帧的边界不连续仍可能发生；加入窗口是否进一步影响表现？

**M：** 在 E3 `win=8,hop=4` 的基础上，将 rect 改为 **periodic Hann**，同时将 `center=False` 改为 **`center=True`**（帧提取前在序列左右加入 `win/2` 零填充）。保持单句级 gate、0/True 和 seeds 0–2。

**R：** 3-seed test Spearman `0.843027±0.000593`、Pearson `0.838083±0.002720`。与 E3 Spearman 均值相差 `+0.000027`，几乎持平。

**C：** 没有看到这一组合带来有意义的性能提升。为了降低结构复杂度，后续优先采用 rectangular、非居中的局部实现；

---

## E6：对最终层 hidden 做 DCT 频带 knockout

**Q：** E3 的早期分数变化是否伴随对不同输入频段的不同依赖？低频是否更重要？

**M：** **不重新训练**。分别读取 seeds 0–2 的 FLaG 与 E3 checkpoint，并在相同 STSB official test 句对上运行 RoBERTa。对最后一层 hidden 中**内容 token**（排除首尾特殊 token）沿 token 位置做 **DCT-II**；用 8 个按几何方式分配宽度的频带 `B0…B7`（代码设 `num_bands=8,base=4.0`，先给每带至少一个系数，再按 `4^b` 的权重分配其余系数并用最大余数法保证数量总和；`B0` 含 DC/最低频，后续频带频率逐渐提高），一次仅将其中一带的全部 hidden 通道 DCT 系数置零，逆 DCT 恢复内容 token hidden，特殊 token 不变，然后用**原模型的 pooling** 重新计算句对 cosine。

只有**句对双方各至少 8 个内容 token** 才进入本探针的八频带分析。按这一 **eligible test 子集** 的预测，逐 seed 计算：

$$
\Delta\rho_b=\rho_{\mathrm{baseline,eligible}}-\rho_{\mathrm{knockout}(b),eligible}.
$$

另外保存每个句对 knockout 前后的 **绝对预测变化**和**绝对平方误差变化**，它们与 Spearman 降幅是不同统计量。

**R：** 已记录的 B0 平均 Spearman 降幅：FLaG `0.146694`，E3 `0.198943`，均明显；但 E3 相对 FLaG 的低频敏感性差异随 seed/子集而变化，其余频带差异没有构成稳健解释。

**C：** B0 是两个模型的重要输入频带。这个实验不足以证明“E3 性能变化由更偏好 DC 造成”；频带扰动的重要性和 attention 分配要分开解释。

## E7：逐内容 token hidden knockout

**Q：** E3 是否比 global FLaG 更重视特定位置的 token，从而表现出不同的句内位置敏感性？

**M：** 使用同一组 **3 个 seeds（0–2）** 的已训练 FLaG/E3，在 STSB **test** 上做推理探针，不更新参数。对于每一个句子的每个 **content token**，单独将其**最后层 hidden 向量全部置零**，保留特殊 token 与原有效 attention mask、其他 token 不变；重新做 pooling，再与句对另一侧**未扰动**的 embedding 算 cosine。

设原预测 `s`、扰动后 `s'`、标签 `y/5`。每个位置记录 `|s'-s|` 与 `|(s'-y/5)^2-(s-y/5)^2|`。下表的**位置响应**为后者：先对同一句、同位置在各 seed 的响应对齐求平均，然后统计句内位置的均值、最大值与总体标准差，最后跨句求平均。将 content tokens 的**相对位置**划为前、中、后等五段，观察位置曲线。

**R：** 以下为报告中保留的 **seeds 0–2 聚合、STSB test** 结果，单位为**绝对平方误差变化：

| 位置响应统计 | FLaG | E3 |
| --- | ---: | ---: |
| 句内平均响应的跨句均值 | 0.003248 | 0.003155 |
| 句内最大响应的跨句均值 | 0.007714 | 0.007616 |
| 句内位置响应标准差的跨句均值 | 0.00195551 | 0.00195521 |

| Content-token 相对位置区域 | FLaG | E3 |
| --- | ---: | ---: |
| 开头段 | 0.003000 | 0.002850 |
| 前部 | 0.002771 | 0.002790 |
| 中部 | 0.002786 | 0.002719 |
| 后部 | 0.002843 | 0.002780 |
| 末尾段 | 0.002256 | 0.002146 |

**C：** E3 与 FLaG 在该扰动下的位置响应统计十分接近，没有证据支持 E3 因为更强的 token 位置选择性而取得早期小幅性能差异。

## E8：DC attention“偏好比”是什么，怎么算？

**Q：** E6 发现两个模型对低频 B0 的 knockout 敏感，那么 E3 的**频域 latent attention** 是否也更偏向 DC token？

**M：** 使用 FLaG/E3 同一 3 个 seeds 的已训练 checkpoint，在 STSB test 运行正常 forward，读取缓存的 spectral tokens 和 latent cross-attention weights。若权重保留 attention heads，先平均 heads；再在 8 个 latent queries 上平均，得到每个有效频率 token 的权重 `A_k`。Global FLaG 的 `k=0` 是唯一 DC token；E3 将**各有效 frame 的 `k=0`** 都作为 DC token，屏蔽无效 frame。

**不能直接比较 DC 的总 attention 质量**：E3 可能有多个 DC tokens。代码先在两个集合内部求**每 token 平均注意力**，对每个句子计算偏好比：

$$
r_{\mathrm{DC}} =
\frac{
    \displaystyle \frac{1}{N_{\mathrm{DC}}}
    \sum_{k\in\mathrm{DC}}A_k
}{
    \displaystyle \frac{1}{N_{\mathrm{nonDC}}}
    \sum_{k\in\mathrm{nonDC}}A_k + 10^{-12}
}.
$$

其中 `N_DC` / `N_nonDC` 是该句**有效频率 token 的数量**。**r>1**：平均每个 DC token 的注意力超过平均每个非 DC token；**r≈1**：两组平均分配接近。对所有 test 句子先各算一个比值，再在各 seed 内对句子求平均。它既不是 DC 频谱**能量**，也不是 E6 的频带**性能重要性**。

**R：** 下表为 **3 个 seeds（0–2）的句均 DC attention 偏好比**，不是相关系数：

| Seed | Global FLaG | E3（8/4 STFT） |
| ---: | ---: | ---: |
| 0 | 2.602 | 1.305 |
| 1 | 1.586 | 1.213 |
| 2 | 2.553 | 1.625 |

**C：** 三个 seeds 上，E3 的 DC **相对 attention 比值**低于 FLaG。该结果与 E6 的“某些设置下 E3 的 B0 knockout 更敏感”**不是同一件事**：一个测 attention 分配，一个测删频带后预测性能；也不能直接对全局 DC 与局部 frame DC 赋予完全相同的频率语义。

---

