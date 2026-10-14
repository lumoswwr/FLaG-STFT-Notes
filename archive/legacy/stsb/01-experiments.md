# STSB：E1–E8

本页使用 **STSB**。E1–E5 重新训练，使用 **seeds 0、1、2**，结果为 **test Spearman / Pearson（mean ± std）**；E6–E8 则使用这些 seed 的**已训练 FLaG/E3 checkpoint** 做分析，不重新训练。共同配置见[实验设置](../methods.md)。

## E1–E5 配置及结果对照

除表内变化，统一采用 **dropout=0、post-pool Norm=True**，8 latents、4 heads、3 epochs。

| 模型 | 与 global FLaG 的主要区别 | test Spearman ↑ | test Pearson ↑ |
| --- | --- | ---: | ---: |
| FLaG | 整句 FFT | 0.841128 ± 0.002177 | 0.836765 ± 0.001782 |
| **E1** | Global FFT 前加 symmetric Hann | 0.837019 ± 0.002763 | 0.836682 ± 0.000282 |
| **E2** | STFT：rect，win16/hop8 | 0.842204 ± 0.001824 | 0.837783 ± 0.003514 |
| **E3** | STFT：rect，win8/hop4 | 0.843000 ± 0.001047 | 0.838685 ± 0.001421 |
| **E4** | E3 + learned frame position | 0.842282 ± 0.001834 | 未汇总 |
| **E5** | E3 + periodic Hann，center=True | 0.843027 ± 0.000593 | 0.838083 ± 0.002720 |

E2/E3 使用 `center=False`，均只生成**一个句级 gate**；E5 **同时改变了窗函数与 center 设置**，不能将变化单独归因于 Hann。

## E1：Global Hann

**Q：** global FFT 前加 Hann，能否减轻边界影响并改善性能？

**M：** 按**每句话自身有效 token 长度**（包括特殊 token）生成 symmetric Hann，乘到 RoBERTa hidden 上，再执行原 global FFT；padding 仍为零。有效长度 ≤2 时使用全 1 窗口。其他设置与 FLaG 一致，**3 seeds**。

**R：** Spearman `0.837019 ± 0.002763`，比同 seed FLaG 平均 **−0.004109**。

**C：** 未改善性能，但不能据此排除频谱泄漏，因为加窗同时减弱了边界 token 的输入。

## E2：首次改为 STFT（16/8）

**Q：** 把整句 FFT 换成局部 STFT 是否改善结果？

**M：** rect 窗，`win=16,hop=8,center=False`，相邻帧重叠 8 token。所有有效帧的频率 token 共同生成一个句级 gate；逐帧逆变换并重建，**3 seeds**。

**R：** Spearman `0.842204 ± 0.001824`，比 FLaG 平均 **+0.001076**，2/3 seeds 正向。

**C：** 观察到小幅正向结果，继续检查窗口大小。

## E3：缩短窗口（8/4）

**Q：** 部分短句在 win16 下只有一个有效帧，缩短窗口后效果如何？

**M：** 延续 E2 的做法，改为 `win=8,hop=4`，其余不变，**3 seeds**。

**R：** Spearman `0.843000 ± 0.001047`；比 FLaG 平均 **+0.001873**（3/3 正向），比 E2 平均 **+0.000796**。

**C：** 早期观察到小幅改善，因此选择 E3 进行 E6–E8 机制分析；3 seeds 不能证明稳定优势。

## E4：添加帧位置编码

**Q：** 局部帧的位置顺序信息是否有帮助？

**M：** 在 E3 的频率 token 进入 latent attention 前，加 **learned frame positional encoding**；窗长及其他配置不变，重新训练 **3 seeds**。

**R：** Spearman `0.842282 ± 0.001834`，比 E3 平均 **−0.000718**。

**C：** 这种帧位置编码没有带来改善，后续不采用。

## E5：局部 Hann

**Q：** 给局部帧加窗是否继续改善 E3？

**M：** 在 E3 的 win8/hop4 上，将 rect 换成 **periodic Hann**，并同时设 `center=True`；重新训练 **3 seeds**。

**R：** Spearman `0.843027 ± 0.000593`，比 E3 平均 **+0.000027**。

**C：** 基本持平，后续优先使用更简单的 rect 实现。本实验未单独拆分 Hann 与 centering 的作用。

---

## E6：频带 Knockout

**Q：** FLaG 与 E3 分别依赖哪些输入频带？

**M：** 固定两组已训练 checkpoint（**seeds 0–2**）。对 RoBERTa **最后一层的内容 token hidden** 沿位置做 DCT-II，按几何比例划分为 **8 个频带 B0–B7**（B0 为最低频）；每次只将一段的系数置零、逆 DCT，再交给原模型预测。特殊 token 不变。只使用**句对两侧均至少有 8 个内容 token** 的 test 子集。

频带的重要性用 **Spearman 降幅**衡量：

$$
\Delta\rho_b=\rho_{\mathrm{before}}-\rho_{\mathrm{after},b}.
$$

**R：** B0 引起的平均 Spearman 降幅最大：FLaG **0.146694**，E3 **0.198943**。

**C：** 两个模型都对最低频敏感；E3 与 FLaG 的相对差异随 seed 变化，不能据此确认 E3 的性能变化由低频造成。

## E7：Token Hidden Knockout

**Q：** E3 是否对特定 token 位置更敏感？

**M：** 固定 FLaG/E3 的 **3-seed checkpoint**。逐个把**内容 token 的最后一层 hidden 向量置零**，其他 token、特殊 token 和 mask 不变，重新计算句对 cosine。

令原预测为 `s`，扰动后为 `s'`，标签为 `y`（原始分数除以 5）。对每个位置计算**绝对平方误差变化**：

$$
R_{\text{token}}=\left|(s'-y)^2-(s-y)^2\right|.
$$

先按相同句子、相同 token 位置对齐三个 seed 的响应取平均，再统计句内位置分布。以下数字**不是** Spearman。

**R：** STSB test，3-seed 聚合：

| 指标 | FLaG | E3 |
| --- | ---: | ---: |
| 句内平均响应的跨句均值 | 0.003248 | 0.003155 |
| 句内最大响应的跨句均值 | 0.007714 | 0.007616 |
| 句内位置响应标准差的跨句均值 | 0.00195551 | 0.00195521 |

**C：** 两组的 token 位置响应接近，没有观察到 E3 更强的位置选择性。

## E8：DC Attention 偏好比

**Q：** E3 是否比 global FLaG 更偏向 DC（零频）？

**M：** 固定 **3-seed checkpoint**，从正常 forward 中读取 latent attention 权重，先在 latent queries（以及需要时的 attention heads）上取平均。FLaG 的全局 `k=0` 是 DC；E3 将**每个有效局部帧**的 `k=0` 都计为 DC。

对每句话，分别计算 DC 与非 DC 的**每 token 平均注意力**，再求比值：

$$
r_{\mathrm{DC}}=
\frac{\left(\sum_{k\in\mathrm{DC}}A_k\right)/N_{\mathrm{DC}}}
{\left(\sum_{k\in\mathrm{nonDC}}A_k\right)/N_{\mathrm{nonDC}}+10^{-12}}.
$$

`r_DC>1` 表示**平均每个 DC token** 的注意力较高。表中的数值为每个 seed 在 **STSB test 句子**上的平均偏好比，不是 DC 能量占比。

**R：**

| Seed | FLaG | E3 |
| ---: | ---: | ---: |
| 0 | 2.602 | 1.305 |
| 1 | 1.586 | 1.213 |
| 2 | 2.553 | 1.625 |

**C：** E3 在三个 seed 上的 DC attention 偏好比都较低。但 attention 分配与 E6 的 knockout 性能敏感性**不是同一个指标**，不能仅凭偏好比判断 DC 的实际重要性。
