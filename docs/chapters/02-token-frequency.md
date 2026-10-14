# STFT 的频率和 Token 响应

在已训练模型上比较频带删除、逐 Token 置零和 DC Attention 权重，三个指标分别对应不同的测试。

## 实验总结

<a id="summary-06"></a>

### 实验六（E6）：Global FLaG 和 STFT-FLaG 分别依赖哪些频率？

- **实验目的：** 探究局部 FFT 是否改变了模型对低频和高频信息的依赖。
- **实验思路：** 固定训练好的模型，对 RoBERTa 最后一层内容 token 的特征进行 DCT-II 变换，将频率划成八个频带，每次删除一个频带，观察 Spearman 下降多少。
- **实验结果：** 删除最低频带 B0 的影响最大。FLaG 的平均 Spearman 降幅为 0.146694，E3 为 0.198943。
- **实验结论：** 删除 B0 对两种模型的影响都最大，其中 E3 的 Spearman 降幅更大，表明它在这项频带删除测试中更依赖最低频部分。

[查看本实验详细论述](#detail-06)

<a id="summary-07"></a>

### 实验七（E7）：局部 FFT 是否让模型更关注某些 token？

- **实验目的：** 验证 STFT-FLaG 是否比全局 FLaG 更重视句子中的特定位置。
- **实验思路：** 固定训练好的模型，每次将一个内容 token 的隐藏向量置零，观察句对预测平方误差发生多大变化。
- **实验结果：** 两种模型的 token 扰动响应非常接近，句内平均响应分别约为 0.003248 和 0.003155。
- **实验结论：** Global FLaG 与 E3 对 token 位置置零的响应接近，没有观察到局部 FFT 带来更强的特定位置依赖。

[查看本实验详细论述](#detail-07)

<a id="summary-08"></a>

### 实验八（E8）：局部 FFT 是否更关注零频率？

- **实验目的：** 比较全局和局部频域模型的注意力，是否更加集中于 DC。

- **实验思路：** 读取 latent attention 权重，比较 DC 频率位置与非 DC 频率位置的平均每位置注意力大小。

- **实验结果：** 三个种子中，E3 的 DC 注意力偏好比都比 Global FLaG 低。

- **实验结论：** E3 给 DC 位置分配的平均注意力相对更低

[查看本实验详细论述](#detail-08)

---

## 实验详细论述

<a id="detail-06"></a>

### 实验六（E6）：Global FLaG 和 STFT-FLaG 分别依赖哪些频率？

**实验设置与评价：** STSB · test 子集 · 3 seeds（0–2）· 固定 checkpoint 后频带删除。

#### 对照问题

FLaG 与 E3 分别依赖哪些输入频带？

#### 实验实施

固定两组已训练 checkpoint（**seeds 0–2**）。对 RoBERTa **最后一层的内容 token hidden** 沿位置做 DCT-II，按几何比例划分为 **8 个频带 B0–B7**（B0 为最低频）；每次只将一段的系数置零、逆 DCT，再交给原模型预测。特殊 token 不变。只使用**句对两侧均至少有 8 个内容 token** 的 test 子集。

频带的重要性用 **Spearman 降幅**衡量：

$$
\Delta\rho_b=\rho_{\mathrm{before}}-\rho_{\mathrm{after},b}.
$$

#### 观测结果

B0 引起的平均 Spearman 降幅最大：FLaG **0.146694**，E3 **0.198943**。

#### 结果解读

两个模型都对最低频敏感；E3 与 FLaG 的相对差异随 seed 变化，不能据此确认 E3 的性能变化由低频造成。


B0 是 **DCT-II 的最低频带**，不是单个 exact FFT DC 系数；E6 的差异是删带后的任务指标变化。

[返回实验 6 的总结](#summary-06)

---

<a id="detail-07"></a>

### 实验七（E7）：局部 FFT 是否让模型更关注某些 token？

**实验设置与评价：** STSB · test · 3 seeds · 固定 checkpoint 后逐 token 置零。

#### 对照问题

E3 是否对特定 token 位置更敏感？

#### 实验实施

固定 FLaG/E3 的 **3-seed checkpoint**。逐个把**内容 token 的最后一层 hidden 向量置零**，其他 token、特殊 token 和 mask 不变，重新计算句对 cosine。

令原预测为 `s`，扰动后为 `s'`，标签为 `y`（原始分数除以 5）。对每个位置计算**绝对平方误差变化**：

$$
R_{\text{token}}=\left|(s'-y)^2-(s-y)^2\right|.
$$

先按相同句子、相同 token 位置对齐三个 seed 的响应取平均，再统计句内位置分布。以下数字**不是** Spearman。

#### 观测结果

STSB test，3-seed 聚合：

| 指标 | FLaG | E3 |
| --- | ---: | ---: |
| 句内平均响应的跨句均值 | 0.003248 | 0.003155 |
| 句内最大响应的跨句均值 | 0.007714 | 0.007616 |
| 句内位置响应标准差的跨句均值 | 0.00195551 | 0.00195521 |

#### 结果解读

两组的 token 位置响应接近，没有观察到 E3 更强的位置选择性。


undefined

[返回实验 7 的总结](#summary-07)

---

<a id="detail-08"></a>

### 实验八（E8）：局部 FFT 是否更关注零频率？

**实验设置与评价：** STSB · 3 seeds · 固定 checkpoint 后读取注意力。

注意力偏好比是 DC 位置的平均每位置权重与非 DC 位置平均每位置权重之比。旧报告记录了三个 seed 的比值，均低于 Global；对应值保留在下表。

#### 对照问题

E3 是否比 global FLaG 更偏向 DC（零频）？

#### 实验实施

固定 **3-seed checkpoint**，从正常 forward 中读取 latent attention 权重，先在 latent queries（以及需要时的 attention heads）上取平均。FLaG 的全局 `k=0` 是 DC；E3 将**每个有效局部帧**的 `k=0` 都计为 DC。

对每句话，分别计算 DC 与非 DC 的**每 token 平均注意力**，再求比值：

$$
r_{\mathrm{DC}}=
\frac{\left(\sum_{k\in\mathrm{DC}}A_k\right)/N_{\mathrm{DC}}}
{\left(\sum_{k\in\mathrm{nonDC}}A_k\right)/N_{\mathrm{nonDC}}+10^{-12}}.
$$

`r_DC>1` 表示**平均每个 DC token** 的注意力较高。表中的数值为每个 seed 在 **STSB test 句子**上的平均偏好比，不是 DC 能量占比。

#### 观测结果

| Seed | FLaG | E3 |
| ---: | ---: | ---: |
| 0 | 2.602 | 1.305 |
| 1 | 1.586 | 1.213 |
| 2 | 2.553 | 1.625 |

#### 结果解读

E3 在三个 seed 上的 DC attention 偏好比都较低。但 attention 分配与 E6 的 knockout 性能敏感性**不是同一个指标**，不能仅凭偏好比判断 DC 的实际重要性。


Attention 权重分配与频带 knockout 后的 Spearman 降幅不是同一种重要性度量。

[返回实验 8 的总结](#summary-08)

