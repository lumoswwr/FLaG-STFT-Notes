### 实验 STSB E08：DC attention 偏好比

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 E3 是否比 global FLaG 更偏向 DC（零频）？

#### 2. 实验设计与控制条件
 固定 **3-seed checkpoint**，从正常 forward 中读取 latent attention 权重，先在 latent queries（以及需要时的 attention heads）上取平均。FLaG 的全局 `k=0` 是 DC；E3 将**每个有效局部帧**的 `k=0` 都计为 DC。

对每句话，分别计算 DC 与非 DC 的**每 token 平均注意力**，再求比值：

$$
r_{\mathrm{DC}}=
\frac{\left(\sum_{k\in\mathrm{DC}}A_k\right)/N_{\mathrm{DC}}}
{\left(\sum_{k\in\mathrm{nonDC}}A_k\right)/N_{\mathrm{nonDC}}+10^{-12}}.
$$

`r_DC>1` 表示**平均每个 DC token** 的注意力较高。表中的数值为每个 seed 在 **STSB test 句子**上的平均偏好比，不是 DC 能量占比。

#### 3. 观测结果


| Seed | FLaG | E3 |
| ---: | ---: | ---: |
| 0 | 2.602 | 1.305 |
| 1 | 1.586 | 1.213 |
| 2 | 2.553 | 1.625 |

#### 4. 基于本次结果的判断
 E3 在三个 seed 上的 DC attention 偏好比都较低。但 attention 分配与 E6 的 knockout 性能敏感性**不是同一个指标**，不能仅凭偏好比判断 DC 的实际重要性。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
