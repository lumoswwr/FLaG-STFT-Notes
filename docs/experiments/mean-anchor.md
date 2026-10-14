### 实验 ANCHOR：正交 Mean-anchor 与无界 Beta

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 3. 第二步：Mean-anchor 与正交残差

将归一化后的 FLaG 表示分解为 Mean 方向上的分量和与 Mean 正交的分量。

首先计算二者的 cosine alignment：

\[
a
=
\langle
\hat{m},
\hat{f}
\rangle
\]

于是：

\[
\hat{f}
=
a\hat{m}
+
r_{\perp}
\]

其中：

\[
r_{\perp}
=
\hat{f}
-
a\hat{m}
\]

满足：

\[
\langle
\hat{m},
r_{\perp}
\rangle
=
0
\]

因此 \(r_{\perp}\) 可以理解为：

> **FLaG 中不能由 Mean 方向解释的补充信息。**

最初的 Mean-anchor 方案只保留这一部分：

\[
z
=
\operatorname{Norm}
\left(
\hat{m}
+
\beta r_{\perp}
\right)
\]

此时：

\[
\beta=0
\quad\Rightarrow\quad
z=\hat{m}
\]

所以这一结构同样具有严格的 Mean endpoint。

#### 3.1 有界 \(\beta\)

最初将 \(\beta\) 限制在有界范围内。

| Method      | STS-B Spearman | Sprint AP | learned \(\beta\) |
| ----------- | -------------: | --------: | ----------------: |
| Mean-anchor |       0.855106 |  0.638445 |     0.197 / 1.000 |

STS-B 上已经略高于 Mean。

但在 Sprint 上，\(\beta\) 几乎直接到达上界 1，说明模型仍然希望进一步增强正交 residual。

这提示：

> Sprint 上并不是“不需要 FLaG”，而是当前结构对 FLaG 信息的利用受到了限制。

#### 3.2 放开 \(\beta\) 上界

随后改为直接学习无界 \(\beta\)。

| Method                 | STS-B Spearman | Sprint AP | learned \(\beta\) |
| ---------------------- | -------------: | --------: | ----------------: |
| Mean-anchor, unbounded |       0.855354 |  0.809582 |    0.200 / 13.124 |

Sprint AP 从 0.638 提升到 0.810，但 \(\beta\) 被放大到约 13。

这说明问题并不是简单的“\(\beta\) 上限太小”，而是当前分解方式本身可能丢掉了 Sprint 所需要的信息。

---

### 4. Mean 与 FLaG 的几何关系

为了理解为什么 Sprint 需要极大的 \(\beta\)，进一步测量 Mean 与 FLaG 表示之间的几何关系。

| Task   | \(\cos(\hat{m},\hat{f})\) |  Angle | \(\|r_{\perp}\|\) |
| ------ | ------------------------: | -----: | ----------------: |
| STS-B  |                    -0.011 |  90.7° |             0.995 |
| Sprint |                    -0.940 | 160.3° |             0.336 |

两种任务呈现出完全不同的几何状态。

#### STS-B

STS-B 中：

\[
a\approx0
\]

即：

\[
\hat{m}
\perp
\hat{f}
\]

Mean 与 FLaG 几乎正交，因此：

\[
\|r_{\perp}\|
\approx1
\]

此时以 Mean 为 anchor，再加入少量正交 FLaG 信息是比较自然的。

#### Sprint

Sprint 中：

\[
a\approx-0.94
\]

Mean 与 FLaG 几乎反向。

同时：

\[
\|r_{\perp}\|
\approx0.336
\]

说明 FLaG 的大部分信息实际上位于 Mean 轴方向上，只不过方向与 Mean 相反。

如果使用：

\[
z
=
\operatorname{Norm}
\left(
\hat{m}
+
\beta r_{\perp}
\right)
\]

那么 FLaG 在 Mean 轴上的分量：

\[
a\hat{m}
\]

被完全删除，只剩下较小的 \(r_{\perp}\)。

在这组训练中，Sprint 学得的参数为：

\[
\beta\approx13
\]

才能重新放大剩余的正交信息。

因此问题进一步明确：

> **不能简单删除 FLaG 与 Mean 共线的部分。共线部分本身也可能包含任务相关信息，只是它应该根据 Mean 与 FLaG 的对齐关系进入最终表示。**

---

### 证据解释的范围

模型几何上的 a 和 r⊥ 描述该 checkpoint 下的样本表示；不能直接将单句夹角解释为分类性能因果机制。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/mean-compatible-flag.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_mean_anchor_flag.sh) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_mean_anchor_unbounded.sh) · [代码与记录 3](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_mean_anchor_geometry.py)

[返回实验导航](index.md)
