### 实验 ALIGNMENT：Alignment-aware 结构与 StopGrad 消融

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

从 FLaG 的几何分解出发：

\[
\hat{f}
=
a\hat{m}
+
r_{\perp}
\]

其中：

\[
a
=
\langle
\hat{m},
\hat{f}
\rangle
\]

不再直接删除 \(a\hat{m}\)，而是将它作为对 Mean 方向的动态修正。

最终得到：

\[
\boxed{
z
=
\operatorname{Norm}
\left(
(1+a)\hat{m}
+
\beta r_{\perp}
\right)
}
\]

也可以写成：

\[
z
=
\operatorname{Norm}
\left(
\hat{m}
+
a\hat{m}
+
\beta r_{\perp}
\right)
\]

这里：

- \(a\) 是每个样本动态计算得到的几何量；
- \(\beta\) 是唯一需要学习的融合标量；
- \(a\) 控制 Mean 与 FLaG 在共享方向上的关系；
- \(\beta\) 控制 FLaG 中正交补充信息的强度。

### 特殊情况

当：

\[
\beta=0
\]

有：

\[
z
=
\operatorname{Norm}
\left(
(1+a)\hat{m}
\right)
\]

只要 \(a>-1\)，其方向就是：

\[
z=\hat{m}
\]

因此模型可以严格退化到 Mean endpoint。

当：

\[
\beta=1
\]

有：

\[
(1+a)\hat{m}
+
r_{\perp}
\]

代入：

\[
\hat{f}
=
a\hat{m}
+
r_{\perp}
\]

得到：

\[
(1+a)\hat{m}
+
r_{\perp}
=
\hat{m}
+
\hat{f}
\]

因此：

\[
\boxed{
\beta=1
\quad\Rightarrow\quad
z
=
\operatorname{Norm}
\left(
\hat{m}
+
\hat{f}
\right)
}
\]

另外，当：

\[
a\approx-1
\]

Mean 轴上的系数：

\[
1+a
\]

会自动趋近于 0。

因此，如果 Mean 与 FLaG 在某个样本上严重冲突，模型不会再无条件保护完整的 Mean 方向。

---

在前期 seed 0 实验中：

| Method                            | STS-B Spearman |    Sprint AP | learned \(\beta\) |
| --------------------------------- | -------------: | -----------: | ----------------: |
| Alignment-aware                   |   **0.858201** | **0.880862** |     0.177 / 0.804 |
| Alignment-aware + StopGrad(\(a\)) |       0.855621 |     0.876518 |     0.180 / 0.821 |

StopGrad 版本在 forward 中使用完全相同的：

\[
a
=
\langle
\hat{m},
\hat{f}
\rangle
\]

但阻断通过 \(a\) 返回 Mean / FLaG branch 的梯度。

Sprint 上普通版本与 StopGrad 版本结果接近，说明主要收益确实来自 Alignment-aware 的前向几何结构。

STS-B 上普通版本更高，则说明允许 Mean/FLaG representation 与 alignment 共同优化还能进一步带来收益。

因此后续主方法采用普通 Alignment-aware，而 StopGrad 作为机制验证。

---

### 证据解释的范围

当 β=0 且 a>-1 时，归一化表示与 Mean 方向相同；a=-1 时表达式退化为零向量，该边界需在数学陈述中单独处理。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/mean-compatible-flag.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_alignment_anchor.sh) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_alignment_anchor_stopgrad.sh)

[返回实验导航](index.md)
