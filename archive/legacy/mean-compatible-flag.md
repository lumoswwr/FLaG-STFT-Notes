# 为什么 FLaG 没有兼顾 Mean：从残差融合到 Alignment-aware

## 1. 问题

这一阶段围绕一个核心问题展开：

> **照理来讲 FLaG 应该能够兼顾 Mean，为什么实际并没有做到？能否让它在特殊情况下严格退化为 Mean？**

现象很直接：

- 在 STS-B 上，Mean 明显优于原始 FLaG；
- 在 Sprint 上，FLaG 又明显优于 Mean。

这说明 FLaG 并不是在所有任务上都比 Mean 更强，而是会改变 Mean 原本已经保留的信息。

原始 FLaG 的完整路径为：

\[
\text{rFFT}
\rightarrow
\text{latent attention}
\rightarrow
\text{frequency gate}
\rightarrow
\text{iFFT}
\rightarrow
\text{max pooling}
\rightarrow
\text{projection}
\]

即使频域中包含 DC / 平均信息，这条完整路径也没有显式约束能够保证最终输出严格回到 Mean pooling。

因此这里的目标不是简单增强 FLaG，而是：

> **给 FLaG 加入一个显式的 Mean endpoint，使模型在 FLaG 无益时能够退化为 Mean，在 FLaG 有益时继续利用其补充信息。**

---

## 2. 第一步：直接加入 Mean residual

先将 Mean 和 FLaG 的输出分别归一化：

\[
\hat{m}
=
\operatorname{Norm}(m),
\qquad
\hat{f}
=
\operatorname{Norm}(f)
\]

最直接的想法是学习一个标量 \(\alpha\)，在两种表示之间进行插值：

\[
z
=
(1-\alpha)\hat{m}
+
\alpha\hat{f},
\qquad
\alpha\in[0,1]
\]

这样具有两个明确端点：

\[
\alpha=0
\quad\Rightarrow\quad
z=\hat{m}
\]

即严格退化为 Mean；

\[
\alpha=1
\quad\Rightarrow\quad
z=\hat{f}
\]

即恢复原始 FLaG。

### 探索结果（seed 0）

| Method        | STS-B Spearman | Sprint AP | learned \(\alpha\) |
| ------------- | -------------: | --------: | -----------------: |
| Mean          |       0.854182 |  0.428922 |                  - |
| FLaG          |       0.841552 |  0.724596 |                  - |
| Mean-residual |       0.850597 |  0.878625 |      0.496 / 0.498 |

这里最后一列依次为 STS-B / Sprint。

两个任务中学习到的 \(\alpha\) 都稳定在约 0.5，并没有自动靠近 Mean 或 FLaG 的某一个端点。

进一步对 \(\alpha\) 进行扫描：

- STS-B validation 上较优区域约为 \(\alpha=0.35\)；
- Sprint validation 上较优区域约为 \(\alpha=0.50\)。

这一结果说明，**显式加入 Mean 路径确实有效**。尤其在 Sprint 上，Mean-residual 相比原始 FLaG 还有明显提升。

但这种方法仍然比较粗糙：

> 一个全局标量 \(\alpha\) 只能决定“Mean 和 FLaG 各占多少”，却无法区分 FLaG 中哪些信息只是 Mean 已经包含的重复信息，哪些才是真正新增的信息。

因此下一步不再直接混合完整的 FLaG 表示，而是分析 Mean 与 FLaG 之间的几何关系。

---

## 3. 第二步：Mean-anchor 与正交残差

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

### 3.1 有界 \(\beta\)

最初将 \(\beta\) 限制在有界范围内。

| Method      | STS-B Spearman | Sprint AP | learned \(\beta\) |
| ----------- | -------------: | --------: | ----------------: |
| Mean-anchor |       0.855106 |  0.638445 |     0.197 / 1.000 |

STS-B 上已经略高于 Mean。

但在 Sprint 上，\(\beta\) 几乎直接到达上界 1，说明模型仍然希望进一步增强正交 residual。

这提示：

> Sprint 上并不是“不需要 FLaG”，而是当前结构对 FLaG 信息的利用受到了限制。

### 3.2 放开 \(\beta\) 上界

随后改为直接学习无界 \(\beta\)。

| Method                 | STS-B Spearman | Sprint AP | learned \(\beta\) |
| ---------------------- | -------------: | --------: | ----------------: |
| Mean-anchor, unbounded |       0.855354 |  0.809582 |    0.200 / 13.124 |

Sprint AP 从 0.638 提升到 0.810，但 \(\beta\) 被放大到约 13。

这说明问题并不是简单的“\(\beta\) 上限太小”，而是当前分解方式本身可能丢掉了 Sprint 所需要的信息。

---

## 4. Mean 与 FLaG 的几何关系

为了理解为什么 Sprint 需要极大的 \(\beta\)，进一步测量 Mean 与 FLaG 表示之间的几何关系。

| Task   | \(\cos(\hat{m},\hat{f})\) |  Angle | \(\|r_{\perp}\|\) |
| ------ | ------------------------: | -----: | ----------------: |
| STS-B  |                    -0.011 |  90.7° |             0.995 |
| Sprint |                    -0.940 | 160.3° |             0.336 |

两种任务呈现出完全不同的几何状态。

### STS-B

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

### Sprint

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

这就解释了为什么 Sprint 中需要：

\[
\beta\approx13
\]

才能重新放大剩余的正交信息。

因此问题进一步明确：

> **不能简单删除 FLaG 与 Mean 共线的部分。共线部分本身也可能包含任务相关信息，只是它应该根据 Mean 与 FLaG 的对齐关系进入最终表示。**

---

## 5. 第三步：Alignment-aware Mean anchor

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

## 6. Alignment-aware 初步验证

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

## 7. 对齐 FLaG 文本实验配置的 5-seed 验证

前面的实验用于一步步确定最终公式。

确定 Alignment-aware 后，重新按照 FLaG 文本实验配置，在三个文本任务上统一进行 5-seed 实验：

- Mean
- FLaG
- Alignment-aware

随机种子为：

\[
0,\ 1,\ 2,\ 3,\ 4
\]

### 共同配置

- backbone：RoBERTa-base
- FLaG latent queries：8
- FLaG attention heads：4
- time pooling：masked max
- residual gate：开启
- FLaG dropout：0
- post-pooling LayerNorm：开启
- optimizer：Adam
- weight decay：0
- scheduler：无

### 三个任务的具体配置

| Setting        | IMDB               | STS-B              | Sprint             |
| -------------- | ------------------ | ------------------ | ------------------ |
| RoBERTa        | unfrozen           | unfrozen           | frozen             |
| Epochs         | 3                  | 3                  | 10                 |
| Max length     | 512                | 128                | 128                |
| Train batch    | 8                  | 32                 | 32                 |
| Eval batch     | 16                 | 64                 | 64                 |
| Backbone LR    | \(1\times10^{-5}\) | \(1\times10^{-5}\) | frozen             |
| Pool / head LR | \(1\times10^{-3}\) | \(1\times10^{-3}\) | \(1\times10^{-3}\) |
| Loss           | CrossEntropy       | MSE                | BCEWithLogits      |
| Checkpoint     | val Accuracy       | val Spearman       | val Accuracy       |

IMDB 使用官方 train 的 90:10 train / validation 划分，官方 test 只用于最终测试。

STS-B 使用官方 development split 作为 validation。

Sprint 使用冻结 RoBERTa 的 adaptation protocol，official test 在 checkpoint 选定后评估。

---

## 8. 最终性能

结果均为：

\[
\text{mean}\pm\text{sample std}
\]

指标排列沿用 FLaG 论文中文本任务表格的形式。

| Method              |          IMDB Acc ↑ |           IMDB F1 ↑ |        Sprint Acc ↑ |         Sprint AP ↑ |         Sprint F1 ↑ |    STS-B Spearman ↑ |     STS-B Pearson ↑ |
| ------------------- | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: |
| Mean                |     0.9525 ± 0.0016 |     0.9528 ± 0.0017 |     0.9901 ± 0.0000 |     0.4289 ± 0.0000 |     0.0000 ± 0.0000 |     0.8521 ± 0.0025 |     0.8567 ± 0.0018 |
| FLaG                |     0.9521 ± 0.0014 |     0.9522 ± 0.0014 |     0.9915 ± 0.0021 |     0.5976 ± 0.0950 |     0.5724 ± 0.0727 |     0.8370 ± 0.0045 |     0.8325 ± 0.0025 |
| **Alignment-aware** | **0.9526 ± 0.0013** | **0.9530 ± 0.0012** | **0.9934 ± 0.0013** | **0.6989 ± 0.0503** | **0.6567 ± 0.0441** | **0.8536 ± 0.0013** | **0.8580 ± 0.0022** |

Alignment-aware 最终学习到的 \(\beta\)：

| Task   | learned \(\beta\) |
| ------ | ----------------: |
| IMDB   |   0.0005 ± 0.0078 |
| STS-B  |   0.1856 ± 0.0025 |
| Sprint |   1.0821 ± 0.0615 |

三个任务呈现出不同的使用模式。

### IMDB

\[
\beta
\approx
0
\]

Alignment-aware 几乎关闭 FLaG 的正交 residual，最终接近 Mean endpoint。

对应性能：

\[
0.9525
\rightarrow
0.9526
\]

Mean、FLaG 和 Alignment-aware 在 IMDB 上总体接近。

### STS-B

原始 FLaG：

\[
0.8370
\]

明显低于 Mean：

\[
0.8521
\]

Alignment-aware 最终达到：

\[
0.8536
\]

说明在 FLaG 本身不适合该任务时，显式 Mean anchor 能够避免原始 FLaG 对 Mean 信息的破坏，同时保留少量有用的补充信息。

### Sprint

Sprint 上 FLaG 本身已经优于 Mean：

\[
\text{AP}:
\quad
0.4289
\rightarrow
0.5976
\]

Alignment-aware 进一步达到：

\[
\text{AP}
=
0.6989
\]

这里学习到：

\[
\beta
=
1.0821
\]

说明 Sprint 需要较强的 FLaG 补充信息，而不是退化回 Mean。

---

## 9. 当前结论

这一系列实验把最初的问题：

> **“如何让 FLaG 兼顾 Mean？”**

逐步收敛为：

> **显式保留 Mean endpoint，并根据 Mean 与 FLaG 的几何关系，只学习 FLaG 相对 Mean 的有效补充信息。**

直接 Mean/FLaG interpolation 虽然提供了 Mean endpoint，但不能区分重复信息与新增信息。

单纯正交 residual 虽然解决了信息重复问题，却会在 Mean 与 FLaG 强反向时错误删除重要的共线分量。

Alignment-aware 最终同时保留：

\[
a\hat{m}
\]

所描述的对齐关系，以及：

\[
r_{\perp}
\]

所描述的新增方向，从而得到：

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

它既包含明确的 Mean endpoint，又能够在 FLaG 有效的任务上继续利用 FLaG 提供的补充表示。
