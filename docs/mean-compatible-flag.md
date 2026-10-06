# 为什么 FLaG 没有兼顾 Mean：从残差融合到 Alignment-aware

## 1. 问题

这一阶段围绕一个核心问题展开：

> **照理来讲 FLaG 应该能够兼顾 Mean，为什么实际并没有做到？能否让它在特殊情况下严格退化为 Mean？**

现象很直接：在 STS-B 上，Mean 明显优于原始 FLaG；而在 Sprint 上，FLaG 又明显优于 Mean。说明 FLaG 并不是“统一更强”的 pooling，而是会在不同任务中改变原本 Mean 已经保留的信息。

原始 FLaG 的完整路径为：

[
	ext{rFFT}
ightarrow
	ext{latent attention}
ightarrow
	ext{frequency gate}
ightarrow
	ext{iFFT}
ightarrow
	ext{max pooling}
ightarrow
	ext{projection}.
]

即使频域中存在 DC / 平均信息，这条完整路径也没有一个显式约束，能够让最终输出严格回到 Mean pooling。因此，这里的目标不是简单增强 FLaG，而是给它加入一个**可退化到 Mean 的结构性端点**。

---

## 2. 第一步：直接加入 Mean residual

先把 Mean 和 FLaG 输出分别归一化：

[
hat m=operatorname{Norm}(m),qquad
hat f=operatorname{Norm}(f).
]

最直接的做法是学习一个标量 (alpha)：

[
z=
(1-alpha)hat m+alphahat f,
qquad
alphain[0,1].
]

这样：

- (alpha=0)：严格回到 Mean；
- (alpha=1)：严格回到原始 FLaG。

### 探索结果（seed 0）

| Method | STS-B Spearman | Sprint AP | learned (alpha) |
| --- | ---: | ---: | ---: |
| Mean | 0.854182 | 0.428922 | — |
| FLaG | 0.841552 | 0.724596 | — |
| Mean-residual | 0.850597 | 0.878625 | 0.496 / 0.498 |

两个任务中 (alpha) 都稳定落在约 0.5，而不是自动靠近某个端点。进一步扫 (alpha) 后，STS-B 的较优区域约在 0.35，Sprint 约在 0.50。

这一版证明了 **显式保留 Mean 路径是有效的**，尤其 Sprint 提升很大，但仍有一个问题：

> 一个全局标量只能控制“Mean 和 FLaG 各占多少”，无法区分 FLaG 中哪些信息只是与 Mean 重复，哪些是真正新增的信息。

因此下一步不再直接混合整条 FLaG 表示，而是先做几何分解。

---

## 3. 第二步：Mean-anchor + 正交残差

将归一化后的 FLaG 分解到 Mean 方向及其正交方向：

[
a=langle hat m,hat fangle,
]

[
hat f=ahat m+r_perp,
]

其中

[
r_perp=hat f-ahat m.
]

只把 FLaG 中与 Mean **不重复**的部分加回来：

[
z=
operatorname{Norm}
left(
hat m+eta r_perp
ight).
]

这里 (eta=0) 时严格回到 Mean。

### 有界 (eta)

最初将 (eta) 限制在有界范围内。

| Method | STS-B Spearman | Sprint AP | learned (eta) |
| --- | ---: | ---: | ---: |
| Mean-anchor | 0.855106 | 0.638445 | 0.197 / 1.000 |

STS-B 已经略高于 Mean，但 Sprint 的 (eta) 几乎顶到边界，说明模型希望使用更强的正交残差。

### 放开 (eta) 上界

改为直接学习无界 (eta)：

| Method | STS-B Spearman | Sprint AP | learned (eta) |
| --- | ---: | ---: | ---: |
| Mean-anchor, unbounded | 0.855354 | 0.809582 | 0.200 / 13.124 |

Sprint 性能恢复很多，但 (eta) 被放大到 13 左右，暴露出新的问题。

对 Mean 与 FLaG 的几何关系做 probe：

| Task | (cos(hat m,hat f)) | angle | (|r_perp|) |
| --- | ---: | ---: | ---: |
| STS-B | -0.011 | 90.7° | 0.995 |
| Sprint | -0.940 | 160.3° | 0.336 |

STS-B 中两条表示几乎正交，因此保留 Mean 再加入少量正交 residual 很自然。

Sprint 中两条表示却几乎反向。此时如果完全丢掉 FLaG 在 Mean 方向上的分量，只保留 (r_perp)，剩余 residual 很小，只能靠巨大的 (eta) 再把它放大。

因此问题进一步明确：

> **不能简单删除 FLaG 与 Mean 共线的部分。共线部分也可能包含任务相关信息，只是应该根据二者的对齐关系决定如何进入最终表示。**

---

## 4. 第三步：Alignment-aware Mean anchor

由

[
hat f=ahat m+r_perp
]

出发，将 FLaG 与 Mean 对齐的部分直接保留在 Mean 轴上，只学习正交新信息的强度：

[
oxed{
z=
operatorname{Norm}
left(
(1+a)hat m+eta r_perp
ight)
}
]

也可以写成：

[
z=
operatorname{Norm}
left(
hat m+ahat m+eta r_perp
ight).
]

这里：

- (a=langlehat m,hat fangle) 是**每个样本动态计算**的几何量，不是额外学习参数；
- (eta) 是唯一需要学习的融合标量；
- (eta=0) 时，输出方向回到 Mean；
- (eta=1) 时：

[
(1+a)hat m+r_perp
=
hat m+hat f,
]

即退化为归一化后的 Mean + FLaG；
- 当 (aapprox-1) 时，Mean 轴系数 (1+a) 自动减小，可抑制与 FLaG 强冲突的 Mean 方向。

### 初步结果（seed 0）

| Method | STS-B Spearman | Sprint AP | learned (eta) |
| --- | ---: | ---: | ---: |
| Alignment-aware | **0.858201** | **0.880862** | 0.177 / 0.804 |
| Alignment-aware + StopGrad((a)) | 0.855621 | 0.876518 | 0.180 / 0.821 |

StopGrad 版本只阻断 (a) 的梯度，forward 几何关系不变。Sprint 上两者接近，说明主要增益来自 alignment-aware 的前向几何结构；STS-B 中允许 alignment 参与共同优化还能带来额外收益。

因此后续主方法采用普通 Alignment-aware 版本。

---

## 5. 对齐 FLaG 文本实验配置的 5-seed 验证

前面的结果用于推进公式设计。最终比较重新按照 FLaG 文本实验配置统一跑 **5 seeds（0–4）**，只比较：

- Mean
- FLaG
- Alignment-aware

### 共同设置

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

### 任务配置

| Setting | IMDB | STS-B | Sprint |
| --- | --- | --- | --- |
| RoBERTa | unfrozen | unfrozen | frozen |
| Epochs | 3 | 3 | 10 |
| Max length | 512 | 128 | 128 |
| Train batch | 8 | 32 | 32 |
| Eval batch | 16 | 64 | 64 |
| Backbone LR | (1	imes10^{-5}) | (1	imes10^{-5}) | frozen |
| Pool / head LR | (1	imes10^{-3}) | (1	imes10^{-3}) | (1	imes10^{-3}) |
| Loss | CrossEntropy | MSE | BCEWithLogits |
| Checkpoint | val Accuracy | val Spearman | val Accuracy |

IMDB 使用官方 train 的 90:10 train/validation 划分，官方 test 仅用于最终评估；STS-B 使用官方 development 作为 validation；Sprint 使用冻结 backbone 的 adaptation protocol，official test 在 checkpoint 选定后评估。

---

## 6. 最终性能

结果为 **mean ± sample std**。指标排列沿用 FLaG 论文文本任务表格的形式。

| Method | IMDB Acc ↑ | IMDB F1 ↑ | Sprint Acc ↑ | Sprint AP ↑ | Sprint F1 ↑ | STS-B Spearman ↑ | STS-B Pearson ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean | 0.9525 ± 0.0016 | 0.9528 ± 0.0017 | 0.9901 ± 0.0000 | 0.4289 ± 0.0000 | 0.0000 ± 0.0000 | 0.8521 ± 0.0025 | 0.8567 ± 0.0018 |
| FLaG | 0.9521 ± 0.0014 | 0.9522 ± 0.0014 | 0.9915 ± 0.0021 | 0.5976 ± 0.0950 | 0.5724 ± 0.0727 | 0.8370 ± 0.0045 | 0.8325 ± 0.0025 |
| **Alignment-aware** | **0.9526 ± 0.0013** | **0.9530 ± 0.0012** | **0.9934 ± 0.0013** | **0.6989 ± 0.0503** | **0.6567 ± 0.0441** | **0.8536 ± 0.0013** | **0.8580 ± 0.0022** |

对应的 learned (eta)：

| Task | (eta) |
| --- | ---: |
| IMDB | 0.0005 ± 0.0078 |
| STS-B | 0.1856 ± 0.0025 |
| Sprint | 1.0821 ± 0.0615 |

最终结果与设计目标一致：

- **IMDB**：(etaapprox0)，Alignment-aware 基本退回 Mean，三者性能接近；
- **STS-B**：原始 FLaG 明显弱于 Mean，Alignment-aware 保住 Mean 并略有提升；
- **Sprint**：FLaG 本身有效，Alignment-aware 进一步利用其补充信息，AP 从 0.5976 提升到 0.6989。

因此，这一系列实验最终把问题从“如何把 Mean 混进 FLaG”收敛为：

> **显式保留 Mean endpoint，并只学习 FLaG 相对 Mean 的有效补充信息。**
