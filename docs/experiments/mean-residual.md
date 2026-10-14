### 实验 RESIDUAL：Mean/FLaG 直接残差插值

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

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

### 证据解释的范围

本节为 seed-0 探索性结果，不能与后续五种子统一协议的指标拼接。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/mean-compatible-flag.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_mean_residual_flag.sh) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_mean_residual_alpha_sweep.py)

[返回实验导航](index.md)
