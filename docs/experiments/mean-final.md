### 实验 FINAL：三项文本任务的统一 5-seed 对照

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

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

### 证据解释的范围

该五种子对照使用特定 Adam/零权重衰减/无调度器协议。不得与之前 STSB 或 Sprint 的 AdamW 结果简单聚合；表中数值仅对应此轮实验条件。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/mean-compatible-flag.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_author_text_5seeds.sh) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_author_text_protocol.py)

[返回实验导航](index.md)
