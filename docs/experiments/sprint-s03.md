### 实验 Sprint S03：Sprint Global/Local Gate × Reconstruction

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 在 Sprint 上，global/local 的输出差异主要来自 gate，还是重建路线？

### 2. 实验设计与控制条件
 固定早期 FLaG 和 E12 各 **3 个 checkpoint（0–2）**，在 **adaptation validation** 上交叉使用两种 gate 来源和重建路线：

| 配置 | Gate 来源 | Gate 作用的频谱及重建 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 中 global 频谱只用于生成 gate，实际被门控、逆变换的仍是 **local 频谱**。同一训练来源的一行实验共享同一个 checkpoint，不重新训练。

### 3. 观测结果
 **3-seed validation 平均绝对 cosine 漂移**：

| 权重来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG-trained | 约 0 | **0.032432 ± 0.003114** |
| E12-trained | 约 0 | **0.037477 ± 0.003556** |

**3-seed validation AP 均值**：

| 权重来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| E12-trained | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

### 4. 基于本次结果的判断
 与 STSB 的 P2 一致：只换 gate 几乎不改变输出，差异主要来自**重建路线**；直接换成 local 重建并未提高当前模型的 validation AP。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/sprint/02-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)

[返回实验导航](index.md)
