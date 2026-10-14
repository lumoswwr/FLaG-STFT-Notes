### 实验 P2：STSB Global/Local Gate × Reconstruction

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 从 global FLaG 换成 E12，预测差异主要来自 gate 的频谱来源，还是重建路线？

### 2. 实验设计与控制条件
 分别加载 FLaG-trained 和 E12-trained 的 **3 个 checkpoint**。对**同一个 checkpoint**交叉更换 gate 的生成来源及实际重建路线：

| 配置 | Gate 来源 | Gate 应用及重建路线 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 表示 global 频谱**只生成 gate**，随后将 gate 应用于 local 频谱，再进行局部逆变换和帧拼接。此处“重建路线”包含频谱组织、逆变换及局部时域拼接，不只是替换 iFFT 函数。

#### 3.1 预测漂移
 STSB validation，**3-seed 平均绝对 cosine 漂移**：

| 权重来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG-trained | \(2.86\times10^{-7}\) | **0.005770** |
| E12-trained | \(2.14\times10^{-7}\) | **0.005970** |

#### 3.2 性能指标
 同批 validation 的 **3-seed Spearman 均值**：

| 权重来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.867756 | 0.867756 | 0.867590 | 0.867590 |
| E12-trained | 0.867938 | 0.867938 | 0.868096 | 0.868096 |

### 4. 基于本次结果的判断
 两种频谱生成的句级 gate 几乎相同，global/local 预测差异主要来自**重建路线**。但是，重建路线改变并不代表 Spearman 一定提高。

**区别：** P1 改变 **global 内部的 FFT 长度**；P2 改变 **global/local 的分析与重建方式**。

### 证据解释的范围

P2 仅定位前向输出差异的来源。它没有证明 local 重建的性能优势。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/03-mechanism.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_global_local_2x2.py) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
