### 实验 STSB E01：Global Hann

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 global FFT 前加 Hann，能否减轻边界影响并改善性能？

### 2. 实验设计与控制条件
 按**每句话自身有效 token 长度**（包括特殊 token）生成 symmetric Hann，乘到 RoBERTa hidden 上，再执行原 global FFT；padding 仍为零。有效长度 ≤2 时使用全 1 窗口。其他设置与 FLaG 一致，**3 seeds**。

### 3. 观测结果
 Spearman `0.837019 ± 0.002763`，比同 seed FLaG 平均 **−0.004109**。

### 4. 基于本次结果的判断
 未改善性能，但不能据此排除频谱泄漏，因为加窗同时减弱了边界 token 的输入。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
