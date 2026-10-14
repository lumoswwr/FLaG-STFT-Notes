### 实验 STSB E13：固定全局 FFT 长度

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 如果 global FLaG 在训练时就使用固定 FFT 长度，能否改善结果？

### 2. 实验设计与控制条件
 global FLaG 在训练及推理时均设 `fixed_fft_length=128`；不引入 STFT，其他设置不变，**3 seeds（0–2）**。

### 3. 观测结果
 test Spearman **0.841856 ± 0.001000**；相对同 seed 动态 FLaG 的配对差为 **+0.000728 ± 0.003079**，仅 **1/3 正向**。

### 4. 基于本次结果的判断
 固定 FFT 长度没有带来稳定的性能改善。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
