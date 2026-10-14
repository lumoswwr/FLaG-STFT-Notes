### 实验 STSB E04：Learned frame position

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 局部帧的位置顺序信息是否有帮助？

### 2. 实验设计与控制条件
 在 E3 的频率 token 进入 latent attention 前，加 **learned frame positional encoding**；窗长及其他配置不变，重新训练 **3 seeds**。

### 3. 观测结果
 Spearman `0.842282 ± 0.001834`，比 E3 平均 **−0.000718**。

### 4. 基于本次结果的判断
 这种帧位置编码没有带来改善，后续不采用。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
