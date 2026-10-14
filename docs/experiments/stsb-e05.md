### 实验 STSB E05：局部 Hann 与 centering

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 给局部帧加窗是否继续改善 E3？

### 2. 实验设计与控制条件
 在 E3 的 win8/hop4 上，将 rect 换成 **periodic Hann**，并同时设 `center=True`；重新训练 **3 seeds**。

### 3. 观测结果
 Spearman `0.843027 ± 0.000593`，比 E3 平均 **+0.000027**。

### 4. 基于本次结果的判断
 基本持平，后续优先使用更简单的 rect 实现。本实验未单独拆分 Hann 与 centering 的作用。

---

### 证据解释的范围

E5 同时改变了 Hann 窗口与 center 参数，单独的加窗贡献无法从该比较分离。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
