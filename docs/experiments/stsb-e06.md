### 实验 STSB E06：最终层频带 Knockout

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 FLaG 与 E3 分别依赖哪些输入频带？

### 2. 实验设计与控制条件
 固定两组已训练 checkpoint（**seeds 0–2**）。对 RoBERTa **最后一层的内容 token hidden** 沿位置做 DCT-II，按几何比例划分为 **8 个频带 B0–B7**（B0 为最低频）；每次只将一段的系数置零、逆 DCT，再交给原模型预测。特殊 token 不变。只使用**句对两侧均至少有 8 个内容 token** 的 test 子集。

频带的重要性用 **Spearman 降幅**衡量：

$$
\Delta\rho_b=\rho_{\mathrm{before}}-\rho_{\mathrm{after},b}.
$$

### 3. 观测结果
 B0 引起的平均 Spearman 降幅最大：FLaG **0.146694**，E3 **0.198943**。

### 4. 基于本次结果的判断
 两个模型都对最低频敏感；E3 与 FLaG 的相对差异随 seed 变化，不能据此确认 E3 的性能变化由低频造成。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
