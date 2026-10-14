### 实验 STSB E14：论文 Text 设置下的十种子对照

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 在论文 text 配置下，E12 相比 FLaG 的早期性能趋势是否仍存在？

### 2. 实验设计与控制条件
 FLaG 与 E12 都改为 **dropout=0.1、post-pool Norm=True**，其余训练设置匹配，重新训练 **10 seeds（0–9）**。

### 3. 观测结果
 **10-seed test**：

| 模型 | Spearman ↑ | Pearson ↑ |
| --- | ---: | ---: |
| FLaG | 0.839065 ± 0.004154 | 0.834980 ± 0.003942 |
| E12 | 0.838944 ± 0.001882 | 0.834015 ± 0.002119 |
| 配对 E12−FLaG | **−0.000121 ± 0.004039** | **−0.000965 ± 0.003878** |

Spearman **5/10 seeds** 正向。

### 4. 基于本次结果的判断
 两者基本持平。E12 在 0/有 Norm 下的小幅正向均值没有在这组 0.1/有 Norm 实验中保持。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
