### 实验 Sprint S04：匹配配置的 Global / Local 对照

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 去掉 dropout/norm 的配置差异后，E12 还有稳定的 AP 优势吗？

#### 2. 实验设计与控制条件
 FLaG、E12 都使用 **dropout=0、post-pool Norm=True**，其余训练条件一致，比较 **10 seeds（0–9）**。两者主要区别是 global FFT 与 E12 的 STFT 16/16。

#### 3. 观测结果
 **10-seed adaptation-validation AP**：

| 模型 | mean ± std |
| --- | ---: |
| FLaG | **0.795830 ± 0.017156** |
| E12 | **0.789136 ± 0.014094** |
| 配对 E12−FLaG | **−0.006693 ± 0.017279** |

**4/10 seeds** 的差值为正。

#### 4. 基于本次结果的判断
 在匹配配置下没有观察到稳定的 E12 优势，S1/S2 的差异不能直接归因于 STFT 算子。

### 证据解释的范围

这里展示的是 adaptation validation AP，不能与 S1/S2 的 official test AP 直接作数值对比。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/sprint/02-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)

[返回实验导航](index.md)
