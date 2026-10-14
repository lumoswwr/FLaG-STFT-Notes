### 实验 Sprint S02：Sprint 十种子扩展

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 S1 的结果是否在更多随机种子下保持？

### 2. 实验设计与控制条件
 延续 S1 两种实际配置，扩至 **10 seeds（0–9）**，其他协议不变。

### 3. 观测结果
 **10-seed official-test（mean ± std）**：

| 指标 ↑ | FLaG（0.1/无 Norm） | E12（0/有 Norm） |
| --- | ---: | ---: |
| AP | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 |
| Accuracy | 0.993857 ± 0.000275 | 0.994485 ± 0.000268 |
| F1 | 0.650671 ± 0.016939 | 0.685348 ± 0.020150 |
| Precision | 0.745202 ± 0.033959 | 0.814274 ± 0.032556 |
| Recall | 0.579800 ± 0.037446 | 0.594100 ± 0.041906 |

配对 E12−FLaG 的 AP 差 **+0.048222 ± 0.017641**，**10/10 seeds 正向**。

### 4. 基于本次结果的判断
 两套**整体配置**之间的 AP 差在 10 seeds 中保持，但仍不能单独归因于局部 STFT。

### 证据解释的范围

该比较中 FLaG 和 E12 的 dropout 与 post-pool Norm 不同。因此观察到的是两套完整配置之间的差别，不是 STFT 算子的独立效果。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/sprint/02-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)

[返回实验导航](index.md)
