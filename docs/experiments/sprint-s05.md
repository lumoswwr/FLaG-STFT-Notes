### 实验 Sprint S05：Global FLaG 的 Dropout × Norm 因子实验

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 dropout 与 post-pool Norm 是否影响 Sprint 的 global FLaG？

### 2. 实验设计与控制条件
 **固定 global FFT**，交叉测试 dropout `{0,0.1}` 和 Norm `{无,有}` 四种设置；每组 **10 seeds（0–9）**，其余训练条件一致。

### 3. 观测结果
 **10-seed adaptation-validation AP**：

| Dropout | Norm | AP（mean ± std） |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 ± 0.025820 |
| 0.1 | **有** | **0.819156 ± 0.015250** |
| 0 | 无 | 0.798974 ± 0.007324 |
| 0 | 有 | 0.795830 ± 0.017156 |

相同 dropout 下**增加 Norm**：dropout=0.1 时 AP 差为 **+0.068220**；dropout=0 时为 **−0.003145**。

### 4. 基于本次结果的判断
 dropout 与 Norm 存在交互，本实验四种 **global FLaG** 配置中 `0.1 + 有 Norm` 的 validation AP 最高。**本实验没有测试 `0.1 + 有 Norm` 的 Sprint E12**，不能据此比较这项配置下的 global 与 local。

### 证据解释的范围

四个条件都使用 Global FFT；本实验没有训练对应的 E12 dropout=0.1 / Norm=True 条件。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/sprint/02-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)

[返回实验导航](index.md)
