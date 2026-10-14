### 实验 Sprint S01：Sprint 早期三种子基线

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 E12 结构用于 Sprint 后，与 Mean、早期 FLaG 相比效果如何？

#### 2. 实验设计与控制条件
 冻结 RoBERTa，分别重新训练 pooling/head，**3 seeds（0–2）**。当时的实际模型配置：

| 模型 | Pooling | Dropout | Post-pool Norm |
| --- | --- | ---: | :---: |
| Mean | 有效 token 均值 | 不适用 | 不适用 |
| FLaG | Global FFT | **0.1** | **无** |
| E12 | STFT win16/hop16 | **0** | **有** |

#### 3. 观测结果
 **3-seed official-test（mean ± std）**：

| 模型 | AP ↑ | Accuracy ↑ | F1 ↑ |
| --- | ---: | ---: | ---: |
| Mean | 0.428907 ± 0.000003 | 0.992040 ± 0 | 0.460390 ± 0 |
| FLaG | 0.721611 ± 0.009747 | 0.993980 ± 0.000298 | 0.667938 ± 0.007364 |
| E12 | 0.765655 ± 0.023788 | 0.994495 ± 0.000423 | 0.690782 ± 0.017933 |

#### 4. 基于本次结果的判断
 E12 早期配置的 AP 更高，但 FLaG/E12 的 dropout 和 norm **不匹配**，还不能判断是否由 STFT 算子造成。

### 证据解释的范围

该比较中 FLaG 和 E12 的 dropout 与 post-pool Norm 不同。因此观察到的是两套完整配置之间的差别，不是 STFT 算子的独立效果。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/sprint/02-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/train_sprint.py)

[返回实验导航](index.md)
