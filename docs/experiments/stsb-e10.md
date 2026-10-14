### 实验 STSB E10：四种 Padding 推理方式

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 E9 的预测变化是否明显影响整个 STSB test 的性能？

#### 2. 实验设计与控制条件
 使用 FLaG/E3 的 **3-seed checkpoint**，每句话只计算一次 RoBERTa hidden，之后采用四种 **pooling 输入长度**，不重新训练：

| 设置 | 处理方式 |
| --- | --- |
| normal | 原始 batch 动态 padding |
| pair_exact | 同一对两句都补到二者较大的有效长度 |
| sentence_exact | 每句分别使用自身有效长度 |
| fixed_128 | 每句统一补零到 128 |

#### 3. 观测结果
 **3-seed mean test Spearman**：

| Padding | FLaG | E3 |
| --- | ---: | ---: |
| normal | 0.841128 | 0.843000 |
| pair_exact | 0.841022 | 0.843000 |
| sentence_exact | 0.841288 | 0.843000 |
| fixed_128 | 0.841458 | 0.843000 |

#### 4. 基于本次结果的判断
 Padding 会改变 FLaG 的预测和少量 test Spearman，但未能解释全部 FLaG/E3 性能差异。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
