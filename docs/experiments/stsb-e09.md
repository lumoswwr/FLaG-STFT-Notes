### 实验 STSB E09：右侧额外 Padding 与输出漂移

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 同一句子只增加右侧零 padding，FLaG 与 E3 的预测是否变化？

#### 2. 实验设计与控制条件
 固定 FLaG/E3 checkpoint，选择有效长度 ≤16 的 STSB 短句。保持 RoBERTa 产生的有效 hidden 不变，分别补零到 16、24、32、64、128，再传入 pooling；计算相对精确有效长度处理的**句对 cosine 绝对差**。

#### 3. 观测结果
 补零至 128 时：

| 指标 | FLaG | E3 |
| --- | ---: | ---: |
| 平均绝对 cosine 漂移 | **0.006003** | 约 0 |
| 最大绝对 cosine 漂移 | **0.035915** | 约 0 |

#### 4. 基于本次结果的判断
 当前 global FLaG 对外部右 padding 敏感；E3 在这项固定 hidden 测试中近似不变。

*记录待核对：现有摘要没有给出这张结果表所对应的完整运行命令。探针脚本默认 seed=0、最多 200 对样本，但正式确定实验数量需要检查原始日志。*

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
