# STSB：E9–E14

E9/E10 使用**已训练的 FLaG 与 E3** 做 padding 测试；E11/E12/E13/E14 是重新训练的配置对照。STSB 基础训练参数见[实验设置](../methods.md)。

## 实验配置对照

| 实验 | 改变什么 | Seeds | 指标 |
| --- | --- | --- | --- |
| E9 | 短句额外右 padding 到 16/24/32/64/128 | 原始运行数量待核对 | cosine 绝对漂移 |
| E10 | normal / pair_exact / sentence_exact / fixed_128 | 0–2（3 个） | test Spearman |
| E11 | STFT win8/hop8，无重叠 | 0–2（3 个） | test Spearman/Pearson |
| E12 | STFT win16/hop16，无重叠 | 先 0–2，后扩至 0–9 | test Spearman/Pearson |
| E13 | 训练与推理都固定 global FFT 长度 128 | 0–2（3 个） | test Spearman/Pearson |
| E14 | 两种模型都用 dropout=0.1、Norm=True | 0–9（10 个） | test Spearman/Pearson |

除 E14 外，STSB 新训练模型统一 **dropout=0、post-pool Norm=True**。

## E9：额外右 Padding 的影响

**Q：** 同一句子只增加右侧零 padding，FLaG 与 E3 的预测是否变化？

**M：** 固定 FLaG/E3 checkpoint，选择有效长度 ≤16 的 STSB 短句。保持 RoBERTa 产生的有效 hidden 不变，分别补零到 16、24、32、64、128，再传入 pooling；计算相对精确有效长度处理的**句对 cosine 绝对差**。

**R：** 补零至 128 时：

| 指标 | FLaG | E3 |
| --- | ---: | ---: |
| 平均绝对 cosine 漂移 | **0.006003** | 约 0 |
| 最大绝对 cosine 漂移 | **0.035915** | 约 0 |

**C：** 当前 global FLaG 对外部右 padding 敏感；E3 在这项固定 hidden 测试中近似不变。

*记录待核对：现有摘要没有给出这张结果表所对应的完整运行命令。探针脚本默认 seed=0、最多 200 对样本，但正式确定实验数量需要检查原始日志。*

## E10：四种 Padding 推理方式

**Q：** E9 的预测变化是否明显影响整个 STSB test 的性能？

**M：** 使用 FLaG/E3 的 **3-seed checkpoint**，每句话只计算一次 RoBERTa hidden，之后采用四种 **pooling 输入长度**，不重新训练：

| 设置 | 处理方式 |
| --- | --- |
| normal | 原始 batch 动态 padding |
| pair_exact | 同一对两句都补到二者较大的有效长度 |
| sentence_exact | 每句分别使用自身有效长度 |
| fixed_128 | 每句统一补零到 128 |

**R：** **3-seed mean test Spearman**：

| Padding | FLaG | E3 |
| --- | ---: | ---: |
| normal | 0.841128 | 0.843000 |
| pair_exact | 0.841022 | 0.843000 |
| sentence_exact | 0.841288 | 0.843000 |
| fixed_128 | 0.841458 | 0.843000 |

**C：** Padding 会改变 FLaG 的预测和少量 test Spearman，但未能解释全部 FLaG/E3 性能差异。

## E11/E12：窗口大小与重叠

**Q：** STFT 是否一定需要窗口重叠？

**M：** 比较 win8/win16 与半窗重叠/无重叠，共四种 rect、`center=False` 的 STFT 配置。沿用 E2/E3，并补充重新训练 E11/E12，**每组 3 seeds（0–2）**，其他参数一致。

**R：** **3-seed test Spearman（mean ± std）**：

| 窗长 | 半窗口重叠 | 无重叠 |
| ---: | ---: | ---: |
| 8 | **E3**：hop4，0.843000 ± 0.001047 | **E11**：hop8，0.842166 ± 0.001423 |
| 16 | **E2**：hop8，0.842204 ± 0.001824 | **E12**：hop16，0.843419 ± 0.000531 |

**C：** 重叠没有一致优势。后续固定使用结构简单的 **E12：rect、win16/hop16、center=False**。

### E12 的 10-seed 稳定性

将原匹配设置（FLaG/E12 都为 **dropout=0、Norm=True**）扩至 **10 seeds（0–9）**：

| 模型 | test Spearman ↑ | test Pearson ↑ |
| --- | ---: | ---: |
| FLaG | 0.841559 ± 0.003538 | 0.838504 ± 0.003444 |
| E12 | 0.842739 ± 0.003214 | 0.839393 ± 0.003467 |
| 配对 E12−FLaG | **+0.001180 ± 0.002107** | **+0.000889 ± 0.003033** |

Spearman **7/10 seeds** 正向，配对 95% CI 约 `[−0.00033, +0.00269]`。虽然平均差为正，但不足以说明稳定优势。

## E13：从训练开始固定 FFT 长度

**Q：** 如果 global FLaG 在训练时就使用固定 FFT 长度，能否改善结果？

**M：** global FLaG 在训练及推理时均设 `fixed_fft_length=128`；不引入 STFT，其他设置不变，**3 seeds（0–2）**。

**R：** test Spearman **0.841856 ± 0.001000**；相对同 seed 动态 FLaG 的配对差为 **+0.000728 ± 0.003079**，仅 **1/3 正向**。

**C：** 固定 FFT 长度没有带来稳定的性能改善。

## E14：统一论文 Text 配置

**Q：** 在论文 text 配置下，E12 相比 FLaG 的早期性能趋势是否仍存在？

**M：** FLaG 与 E12 都改为 **dropout=0.1、post-pool Norm=True**，其余训练设置匹配，重新训练 **10 seeds（0–9）**。

**R：** **10-seed test**：

| 模型 | Spearman ↑ | Pearson ↑ |
| --- | ---: | ---: |
| FLaG | 0.839065 ± 0.004154 | 0.834980 ± 0.003942 |
| E12 | 0.838944 ± 0.001882 | 0.834015 ± 0.002119 |
| 配对 E12−FLaG | **−0.000121 ± 0.004039** | **−0.000965 ± 0.003878** |

Spearman **5/10 seeds** 正向。

**C：** 两者基本持平。E12 在 0/有 Norm 下的小幅正向均值没有在这组 0.1/有 Norm 实验中保持。
