# FLaG / STFT 实验报告

研究目标：在原始 **FLaG（Global FFT）** 的基础上尝试 **STFT 局部频域池化**，比较性能，并分析 global/local 的实际结构差异。

| 内容 | 说明 |
| --- | --- |
| [实验设置与指标](methods.md) | 模型、数据集、训练配置、评价指标 |
| [STSB E1–E8](stsb/01-experiments.md) | Hann、STFT、窗长、频带和 token 分析 |
| [STSB E9–E14](stsb/02-padding.md) | Padding、窗口重叠、固定 FFT、10-seed 配置对照 |
| [循环反射与 P1/P2](stsb/03-mechanism.md) | 分析 padding 和 global/local 预测差异的来源 |
| [Sprint 设置](sprint/01-protocol.md) | 冻结 RoBERTa 后的分类训练和指标 |
| [Sprint S1–S5](sprint/02-experiments.md) | 跨任务性能、机制验证及配置控制 |
| [结果总结](findings.md) | 整体实验结论 |
| [DC / 频带机制](dc-frequency.md) | Frozen RoBERTa 下的 DC readability、band-only 与 band-knockout 机制分析 |
| [FLaG 门控与输出线性投影](flag-gate-projection.md) | 逐阶段转角、Gate 输入依赖性、高能通道、简化模型与严格 Projection 匹配的三种子实验 |

**阅读约定**：各实验采用 **Q（问题）、M（方法）、R（结果）、C（结论）**。性能表写明是 **test** 还是 **validation**、使用几个随机种子；机制实验的“预测漂移”不等于性能提高。
