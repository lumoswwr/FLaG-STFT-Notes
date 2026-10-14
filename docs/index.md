# FLaG / STFT 研究记录

这个站点按**协议 → 逐项实验 → 事实结论**组织。每项实验单独成页，详细论述使用已有报告与代码仓库的可核对记录。各页顶部保留“实验目的/实验思路/实验结果/实验结论”四字段，待粘贴作者正式总结，**不会自动改写原文**。

[进入实验逐项导航](experiments/index.md) · [查看对照协议](methods.md) · [查看结果索引](findings.md)

## 实验主线

| 阶段 | 原始问题 | 已有记录范围 | 浏览入口 |
| --- | --- | --- | --- |
| STSB 结构试验 | Global FFT 与 STFT 的窗口、重叠、位置编码差别 | E1–E5、E11–E14 | [实验目录](experiments/index.md#stsb-e1e14) |
| STSB 机制分析 | 频带、token、padding、循环反射 | E6–E10、P1/P2 | [实验目录](experiments/index.md#p1p2-与机制) |
| Sprint 跨任务 | 早期对比是否可归因于算子，配置影响有多大 | S1–S5 | [实验目录](experiments/index.md#sprint-s1s5) |
| DC / Gate / Projection | 低频是否有可读信息，Gate 和输出层分别做什么 | 冻结 backbone 与干预对照 | [实验目录](experiments/index.md#dc-与频带) |
| Mean-compatible | 融合方式能否保留 Mean endpoint | 插值、正交、Alignment-aware、5-seed | [实验目录](experiments/index.md#mean-compatible-与-alignment-aware) |

## 核心区分

- **性能比较**：只比较同一 split、相同评价指标及明确控制条件下的数值。
- **前向机制**：固定 checkpoint 下的输出漂移说明算子改变了什么，不直接说明重新训练后的性能归因。
- **探索与复核**：3-seed/seed-0、10-seed、单独更换 checkpoint 规则的实验分开记载。

[代码与实验日志索引](https://github.com/lumoswwr/AMPCliff/tree/FLaG-STFT-mechanism/text_repro)；旧版 Markdown 保留在仓库中供溯源，但不再作为主导航入口。
