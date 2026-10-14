# FLaG / STFT 实验报告

本报告围绕 **Global FLaG 的全局频域处理与 STFT-FLaG 的局部频域处理**，记录模型性能对照、Padding 与重建机制、频带信息、Mean 融合和 Gate / Projection 的作用。内容按照研究问题组织，涵盖 38 项已完成的实验。

每章先集中呈现实验总结，随后逐项列出实验设置、结果数据和基于结果的分析。实验总结保持原文不变，可在总结与详细论述之间直接跳转。

## 研究内容

| 章节 | 实验范围 |
| --- | --- |
| [STFT 窗口设计与 STSB 配置对照](chapters/01-stft-window.md) | 1、2、3、4、5、11、12、14 |
| [STFT 的频率和 Token 响应](chapters/02-token-frequency.md) | 6、7、8 |
| [Padding 敏感性与 Global/Local 重建机制](chapters/03-padding-reconstruction.md) | 9、10、13、15、16、17 |
| [Sprint 跨任务表现与配置控制](chapters/04-sprint.md) | 18、19、20、21、22 |
| [最低频带信息与 DC 训练干预](chapters/05-dc-bands.md) | 23、24、25、26 |
| [Mean 与 FLaG 的向量融合](chapters/06-mean-fusion.md) | 27、28、29、30、31 |
| [Gate 与 Projection 的逐阶段作用](chapters/07-gate-projection.md) | 32、33、34、35 |
| [Mean + Projection 的简化结构对照](chapters/08-simplified-pooling.md) | 36、37、38 |

## 主要观测

| 对照 | 已报告的结果 |
| --- | --- |
| STSB：E12 与 Global FLaG，匹配 0 dropout、启用 Norm，10 seeds | Test Spearman 0.842739 vs 0.841559，均值差 +0.001180，[实验 12](chapters/01-stft-window.md#detail-12) |
| STSB：E14，匹配 0.1 dropout、启用 Norm，10 seeds | E12 0.838944 vs Global 0.839065，基本持平，[实验 14](chapters/01-stft-window.md#detail-14) |
| Sprint：S1/S2 早期配置，10 seeds | E12 test AP 0.761344 vs Global 0.713121；两模型 dropout/Norm 不匹配，[实验 19](chapters/04-sprint.md#detail-19) |
| Sprint：S4 匹配配置，10 seeds | E12 validation AP 0.789136 vs Global 0.795830，[实验 21](chapters/04-sprint.md#detail-21) |
| Sprint：严格匹配初始 Projection，3 seeds | Mean+Projection test AP 0.841386，FLaG 0.830539；validation 则 FLaG 略高，[实验 38](chapters/08-simplified-pooling.md#detail-38) |

**指标口径：** STSB 使用 Spearman，Sprint 使用 AP；test 与 validation 分别标注。对不同训练配置的比较只陈述已经观测到的整体结果，不将差异归因于未单独控制的因素。
