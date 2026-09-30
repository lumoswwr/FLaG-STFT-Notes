# FLaG / STFT 实验报告

本报告记录 **FLaG 全局 FFT 与局部 STFT-FLaG **在两个英文句对任务上的探索、消融与机制分析。

!!! info "建议阅读顺序"
    **第一次阅读**：先看[数据与评价协议](methods.md)，再浏览[研究结论综述](findings.md)，之后按导航依次阅读 STSB E1–E8、E9–E14、循环反射/P1/P2，最后阅读 Sprint 迁移与 S1–S5。每个独立实验均以 **Q / M / R / C** 编写；章节开头还有**实验矩阵**，方便直接对照参数和 seeds。

## 核心研究路线

| 阶段 | 数据集/任务 | 核心问题 | 章节 |
| --- | --- | --- | --- |
| 统一协议 | STSB + Sprint | 使用什么数据、分割、损失、指标与 seeds？ | [协议与指标](methods.md) |
| 第一阶段 | STSB，句对语义相似度 | 边缘 Hann、局部 STFT、窗口/重叠/帧位置有何影响？如何定义 knockout 和 DC attention？ | [E1–E8](stsb/01-experiments.md) |
| 第二阶段 | STSB | padding 是什么问题？固定长度训练与 E12 的 10-seed 结果如何？原论文 text 配置下如何？ | [E9–E14](stsb/02-padding.md) |
| 第三阶段 | STSB | 实/虚门控为什么会产生循环反射？P1 与 P2 各隔离了什么？ | [循环反射、P1/P2](stsb/03-mechanism.md) |
| 第四阶段 | Sprint，重复问题检测 | 冻结 RoBERTa 和分类器之后，数据划分/阈值/AP 怎么算？ | [Sprint 协议](sprint/01-protocol.md) |
| 第五阶段 | Sprint | 早期结果、推理重建干预、matched global、dropout×norm 交互分别说明什么？ | [S1–S5](sprint/02-experiments.md) |
| 结论 | 两个任务 | 哪些是可复核结构发现、哪些性能结论仍受配置影响？ | [研究结果综述](findings.md) |


