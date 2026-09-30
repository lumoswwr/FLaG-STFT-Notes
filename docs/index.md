# FLaG / STFT 频域池化与机制实验报告

本报告记录 **FLaG 全局 FFT 与局部 STFT-FLaG（E3 / E12）** 在两个英文句对任务上的探索、消融与机制分析。重点不是单独展示模型成绩，而是把**问题、方法、完整配置、随机种子数、数据划分、评价指标、原始结果及结论边界**关联起来，让每一个实验都可以沿代码和日志复核。

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

## 读报告时最容易混淆的三点

**第一，实验报告不是“每组谁的数字高就一定是哪项结构起作用”。** STSB E1–E5 / E11–E14 有部分重新训练的候选；E6–E10 及 P1/P2 大多在固定 checkpoint 下改变推理条件。两种证据层次不能互相代替。

**第二，S1/S2 和 S4 的数值不属于同一控制条件。** Sprint 早期 FLaG \`dropout=0.1,post_pool_norm=False\`，E12 \`0/True\`；S4 才把两者匹配为 \`0/True\`。S5 的 \`0.1/True\` 是**仅 global FLaG**在 adaptation validation 上的控制，不能与 STSB E14 的 **test Spearman** 直接并列比较。

**第三，所有均值都必须带限定词。** 比如“STSB 3-seed test Spearman”“Sprint 10-seed adaptation validation AP”“P2 3-seed validation cosine 绝对漂移”；本文不会只给一个无数据集/seed/split 说明的数字。

## 文件与证据

正文由 MkDocs + Material 组织，可点击左侧章节与右侧页内目录。本网站是可读性整理版；**运行脚本、模型源码和研究原始摘要位于 [AMPCliff 实验分支](https://github.com/lumoswwr/AMPCliff/tree/FLaG-STFT-mechanism)**。具体每次实验仍以训练日志和 \`config.json\` 为准。新增结果应同时记录运行 commit、任务 split、seed 清单和训练/推理设置，避免后续追溯歧义。
