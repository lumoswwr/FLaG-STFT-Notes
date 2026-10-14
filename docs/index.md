# FLaG / STFT 已完成实验报告

本版依据上传的 **38 项实验总结** 重组，每项实验独立成页。每一页**开头完整保留用户提交的总结原文**，然后进入详细实验记录。文首可跳到详情，文末可回到文首。

按研究问题而非脚本或实验时间拆分：相同实验目的的多个控制组或补充测试保留在一篇文档内。

## 研究阶段

| 阶段 | 已完成实验 | 核查入口 |
| --- | --- | --- |
| STSB：早期局部频域设计 | 1–5 | [实验 1](experiments/01.md) |
| STSB：频率与 token 探针 | 6–8 | [实验 6](experiments/06.md) |
| STSB：Padding 行为 | 9–10 | [实验 9](experiments/09.md) |
| STSB：窗口与稳定性 | 11–14 | [实验 11](experiments/11.md) |
| STSB：循环反射与重建机制 | 15–17 | [实验 15](experiments/15.md) |
| Sprint：跨任务与配置对照 | 18–22 | [实验 18](experiments/18.md) |
| DC：频带信息与训练干预 | 23–26 | [实验 23](experiments/23.md) |
| Mean：兼容性与融合 | 27–31 | [实验 27](experiments/27.md) |
| Gate / Projection：机制与简化对照 | 32–38 | [实验 32](experiments/32.md) |

## 阅读规则

- **先读原始总结，再读数据与方法。** 总结为用户原文，详细部分呈现既有报告可核对的操作、样本、指标与控制条件。
- **不同指标不可混用。** STSB 的 Spearman、Sprint 的 AP，以及 validation/test 必须区分。
- **不同训练协议不可直接归因。** Sprint S1/S2 的 Global 与 Local 除 pooling 算子外还存在 dropout、Norm 差异；配齐设置后的 S4 单独报告。
- **预测变化不等于性能收益。** 门控交换或频谱重建方式改变产生的 cosine 漂移，只说明输出发生改变。
- **旧稿留档。** 原 Markdown 全部备份到仓库根目录 `archive/legacy/`，不再与现版混用。

参见 [实验协议与指标](methods.md) 和 [关键观测索引](findings.md)。
