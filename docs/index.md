# FLaG / STFT 研究与实验报告

## 目前研究进展

最初，我们想把 FLaG 的整句 FFT 改成分窗口 STFT，看看局部频率信息能不能改善文本预测。早期一些实验中，STFT 的分数确实略高，但把训练配置统一以后，这个优势没有保留下来。

为了弄清原因，我们检查了 Padding、Gate 和频谱重建。结果发现，Global FLaG 对 Padding 长度敏感，主要与逆变换时出现的循环反射有关；Global 和 Local 的预测差异也主要来自重建路线，而不是它们生成了明显不同的 Gate。

随后我们研究频率本身。删除最低频带会明显伤害预测，而只保留 DC 并从头训练，仍然可以获得很好的结果。说明 Mean/DC 中已有大量任务相关信息，关键是后续模型能否把它们利用起来。

后续的实验继续把 FLaG 拆开：训练后的 Gate 在不同句子间几乎不变，少数低 Gate 值通道对应大量 DC 平方能量。最终在 Sprint 上，直接 Mean 加一个可训练线性层就得到了与 FLaG 接近甚至更高的 Test AP。

## 实验报告

| 研究阶段 | 具体报告 | 这一阶段回答了什么？ |
| --- | --- | --- |
| 1. 尝试局部 FFT | [STSB：E1–E8，尝试不同 STFT 窗口](stsb/01-experiments.md) | 短窗口 E3 曾小幅领先；Hann 窗和位置编码没有进一步提高性能 |
| 2. 检查 STFT 的表现 | [STSB：E9–E14，Padding 和训练配置](stsb/02-padding.md) | Global FLaG 对补零长度敏感；E12 在匹配配置后没有保持明显优势 |
| 3. 解释模型差异 | [STSB：循环反射及 P1/P2](stsb/03-mechanism.md) | 确认 padding 和 Global/Local 输出差异主要来自重建路径 |
| 4. 换任务验证 | [Sprint：S1–S5](sprint/02-experiments.md) | 早期 E12 AP 较高，统一训练配置后不再领先 |
| 5. 分析频率信息 | [DC / 频带机制](dc-frequency.md) | 最低频带很重要；只保留精确 DC 训练仍得到很高的预测成绩 |
| 6. 设计融合方法 | [Mean 与 FLaG 融合](mean-compatible-flag.md) | 保留 Mean 路径并补充 FLaG 信息，改进了 STSB 和 Sprint 的表现 |
| 7. 拆解内部结构 | [Gate 和 Projection 的作用](flag-gate-projection.md) | Gate 训练后几乎不随句子变化；Mean + 可训练线性层达到很高的 Sprint AP |

## 关于指标和实验设置

STSB 的主要指标是 **Spearman**（预测相似度排序和真实排序的一致程度）；Sprint 的主要指标是 **AP**（按预测分数排序寻找重复问题的能力）；IMDB 主要看 Accuracy 和 F1。这些分数衡量不同任务，不能直接相互比较。

**完整索引：** [实验设置](methods.md) · [任务理解](MTEB.md) · [总结果](findings.md)
