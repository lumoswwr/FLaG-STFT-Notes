# 实验结果总结

## 整个研究过程

最初，我们想把 FLaG 的整句 FFT 改成分窗口 STFT，看看局部频率信息能不能改善文本预测。早期一些实验中，STFT 的分数确实略高，但把训练配置统一以后，这个优势没有保留下来。

为了弄清原因，我们检查了 Padding、Gate 和频谱重建。结果发现，Global FLaG 对 Padding 长度敏感，主要与逆变换时出现的循环反射有关；Global 和 Local 的预测差异也主要来自重建路线，而不是它们生成了明显不同的 Gate。

随后我们研究频率本身。删除最低频带会明显伤害预测，而只保留 DC 并从头训练，仍然可以获得很好的结果。说明 Mean/DC 中已有大量任务相关信息，关键是后续模型能否把它们利用起来。


## 六个阶段分别得到了什么结果？

1. **[第一次尝试 STFT：STSB E1–E8](stsb/01-experiments.md)**：分窗口 FFT 的 E3 在早期三个种子中比 Global FLaG 的 Spearman 高 0.001873；增加 Hann 窗和帧位置编码没有明显帮助。
2. **[检查窗口、Padding 和配置：STSB E9–E14](stsb/02-padding.md)**：Global FLaG 只因后面补零就会改变预测；E12 在十种子匹配配置下平均高 0.001180，而统一另一组论文配置后比 Global 低 0.000121。
3. **[找到输出差异的原因：循环反射与 P1/P2](stsb/03-mechanism.md)**：更换 Gate 生成来源几乎不改变预测；改变频谱重建长度或重建方式则会产生明显变化，循环反射解释了主要的 Padding 现象。
4. **[换到 Sprint 验证：S1–S5](sprint/02-experiments.md)**：早期 E12 平均 Test AP 比 FLaG 高 0.048222，但训练配置不同；统一 dropout 和 LayerNorm 后，两者的平均 Validation AP 分别为 0.789136 与 0.795830，E12 不再领先。
5. **[研究频率到底包含什么：DC 与频带](dc-frequency.md)**：在最后一层只保留最低频带，STSB 和 Sprint 分别保留完整模型 96.4% 的 Spearman 和 91.4% 的 AP；进一步只保留精确 DC 从头训练，两个任务的 seed-0 分数都超过了完整 FLaG。
6. **[让 Mean 和 FLaG 互补：Alignment-aware](mean-compatible-flag.md)**：五种子实验中，改进后的融合方法 STSB Spearman 达到 0.8536，Sprint AP 达到 0.6989，高于同批 FLaG 的 0.8370 与 0.5976。

## 详细数值和机制记录

下面保留原来的汇总表、机制分析及不同实验之间的配置区别。具体的单项实验请进入上面的对应页面。

### 性能

| 数据集与设置 | Global FLaG | E12（STFT 16/16） | 同 seed E12−FLaG |
| --- | ---: | ---: | ---: |
| **STSB test Spearman**，0 dropout / 有 norm，10 seeds | 0.841559 ± 0.003538 | 0.842739 ± 0.003214 | +0.001180 ± 0.002107（7/10 正向） |
| **STSB test Spearman**，0.1 dropout / 有 norm，10 seeds | 0.839065 ± 0.004154 | 0.838944 ± 0.001882 | −0.000121 ± 0.004039（5/10 正向） |
| **Sprint validation AP**，两者均 0 dropout / 有 norm，10 seeds | 0.795830 ± 0.017156 | 0.789136 ± 0.014094 | −0.006693 ± 0.017279（4/10 正向） |



### 机制

- **E9/E10**：仅改变 pooling 输入的右 padding，global FLaG 预测会变；当前 local STFT 实现近似不变。
- **循环反射与 P1**：实、虚 gate 不同会产生循环反射；global 对 padding 的预测漂移主要来自重建长度，而非 gate 变化。
- **P2 与 Sprint S3**：在固定 checkpoint 下，global/local 的 gate 几乎相同，预测差异主要来自**完整重建路线**。
- **Sprint S5**：global FLaG 的 dropout 与 post-pool Norm 存在明显交互；四种测试配置中，`0.1 dropout + 有 norm` 的 validation AP 为 **0.819156 ± 0.015250**。

**目前可以确定的是结构差异的来源；尚未证明 local STFT 能稳定提高下游性能。** 具体数值和配置见各实验页面。

