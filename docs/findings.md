# 实验结果总结

## 性能

| 数据集与设置 | Global FLaG | E12（STFT 16/16） | 同 seed E12−FLaG |
| --- | ---: | ---: | ---: |
| **STSB test Spearman**，0 dropout / 有 norm，10 seeds | 0.841559 ± 0.003538 | 0.842739 ± 0.003214 | +0.001180 ± 0.002107（7/10 正向） |
| **STSB test Spearman**，0.1 dropout / 有 norm，10 seeds | 0.839065 ± 0.004154 | 0.838944 ± 0.001882 | −0.000121 ± 0.004039（5/10 正向） |
| **Sprint validation AP**，两者均 0 dropout / 有 norm，10 seeds | 0.795830 ± 0.017156 | 0.789136 ± 0.014094 | −0.006693 ± 0.017279（4/10 正向） |

作为参照，STSB Mean pooling 在相同早期协议下的 10-seed test Spearman 为 **0.851201 ± 0.002000**。STSB 上 E12 的小幅正向趋势未在不同配置下稳定出现。Sprint 早期 E12 的 **official-test AP** 高于早期 FLaG，但当时 dropout/norm 不匹配，不能据此认定算子带来增益。

## 机制

- **E9/E10**：仅改变 pooling 输入的右 padding，global FLaG 预测会变；当前 local STFT 实现近似不变。
- **循环反射与 P1**：实、虚 gate 不同会产生循环反射；global 对 padding 的预测漂移主要来自重建长度，而非 gate 变化。
- **P2 与 Sprint S3**：在固定 checkpoint 下，global/local 的 gate 几乎相同，预测差异主要来自**完整重建路线**。
- **Sprint S5**：global FLaG 的 dropout 与 post-pool Norm 存在明显交互；四种测试配置中，`0.1 dropout + 有 norm` 的 validation AP 为 **0.819156 ± 0.015250**。

**目前可以确定的是结构差异的来源；尚未证明 local STFT 能稳定提高下游性能。** 具体数值和配置见各实验页面。

## 后续：FLaG Gate 与 Projection 的作用（Mean-pooling 对照）

[新增机制章节](flag-gate-projection.md) 在**Global FLaG、time pooling=Mean、关闭 post-pool LayerNorm、dropout=0** 的单独配置下，继续分析了 Gate 与输出线性层的实际作用。与本页 Global/Local STFT 结论属于不同实验协议，不能直接混合数值。

- **Gate 输出及高精度复核**：训练后的 Gate 高度饱和；用训练集平均 Gate 或其他句子的 Gate 替换时，STSB 和 Sprint 指标在六位小数精度下不变。高精度 validation 统计进一步发现，Sigmoid 前的 Gate logits 仍随句子变化（Sprint FLaG 的平均跨句子通道标准差约 **0.4394**），而实际 Gate 相对训练集平均值的平均绝对差仅约 **7.58×10⁻¹⁴**。因此结论是**实际 Gate 的输入相关变化极小**，不是证明 Gate 严格不变；极小方差的数值计算也可能受浮点精度影响。
- **DC 通道权重**：Sprint seed0 中，约 2.21%（FLaG）/3.78%（B2）的低 Gate 值实部通道，承载原始 DC 平方能量约 93.72%/95.49%。
- **从头训练的对照**：Sprint 采用验证集 AP 选 checkpoint，且强制对齐初始 Projection 的 3-seed test AP：FLaG **0.830539 ± 0.016711**，Mean + 随机初始化、可训练 Projection **0.841386 ± 0.008833**。配对 FLaG−MeanProjRand 为 **−0.010847 ± 0.014922**，0/3 个种子为正。验证集 AP 均值则为 FLaG 0.853535、MeanProjRand 0.851664，顺序相反。

因此，目前可以说在这个专门控制设置下**没有观察到复杂 FLaG 相对 Mean+Linear 的额外 test AP 优势**。它并不证明 FFT 和 latent attention 在训练中必然无用，也不能推广到 Max-pooling 原配置。

