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
