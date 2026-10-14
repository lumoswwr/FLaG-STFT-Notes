# 按证据整理的实验结果

本页只陈述现有实验输出，按可比性归类；各项的对照设计和原始出处在[逐项实验目录](experiments/index.md)。

## 一、STSB：同协议 Global / Local

| 条件（10 seeds，test Spearman） | Global FLaG | E12（16/16） | 配对 E12−FLaG |
| --- | ---: | ---: | ---: |
| dropout=0、post-pool Norm=True | 0.841559 ± 0.003538 | 0.842739 ± 0.003214 | +0.001180 ± 0.002107；7/10 正向 |
| dropout=0.1、post-pool Norm=True | 0.839065 ± 0.004154 | 0.838944 ± 0.001882 | −0.000121 ± 0.004039；5/10 正向 |

两组匹配对照的均值方向不同。见 [E12](experiments/stsb-e12.md)、[E14](experiments/stsb-e14.md)。作为单独参照，早期 STSB Mean 10-seed test Spearman = 0.851201 ± 0.002000。

## 二、Sprint：先区分不同配置，再看匹配对照

| 对照 | 范围与指标 | FLaG | E12 | 配对 E12−FLaG |
| --- | --- | ---: | ---: | ---: |
| 早期 dropout/norm 不匹配 | 10-seed **official test AP** | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 | +0.048222 ± 0.017641；10/10 正向 |
| dropout=0、Norm=True 匹配 | 10-seed **adaptation validation AP** | 0.795830 ± 0.017156 | 0.789136 ± 0.014094 | −0.006693 ± 0.017279；4/10 正向 |

两行不在同一个评估 split，不能直接相减归因。见 [S2](experiments/sprint-s02.md)、[S4](experiments/sprint-s04.md)。[S5](experiments/sprint-s05.md) 仅在 Global FLaG 上做 dropout × Norm 2×2，其中 dropout=0.1、Norm=True 的 validation AP 为 0.819156 ± 0.015250。

## 三、固定 checkpoint 的前向机制

- [STSB P1](experiments/mech-p1.md)：仅改变 gate 频谱长度，3-seed 平均 cosine 漂移约 4.80×10⁻⁷；改变重建长度则为 0.005520 ± 0.000805。
- [STSB P2](experiments/mech-p2.md)：仅变 gate 来源的平均漂移约 2.86×10⁻⁷ / 2.14×10⁻⁷；改重建路线约 0.005770 / 0.005970。
- [Sprint S3](experiments/sprint-s03.md)：仅变 gate 来源几乎无影响，重建路线对应 0.032432 ± 0.003114 / 0.037477 ± 0.003556 的平均 cosine 漂移。

以上为**预测层面**的变化，不是 Spearman 或 AP 的收益。[循环反射公式](experiments/mech-reflection.md) 给出长度依赖的数学分解。

## 四、DC、Gate、Projection 与融合

- [DC 频带](experiments/dc-band.md)：frozen RoBERTa 的 L12 B0-only 分别保留 STSB 96.4%、Sprint 91.4% 的对应 baseline 指标。DCT 的 B0 不等于 exact DC。
- [Exact DC 训练](experiments/dc-exact.md)：seed-0 的 DC-only FLaG 在 STSB test Spearman = 0.746003、Sprint official test AP = 0.894565；分别来自独立重训练条件。
- [Gate 输入相关性](experiments/gate-input.md)：实际 Gate 的跨句变化极小，但并非已证明严格常数。
- [严格共享 Projection 初始化](experiments/gate-projection.md)：Sprint 3-seed test AP：FLaG 0.830539 ± 0.016711，Mean+Projection 0.841386 ± 0.008833；validation AP 均值排序反向。
- [Alignment-aware 统一五种子](experiments/mean-final.md)：IMDB Accuracy 0.9526 ± 0.0013、STS-B Spearman 0.8536 ± 0.0013、Sprint AP 0.6989 ± 0.0503，均属于专门统一协议。

## 证据边界

- “算子导致输出变化”与“算子导致性能提高”不是同一结论。
- 一次 seed-0 结果不能等同于多种子稳定性。
- 配置不匹配的模型整体比较，不应命名为纯 STFT 算子消融。
- Sprint AP 与 STSB Spearman、validation 与 official test，均需分别列示。
