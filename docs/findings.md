# 关键观测索引

这里仅索引各项实验已经报告的具体数值与比较口径，不把不同协议合成同一个总体因果结论。

| 比较 | 已观察到的数值 | 对应实验 |
| --- | --- | --- |
| STSB E12（win16/hop16）vs Global，0 dropout、Norm=True，10 seeds | test Spearman 0.842739 vs 0.841559，配对均值 +0.001180；7/10 seeds 正向 | [12](experiments/12.md) |
| STSB E12 vs Global，0.1 dropout、Norm=True，10 seeds | test Spearman 0.838944 vs 0.839065，差 −0.000121 | [14](experiments/14.md) |
| Sprint 早期配置对照，10 seeds（dropout/norm 不匹配） | test AP E12 0.761344 vs FLaG 0.713121 | [19](experiments/19.md) |
| Sprint 匹配 dropout=0、Norm=True，10 seeds | **validation** AP E12 0.789136 vs FLaG 0.795830 | [21](experiments/21.md) |
| STSB P1：只改 Gate 长度 vs 只改重建长度 | validation 平均绝对 cosine 漂移 4.8×10⁻⁷ vs 0.005520 | [16](experiments/16.md) |
| Sprint：严格匹配初始 Projection，3 seeds | **test AP** Mean+Projection 0.841386 vs FLaG 0.830539；**validation AP** FLaG 0.853535 vs Mean+Projection 0.851664 | [38](experiments/38.md) |

这几项结果的详细协议与其他实验记录，以[38 项实验目录](index.md#研究阶段)为准。
