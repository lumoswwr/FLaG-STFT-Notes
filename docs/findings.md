# 研究结果综述：哪些结论站得住？



## 1. 为什么开展这组实验？

在原始 FLaG 研究背景下，STSBenchmark 是一个 global FLaG 在当前协议下不如 Mean pooling 的文本任务。我们先尝试 Hann、局部 STFT、窗口长度、重叠和位置编码，发现局部方案在少量 seeds 上有微小正向信号；随后通过 knockout、padding probe、时域等式与 2×2 干预，发现**local/global 最清晰的差异在重建路径，而非 gate 观测**。把结构固定的 E12 转到 Sprint 后，又发现**dropout 与 post-pool normalization 会显著影响性能比较**，因此原始整套方案增益不能解释为局部算子的独立增益。

## 2. 性能结果一览

| 实验 | 设置及数据位置 | Global FLaG | E12 local STFT | 配对 E12−Global | Seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| STSB 原机制线 | **test Spearman**，两者 `dropout=0,norm=True` | 0.841559 ± 0.003538 | 0.842739 ± 0.003214 | **+0.001180 ± 0.002107**，7/10 正向；区间跨 0 | **10** |
| STSB E14 | **test Spearman**，两者论文文本配置 `0.1/True` | 0.839065 ± 0.004154 | 0.838944 ± 0.001882 | **−0.000121 ± 0.004039**，5/10 正向 | **10** |
| Sprint **早期 S2** | **official-test AP**，FLaG `0.1/False`、E12 `0/True`，**配置不匹配** | 0.713121 ± 0.019140 | 0.761344 ± 0.016899 | **+0.048222 ± 0.017641**，10/10 正向；**不能归因于 local 算子** | **10** |
| Sprint **匹配 S4** | **adaptation validation AP**，两者 `0/True` | 0.795830 ± 0.017156 | 0.789136 ± 0.014094 | **−0.006693 ± 0.017279**，4/10 正向；区间跨 0 | **10** |



## 3. 机制实证一览

| 发现 | 支持实验和证据 | 能够说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| Global FLaG 会对额外右 padding 长度敏感 | E9/E10：同 hidden 仅改 pooling 输入补零，global cosine 漂移、local E3 近似不变 | 当前实现的 global FFT length 会影响输出 | 不表示所有 STFT 实现对所有层面的 padding 都不变 |
| 实/虚分别做跨频率共享门控可写成缩放与循环反射 | 时域推导；float64 验证最大误差约 `3.55×10^-15` | 输出的 `N` 依赖有明确的代数来源 | 不表示整个学习网络是输入无关的线性模型 |
| padding-length 的预测漂移以重建长度为主 | P1：仅换 gate 观测长度约 `4.80e-7`，仅换重建长度约 `0.005520` | 所测 STSB checkpoint 中 reconstruction length 是主要路径 | 不证明“固定长度训练一定涨分”；E13 恰好没有稳定增益 |
| global/local 的**完整路线**差异主要在重建侧 | STSB P2：gate 来源切换约 `2–3e-7`，重建路线切换约 `0.0058`；Sprint S3 有同向结构结果 | 两组频谱在当前 checkpoint 下生成非常接近的句级 gate，逆变换/时域支撑的切换影响明显 | 不证明 local 重建会导致指标提高 |

注意，P1 和 P2 **不是同一个实验**：P1 是**global 内部** `N_gate×N_recon`，P2 是**global/local 算子间**的 `gate observation×analysis/reconstruction`；它们从不同角度指向 reconstruction/temporal support。

## 4. 正在保留与暂不支持的解释

已完成的实验显示：

- STSB E1 的 Hann 没有改善结果，但并未隔离或排除频谱泄漏；
- E6 的 B0/低频 knockout 对两者影响较大，但**跨 seed 的相对敏感性不足以解释 E3 的微小收益**；
- E7 的 token 位置响应差异很小；E8 的 DC attention **偏好比**不等于 DC 的预测重要性；
- STSB E12 早期三 seed 的积极信号，在 10 seeds 和不同 dropout 配置下**不构成稳定 local 优势**；
- Sprint 早期 E12 配置的 AP 确实高于早期 global 配置，但 S4/S5 证明**混杂的非算子因素不能忽略**，不能将整个差额归因于 STFT；
- 当前结果**不支持**“local STFT 普遍优于 FLaG”“reconstruction 差异已经构成性能因果证明”“在 Sprint 的 `0.1/True` 条件下已完成 global/local 匹配测试”等说法。

