# Sprint：S1–S5

## S1

**Q：** 对比 FLaG、Mean、STFT（E12）性能。

**M：**

| 模型 |              pooling方式 | pooling内dropout | pooling后LayerNorm |
| ---- | -----------------------: | ---------------: | -----------------: |
| Mean | 对有效token hidden求平均 |                - |                  - |
| FLaG |                  全局FFT |              0.1 |                 无 |
| E12  |           局部STFT,16/16 |                0 |                 有 |

**R：**

| 模型 |           test AP |     test Accuracy |           test F1 |
| ---- | ----------------: | ----------------: | ----------------: |
| Mean | 0.428907±0.000003 |        0.992040±0 |         0.46039±0 |
| FLaG | 0.721611±0.009747 | 0.993980±0.000298 | 0.667938±0.007364 |
| E12  | 0.765655±0.023788 | 0.994495±0.000423 | 0.690782±0.017933 |

## S2

**M：** 在 S1 基础上扩展到 10 seed。

**R：**

|      |                   |                   |                   |
| ---- | ----------------: | ----------------: | ----------------: |
| FLaG | 0.713121±0.019140 |  0.93857±0.000275 | 0.650671±0.016939 |
| E12  | 0.761344±0.016899 | 0.994485±0.000268 | 0.685348±0.020150 |

## S3

**Q：** 在 Sprint 上施加上述 P2 的 global/local gate/reconstruction 实验。

**R：预测漂移**

| 参数来源    | 只换gate:平均绝对LG-GG cosine差 | 只换重建: 平均绝对GL-GG cosine差 | 全部换成local: 平均绝对LL-GG cosine差 |
| ----------- | ------------------------------: | -------------------------------: | ------------------------------------: |
| Sprint FLaG |                               0 |                         0.032432 |                              0.032432 |
| Sprint E12  |          $8.30 \times 10^{-13}$ |                         0.037477 |                              0.037477 |

**R：validation AP**

| 参数来源    |       GG |       LG |       GL |       LL |
| ----------- | -------: | -------: | -------: | -------: |
| Sprint FLaG | 0.766287 | 0.766287 | 0.764097 | 0.764097 |
| Sprint E12  | 0.800695 | 0.800695 | 0.792300 | 0.792300 |

**C：** 预测变化主要来自 reconstruction，但 local reconstruction 没有带来更高 AP。

原报告随后记录：STSB 数据集上所有模型设置为 `dropout=0` 且有 post norm；Sprint 数据集上所有模型设置为 `dropout=0.1` 且无 post norm。返回确认 FLaG 论文后，发现 text 领域配置为 `dropout=0.1` 且有 post norm。

## S4

**Q：** 在 Sprint 上，global FLaG 与 local STFT 的性能差异是否来自频域算子本身，而不是 dropout、post-pool normalization 等非算子设置？

**M：** 将 global FLaG 与 E12 的设置统一为 `dropout=0`、`post-pool LayerNorm=True`。

**R：10-seed validation AP**

| 模型 | Validation AP |
| --- | ---: |
| Global FLaG | 0.795830 ± 0.017156 |
| E12 | 0.789136 ± 0.014094 |
| E12 − Global | −0.006693 ± 0.017279 |

4/10 seed 为正。

**C：** 没有观察到稳定的 local STFT 性能优势。因此此前 global/local 的明显性能差异不能简单归因于 STFT 算子本身。

## S5

**Q：** Global FLaG 对 dropout / norm 是否敏感？做 $2\times2$ 控制实验会是什么效果？

**M：** `dropout=0/0.1`，`norm=True/False`。

**R：**

| Dropout | Post Norm | Validation AP |
| ---: | :---: | ---: |
| 0.1 | 无 | 0.750936 |
| 0.1 | 有 | 0.819156 |
| 0 | 无 | 0.798974 |
| 0 | 有 | 0.795830 |

**C：** dropout 和 post norm 的效果依赖彼此的设置。
