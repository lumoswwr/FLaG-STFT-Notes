# STSB：循环反射、P1 与 P2

本页使用 STSB 早期训练的 checkpoint（`dropout=0`、`post-pool norm=True`），**固定模型参数，仅改变推理计算**。P1/P2 均使用 **3 seeds（0、1、2）**，结果来自 **validation**。这里的“预测漂移”指同一句对两种处理下 cosine 预测的平均绝对差，不是 Spearman 的变化。

## 循环反射的时域推导

**Q：** 为什么给 global FLaG 增加右侧零 padding，也会改变有效 token 的输出？

**M：** 当前 FLaG 在不同频率位置共享同一通道 gate，但对 FFT 的实部、虚部分别使用乘数 `a`、`b`。固定某个样本及 hidden 通道，令输入为 `x[t]`，FFT 长度为 `N`：

$$
Y[k]=a\,\mathrm{Re}(X[k])+ib\,\mathrm{Im}(X[k])
=\frac{a+b}{2}X[k]+\frac{a-b}{2}\overline{X[k]}.
$$

逆 FFT 后得到：

$$
\boxed{y[t]=\frac{a+b}{2}x[t]+\frac{a-b}{2}x[(-t)\bmod N]}
$$

第一项是原位置缩放，第二项是**循环反射位置**的信息混入。反射位置取决于 FFT 长度 `N`，并非普通倒序。

**R：** 固定 `x=[10,20,30,40]`、`a=1.8`、`b=1.2`，则：

| 处理 | 位置 1 的反射来源 | 重建结果 `y[1]` |
| --- | --- | ---: |
| `N=4` | 原位置 3，数值 40 | `1.5×20+0.3×40=42` |
| 右侧补零至 `N=8` | 位置 7，数值 0 | `1.5×20+0.3×0=30` |

即使 gate 完全相同，改变 `N` 也能改变输出。独立 float64 检查 `N=1–129`，时域等式与 rFFT/irFFT 实现的最大误差约 `3.55×10^{-15}`。

**C：** 实、虚门控不一致时，会出现与 FFT 长度有关的循环反射。若 `a=b`，反射项消失。公式只描述**给定 gate 时**的重建，实际模型的 gate 还可能因输入频谱改变，因此继续做 P1。

## P1：Padding 的影响来自 gate 还是重建？

**Q：** 改变 global FFT 长度后，预测漂移主要来自 gate 变化，还是重建长度变化？

**M：** 在同一 FLaG checkpoint 上，分别控制**生成 gate 的 FFT 长度** `N_g` 和**应用 gate、逆 FFT 的长度** `N_r`。设 `T` 是正常 batch padding 长度，固定长度为 128。

| 配置 | `N_g` | `N_r` |
| --- | ---: | ---: |
| TT（原始） | T | T |
| 128T（只换 gate 输入） | 128 | T |
| T128（只换重建长度） | T | 128 |
| 128128（两个都换） | 128 | 128 |

**R：** STSB validation，3-seed **相对 TT 的平均绝对 cosine 漂移**：

| 配置 | mean ± std |
| --- | ---: |
| TT | 0 |
| 128T | \((4.80\pm2.84)\times10^{-7}\) |
| T128 | 0.005520 ± 0.000805 |
| 128128 | 0.005520 ± 0.000805 |

**C：** 改变 gate 输入频谱长度几乎不影响预测；改变重建长度产生明显漂移。当前模型的 padding 敏感性主要来自重建路径。

## P1 补充：去掉循环反射

**Q：** 去掉循环反射项，能否重现固定重建长度带来的预测变化？

**M：** 在相同 checkpoint 上，将实部、虚部乘数都设为 `(a+b)/2`，令反射项系数为零，记作 **SYM**。比较 SYM、原始 TT 与 P1 的 T128。

**R：** STSB validation，3-seed 平均绝对 cosine 差：

| 比较 | mean ± std |
| --- | ---: |
| SYM − TT | 0.005576 ± 0.000761 |
| T128 − TT | 0.005520 ± 0.000805 |
| SYM − T128 | 0.000295 ± 0.000003 |

两种干预引起的句对预测变化相关系数为 `0.99853±0.00046`。

**C：** SYM 与 T128 引起的变化高度接近，支持**循环反射是 padding 漂移的主要来源**。注意固定长度 128 不代表所有 token 的反射项都严格为零，位置 0 始终自反。该结果也不意味着去掉反射会提高 Spearman。

## P2：Global 与 Local 的差异来自哪里？

**Q：** 从 global FLaG 换成 E12（STFT 16/16），改变预测的主要是 **gate 生成**还是**重建路线**？

**M：** 分别使用 3 个 FLaG-trained 和 3 个 E12-trained checkpoint。对**同一个 checkpoint**，交叉改变 gate 的频谱来源与实际重建路线：

| 配置 | Gate 来源 | 施加 gate 的频谱及重建 |
| --- | --- | --- |
| GG | Global | Global FFT → global iFFT |
| LG | Local | Global FFT → global iFFT |
| GL | Global | Local STFT → 局部 iFFT + 帧拼接 |
| LL | Local | Local STFT → 局部 iFFT + 帧拼接 |

其中，**GL 只使用 global 频谱计算 gate，实际被重建的仍是 local 频谱**。每一组对照共享同一 checkpoint 权重，不重新训练。这里的“重建”包括频谱组织、逆变换及局部帧拼接，不只是替换 iFFT 函数。

**R1：** STSB validation，3-seed **相对 GG 的平均绝对 cosine 漂移**：

| 训练来源 | 只换 gate（LG−GG） | 只换重建路线（GL−GG） |
| --- | ---: | ---: |
| FLaG-trained | \(2.86\times10^{-7}\) | 0.005770 |
| E12-trained | \(2.14\times10^{-7}\) | 0.005970 |

**R2：** 同一组 validation 上的 **3-seed Spearman 均值**：

| 训练来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.867756 | 0.867756 | 0.867590 | 0.867590 |
| E12-trained | 0.867938 | 0.867938 | 0.868096 | 0.868096 |

**C：** 两种频谱生成的 gate 在当前 checkpoint 中几乎相同，global/local 预测差异主要来自**重建路线**。但切换路线并未一致提高 Spearman。

**P1 与 P2 的区别：** P1 在 **global 内部**改变 FFT 长度；P2 在**同一模型权重下**切换 global/local 路线。二者都定位了预测差异，但都没有证明 local STFT 会稳定提高性能。
