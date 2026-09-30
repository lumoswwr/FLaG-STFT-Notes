# STSB：循环反射与 P1/P2

本页在**固定 checkpoint** 下分析预测变化，不重新训练。P1/P2 均使用 **STSB validation、seeds 0–2**；指标“平均绝对预测漂移”指同一句对两种处理方式下 cosine 预测差的绝对值均值，**不是 Spearman 差**。

## 循环反射的时域推导

**Q：** 为什么只增加右侧零 padding，global FLaG 的预测也会变化？

**M：** 固定某个样本和 hidden 通道。FLaG 的 gate 在各频率位置共享，但对 FFT **实部、虚部**分别使用乘数 `a`、`b`：

$$
Y[k]=a\,\operatorname{Re}X[k]+ib\,\operatorname{Im}X[k].
$$

逆 FFT 后可以化为：

$$
\boxed{y[t]=\frac{a+b}{2}x[t]+\frac{a-b}{2}x[(-t)\bmod N]}
$$

其中 `N` 为 FFT 长度。第一项是原位置缩放，第二项是**循环反射**。改变 `N` 就会改变反射位置。

**R：** 取 `x=[10,20,30,40]`、`a=1.8,b=1.2`，位置 1 的重建结果：

| FFT 长度 | 反射来源 | `y[1]` |
| --- | --- | ---: |
| N=4 | `x[3]=40` | 42 |
| 右侧补零到 N=8 | `x[7]=0` | 30 |

独立 float64 检查 `N=1–129`，公式和 rFFT/irFFT 的最大误差约 `3.55×10^{-15}`。

**C：** 给定 gate 时，**实虚不对称门控**会产生随 FFT 长度改变的循环反射；实际 gate 也可能随输入改变，因此做 P1 分离两种来源。

## P1：Gate 长度 × 重建长度

**Q：** padding 导致的预测变化主要来自 gate 变化，还是重建长度变化？

**M：** 使用原始动态 FLaG **3 个 checkpoint**，分别控制生成 gate 的 FFT 长度 `Ng` 和应用 gate、逆 FFT 的长度 `Nr`。`T` 是正常 batch padding 长度。

| 配置 | Ng | Nr |
| --- | ---: | ---: |
| TT（原始） | T | T |
| 128T | 128 | T |
| T128 | T | 128 |
| 128128 | 128 | 128 |

**R：** STSB validation，**3-seed 相对 TT 的平均绝对 cosine 漂移**：

| 配置 | mean ± std |
| --- | ---: |
| TT | 0 |
| 128T（只换 gate 来源） | \((4.80\pm2.84)\times10^{-7}\) |
| T128（只换重建长度） | **0.005520 ± 0.000805** |
| 128128 | **0.005520 ± 0.000805** |

**C：** 只改变 gate 输入频谱的长度几乎没有影响，当前 padding 敏感性主要来自**重建长度**。

## P1 补充：去掉反射项

**Q：** 如果强制实、虚 gate 相同，能否重现固定重建长度的预测变化？

**M：** 对同一 checkpoint 令两个实际乘数都等于 `(a+b)/2`，使反射项系数为零，记为 **SYM**。比较 SYM、TT、T128。

**R：** STSB validation，3-seed 平均绝对 cosine 差：

| 对比 | mean ± std |
| --- | ---: |
| SYM 与 TT | 0.005576 ± 0.000761 |
| T128 与 TT | 0.005520 ± 0.000805 |
| SYM 与 T128 | **0.000295 ± 0.000003** |

**C：** SYM 和 T128 的预测变化高度接近，支持**循环反射是当前 padding 漂移的主要来源**。固定到 128 不等于每个有效 token 的反射项都为零：位置 0 始终自反。

## P2：Global / Local 的 Gate 与重建

**Q：** 从 global FLaG 换成 E12，预测差异主要来自 gate 的频谱来源，还是重建路线？

**M：** 分别加载 FLaG-trained 和 E12-trained 的 **3 个 checkpoint**。对**同一个 checkpoint**交叉更换 gate 的生成来源及实际重建路线：

| 配置 | Gate 来源 | Gate 应用及重建路线 |
| --- | --- | --- |
| GG | Global | Global |
| LG | Local | Global |
| GL | Global | Local |
| LL | Local | Local |

**GL** 表示 global 频谱**只生成 gate**，随后将 gate 应用于 local 频谱，再进行局部逆变换和帧拼接。此处“重建路线”包含频谱组织、逆变换及局部时域拼接，不只是替换 iFFT 函数。

**R1：** STSB validation，**3-seed 平均绝对 cosine 漂移**：

| 权重来源 | 只换 gate：LG−GG | 换重建路线：GL−GG |
| --- | ---: | ---: |
| FLaG-trained | \(2.86\times10^{-7}\) | **0.005770** |
| E12-trained | \(2.14\times10^{-7}\) | **0.005970** |

**R2：** 同批 validation 的 **3-seed Spearman 均值**：

| 权重来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.867756 | 0.867756 | 0.867590 | 0.867590 |
| E12-trained | 0.867938 | 0.867938 | 0.868096 | 0.868096 |

**C：** 两种频谱生成的句级 gate 几乎相同，global/local 预测差异主要来自**重建路线**。但是，重建路线改变并不代表 Spearman 一定提高。

**区别：** P1 改变 **global 内部的 FFT 长度**；P2 改变 **global/local 的分析与重建方式**。
