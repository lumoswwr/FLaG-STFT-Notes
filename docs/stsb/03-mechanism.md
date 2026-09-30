# STSB 机制：循环反射、P1 与 P2

**本页只讨论“为什么输出会改变、改变发生在哪个模块”。** 一个重建路径会显著改变句向量，并不代表该路径的下游 Spearman 必然更高。所有实证 P1/P2 都是在**固定已训练 checkpoint、推理模式**下进行的干预，而非为每一种杂交架构重新训练。

| 实验 | 问题 | 模型与数据 | Seeds | 主要指标 |
| --- | --- | --- | --- | --- |
| 循环反射推导 | 共享的实/虚门控在时域究竟对应什么？ | 数学恒等式；另用 float64 数值检查 | 数值检查 `N=1…129` | 数值最大误差 |
| P1 | **同一个 global FLaG**：改变外部 padding/FFT length，预测漂移由 gate 还是 reconstruction 造成？ | STSB 原始动态 FLaG checkpoint；推理探针 | **0、1、2** | 预测 cosine 的平均绝对漂移 |
| P1 补充 | 对称实虚门控、移除反射项，能否重现固定长度带来的变化？ | 同一 checkpoint 的对称门控推理干预 | **0、1、2** | 预测漂移、相似性与验证指标 |
| P2 | **global vs local**：两种完整频域路线的输出差异来自 gate 的生成还是完整的 analysis/reconstruction 路径？ | FLaG-trained 和 E12-trained checkpoints，分别重放 global/local 组合 | 每组 **0、1、2** | cosine 漂移；辅助 validation Spearman |

以上实验使用 STSB 早期匹配配置 **dropout=0、post-pool LayerNorm=True**，P2 的局部路线固定为 **E12（win=16, hop=16, rect, center=False，无重叠）**，与后续论文配置 E14 不是同一组 checkpoint。

---

## 一、循环反射的数学推导

### Q：为什么只是增加右侧零 padding，global FLaG 的有效 token 输出也可能改变？

原 global FLaG 先按 attention mask 将 padding hidden 置零，再对整个 batch padded 长度 `N` 做 global rFFT。虽然新增加的数值全是零，但**变换长度 `N` 会变**。关键在于 FLaG 的 gate 对每个 hidden 通道的 Fourier **实部与虚部分别给出乘数**，且乘数在频率 bin 之间共享。

### M：把一维、一个通道的操作写成时域形式

考虑进入 global FFT 的一条实值序列 `x[t]`，长度为 `N`，无效位置已补零。它的 DFT 为：

$$
X[k]=\sum_{t=0}^{N-1}x[t]e^{-i2\pi kt/N}.
$$

对于**给定样本和给定通道**，把 gate 的实际实部乘数、虚部乘数记为：

$$
a=1+g_{\mathrm{Re}},\qquad b=1+g_{\mathrm{Im}}.
$$

`g_Re` 与 `g_Im` 来自样本条件 latent gate；对于**同一条序列/同一通道**，`a,b` 在各个频率 bin 共享，**不是**在不同样本或不同通道上恒定。门控后频谱：

$$
Y[k]=a\,\mathrm{Re}\,X[k]+ib\,\mathrm{Im}\,X[k].
$$

用原频谱及其共轭展开：

$$
Y[k]=\frac{a+b}{2}X[k]+\frac{a-b}{2}\overline{X[k]}.
$$

记 \(\alpha=(a+b)/2\)，\(\beta=(a-b)/2\)。对实值序列，共轭频谱的逆 DFT 对应**循环反射**，因此：

$$
\boxed{y[t]=\alpha x[t]+\beta x[(-t)\bmod N].}
$$

它表示两条路径：原位置 token 缩放（\(\alpha x[t]\)）与来自**循环反射位置**的 token 混入（\(\beta x[(-t)\bmod N]\)）。这描述的是**逆 FFT 后、最终 masked max pooling 前**的 hidden，**不是**直接描述最终句向量的线性公式。完整实值 DFT 的证明与实现的 rFFT/irFFT 半谱重建在这一条件下等价。

**循环反射不是普通倒序。** 例如 `N=4` 时，位置 `[0,1,2,3]` 反射到 `[0,3,2,1]`，其中位置 0 自反。当 `a=b` 时，\(\beta=0\)，反射混合消失，只剩逐通道缩放。

### R：改变 N 就能改变反射来源

构造固定 gate 的例子：`x=[10,20,30,40]`，令 `a=1.8,b=1.2`，即 \(\alpha=1.5,\beta=0.3\)。

- `N=4` 时，`y[1]=1.5×20+0.3×40=42`。
- 相同有效序列后再补 4 个零，`N=8` 时，位置 1 的反射来源变成位置 7（零），故 `y[1]=1.5×20+0.3×0=30`。

即使**暂时固定 gate**，仅改变 `N`，重建后的有效位置 hidden 也可以改变。更一般地，若有效长度 `L` 且 `N≥2L−1`，则所有**有效位置 `t>0`** 的反射来源均位于零 padding 区。

实现校验：在 float64 下比较原 `rFFT → 实虚乘 gate → irFFT` 与上述等式，在 `N=1…129` 的数值检查中记录的最大绝对误差约 `3.55×10^-15`。

### C：这是结构解释，不是完整性能解释

global FLaG 对 FFT length 的敏感性，在结构上可通过**实虚非对称 gate 引发的、与 `N` 有关的循环反射**理解。不过实际模型还会根据频谱**重新生成 gate**。因此下一步 P1 要把“gate 变了”和“同 gate 下 reconstruction 变了”分开检查。

---

## 二、P1：global FLaG 内部，gate length × reconstruction length

### Q

改变外部 padding 时，最终预测可能经两条路线改变：**①** 不同 FFT length 的频谱使 latent attention/gate `a,b` 改变；**②** gate 即使近似不变，reconstruction length 变化也会改变循环反射位置。哪个在当前 checkpoint 中占主要作用？

### M：2×2 推理干预

读取 **3 个 seeds（0–2）** 的原始动态 global FLaG 已训练 checkpoint（早期 `0/True`），保持模型权重、数据及其他参数不变。对同一批样本分别构造：

- `N_g`：仅用于获取频谱并**生成 gate**的 FFT length；
- `N_r`：gate 应用于该长度的**global 频谱**，并使用该长度做 irFFT reconstruction；
- `T`：当前正常 batch padded length；`128`：固定长度。

| 条件 | gate 观测 `N_g` | gated spectrum 与 reconstruction `N_r` | 想隔离什么 |
| --- | ---: | ---: | --- |
| **TT** | T | T | 原生 FLaG |
| **128T** | 128 | T | 只改变生成 gate 所看频谱的长度 |
| **T128** | T | 128 | gate 来源不变，只换 reconstruction 长度 |
| **128128** | 128 | 128 | 两个长度同时改变 |

对每个句对依次计算这四种条件的 cosine，报告**与 TT 的平均绝对预测差**；它测的是同 checkpoint、同输入下的**推理漂移**，不是训练出的四个模型 Spearman 差异。P1 默认使用 **STSB validation** 的推理探针协议；这些数值不可直接与 STSB official **test Spearman** 对照。

### R

| 配置 | 相对 TT 的预测 cosine 平均绝对漂移（3-seed mean ± std） |
| --- | ---: |
| TT | 0 |
| 128T | \((4.80\pm2.84)\times10^{-7}\) |
| T128 | **0.005520 ± 0.000805** |
| 128128 | **0.005520 ± 0.000805** |

门控观测长度的变化几乎不影响当前这批模型的 gate/预测，而重建长度的变化带来更大的预测漂移。所测 gate cosine 近似 1，但**不是关于所有可能 checkpoint 的数学定理**。

### C

**P1 的结论只针对 global FLaG 的 padding/FFT-length 敏感性：当前预测漂移主要由 reconstruction length/反射支撑改变造成，而非 latent attention 因变换长度变化得到明显不同的 gate。** 这还没有比较 local STFT。

---

## 三、P1 补充：实虚对称门控与反射项

### Q

既然门控的实虚不对称会产生 \(\beta x[(-t)\bmod N]\) 反射混合，那么在相同 checkpoint 下**强制对称门控**，能否重现 P1 中固定 reconstruction length=128 带来的大部分预测变化？

### M

使用同样的 global FLaG checkpoint 做推理反事实，不重新训练。保留 gate 给出的两个实际乘数 `a,b`，将它们共同替换为 **\((a+b)/2\)**，使新乘数满足 `a=b`，从而令反射系数 \(\beta=0\)，记为 `SYM`。将 `SYM` 与原生 `TT`、P1 的 `T128` 比较。

注意：**T128 本身不保证反射项对所有有效 token 精确为零**；只有在满足前面 `N≥2L−1` 等条件的位置上，反射源才一定落入零 padding。这个实验检验的是**相似的预测扰动是否来自反射混合**。

### R

| 推理比较 | 3-seed 平均绝对 cosine 差 |
| --- | ---: |
| SYM − TT | **0.005576 ± 0.000761** |
| T128 − TT | **0.005520 ± 0.000805** |
| SYM − T128 | **0.000295 ± 0.000003** |

句对层面，SYM 和 T128 引发的扰动相关约 `0.99853±0.00046`；它们之间的平均绝对差约为相对 TT 的大漂移的 `5.4%`。当前材料还记录 SYM 对 validation Spearman 的改变仅在 `1e-4` 数量级，未见一致性能收益。

### C

移除实虚非对称门控诱发的反射项，能**高度近似**固定 reconstruction length 的预测扰动。因此循环反射是当前 global FLaG padding 漂移的主要结构性解释；但这**不能证明反射导致 STSB 性能落后**，也不能据此声称去掉反射一定更好。

---

## 四、P2：global vs local，到底哪条路径变了？

### Q

P1 研究**同一种 global 算子内部**的 FFT length。P2 研究**两个不同算子**：把 global FLaG 换成 E12 local STFT 后，最终句向量/预测差异主要来自**gate 的频谱观测**，还是来自**完整的 global/local analysis + inverse reconstruction 路线**？

### M：在同一套权重上做 GG / LG / GL / LL

同一组 hidden 同时生成 global 频谱 `F_G` 和 E12 局部频谱 `F_L`。分别让同一份权重的 latent attention 读取两种频谱，得到**样本级通道 gate** `g_G`、`g_L`。这个 gate 对频率 bin 共享，形状为 `[2D]`，因此 global/local 生成的 gate 都能乘到两种频谱上。之后独立改变 gated spectrum **及其匹配的完整 inverse reconstruction**：

| 模式 | 生成 gate 的观测来源 | 实际 gated spectrum | 逆变换/重建 |
| --- | --- | --- | --- |
| **GG** | Global `F_G` | Global `F_G` | global irFFT |
| **LG** | Local `F_L` | Global `F_G` | global irFFT |
| **GL** | Global `F_G` | Local `F_L` | **逐帧 local irFFT + frame stitching** |
| **LL** | Local `F_L` | Local `F_L` | 逐帧 local irFFT + stitching |

**GL 有明确的实验意义**：global 频谱**只负责生成 gate**；真正被 gate 调制并做局部逆变换的仍然是**local 频谱**。我们没有把 global rFFT 的长频谱直接塞给长度 16 的 local irFFT。LG 则保持 global 重建，只替换 gate 的来源。这样，`LG−GG` 观察**固定 global 重建时**的 gate 来源效应；`GL−GG` 观察**固定 global gate 时**的完整 global/local 重建路线效应。

该方案分别用两种训练来源执行：**FLaG-trained** 与 **E12-trained** checkpoints，各使用 **seeds 0–2**。每一种来源内，将**相同 state_dict**加载到 global/local operator，eval 模式、同 batch 同 hidden、统一 post-norm 和 projection；核对**手写 GG / LL**是否匹配各自 native forward（实现容差 `2×10^-5`）。**不同训练来源的 checkpoint 权重不能直接混成“固定权重”比较。**

此处的“reconstruction effect”是简写，严格说包含**频谱组织方式 + gate 应用到对应谱 + inverse transform + frame stitching/temporal support**，并非只替换 `irfft()` 单独一个函数。

### R：两个层次，不要混为一谈

第一层是**同一 checkpoint 内句对 cosine 的平均绝对变化**，3-seed 汇总：

| 权重来自 | 仅换 gate 来源 \(\mathrm{mean}|LG-GG|\) | 固定 G gate，仅换完整重建路线 \(\mathrm{mean}|GL-GG|\) |
| --- | ---: | ---: |
| FLaG-trained | \(2.86\times10^{-7}\) | **0.005770** |
| E12-trained | \(2.14\times10^{-7}\) | **0.005970** |

因此在这组固定 checkpoint 中，`LG≈GG`、`GL≈LL`；完整 local/global 输出差异主要体现在**重建路线**，而不是两套频谱生成了显著不同的句级 gate。

第二层是**辅助性能结果**。本探针默认在 **STSB validation** 上执行；3-seed 汇总的 Spearman 为：

| 训练来源 | GG | LG | GL | LL |
| --- | ---: | ---: | ---: | ---: |
| FLaG-trained | 0.867756 | 0.867756 | 0.867590 | 0.867590 |
| E12-trained | 0.867938 | 0.867938 | 0.868096 | 0.868096 |

这些差值很小；其目的在于展示“表示/预测明显变化”和“指标可能只变化一点”能够同时成立。**表内是验证集干预结果，不是四组重新训练的 test 性能。**

### C：P1 与 P2 的区别

**P1**：同一个 global FLaG 面对不同外部 padding length，为什么变？答案：当前 checkpoint 的漂移以**length-dependent global reconstruction** 为主，循环反射提供数学解释。

**P2**：从 global FLaG 换成 local E12，两条算子路线哪里变？答案：当前两组 checkpoint 中，**gate 观测来源的差异很小，完整重建路线的影响明显**。

这两点可以构成一致的机制叙事，但**不能进一步推出 local reconstruction 在 STSB、Sprint 或其他任务上一定位优**。另外，P1/P2 的 gate 近似相同是**所测权重与输入的经验观察**，不是所有模型/所有样本的无条件命题。

---

**实现与核对：** [STSB P1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_gate_reconstruction_2x2.py)、[STSB P2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_global_local_2x2.py)、[global/local pooling](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/factory/pooling/flag_pooling.py)。
