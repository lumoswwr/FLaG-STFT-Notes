### 实验 REFLECTION：实虚门控的循环反射恒等式

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 为什么只增加右侧零 padding，global FLaG 的预测也会变化？

### 2. 实验设计与控制条件
 固定某个样本和 hidden 通道。FLaG 的 gate 在各频率位置共享，但对 FFT **实部、虚部**分别使用乘数 `a`、`b`：

$$
Y[k]=a\,\operatorname{Re}X[k]+ib\,\operatorname{Im}X[k].
$$

逆 FFT 后可以化为：

$$
\boxed{y[t]=\frac{a+b}{2}x[t]+\frac{a-b}{2}x[(-t)\bmod N]}
$$

其中 `N` 为 FFT 长度。第一项是原位置缩放，第二项是**循环反射**。改变 `N` 就会改变反射位置。

### 3. 观测结果
 取 `x=[10,20,30,40]`、`a=1.8,b=1.2`，位置 1 的重建结果：

| FFT 长度 | 反射来源 | `y[1]` |
| --- | --- | ---: |
| N=4 | `x[3]=40` | 42 |
| 右侧补零到 N=8 | `x[7]=0` | 30 |

独立 float64 检查 `N=1–129`，公式和 rFFT/irFFT 的最大误差约 `3.55×10^{-15}`。

### 4. 基于本次结果的判断
 给定 gate 时，**实虚不对称门控**会产生随 FFT 长度改变的循环反射；实际 gate 也可能随输入改变，因此做 P1 分离两种来源。

### 证据解释的范围

这是在给定频率共享通道 gate 时成立的数学分解，不是对任意频率独立 gate 的一般结论。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/03-mechanism.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/factory/pooling/flag_pooling.py)

[返回实验导航](index.md)
