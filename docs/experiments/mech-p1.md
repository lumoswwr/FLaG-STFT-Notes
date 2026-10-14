### 实验 P1：FFT Gate 长度与重建长度分离

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 padding 导致的预测变化主要来自 gate 变化，还是重建长度变化？

### 2. 实验设计与控制条件
 使用原始动态 FLaG **3 个 checkpoint**，分别控制生成 gate 的 FFT 长度 `Ng` 和应用 gate、逆 FFT 的长度 `Nr`。`T` 是正常 batch padding 长度。

| 配置 | Ng | Nr |
| --- | ---: | ---: |
| TT（原始） | T | T |
| 128T | 128 | T |
| T128 | T | 128 |
| 128128 | 128 | 128 |

### 3. 观测结果
 STSB validation，**3-seed 相对 TT 的平均绝对 cosine 漂移**：

| 配置 | mean ± std |
| --- | ---: |
| TT | 0 |
| 128T（只换 gate 来源） | \((4.80\pm2.84)\times10^{-7}\) |
| T128（只换重建长度） | **0.005520 ± 0.000805** |
| 128128 | **0.005520 ± 0.000805** |

### 4. 基于本次结果的判断
 只改变 gate 输入频谱的长度几乎没有影响，当前 padding 敏感性主要来自**重建长度**。

### P1 补充：去掉反射项

#### 1. 问题与比较对象
 如果强制实、虚 gate 相同，能否重现固定重建长度的预测变化？

#### 2. 实验设计与控制条件
 对同一 checkpoint 令两个实际乘数都等于 `(a+b)/2`，使反射项系数为零，记为 **SYM**。比较 SYM、TT、T128。

#### 3. 观测结果
 STSB validation，3-seed 平均绝对 cosine 差：

| 对比 | mean ± std |
| --- | ---: |
| SYM 与 TT | 0.005576 ± 0.000761 |
| T128 与 TT | 0.005520 ± 0.000805 |
| SYM 与 T128 | **0.000295 ± 0.000003** |

#### 4. 基于本次结果的判断
 SYM 和 T128 的预测变化高度接近，支持**循环反射是当前 padding 漂移的主要来源**。固定到 128 不等于每个有效 token 的反射项都为零：位置 0 始终自反。

### 证据解释的范围

预测漂移是相同 checkpoint 下的句对 cosine 绝对差，不等于 test Spearman 的提高或下降。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/03-mechanism.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_gate_reconstruction_2x2.py) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
