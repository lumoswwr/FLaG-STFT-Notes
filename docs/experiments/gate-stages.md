### 实验 STAGES：DC、Gate 和 Projection 的向量方向变化

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 从一个疑问开始：直接 Mean 为什么有时不如 FLaG？

对一句话，RoBERTa 输出 $T$ 个 token 的特征矩阵：

$$
H\in\mathbb R^{T\times D},\qquad D=768.
$$

直接 Mean pooling 很简单：把有效 token 对应的向量平均，得到句向量 $m$：

$$
m=\frac{1}{L}\sum_{t\in\mathrm{valid}}h_t.
$$

随后根据两句话的句向量计算 cosine similarity（余弦相似度）。

而 FLaG 的路径复杂得多：

$$
H
\rightarrow \mathrm{rFFT}
\rightarrow \text{频谱}
\rightarrow \text{Gate 门控}
\rightarrow \mathrm{iFFT}
\rightarrow \text{时域池化}
\rightarrow \text{输出线性层}.
$$

其中 Gate 不是固定写死的系数：模型会先用 **latent attention（可学习查询注意力）**汇集频谱信息，再通过多层感知机生成每个隐藏特征通道的缩放系数。

此前实验有一个值得追问的现象：**在 STSB 上直接 Mean 有时好于 FLaG，而在 Sprint 上 FLaG 又明显强于直接 Mean**。于是我们首先尝试将 FLaG 最后的 Max pooling 改成 Mean pooling。它在不同任务上改变了表现，却没有完全解释差异。

这使我们意识到，应该把 FLaG 的复杂路径拆开看：即使最后做 Mean，**Gate 和最后的线性层仍然会改变向量**。

### 2. 第一步：找到 Mean 向量在哪里被改变

#### 2.1 DC 与 Mean 究竟是什么关系？

rFFT 是沿 token 序列这个维度进行的。输出第一个频率位置称为 **DC（零频率分量）**。

对于 FFT 前已将 padding 位置置零的输入，有：

$$
X_0=\sum_{t\in\mathrm{valid}}h_t=Lm.
$$

所以 DC 的**实部**就是有效 token 特征的总和，与 Mean 向量只差一个正的倍数。DC 的虚部理论上为零。

为了找出方向在哪里改变，我们在同一句话上记录原始 Mean、原始 DC、经过 Gate 的 DC、iFFT 后的 masked Mean，以及最终线性层的输出，分别计算向量夹角。

这里的夹角是“同一句话在两个步骤中的表示方向差了多少”，**不是两句话之间的相似度**。

#### 2.2 为什么还要研究 Gate 和线性层的初始化？

仅观察训练完成的 FLaG，难以分清方向变化来自起点还是训练过程。因此比较了以下模型：

| 实验代号 | Gate 的缩放公式 | 输出线性层如何初始化 | 初始化是否严格等于 Mean |
| --- | --- | --- | --- |
| FLaG（原始参数化） | $g=1+\sigma(\ell)$ | 随机矩阵 | 否 |
| A1 | $g=1+\sigma(\ell)$ | 单位矩阵 | 否 |
| B1 | $g=2\sigma(\ell)$ | 单位矩阵 | 否 |
| B2 | $g=2\sigma(\ell)$，且 Gate 最后一层初始化为零 | 单位矩阵 | **是** |

$\sigma$ 表示 sigmoid 函数，$\ell$ 是 Gate 的线性层输出。



#### 2.3 实际测到哪些位置转了角度？

B2、seed 0、完成第一个 epoch 后：

| 同一句话的两个阶段 | STSB 平均角度 | Sprint 平均角度 |
| --- | ---: | ---: |
| 直接 Mean vs. 原始 DC | 约 0° | 约 0° |
| 原始 DC vs. Gate 处理后的 DC | **64.715°** | **77.867°** |
| 直接 Mean vs. iFFT 后的 masked Mean | 64.523° | 77.655° |
| 输出线性层的输入 vs. 输出 | 34.657° | 83.668° |

**我们得到的第一个解释**：FFT 自身保留 DC / Mean 的方向，重要的方向变化主要发生在 **Gate 缩放**和**最后的可学习线性层**。

但这并不是说“大角度导致性能下降”。如果所有句向量被一个共同正交变换旋转，每个句子相对于原来的方向可能变化很大，句子之间的余弦相似度却可以保持不变。

还有一个数学约束：原始 FLaG 的 Gate 位于 $(1,2)$，对 DC 做逐通道正缩放所能造成的最大角度约为 $19.47^\circ$；B2 位于 $(0,2)$，允许部分通道接近零，角度就能接近 $90^\circ$。因此，B2 比原始 FLaG 转得远，部分原因是两种 Gate **允许的缩放范围不同**，不能直接判断模型优劣。


### 证据解释的范围

大角度是单句向量相对于原方向的夹角，不能直接推断句对 cosine 的变化或模型性能优劣。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/flag-gate-projection.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_flag_deep_mechanism.py) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/run_flag_stage_angle_audit_seed0.sh)

[返回实验导航](index.md)
