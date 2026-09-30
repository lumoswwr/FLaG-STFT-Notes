# STSB：E1–E8 早期实验

## 统一实验设置

本页的 E1–E5 属于 **STSB 上的 pooling 算子探索实验**；E6–E8 不重新训练模型，而是在已经训练好的 FLaG 和 E3 checkpoint 上做机制探针。除表中注明的改变外，早期 STSB 实验保持同一训练协议。**本页结果来自探索性实验，不能仅凭少量 seed 的差异认定 STFT 具有稳定性能优势。**

| 项目 | 配置 |
| --- | --- |
| 数据集及任务 | **STSBenchmark（STSB）**；输入一对英文句子，预测其语义相似程度 |
| 数据划分 | STSB 官方 train / validation（development）/ test；训练使用 train，以 validation Spearman 选择 checkpoint，在 test 上报告结果 |
| 标签与预测 | 原始相似度标签范围 **0–5**，训练时除以 5；两句分别经过 RoBERTa + pooling，预测值为两个句向量的 **cosine similarity** |
| Backbone | RoBERTa-base，**端到端参与微调** |
| 训练目标 | cosine similarity 与归一化金标准分数之间的 **MSE loss** |
| 训练预算 | **3 epochs**；batch size = 4 对句子、gradient accumulation = 4，等效 batch = 16 对句子；最大 token 长度 = 128 |
| 优化配置 | AdamW；backbone learning rate = `1e-5`；pooling learning rate = `1e-3`；weight decay = 0.01；10% warmup，linear learning-rate schedule；gradient clipping = 1.0 |
| FLaG/STFT 共有设置 | 8 个 latent queries、4 个 attention heads、residual gate（`1 + sigmoid`）、最后 masked max pooling |
| **本页早期实验设置** | **pooling dropout = 0，post-pool LayerNorm = True**。这是本轮探索的统一控制条件，**不是**论文中 `dropout=0.1` 的 text 主配置 |
| 早期实验随机种子 | E1–E5 各 **3 seeds（0、1、2）**；E6–E8 读取相同 seed 的 FLaG/E3 checkpoint，不重新训练 |
| 主性能指标 | **test Spearman**：预测相似度与金标准的等级相关系数；**test Pearson**：两者的线性相关系数；均为越大越好 |
| 数值报告方式 | 若写作 `均值 ± 标准差`，则指上述 **3 个 seed 的 test 指标**的均值与样本标准差；配置之间优先报告相同 seed 的配对差值 |

**注意：** 此处 E1–E5 的对照应使用相同 seeds 0–2 的 FLaG 基线（test Spearman `0.841128 ± 0.002177`），不能直接拿后续 **10-seed** FLaG 均值或另一种 dropout/norm 配置当作配对基线。E6 的 Spearman 降幅属于 knockout 后的性能变化，E7 的响应量属于单 token 扰动，E8 的偏好比属于 attention 统计，三者都不是模型原始 test Spearman。

## E1–E8 配置对照

| 编号 | 类型 / 所用模型 | 关键方法及与基线的差别 | Seeds | 主要观察量 |
| --- | --- | --- | --- | --- |
| 基线 | Global FLaG | **整句 rFFT / irFFT**；不加 Hann；共享设置见上表 | 0、1、2 | test Spearman / Pearson |
| E1 | Global FLaG + Hann | global FFT 前按有效句长施加 **Hann window**，其余保持不变 | 0、1、2 | test Spearman / Pearson |
| E2 | STFT-FLaG | **rectangular**；`win=16, hop=8`；overlap = 8；`center=False`；仅生成一个句级 gate | 0、1、2 | test Spearman / Pearson |
| E3 | STFT-FLaG | **rectangular**；`win=8, hop=4`；overlap = 4；`center=False`；仅生成一个句级 gate | 0、1、2 | test Spearman / Pearson |
| E4 | E3 + frame position | 在 E3 上，为送入 latent attention 的局部频谱加入 **learned frame positional encoding** | 0、1、2 | test Spearman / Pearson |
| E5 | E3 + window/centering | 从 E3 改为 **periodic Hann + `center=True`**，仍为 `win=8, hop=4`；**同时改变了窗函数与居中方式** | 0、1、2 | test Spearman / Pearson |
| E6 | FLaG vs E3 探针 | 读取已训练 checkpoint；对最后一层 **content-token hidden** 做 DCT，分 8 个频带，每次置零一个频带、逆变换后重新预测；特殊 token 不动 | 0、1、2 | 各频带 knockout 前后 **Spearman 降幅** |
| E7 | FLaG vs E3 探针 | 读取已训练 checkpoint；逐个把 **content token 的 hidden 向量置零**，保持其他 token 和有效 mask 不变，观察句对预测扰动 | 0、1、2 | 预测及平方误差响应、句内位置响应分布 |
| E8 | FLaG vs E3 探针 | 读取已训练 checkpoint；取频谱 latent-attention 权重，对有效频率 token 统计 **每 token 的 DC / 非 DC 平均注意力之比** | 0、1、2 | 各 seed 的 **DC attention 偏好比** |

**E6 的样本范围：** 使用 test split，但八频带划分只对句对两侧各自至少有 **8 个 content tokens** 的样本实施；因此 E6 的 knockout Spearman 是在符合条件的子集上计算的，不等于完整 test 集的原始 Spearman。**E7 / E8** 则属于已训练模型的推理期诊断，不是额外训练出的新模型。

---


## E1

**Q (Question)：** 整句 FFT 的边界处理会有效果吗？

**M (Method)：** 在 global FLaG 进行 FFT 前，按句子长度加 Hann window。

**R (Result)：** 0.837019 ± 0.002763；相对同 seed FLaG 平均 −0.004109。

**C (Conclusion)：** 未改善，但不等于已排除频谱泄漏。

## E2

**Q：** 使用 STFT 是否会有改善？

**M：** 使用 rectangular window，`win=16, hop=8`，生成一个句级 gate，并非每个窗口一个 gate。

**R：** 0.842204 ± 0.001824；相对 FLaG +0.001076，3 seed 上 2/3 正向。

## E3

**Q：** 检查数据集 sentence length 后发现大部分句子 `length < 16`，也就是 STFT 并没有起到多窗口作用，因此修改 `window` 与 `hop`。

**M：** `win=8, hop=4`。

**R：** 0.843000 ± 0.001047；相对 FLaG +0.001873，3/3 正向；相对 E2 平均 +0.000796。

## E4

**Q：** 上述局部频谱送入 attention 时没有加入帧位置编码，探究帧顺序信息是否有用。

**M：** 在 E3 基础上加入 learned frame position。

**R：** 0.842282 ± 0.001834；相对 E3 平均 −0.000718。

**C：** 放弃帧顺序信息。

## E5

**Q：** 继续探究 STFT 是否造成频谱泄漏影响。

**M：** 在 E3 基础上使用 periodic Hann，`center=True`。

**R：** 0.843027 ± 0.000593；相对 E3 平均 +0.000027，基本持平。

**C：** 后续放弃 Hann 使用。

## E6

**Q：** 尝试解释 E3 带来的提升，使用频带 knockout。

**M：** 沿用 FLaG 论文中的方法，DCT 后频带分组置 0。

**R：** B0 的平均 Spearman 降幅最大；FLaG 0.146694，E3 0.198943。

**C：** 除 B0 影响最大外，在不同 seed 中 E3 和 FLaG 的相对敏感性不稳定。使用 STFT 后与 FLaG 结论几乎持平。

## E7

**Q：** Token hidden knockout。

**M：** 依旧沿用 FLaG。

**R：**

| 汇总指标                   |       FLaG |         E3 |
| :------------------------- | ---------: | ---------: |
| 句内平均相应的平均值       |   0.003248 |   0.003155 |
| 句内最大相应的平均值       |   0.007714 |   0.007616 |
| 句内位置相应标准差的平均值 | 0.00195551 | 0.00195521 |

| 相对位置区域 |     FLaG |       E3 |
| ------------ | -------: | -------: |
| 开头段       | 0.003000 | 0.002850 |
| 前部         | 0.002771 | 0.002790 |
| 中部         | 0.002786 | 0.002719 |
| 后部         | 0.002843 | 0.002780 |
| 末尾段       | 0.002256 | 0.002146 |

**C：** 和 FLaG 模型表现无明显差异。

## E8

**Q：** STFT 是否相较于 FLaG 更关注 DC？

**M：** 读取相同 seeds 0–2 的 FLaG 与 E3 已训练 checkpoint，在 **STSB test** 上提取 latent cross-attention 权重。先对各 latent query 的 attention 取平均（如存在逐 head 权重，也先取 head 均值），得到各有效频率 token 的权重。Global FLaG 仅有一个 DC token（`k=0`）；E3 将各有效局部 frame 的 `k=0` 都计为 DC，并忽略无效 frame。

**“DC 偏好比”定义：** 对**每个句子**，分别计算 DC 组内和非 DC 组内的**每个 token 平均 attention**，再相除：

$
r_{\mathrm{DC}} =
\frac{\left(\sum_{k \in \mathrm{DC}} A_k\right)/N_{\mathrm{DC}}}
{\left(\sum_{k \in \mathrm{nonDC}} A_k\right)/N_{\mathrm{nonDC}}+10^{-12}}
$

其中 `A_k` 是平均后的 attention weight，`N_DC` / `N_nonDC` 分别为对应的有效频率 token 数。**这样除以各组 token 数，是为了避免 E3 的局部 frame 数量更多导致 DC token 更多，从而使单纯总注意力质量不可比。** `r_DC>1` 表示平均每个 DC token 获得的 attention 高于平均每个非 DC token，`r_DC≈1` 表示两组平均接近。表中每个 seed 的值为该 seed 在 test 句子上的偏好比平均值。

该值描述 **attention 分配**，不是频谱能量占比，也不等于 knockout 的性能重要性；不同 global/local 频谱 token 的语义和分辨率也不同，所以不能把比值差异直接当作因果解释。

**R：**

| Seed | FLaG的DC偏好比 | E3的DC偏好比 |
| ---- | -------------: | -----------: |
| 0    |          2.602 |        1.305 |
| 1    |          1.586 |        1.213 |
| 2    |          2.553 |        1.625 |

**C：** 在这三个 seed 的 attention 诊断中，E3 的**每 token DC 注意力偏好比**都低于 FLaG。不过，E6 的 B0 knockout 在汇总统计上曾表现出 E3 的较大性能降幅，两种观察**不能简单认为相互印证**：E6 衡量删除频带后的预测/性能敏感性，E8 衡量模型内部的 attention 分配，二者不是同一种指标。现有 E8 结果只支持“该诊断下 E3 的 DC 相对 attention 较低”，不能推出 DC 对 E3 不重要。
