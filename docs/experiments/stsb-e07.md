### 实验 STSB E07：Token hidden knockout

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 问题与比较对象
 E3 是否对特定 token 位置更敏感？

### 2. 实验设计与控制条件
 固定 FLaG/E3 的 **3-seed checkpoint**。逐个把**内容 token 的最后一层 hidden 向量置零**，其他 token、特殊 token 和 mask 不变，重新计算句对 cosine。

令原预测为 `s`，扰动后为 `s'`，标签为 `y`（原始分数除以 5）。对每个位置计算**绝对平方误差变化**：

$$
R_{\text{token}}=\left|(s'-y)^2-(s-y)^2\right|.
$$

先按相同句子、相同 token 位置对齐三个 seed 的响应取平均，再统计句内位置分布。以下数字**不是** Spearman。

### 3. 观测结果
 STSB test，3-seed 聚合：

| 指标 | FLaG | E3 |
| --- | ---: | ---: |
| 句内平均响应的跨句均值 | 0.003248 | 0.003155 |
| 句内最大响应的跨句均值 | 0.007714 | 0.007616 |
| 句内位置响应标准差的跨句均值 | 0.00195551 | 0.00195521 |

### 4. 基于本次结果的判断
 两组的 token 位置响应接近，没有观察到 E3 更强的位置选择性。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
