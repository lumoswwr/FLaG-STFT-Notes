### 实验 COUNTERFACTUAL：训练后关闭 DC/AC Gate 与 Projection

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 4. 第三步：这些操作对预测性能真的有用吗？

前面说明了 Gate 怎么改变向量。接着我们想知道：**如果临时关闭某部分 Gate，训练好的 FLaG 还能正常预测吗？**

我们固定原 checkpoint，不再训练，分别把 DC 的 Gate 改为 1、把非 DC 的 Gate 改为 1、全部改为 1，或者直接跳过输出线性层。

Sprint official test（seed 0）：

| 修改方式 | 原始 FLaG AP | B2 AP |
| --- | ---: | ---: |
| 什么都不改 | **0.848503** | **0.833946** |
| 仅取消 DC Gate 的缩放 | 0.838674 | 0.678850 |
| 仅取消非 DC 频率的 Gate 缩放 | 0.849550 | 0.835823 |
| 取消全部 Gate 缩放 | 0.840069 | 0.784842 |
| 保留 Gate、跳过最后的可学习线性层 | 0.464409 | 0.494782 |

这张表说明：

- **B2 的已训练参数高度依赖 DC Gate**：取消 DC 缩放，AP 从 0.833946 降为 0.678850。
- **在当前 Sprint 设置下，单独取消非 DC 缩放几乎没有损失**，但不能因此说所有非 DC 信息在所有任务中均无意义；STSB B2 的对应修改就出现了性能下降。
- **已训练的模型非常依赖最后的可学习线性层**。不经过它，Sprint AP 大幅降低。

这里尤其要区分两件事：训练完成后把 Projection 删除，模型当然可能崩，因为前面各层已与它共同适应；这并不表示“如果从一开始就不用复杂 Gate，模型就学不到好结果”。

因此，真正有必要做的是：**从一开始就训练一个不含复杂频域模块的简单模型**，而不是继续在现有 checkpoint 中删除组件。

### 证据解释的范围

已有模型对某个模块的后验依赖，不等于该模块在重新训练的新架构中不可替代。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/flag-gate-projection.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/GATE_COUNTERFACTUAL_README.md) · [代码与记录 2](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/probe_flag_gate_counterfactual.py)

[返回实验导航](index.md)
