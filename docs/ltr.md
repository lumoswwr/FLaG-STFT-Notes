# 第四阶段：STSB 上的经典 LTR 排序目标对照

## 本页实验概览（1 项）

<a id="exp-18-summary"></a>

### 实验十八：六种经典排序目标在三种池化模型上的对照

- **实验目的：** 在 STSB 任务上，比较不同 Learning-to-Rank（LTR）训练目标下 Mean、FLaG-Zero 与原始 FLaG 的排序指标。
- **实验思路：** 固定数据集划分与训练协议，比较 RankNet、Scaled RankNet、Pairwise Hinge、ListNet、ListMLE 和 LambdaRank-style 六种排序目标，每种模型与目标组合使用 5 个随机种子。
- **实验结果：** 表格如下。数值为 5 个随机种子的均值 ± 样本标准差。

**[↓ 查看详细实验与数据](#exp-18-detail)**

---

## 详细实验记录与数据

<a id="exp-18-detail"></a>

### 1. 模型与排序目标

**模型设置**

- **Mean：** Masked Mean Pooling。
- **FLaG-Zero：** 使用 Mean 等价初始化的 FLaG_B2 控制模型。
- **原始 FLaG：** 按原始实验协议实现的 FLaG。



**排序目标**

1. **RankNet：** 对余弦相似度直接使用 pairwise logistic loss。
2. **Scaled RankNet：** 使用可学习仿射评分 \(s=\exp(\alpha)\,\mathrm{cosine}+b\) 的 pairwise logistic loss。
3. **Pairwise Hinge：** 使用间隔式 pairwise 排序损失；相同标签的样本对不参与损失计算。
4. **ListNet：** 对标签和预测分数的 softmax 分布计算 top-one 概率交叉熵。
5. **ListMLE：** 使用 Plackett–Luce 排列似然；每个 minibatch 按标签降序排列。
6. **LambdaRank-style：** 使用由交换样本对所引起的 NDCG 变化幅度加权的 pairwise logistic loss。



### 2. 训练与评估设置

| 设置 | 配置 |
| --- | --- |
| 数据集 | STS Benchmark（STSB）官方 train / validation / test 划分 |
| 模型 | Mean、FLaG-Zero、原始 FLaG |
| 随机种子 | 0、1、2、3、4 |
| 训练轮数 | 3 epochs |
| 训练 batch size | 32 |
| 评估 batch size | 64 |
| 最大序列长度 | 128 |
| 优化器 | Adam |
| Encoder 学习率 | \(1\times10^{-5}\) |
| Pooling / head 学习率 | \(1\times10^{-3}\) |
| Weight decay | 0 |
| Checkpoint 选择 | Validation Spearman |
| 本页展示指标 | Train / Validation / Test Spearman |

测试集指标在根据验证集 Spearman 选择 checkpoint 后计算。下表均为 5 个随机种子的 **mean ± sample std**。

### 3. Mean 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| RankNet | 0.8294 ± 0.0184 | 0.8129 ± 0.0033 | 0.7673 ± 0.0062 |
| Scaled RankNet | 0.8622 ± 0.0147 | 0.8246 ± 0.0029 | **0.7840 ± 0.0062** |
| Pairwise Hinge | **0.8659 ± 0.0033** | 0.8219 ± 0.0035 | 0.7821 ± 0.0071 |
| ListNet | 0.8341 ± 0.0288 | **0.8281 ± 0.0050** | 0.7836 ± 0.0082 |
| ListMLE | 0.8625 ± 0.0039 | 0.8200 ± 0.0050 | 0.7814 ± 0.0055 |
| LambdaRank-style | 0.7334 ± 0.0082 | 0.7697 ± 0.0053 | 0.7063 ± 0.0067 |

### 4. FLaG-Zero 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| RankNet | 0.8322 ± 0.0072 | 0.8136 ± 0.0058 | 0.7734 ± 0.0069 |
| Scaled RankNet | 0.8572 ± 0.0201 | **0.8190 ± 0.0071** | 0.7863 ± 0.0085 |
| Pairwise Hinge | **0.8996 ± 0.0023** | 0.8181 ± 0.0069 | **0.7916 ± 0.0046** |
| ListNet | 0.8429 ± 0.0209 | 0.8157 ± 0.0080 | 0.7807 ± 0.0096 |
| ListMLE | 0.8626 ± 0.0157 | 0.8159 ± 0.0074 | 0.7809 ± 0.0139 |
| LambdaRank-style | 0.7618 ± 0.0255 | 0.7679 ± 0.0225 | 0.7140 ± 0.0252 |

### 5. 原始 FLaG 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| RankNet | 0.8162 ± 0.0294 | 0.7873 ± 0.0083 | 0.7366 ± 0.0199 |
| Scaled RankNet | 0.8628 ± 0.0310 | 0.7955 ± 0.0071 | 0.7623 ± 0.0143 |
| Pairwise Hinge | **0.8849 ± 0.0047** | 0.8094 ± 0.0041 | 0.7700 ± 0.0115 |
| ListNet | 0.8409 ± 0.0138 | **0.8167 ± 0.0086** | **0.7759 ± 0.0072** |
| ListMLE | 0.8431 ± 0.0131 | 0.7991 ± 0.0096 | 0.7565 ± 0.0142 |
| LambdaRank-style | 0.7755 ± 0.0027 | 0.7714 ± 0.0134 | 0.7123 ± 0.0076 |

### 6. 结果观察

宏观上看使用LTR后FLaG-Zero>Mean>FLaG
但使用LTR后模型整体表现低于MSE方法:0.7916 ± 0.0046(LTR best model)<0.852110 ± 0.002489(Mean mse)

**各模型在不同数据划分上的最高均值**

| 模型 | Train Spearman 最高 | Validation Spearman 最高 | Test Spearman 最高 |
| --- | --- | --- | --- |
| Mean | Pairwise Hinge：**0.8659 ± 0.0033** | ListNet：**0.8281 ± 0.0050** | Scaled RankNet：**0.7840 ± 0.0062** |
| FLaG-Zero | Pairwise Hinge：**0.8996 ± 0.0023** | Scaled RankNet：**0.8190 ± 0.0071** | Pairwise Hinge：**0.7916 ± 0.0046** |
| 原始 FLaG | Pairwise Hinge：**0.8849 ± 0.0047** | ListNet：**0.8167 ± 0.0086** | ListNet：**0.7759 ± 0.0072** |

**三种模型合并比较时的最高均值**

- **Train Spearman：** FLaG-Zero + Pairwise Hinge，0.8996 ± 0.0023。
- **Validation Spearman：** Mean + ListNet，0.8281 ± 0.0050。
- **Test Spearman：** FLaG-Zero + Pairwise Hinge，0.7916 ± 0.0046。

**Test Spearman 按模型查看**

- Mean：六种排序目标中，Scaled RankNet 的均值最高（0.7840）。
- FLaG-Zero：Pairwise Hinge 的均值最高（0.7916）。
- 原始 FLaG：ListNet 的均值最高（0.7759）。

**同一种排序目标下的模型对比**

- RankNet、Scaled RankNet、Pairwise Hinge 和 LambdaRank-style 下，FLaG-Zero 的 Test Spearman 均值高于 Mean。
- ListNet 和 ListMLE 下，Mean 的 Test Spearman 均值高于 FLaG-Zero。
- RankNet、Scaled RankNet、Pairwise Hinge、ListNet 和 ListMLE 下，原始 FLaG 的 Test Spearman 均值低于对应的 Mean；LambdaRank-style 下，原始 FLaG（0.7123）高于 Mean（0.7063）。

**[↑ 返回实验概览](#exp-18-summary)**
