# 第四阶段：STSB 上的经典 LTR 排序目标对照

## 本页实验概览（1 项）

<a id="exp-18-summary"></a>

### 实验十八：七种排序目标在三种池化模型上的对照

- **实验目的：** 在 STSB 任务上，比较 Mean、FLaG-Zero 和原始 FLaG 使用不同 Learning-to-Rank（LTR）目标训练时的排序表现。
- **实验思路：** 固定数据划分、模型训练设置和 checkpoint 选择方式，测试 RankNet、两种 Pairwise Hinge、LambdaRank-style、ListNet KL、Grouped ListMLE、Soft Spearman 七种排序目标，并以 MSE 作为对比，运行 5 个随机种子。
- **实验结果：** 共 3 种模型 × 8 种训练目标 × 5 seeds = **120 组**。下面展示 Train、Validation 和 Test Spearman 的五种子均值 ± 样本标准差。

**[↓ 查看详细实验与数据](#exp-18-detail)**

---

## 详细实验记录与数据

<a id="exp-18-detail"></a>

### 1. 模型与排序目标

**模型设置**

- **Mean**
- **FLaG-Zero：** 从与 Mean 等价的池化初始化出发。
- **原始 FLaG**

**排序目标**

1. **RankNet（Pairwise）：** 对不同标签的样本对使用 pairwise logistic loss。
2. **Pairwise Hinge（Fixed）：** 使用固定 margin 的 pairwise hinge loss。
3. **Pairwise Hinge（Label-aware）：** 使 margin 随标签差距变化。
4. **LambdaRank-style：** 根据 minibatch 内样本交换对应的 NDCG 变化，为 pairwise logistic loss 加权。
5. **ListNet KL（Listwise）：** 比较真实标签与预测分数的 softmax 概率分布。
6. **Grouped ListMLE（Listwise）：** 按相同标签分组，使用有序相关性组的排列概率代理目标。
7. **Soft Spearman（Listwise）：** 基于可微 SoftRank，直接优化排序相关性的近似目标。

同时保留 **MSE** 作为基线。纯排序目标均基于 raw cosine，不混合 MSE 损失。LambdaRank-style 和 Grouped ListMLE 分别是 minibatch 加权版本与并列标签分组代理版本，不等同于所有标准 LTR 设定。

### 2. 训练与评估设置

| 设置 | 配置 |
| --- | --- |
| 数据集 | STS Benchmark（STSB）官方 train / validation / test 划分 |
| 模型 | Mean、FLaG-Zero、原始 FLaG |
| 随机种子 | 0、1、2、3、4 |
| Encoder | RoBERTa-base，端到端训练（不冻结） |
| 训练轮数 | 最多 20 epochs；至少 12 epochs；early stopping patience = 5 |
| 训练 batch size | 32 |
| 评估 batch size | 64 |
| 最大序列长度 | 128 |
| 优化器 | Adam |
| Encoder 学习率 | \(1\times10^{-5}\) |
| Pooling 学习率 | \(1\times10^{-3}\) |
| Weight decay | 0 |
| 排序分数 | Raw cosine；score temperature = 0.05 |
| ListNet 标签温度 | 0.20（标签归一化至 0–1） |
| Checkpoint 选择 | Validation Spearman |
| 本页展示指标 | Train / Validation / Test Spearman |

每个 seed 先根据 **Validation Spearman** 选择最佳 checkpoint，再固定该 checkpoint 评估完整 Train 和 Test；测试集**不参与训练和 checkpoint 选择**。下表均为 5 个随机种子的 **mean ± sample std**；各列最优结果加粗。

### 3. Mean 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| MSE（基线） | 0.980883 ± 0.006974 | 0.887813 ± 0.002137 | 0.859666 ± 0.002207 |
| Soft Spearman | 0.982833 ± 0.006043 | 0.890544 ± 0.001242 | **0.871787 ± 0.002330** |
| RankNet | 0.983139 ± 0.004733 | 0.891177 ± 0.000950 | 0.866908 ± 0.003161 |
| Pairwise Hinge（Fixed） | **0.984306 ± 0.004929** | **0.891492 ± 0.000930** | 0.868886 ± 0.002233 |
| Pairwise Hinge（Label-aware） | 0.978632 ± 0.012509 | 0.889612 ± 0.001938 | 0.870153 ± 0.002685 |
| LambdaRank-style | 0.949146 ± 0.027091 | 0.878351 ± 0.001351 | 0.847456 ± 0.002914 |
| ListNet KL | 0.972930 ± 0.012976 | 0.879715 ± 0.002021 | 0.864924 ± 0.002385 |
| Grouped ListMLE | 0.969734 ± 0.010516 | 0.890560 ± 0.001272 | 0.865785 ± 0.001285 |

### 4. FLaG-Zero 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| MSE（基线） | 0.982181 ± 0.002768 | 0.880521 ± 0.002249 | 0.852086 ± 0.003844 |
| Soft Spearman | 0.980935 ± 0.002762 | 0.882284 ± 0.001639 | 0.855805 ± 0.003278 |
| RankNet | 0.980722 ± 0.003220 | 0.881677 ± 0.001521 | 0.853477 ± 0.001978 |
| Pairwise Hinge（Fixed） | **0.983110 ± 0.002503** | 0.881495 ± 0.001188 | 0.854851 ± 0.003354 |
| Pairwise Hinge（Label-aware） | 0.980794 ± 0.003802 | **0.882708 ± 0.000766** | **0.860249 ± 0.001024** |
| LambdaRank-style | 0.953559 ± 0.012417 | 0.868304 ± 0.001971 | 0.839680 ± 0.004394 |
| ListNet KL | 0.980037 ± 0.004266 | 0.871968 ± 0.003235 | 0.851010 ± 0.004716 |
| Grouped ListMLE | 0.963735 ± 0.006698 | 0.874172 ± 0.002612 | 0.848842 ± 0.005220 |

### 5. 原始 FLaG 结果

| 排序目标 | Train Spearman | Validation Spearman | Test Spearman |
| --- | ---: | ---: | ---: |
| MSE（基线） | 0.979631 ± 0.001627 | 0.874077 ± 0.002288 | 0.845427 ± 0.002017 |
| Soft Spearman | 0.975530 ± 0.005502 | 0.872682 ± 0.002478 | 0.848276 ± 0.002470 |
| RankNet | 0.974177 ± 0.007443 | 0.871636 ± 0.000625 | 0.843232 ± 0.004246 |
| Pairwise Hinge（Fixed） | 0.976544 ± 0.005794 | 0.873738 ± 0.001429 | 0.848093 ± 0.006993 |
| Pairwise Hinge（Label-aware） | **0.982147 ± 0.002159** | **0.877168 ± 0.003000** | **0.852488 ± 0.004229** |
| LambdaRank-style | 0.957079 ± 0.012692 | 0.860642 ± 0.002489 | 0.826439 ± 0.003789 |
| ListNet KL | 0.975730 ± 0.006503 | 0.867436 ± 0.001452 | 0.842925 ± 0.007009 |
| Grouped ListMLE | 0.951730 ± 0.011394 | 0.866297 ± 0.003806 | 0.831854 ± 0.009157 |

### 6. 结果观察

**各模型的最高均值**

| 模型 | Train Spearman 最高 | Validation Spearman 最高 | Test Spearman 最高 |
| --- | --- | --- | --- |
| Mean | Pairwise Hinge（Fixed）：**0.984306** | Pairwise Hinge（Fixed）：**0.891492** | Soft Spearman：**0.871787** |
| FLaG-Zero | Pairwise Hinge（Fixed）：**0.983110** | Pairwise Hinge（Label-aware）：**0.882708** | Pairwise Hinge（Label-aware）：**0.860249** |
| 原始 FLaG | Pairwise Hinge（Label-aware）：**0.982147** | Pairwise Hinge（Label-aware）：**0.877168** | Pairwise Hinge（Label-aware）：**0.852488** |

**同一种排序目标下的模型对比**

- 在 **Validation** 和 **Test** 上，七种 LTR 排序目标以及 MSE 基线均呈现 **Mean > FLaG-Zero > 原始 FLaG**。
- 从五个种子的配对结果看，所有这些训练目标下，FLaG-Zero 和原始 FLaG 在 Test Spearman 上均没有超过相应的 Mean。
- 三种模型各自的最佳 Test Spearman 均值分别是 **Mean 0.871787**、**FLaG-Zero 0.860249**、**原始 FLaG 0.852488**。

**实验观察：** 即使采用不同的 Pairwise / Listwise 排序目标，**FLaG 和 FLaG-Zero 在 STSB 上仍然不如 Mean**。

**[↑ 返回实验概览](#exp-18-summary)**
