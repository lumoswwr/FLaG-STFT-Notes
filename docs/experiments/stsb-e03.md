### 实验 STSB E03：STFT 8/4（缩短窗口）

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 部分短句在 win16 下只有一个有效帧，缩短窗口后效果如何？

#### 2. 实验设计与控制条件
 延续 E2 的做法，改为 `win=8,hop=4`，其余不变，**3 seeds**。

#### 3. 观测结果
 Spearman `0.843000 ± 0.001047`；比 FLaG 平均 **+0.001873**（3/3 正向），比 E2 平均 **+0.000796**。

#### 4. 基于本次结果的判断
 早期观察到小幅改善，因此选择 E3 进行 E6–E8 机制分析；3 seeds 不能证明稳定优势。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
