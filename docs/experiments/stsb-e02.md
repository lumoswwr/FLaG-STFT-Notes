### 实验 STSB E02：STFT 16/8（初次局部变换）

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 问题与比较对象
 把整句 FFT 换成局部 STFT 是否改善结果？

#### 2. 实验设计与控制条件
 rect 窗，`win=16,hop=8,center=False`，相邻帧重叠 8 token。所有有效帧的频率 token 共同生成一个句级 gate；逐帧逆变换并重建，**3 seeds**。

#### 3. 观测结果
 Spearman `0.842204 ± 0.001824`，比 FLaG 平均 **+0.001076**，2/3 seeds 正向。

#### 4. 基于本次结果的判断
 观察到小幅正向结果，这一结果只描述当前三种子配置之间的数值差。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/01-experiments.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
