### 实验 STSB E11：无重叠 STFT 8/8

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

#### 1. 研究问题

STFT 的窗口重叠在相同窗口尺寸下是否提高 STSB test Spearman？

 STFT 是否一定需要窗口重叠？

 比较 win8/win16 与半窗重叠/无重叠，共四种 rect、`center=False` 的 STFT 配置。沿用 E2/E3，并补充重新训练 E11/E12，**每组 3 seeds（0–2）**，其他参数一致。

 **3-seed test Spearman（mean ± std）**：

| 窗长 | 半窗口重叠 | 无重叠 |
| ---: | ---: | ---: |
| 8 | **E3**：hop4，0.843000 ± 0.001047 | **E11**：hop8，0.842166 ± 0.001423 |
| 16 | **E2**：hop8，0.842204 ± 0.001824 | **E12**：hop16，0.843419 ± 0.000531 |

 重叠没有一致优势。后续固定使用结构简单的 **E12：rect、win16/hop16、center=False**。



#### 4. 基于本次结果的判断

win8 时 hop4 为 0.843000、hop8 为 0.842166；窗口重叠在 win8 条件下对应 +0.000834 的三种子均值差，不能推广为所有窗长上的优势。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
