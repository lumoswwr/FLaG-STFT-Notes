### 实验 STSB E11：无重叠 STFT 8/8

- **实验目的：**
- **实验思路：**
- **实验结果：**
- **实验结论：**

[查看本实验的详细论述](#detail)

> **总结原文预留区**：尚未收到四项正式实验总结。以上四项保持空白，等待粘贴你的原文；下文仅重构现有实验记录，不代写总结。

---

## 实验详细论述 {#detail}

### 1. 实验问题

比较窗口长度为 8 或 16 时，有重叠和无重叠 STFT 的 STSB test Spearman 是否相同。

### 2. 配置与训练

四种实验均使用 rectangular window、`center=False`。E3/E2 分别为 win8/hop4、win16/hop8；E11/E12 分别为 win8/hop8、win16/hop16。各重新训练 3 seeds（0–2），除此之外保持原 STSB 协议。

### 3. 实验结果

**Test Spearman，3-seed mean ± sample std**：

| 窗长 | 半窗口重叠 | 无重叠 |
| ---: | ---: | ---: |
| 8 | E3：0.843000 ± 0.001047 | E11：0.842166 ± 0.001423 |
| 16 | E2：0.842204 ± 0.001824 | E12：0.843419 ± 0.000531 |

### 4. 本实验支持的判断

win8 的重叠版本均值高出 0.000834；win16 的无重叠版本均值高出 0.001215。两种窗长的差异方向相反，因此此对照不支持“重叠总是更优”的一般性判断。

### 原始证据与实现

[旧版实验记录](https://github.com/lumoswwr/FLaG-STFT-Notes/blob/report-old-version/docs/stsb/02-padding.md) · [代码与记录 1](https://github.com/lumoswwr/AMPCliff/blob/FLaG-STFT-mechanism/text_repro/STFT_MECHANISM_SUMMARY.md)

[返回实验导航](index.md)
