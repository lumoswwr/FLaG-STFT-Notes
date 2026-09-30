# FLaG / STFT Mechanism Experiments

## 当前结构

### STSB

- **E1–E8**：Hann window、STFT、多窗口、位置编码、frequency/token knockout、DC attention。
- **E9–E14**：padding 敏感性、不重叠窗口、固定 padding、dropout / post-pool LayerNorm 配置补充。
- **机制分析**：循环反射的时域推导，以及 P1 / P1 补充 / P2 交叉验证。

### Sprint

- **迁移协议**：STSB 与 Sprint 任务设置差异。
- **S1–S5**：Mean / FLaG / E12 对比、10-seed、global/local gate-reconstruction、配置对齐和 dropout × norm 控制实验。

## 当前研究阶段

目前实验包括：

- STSB 上的 E1-E14
- 循环反射机制分析
- P1 / P2 验证
- Sprint 上的 S1-S5
