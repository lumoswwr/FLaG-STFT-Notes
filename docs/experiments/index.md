# 逐项实验导航

本目录以**每个实验一页**组织，先预留你要求的四项原始实验总结，再提供页内 [详细论述](#reading) 链接。所有原始总结尚未提供，因而模板字段为空，未代写任何总结。

## 阅读次序 {#reading}

1. [实验协议与指标](../methods.md)：先确认 test / validation、是否冻结 backbone、checkpoint 选择方式。
2. STSB：全局频域 → 局部窗口 → 机制探针 → Padding / 匹配设置。
3. Sprint：跨任务迁移 → 扩种子 → 算子分离 → 配置对齐。
4. DC、Gate、Projection 和 Mean-compatible：独立训练协议的读出与结构研究。

## STSB E1–E14

- [Global Hann](stsb-e01.md)
- [STFT 16/8（初次局部变换）](stsb-e02.md)
- [STFT 8/4（缩短窗口）](stsb-e03.md)
- [Learned frame position](stsb-e04.md)
- [局部 Hann 与 centering](stsb-e05.md)
- [最终层频带 Knockout](stsb-e06.md)
- [Token hidden knockout](stsb-e07.md)
- [DC attention 偏好比](stsb-e08.md)
- [右侧额外 Padding 与输出漂移](stsb-e09.md)
- [四种 Padding 推理方式](stsb-e10.md)
- [无重叠 STFT 8/8](stsb-e11.md)
- [无重叠 STFT 16/16 与十种子复核](stsb-e12.md)
- [固定全局 FFT 长度](stsb-e13.md)
- [论文 Text 设置下的十种子对照](stsb-e14.md)

## Sprint S1–S5

- [Sprint 早期三种子基线](sprint-s01.md)
- [Sprint 十种子扩展](sprint-s02.md)
- [Sprint Global/Local Gate × Reconstruction](sprint-s03.md)
- [匹配配置的 Global / Local 对照](sprint-s04.md)
- [Global FLaG 的 Dropout × Norm 因子实验](sprint-s05.md)

## P1/P2 与机制

- [实虚门控的循环反射恒等式](mech-reflection.md)
- [FFT Gate 长度与重建长度分离](mech-p1.md)
- [STSB Global/Local Gate × Reconstruction](mech-p2.md)

## DC 与频带

- [DCT 频带 Knockout、Band-only 与层深变化](dc-band.md)
- [训练时删除或仅保留 exact DC](dc-exact.md)

## Gate 与 Projection

- [DC、Gate 和 Projection 的向量方向变化](gate-stages.md)
- [跨句 Gate 变化与 Sigmoid 饱和](gate-input.md)
- [训练后关闭 DC/AC Gate 与 Projection](gate-counterfactual.md)
- [从头训练 Mean + Projection 与严格同初始化对照](gate-projection.md)

## Mean-compatible 与 Alignment-aware

- [Mean/FLaG 直接残差插值](mean-residual.md)
- [正交 Mean-anchor 与无界 Beta](mean-anchor.md)
- [Alignment-aware 结构与 StopGrad 消融](mean-alignment.md)
- [三项文本任务的统一 5-seed 对照](mean-final.md)

## 数据口径

所有性能结论只对应各实验记录中列明的 split、seed、训练/推理协议。test Spearman、test AP、validation AP、cosine 预测漂移、Gate 张量漂移各自代表不同量，不在同一张表里直接比较大小。
