# 市场状态：方法选择与研究结论

本页只保留方法取舍和有明确样本边界的研究结果，操作见[市场状态](../regimes/README.md)。原资料检索于 2026-09-07 至 09-16，本次未重新联网核验；不能将当时行业介绍或代码缺口当作当前状态。

## 长期决定

| 问题 | 保留的决定与理由 |
|---|---|
| 历史状态还是实时信号 | 先定义不可变历史参考，再研究因果识别；后者不能反过来替历史参考定义长期资本假设 |
| 规则、HMM/GMM 与组合 | 规则保留可解释边界，模型后验只说明内部模型；准确性必须对指定参考与独立样本检验 |
| 聚类如何命名 | 统计簇不自动具有牛／熊／复苏含义；状态字典和映射显式登记，不事后排列标签提高分数 |
| 平滑是否实时 | 单边滤波与双边／全样本平滑分别管理；最终时点能力由完整依赖图决定 |
| 算法颗粒度 | 分类、确认、区间与展示能独立替换就展开；只有递推或拟合联合状态保留耦合内核 |
| Horizon | 历史按完整 episode 测量；模拟期数仅代表网格，终点回到基线也不证明经济持续期 |
| 是否推荐模板 | 数据、长期含义和实时证据不足时保留研究用途，不因技术运行通过进入可信模板推荐 |

当前状态族是 Trend、Risk、Drawdown、Macro。宏观 PIT 发布日未齐时不认证严谨 realtime。Style Cycle 保留旧研究兼容，不作为新建推荐。

## Drawdown Cycle 研究

### 为什么没有使用第一版 HMM 聚类

第一版用 Drawdown、20D return、20D volatility 做 HMM，虽然可以得到统计簇，但其中一个簇的收益/路径含义并不符合“Recovery”。

不能因为有三个 cluster 就把它们人为命名为：

```text
Normal / Stress / Recovery
```

因此舍弃该方案。

### Historical Reference

最终采用明确的经济语义：

- 月频局部峰谷；
- 峰 → 谷，完整跌幅达到 12%：`Stress`；
- 该 Stress 谷底 → 后续完整峰：`Recovery`；
- 其他完整/非压力区间：`Normal`。

它使用未来确认的完整峰谷，因此明确属于事后 Reference。

正式系统对象：

```text
Historical Definition
regime-ca801859cf1041fa9a0660618cc956c2 r1

Run
regime-run-a836fe6aa00c49ce89b6e5be4525690f

Publication
publication-cc3e14f1489d43dcaa9c8792731e663e

Quality Report
reference-quality-6e9359817eebedc950d32540beafaf27f67a5008e5f5a0b5e7e433f8da3347fb
```

Historical Horizon Profile：

| State | 完整 Episodes | Median calendar duration |
|---|---:|---:|
| Normal | 9 | 123 天 |
| Recovery | 9 | 31 天 |
| Stress | 10 | 124.5 天 |

整体转换约 `1.75 次/年`。

三个状态均达到 Historical conditional-estimation minimum episode requirement。

### Realtime Recognition

因果状态机：

- 使用过去 9 个月最高点计算当前 drawdown；
- drawdown ≤ -12% → Stress；
- 压力谷底后反弹 ≥ 5% → Recovery；
- 回撤收窄到 -8% 以内 → Normal；
- 全部只使用当前及过去信息。

正式系统对象：

```text
Realtime Definition
regime-3b710e68ac7f4c5ca370ce31d46087aa r1

Run
regime-run-c198f545d7be46ad8ad34131be58cd4f

Publication
publication-bdfa9bf588b741d5b2c15f2d68d9abb1

Reliability Report
reliability-e573d94a38837fa9dd0d9e8b282913d2a4984cbbc3b9cae03cd0e9c560d1a4a2
```

全比较样本 classification：

- Accuracy: 80.10%
- Balanced Accuracy: 78.49%
- Macro F1: 74.86%

但最终留出验证的独立 Episode 仍不足：

- Normal: 0 complete holdout episodes；
- Recovery: 1；
- Stress: 1。

所以正式状态是：

```text
verification.status = insufficient_evidence
recognition_ready = false
production_eligible = false
verified_states = []
```

这套算法已进入系统作为研究模板和研究对象，但**未授权 TAA 实时决策**。

---

## Style Cycle 研究结论

研究了：

- Small / Large relative cycle；
- Growth / Value relative cycle；
- 三状态版本（含 Balanced）；
- 两状态版本；
- slope / momentum / SMA / EMA 等因果 realtime approximation。

结论：

### Size Cycle

两状态版本比三状态清楚，但不同时间段一致性明显波动，最终留出并不足以形成稳定的“可信实时市场状态”。

### Growth / Value Cycle

2019 后若干参数表现较好，但训练期 Macro-F1 仍偏低，跨时期稳定性不足；不符合“先定义可信 Historical Reference，再验证 realtime recognition”的要求。

因此本轮：

```text
不新增正式 Style Cycle Historical/Realtime 模板
```

现有 `size-rotation-v2` / `growth-value-rotation-v2` 保持旧研究兼容，但从新建 Market State authoring 隐藏。

这是主动拒绝弱模型，不是未完成开发。

---

## 独立研究记录

- [沪深300参考与识别实验](csi300-reference-recognition-study-2026-09-15.md)：冻结训练／验证及资格边界。
- [连续市场状态](../regimes/continuous-state.md)：保留递推公式、覆盖率和真实样本，条件覆盖率不能直接与全覆盖率比较。
- [峰谷方法](../regimes/peak-trough.md)、[平滑方法](../regimes/smoothing.md)：独立专业算法合同。

旧调研提出的 BOCPD、PELT、更复杂持续期／宏观模型属于候选方向，除非能力目录和验证证据明确支持，不写成已实现。历史文献引用保留在原 Git 文档；采用方法的关键来源保留于各专业合同。
