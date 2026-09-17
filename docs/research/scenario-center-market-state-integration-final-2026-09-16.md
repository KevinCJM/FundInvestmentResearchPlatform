# 情景算法中心：市场状态研究融合、Horizon 与状态算法治理（2026-09-16）

## 1. 本轮目标

本轮不重写已有情景算法执行引擎，而是完成四件事：

1. 从人类用户视角，把“历史状态定义 → 实时状态识别 → 识别有效性验证”融合成一个连续的市场状态研究流程。
2. 明确 Historical Reference 与 Realtime Recognition 的下游边界：Realtime Recognition 不再作为 LTCMA 输入。
3. 用真实数据和当前执行链清理低质量旧模板，旧不可变研究仍能精确读取，但不再进入新建算法目录。
4. 深入研究 Drawdown Cycle 与 Style Cycle；只有研究证据足够的方案进入系统。

---

## 2. 产品结构：情景算法中心从四个并列页签收敛为三个一级模块

新的一级结构：

```text
情景算法中心
├── 市场状态研究
├── 全球历史事件库
└── 情景模拟与压力测试
```

“历史状态定义”和“实时状态识别”不再作为两个平级一级页签暴露给用户。

### 2.1 市场状态研究的人类流程

页面顶部固定展示三步：

```text
① 定义历史参考
        ↓
② 建立实时识别
        ↓
③ 验证识别能力
```

#### 第一步：定义历史参考

用户看到的是“先定义什么叫这个市场状态”，而不是技术上的 retrospective mode。

核心操作：

- 选择或创建 Historical Reference 算法；
- 查看完整历史区间；
- 检查状态支持、独立 Episode 和 Horizon Profile；
- 保存为不可变 Historical Reference；
- 直接点击“下一步：建立实时识别”。

#### 第二步：建立实时识别

核心操作：

- 选择精确 Historical Reference 版本；
- 构建只使用当时可得信息的识别模型；
- 运行 realtime replay；
- 保存模型版本；
- 直接进入“验证识别能力”。

Historical Reference 与 Realtime Model 是不同对象，不做隐式复制或覆盖。

#### 第三步：验证识别能力

第三步复用第二步同一份实时模型草稿，不重新初始化算法。

展示：

- 与 Historical Reference 的状态级匹配；
- calibrated probability；
- 每状态独立 Episode；
- `verified / insufficient_evidence / failed`；
- prospective qualification。

### 2.2 用户体验原则

- 一级模块少：用户只需理解“市场状态 / 历史事件 / 模拟压测”。
- 三步研究关系一直可见，不需要来回跳两个工作区。
- 桌面、平板、320px 手机使用同一逻辑。
- Step 切换支持鼠标、键盘左右键、Home、End。
- Historical 草稿与 Realtime 草稿分别保留。
- Realtime → Validation 复用同一草稿。
- 跨 Historical / Realtime 时，不携带另一个对象的 `definition/revision/template`，避免错误加载。
- 旧 `center=historical`、`center=realtime` 深链接继续兼容到新流程。

底层 immutable artifact 不融合：

```text
Historical Definition / Run / Publication
Realtime Definition / Run / Publication
Reliability Report
Prospective Qualification
```

仍保持各自版本、hash 和血缘。

---

## 3. LTCMA / TAA 消费边界

正式边界：

```text
Historical Regime ──→ LTCMA Research
Realtime Regime   ──→ TAA / Product PIT / Monitoring
```

Realtime Reliability 新结果使用 `recognition_evidence` 语义，不再把 realtime 报告称为 CMA evidence。

LTCMA 后续应消费 Historical Reference 侧的：

- reference identity / hash；
- independent episode count；
- duration statistics；
- state occupancy；
- conditional-estimation readiness。

而不是 realtime confidence。

---

## 4. Horizon：不能由用户声明，必须有证据

### 4.1 Historical Market State Horizon

Historical Regime 的 Horizon 由完整独立 Episode 实际测量，不使用“用户填 long-term / short-term”作为事实。

质量报告输出：

- observation frequency；
- complete independent episodes；
- duration observations P25 / Median / P75；
- calendar duration P25 / Median / P75；
- state occupancy；
- annualized transition rate；
- classified coverage。

首尾未完成区间、Unknown 隔断区间不进入完整 Episode Duration 估计。

因此消费者可以基于真实持续期判断适用性，而不能靠命名绕过门禁。

### 4.2 Scenario / Stress Horizon

未来情景没有已经发生的 Episode，因此不能用 Historical Regime 的方式“测出真实持续时间”。

系统目前明确拆成两件事：

1. `frequency × horizon`：只定义模拟网格长度；
2. `Horizon Evidence`：检查路径在终点是否仍存在冲击。

预览现在展示：

- 频率 × 期数；
- market path 是否在终点回到 baseline；
- baseline tail periods；
- `economic_horizon_status = not_empirically_established`。

如果终点仍有冲击：

> 当前模拟窗口可能截断了传播或恢复过程。

即使终点回基线：

> 也只能说明“模型路径在窗口内结束”，不能证明现实经济冲击必然持续相同时间。

未来若要把 Scenario Horizon 升级为“经济持续期已验证”，还需要额外历史类比、衰减模型、流动性/传导证据；本轮没有伪造这一结论。

---

## 5. 老算法治理

原则：

> 不删除不可变历史研究的读取能力；从新建算法目录中隐藏被证据否定的模板。

已隐藏：

| 模板 | 原因 |
|---|---|
| `historical_gmm_volatility_v1` | CSI300 retrospective / realtime 一致仅约 35%，缺少状态持续结构 |
| `historical_window_mean_change_v1` | 当前真实数据几乎全部落入 Stable，区分能力弱 |
| `historical_trend_ensemble_v1` | 大量 Unknown，状态多数只有 1–2 天，不像稳定市场状态 |
| `size-rotation-v2` | 严谨 ex-post reference 配对后 realtime 稳定性不足 |
| `growth-value-rotation-v2` | 后段表现改善，但跨时期不稳定，训练期不足以支持“可信实时状态” |
| `peak-trough-daily-v2` | CSI300 约 241 次区间转换；牛/熊中位仅 14/12 个交易日，过度切分 |
| `peak-trough-daily-legacy-v1` | 约 189 次转换；属于旧日频兼容逻辑，不再作为默认状态研究 |

这些 ID 仍可 exact lookup，用于旧定义、旧运行、审计和回放。

默认新建目录保留的是可解释的基础构建能力与当前研究价值较高的状态族，例如：

- CSI300 主趋势 Historical Reference；
- CSI300 SMA9 realtime recognition；
- HMM volatility Historical Reference；
- fixed-threshold volatility realtime recognition；
- Macro Clock historical research；
- Drawdown Cycle Historical / Realtime research templates；
- generic blank/graph primitives。

---

## 6. Drawdown Cycle 研究

### 6.1 为什么没有使用第一版 HMM 聚类

第一版用 Drawdown、20D return、20D volatility 做 HMM，虽然可以得到统计簇，但其中一个簇的收益/路径含义并不符合“Recovery”。

不能因为有三个 cluster 就把它们人为命名为：

```text
Normal / Stress / Recovery
```

因此舍弃该方案。

### 6.2 Historical Reference

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

### 6.3 Realtime Recognition

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

## 7. Style Cycle 研究结论

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

## 8. 当前市场状态研究库的定位

目前建议优先研究的状态族：

1. **Market Trend**：Bull / Sideways / Bear；
2. **Market Risk**：High / Normal / Low volatility；
3. **Drawdown Cycle**：Normal / Stress / Recovery（研究可用，实时证据仍不足）；
4. **Macro Cycle**：Recovery / Overheat / Stagflation / Recession（Historical 可研究，PMI/CPI PIT 发布时间未补齐前禁止严谨 realtime）。

Style Cycle 暂不进入可信状态库。

---

## 9. 验收结果

### Backend

- 情景 / Reliability / TAA 主链：265 passed；
- Granularity / Peak-Trough / Source Continuation 等底层：266 passed；
- 被淘汰模板隐藏兼容专项：23 passed；
- Scenario / Horizon / Published Risk 相关回归此前：89 passed；
- Drawdown kernel / template 等相关专项此前：121 passed。

### Frontend

- Full Vitest：146 files / 1119 tests passed；
- TypeScript `tsc --noEmit`：passed；
- design check：无新增回归；
- i18n check：passed；
- production build：passed。

### Browser

新的市场状态三步流程：

- mobile 320：passed；
- tablet 768：passed；
- desktop 1440：passed；
- old historical/realtime deep link compatibility：passed；
- 合计 6/6。

旧 Historical Regime Workbench：

- 10 passed；
- 2 个按测试设计跳过；
- 三种响应式宽度覆盖。

---

## 10. 最终业务边界

```text
                    情景算法中心
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     市场状态研究      历史事件库      情景模拟/压测
          │
   ┌──────┴──────┐
   │             │
Historical     Realtime
Reference      Recognition
   │             │
   │             ├── Product PIT
   │             ├── TAA
   │             └── Monitoring
   │
   ├── Product Historical Analysis
   └── LTCMA Research
```

严格取消：

```text
Realtime Recognition → LTCMA
```

并保持：

```text
Historical Regime → LTCMA Research → LTCMA → SAA
Realtime Regime → Tactical Assumptions / TAA
```
