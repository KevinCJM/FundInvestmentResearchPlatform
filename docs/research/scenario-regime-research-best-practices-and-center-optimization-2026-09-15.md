# 情景 / Regime 研究最佳实践与情景算法中心后续优化

日期：2026-09-15\
分支：`ISSUE2609/BetterSaaTaa`

> 本文结合经典 Regime / Business Cycle 研究、概率预测与机构 CMA 实践，以及当前代码事实，给出情景算法中心下一阶段的产品与技术优化方案。核心原则：Historical Reference 是研究定义产生的参考 chronology，不是绝对真值；Realtime Recognition 是在当时时点信息下对该定义的识别；CMA 只消费经过可靠性约束的情景信息，并对样本不足的状态保守回退。

---

## 1. 调研结论

### 1.1 Historical Regime 与 Realtime Recognition 是两个不同问题

Bry–Boschan、Harding–Pagan、Pagan–Sossounov 一类研究主要解决：利用完整历史样本，在事后识别周期、峰谷与完整状态区间。它适合建立 **Historical Reference**，但允许使用后续信息，因此不能直接当成实时信号。

Hamilton 的 Markov Switching 与 Chauvet–Hamilton 的实时周期识别强调另一个问题：在时点 `t` 只能使用当时可得信息，估计当前状态或转折是否已经发生。等待额外一个月/季度通常会降低误判，但带来 recognition delay。

因此平台必须保持：

```text
Regime Definition
      ↓
Historical Reference（事后成熟标签）
      ↓
Realtime Recognition（PIT / causal / no repaint）
```

不能要求实时结果逐日、逐月完全复刻事后峰谷边界。

### 1.2 评估实时算法不能只看逐期 Accuracy / Macro-F1

对 Regime 来说，连续多个月属于同一个状态区间，这些观测并不是独立样本。真正独立的研究证据更接近 **Regime Episode / Transition**。

例如：历史上只有 1 个 Sideways 区间，即使它持续 18 个月，也不能把它当成 18 个独立震荡样本。若实时算法没有识别好这个唯一震荡区间，合理结论应是：

> `Sideways = Insufficient Evidence`

而不是：

> `Sideways = Failed`，进而判整个模型失败。

因此状态验证至少应同时展示：

- Observation Count
- Independent Episode Count
- Transition Count
- Precision / Recall / F1
- Interval IoU
- Recognition Delay
- False / Missed Transition

只有在 **独立区间数量足够** 后，低指标才应进入 `Failed`。

### 1.3 概率与置信度必须校准

模型内部 posterior、vote share、规则 one-hot 都不能直接解释为“当前市场状态正确概率”。

正确概念是：

\[
P(S_t^{Reference}=\hat S_t^{Realtime}\mid\mathcal F_t)
\]

即：在给定 Historical Reference 定义下，历史上类似实时判断最终与成熟参考一致的概率。

应使用：

- Brier Score
- Log Loss
- Reliability Diagram
- ECE
- Calibration / Holdout / Prospective split

并继续区分：

```text
Raw Evidence
≠ Calibrated Reference Agreement Probability
≠ Investment Return Probability
```

### 1.4 HMM / Markov 的事后与实时概率必须分开

事后研究可以使用 smoothed probability：

\[
P(S_t\mid\mathcal F_T)
\]

实时识别只能使用 filtered probability：

\[
P(S_t\mid\mathcal F_t)
\]

因此平台现在坚持 retrospective / realtime 的时点隔离是必要的，未来新增任何模型也不能绕过此约束。

### 1.5 Regime 可以影响资产配置，但 CMA 不应变成 TAA

Guidolin & Timmermann 的研究表明，不同 Regime 下股票、债券的联合分布和最优资产配置明显不同，资产配置会随着状态概率变化。

但 J.P. Morgan LTCMA 明确强调 CMA 服务于约 10–15 年的战略资产配置，不是短期战术配置工具。BlackRock 也用不同场景表达长期收益路径的不确定性；MSCI Macro-Finance Model 则把宏观情景、现金流/贴现率、长期风险收益和资产配置连接起来。

因此平台正确关系应是：

```text
Base CMA
   +
Historical / Scenario Conditional CMA
   +
Realtime Regime Probability（只作为条件化证据）
   ↓
CMA Ensemble
   ↓
SAA

TAA = 另一个更短周期的下游决策层
```

Realtime Regime 不能直接重写长期 CMA，也不能把 CMA 做成短期择时。

### 1.6 稀有 Regime 的正确处理是 Shrinkage，而不是硬估计

如果某状态只有 1–2 个独立 Episode，直接用该状态历史平均收益估计 `mu_state` 会产生巨大估计误差。

更合理的是：

\[
\mu_s^* = w_s\mu_s + (1-w_s)\mu_{base}
\]

样本越少，`w_s` 越小；最终逐步回退长期 Base CMA。

Covariance / Correlation 也应采用类似 shrinkage，而不是因为存在一个 Regime 标签就认为条件分布估计可靠。

---

## 2. 机构实践对平台设计的启示

### BlackRock

2026 CMA 使用多个经济场景表达未来路径，并强调 CMA 本身需要反映平均估计值的不确定性，而不是绑定单一预测结果。

平台启示：

- Base CMA 必须独立存在。
- Scenario / Regime CMA 是条件化版本，不是唯一真值。
- 不确定性应进入 CMA 输出和组合优化，而不是只给一个点估计。

### J.P. Morgan

LTCMA 用于 10–15 年战略配置，并明确不是短期 tactical allocation 工具。

平台启示：

- CMA 与 TAA 必须保持业务边界。
- Realtime Regime 可以改变情景权重，但不能把长期预期收益变成月度择时信号。

### MSCI

Macro-Finance Model 将宏观场景映射到现金流、贴现率、风险和长期预期收益，用于 CMA、Risk Management、Scenario Analysis 与 Asset Allocation。

平台启示：

- 情景算法中心应输出标准化状态 / 概率 / 可靠性证据。
- CMA Center 负责把这些情景映射到 `mu / sigma / correlation / distribution`。
- 两个中心不能相互嵌入数值实现。

---

## 3. 当前代码事实

当前情景算法中心已经具备较完整的基础架构。

### 3.1 顶层工作区已经合理分离

`frontend/src/pages/ScenarioCenters.tsx` 当前分为：

1. **历史状态定义**
2. **实时状态识别**
3. **情景模拟与压测**
4. **全球历史事件库**

历史与实时共用 `HistoricalRegimeWorkbench`，但通过 `purpose` 固定语义，没有重新复制一整套工作台。这一架构应保留。

### 3.2 历史 Reference 已有质量检查

`RegimeQualityPanel` 已经展示：

- 分类覆盖
- 首尾未分类
- 每个状态的 Observation 数
- **每个状态的独立 Segment 数**
- 最短 / 中位 / 最长 / 平均持续长度
- 区间收益
- 参数稳定性

这意味着系统实际上已经具备判断“Sideways 只有 1 个独立 Episode”的数据基础。

### 3.3 Realtime Reliability 已有较丰富的验证指标

`RegimeReliabilityReportView` 当前已经展示：

- Overall Accuracy
- Balanced Accuracy / Macro F1
- 每状态 Precision / Recall / F1 / IoU
- Confusion Matrix
- Interval IoU
- Recognition Delay
- Missed / False Transition
- Stability
- Bootstrap / Confidence Interval
- Brier / LogLoss / ECE
- Calibrated Confidence
- Selective Classification Coverage / Error

这部分指标已经足够丰富，问题主要不是“缺指标”，而是**如何解释和汇总这些指标**。

### 3.4 当前主要语义缺口

当前 `Reliability` 仍然存在几个重要问题：

1. **Block-level `sufficient` 仍依赖每类 observation sample 门槛。**
2. `Macro F1` 仍作为显眼整体指标展示，Rare Regime 会被等权放大。
3. 后端 / 前端只有整体 `retrospective_only / insufficient_evidence / eligible`，没有明确的 **per-state verification status**。
4. 当前报告能显示每状态 support，但没有把 Historical Reference 中的 **独立 Episode Count** 直接带到每状态实时校验结论。
5. `Failed` 与 `Insufficient Evidence` 尚未形成严格机器契约。
6. CMA 还没有一个正式的、机器可消费的 Regime Evidence 输出契约。

因此现在最重要的优化不是新增更多算法，而是完善验证语义与下游契约。

### 3.5 Prospective Validation 已存在

`ProspectiveService` 已经实现：

```text
登记冻结候选
→ 捕获未来真实判断
→ 等未来参考成熟
→ 评估固定窗口
→ 生成 qualification
```

并且 TAA 已经能够消费 qualification。

因此无需再设计第二套前瞻系统；下一步应该复用同一资格体系，为 CMA 增加“研究可用”和“实时条件化可用”的分层权限。

---

## 4. 情景算法中心下一阶段优化方案

## P0：把验证从“模型一次性 Pass/Fail”改成“每状态证据状态”

新增状态级验证契约：

```yaml
state_validation:
  bull:
    status: verified | insufficient_evidence | failed
    observations: 70
    episodes: 8
    transitions: 8
    precision: 0.74
    recall: 0.53
    interval_iou: 0.45

  sideways:
    status: insufficient_evidence
    observations: 19
    episodes: 1
    reason: insufficient_independent_episodes
```

### 推荐判定原则

`Insufficient Evidence`：

- 独立 Episode 数不足；或
- 该状态转折事件不足；或
- 有效 OOS 预测样本不足。

`Failed`：

- **只有在样本充分后**，Precision / Recall / IoU / Transition 指标仍未达到策略门槛时才能判定。

`Verified`：

- 独立 Episode / Observation / Transition 均满足门槛；
- 状态级识别和概率表现满足策略。

### 整体模型状态

建议增加：

```text
Verified
Partially Verified
Insufficient Evidence
Failed
```

例如 Bull、Bear Verified，Sideways Insufficient Evidence：

> **Overall = Partially Verified**

而不是 Failed。

---

## P0：Episode Count 必须进入实时报告，不只停留在 Historical Quality

当前历史质量报告已有 `segments.per_state`，应复用该结果或同一后端统计内核，在 Reliability Report 中新增：

```yaml
per_state:
  observations
  independent_reference_episodes
  predicted_episodes
  transition_events
  complete_reference_episodes
  evidence_status
```

前端表格从：

```text
状态 / 参考样本 / Precision / Recall / F1 / IoU
```

改成：

```text
状态 / 独立区间 / 观测 / 验证状态 / Precision / Recall / IoU / 转折延迟
```

**独立区间应放在观测数前面。**

---

## P0：整体指标不能让 Rare Regime 错误否决模型

保留 Macro-F1 作为诊断指标，但不要再让它单独决定模型资格。

推荐同时显示三类汇总：

1. `All-observation Accuracy`：描述历史整体匹配。
2. `Verified-state Performance`：只描述已有充分证据的状态。
3. `Coverage of Verified States`：说明模型有多少 Reference 状态真正经过验证。

禁止把 `Insufficient Evidence` 状态从概率质量计算中静默删掉后再把剩余状态重新归一化，避免人为抬高置信度。

---

## P0：增加 CMA-ready Evidence Contract

情景算法中心不要直接计算 CMA，只输出标准化证据：

```yaml
regime_evidence:
  reference_version
  recognition_model_version
  as_of

  state_probabilities:
    bull: 0.68
    sideways: 0.18
    bear: 0.14

  calibrated_reference_agreement:
    bull: 0.74
    sideways: null
    bear: 0.69

  state_validation:
    bull: verified
    sideways: insufficient_evidence
    bear: verified

  overall_status: partially_verified
  qualification_status: research_only | prospectively_qualified
```

CMA Center 只消费这个契约，不直接读取 Regime 内部模型节点。

---

## P0：CMA 的 fallback 规则要提前固化

CMA 推荐使用：

\[
\mu_t=
\sum_{s\in V}p_{s,t}\mu_s
+
\left(1-\sum_{s\in V}p_{s,t}\right)\mu_{base}
\]

其中 `V` 是已验证且 conditional CMA 估计也具备充分样本的状态。

这意味着：

- Verified Bull/Bear → 可以参与条件化 CMA。
- Sideways Insufficient Evidence → 对应概率质量进入 Base CMA。
- **不能把剩余 Bull/Bear 概率重新归一化到100%。**

否则会把“我们不知道 Sideways”错误解释成“市场一定属于 Bull/Bear”。

---

## P1：Historical Reference 增加“用途级样本充分性”

同一个 Reference 可以适合描述历史，却不一定适合估计 CMA。

建议 Historical Quality 增加两层状态：

```text
Definition Quality
- 这套定义能否合理划分历史？

Conditional Estimation Readiness
- 每个状态是否有足够独立区间估计 return / vol / corr？
```

例如：

```text
Sideways
Definition: Valid
Realtime Recognition Validation: Insufficient Evidence
CMA Parameter Estimation: Insufficient Evidence
```

三件事不能混为一个 Pass/Fail。

---

## P1：CMA 条件参数必须 Shrinkage

未来 CMA Center 对每个状态估计：

- Expected Return
- Volatility
- Correlation / Covariance
- Distribution / Tail Risk

都应带：

```yaml
raw_estimate
base_estimate
shrinkage_weight
posterior_or_shrunk_estimate
sample_observations
independent_episodes
uncertainty
```

Rare Regime 自动向 Base CMA 收缩。

这样 Historical Regime 稀缺不会阻塞整个 CMA。

---

## P1：Realtime 页面顶层结论需要更“人类友好”

当前页面已有很多指标，但用户需要先看到结论。

建议报告顶部改为：

```text
模型结论：部分验证通过

Bull      已验证
Bear      已验证
Sideways  证据不足（仅1个独立历史区间）

当前识别：Sideways
当前状态尚未验证 → CMA 回退 Base CMA
```

然后再展开：

- Classification Details
- Transition Details
- Calibration
- Stability
- Lineage

避免用户先看到 Macro-F1、Brier、IoU 后自己推断能不能用。

---

## P1：Prospective Qualification 改成“状态级积累证据”

当前 Prospective Service 是整体固定窗口资格。

下一阶段无需推翻，只需要扩展报告：

```yaml
prospective_state_evidence:
  bull:
    future_episodes: 2
    status: verified
  sideways:
    future_episodes: 0
    status: insufficient_evidence
```

这样真实未来数据到来时，可以逐步从：

`Partially Verified → Verified`

而不是要求一次性等所有 Rare Regime 都出现。

---

## P2：进一步增加模型不确定性，但不要先扩算法数量

目前已有：

- SMA / filters
- Trend Regime
- HMM / Markov / GMM
- Change Point
- Ensemble
- Stability diagnostics
- Calibration

现阶段算子已经足以研究，大量新增模型的边际价值低于验证语义改造。

优先级应该是：

```text
验证语义
→ CMA消费契约
→ Shrinkage
→ Prospective状态级证据
→ 最后才考虑新算法
```

如果未来新增算法，应继续遵守：

- 可编辑组合优先
- 新黑盒节点必须有不可分离数值语义
- PIT / filtered probability
- fixed-signature NJIT
- 不为提高历史评分修改 Historical Reference

---

## 5. 推荐最终产品流程

```text
1. Regime Definition
   ↓
2. Historical Reference
   ↓
3. Reference Quality
   ├─ Definition Quality
   └─ Conditional Estimation Readiness
   ↓
4. Realtime Recognition
   ↓
5. Recognition Validation
   ├─ Per-state Evidence Status
   ├─ Episode / Transition Metrics
   ├─ Stability
   └─ Probability Calibration
   ↓
6. Prospective Validation
   ↓
7. Regime Evidence Snapshot
   ↓
8. CMA Center
   ├─ Base CMA
   ├─ Regime Conditional CMA
   ├─ Shrinkage / Parameter Uncertainty
   └─ Probability-weighted Ensemble
   ↓
9. SAA
```

---

## 6. 下一阶段开发顺序

### Milestone 1 — Validation Semantics

- 新增 per-state independent episode count。
- 新增 `verified / insufficient_evidence / failed` 状态级契约。
- 新增 overall `verified / partially_verified / insufficient_evidence / failed`。
- Macro-F1 降级为诊断指标，不再隐式承担整体通过语义。

### Milestone 2 — CMA Evidence Contract

- 新增只读 `RegimeEvidenceSnapshot`。
- 绑定 reference/model/calibration/qualification hash。
- 明确当前状态、校准概率、每状态验证状态及 fallback。

### Milestone 3 — CMA Conditional Readiness

- Historical Quality 输出每状态 conditional-estimation readiness。
- episode 数不足时标记 `insufficient_evidence`。
- CMA 后端实现 Base shrinkage，而不是硬状态均值。

### Milestone 4 — Prospective State Evidence

- 在现有 Prospective Journal 上增加每状态未来 episode 证据统计。
- 不新增第二套资格系统。

### Milestone 5 — UI 收口

Realtime 页面首先回答：

1. 当前识别什么？
2. 这个状态是否验证过？
3. 置信度是多少？
4. CMA 能不能用？
5. 如果不能，下一步是什么？

技术细节继续放在折叠区。

---

## 7. 当前推荐结论

当前情景算法中心的**大架构已经基本正确，不建议再做大规模模块重构**。

下一步最应该修的是：

> **把“状态样本不足”从“模型识别失败”中彻底分离，并把验证结果做成 CMA 可消费的状态级 Evidence Contract。**

做到这一点后，就可以开始 CMA Center，而不需要等历史上所有 Rare Regime 都积累到大量样本。

### 7.1 2026-09-15 实施状态

上述 P0/P1 核心语义已经落地：

- Historical Quality 按状态输出 `independent_complete_episodes` 与 `conditional_estimation_status`，默认至少3个完整独立区间才进入基础条件估计；不足状态要求在 CMA 侧做 shrinkage 或回退 Base CMA。
- Realtime Reliability 在最终独立评价块按校准门槛后的 accepted prediction 做逐状态验证；默认要求至少3个独立完整区间、5个高置信预测、Precision ≥65%。结果为 `verified / insufficient_evidence / failed`，整体允许 `partially_verified`。
- 样本不足不再进入 Failed；只有独立证据达到门槛但 Precision 仍低于门槛才是 Failed。
- 新增只读 `regime_cma_evidence` 契约。CMA 只读取模型/参考/报告版本、逐状态证据、verified/fallback states 与置信门槛，不读取情景计算图内部实现。
- fallback 规则固定为：未验证状态的概率质量进入 Base CMA，**不能重新归一化到已验证状态**。
- Prospective Validation 也改为逐状态资格；Bull/Bear 可以先取得未来资格，Rare Sideways 可以继续保持 `insufficient_evidence`。消费端在每次决策时检查当前 state 是否属于 `qualified_states`。
- 旧报告没有新字段时仍可读取，但不会被自动升级成新的 CMA Evidence；需要重新验证才能生成新契约。

默认阈值是平台研究默认值，不是学术界的普适常数，允许按资产类别/频率建立治理后的 policy version；不得为了某个模型“通过”而事后降低。

---

## 8. 主要参考资料

### 学术研究

- Bry, G. & Boschan, C. (1971), *Cyclical Analysis of Time Series: Selected Procedures and Computer Programs*. NBER.\
  https://www.nber.org/books-and-chapters/cyclical-analysis-time-series-selected-procedures-and-computer-programs
- Hamilton, J. D. (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica.
- Harding, D. & Pagan, A. (2002), *Dissecting the Cycle: A Methodological Investigation*. Journal of Monetary Economics.
- Pagan, A. & Sossounov, K. (2003), *A Simple Framework for Analysing Bull and Bear Markets*. Journal of Applied Econometrics.\
  https://ideas.repec.org/a/jae/japmet/v18y2003i1p23-46.html
- Chauvet, M. & Hamilton, J. D. (2005), *Dating Business Cycle Turning Points*. NBER Working Paper 11422.\
  https://www.nber.org/papers/w11422
- Guidolin, M. & Timmermann, A. (2007), *Asset Allocation under Multivariate Regime Switching*. Journal of Economic Dynamics and Control.\
  https://www.sciencedirect.com/science/article/pii/S0165188906002272
- Gneiting, T. & Raftery, A. E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA.

### 机构实践

- BlackRock, *Capital Market Assumptions*, 2026.\
  https://www.blackrock.com/us/financial-professionals/insights/capital-market-assumptions
- J.P. Morgan Asset Management, *Long-Term Capital Market Assumptions*.\
  https://am.jpmorgan.com/ca/en/asset-management/adv/insights/portfolio-insights/ltcma/
- MSCI, *The MSCI Macro-Finance Model*, 2024.\
  https://www.msci.com/research-and-insights/paper/the-msci-macro-finance-model

调研时间：2026-09-15。
