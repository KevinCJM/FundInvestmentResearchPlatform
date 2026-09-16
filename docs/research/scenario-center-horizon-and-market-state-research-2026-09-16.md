# 情景算法中心：Horizon 证据与市场状态研究

日期：2026-09-16\
分支：`ISSUE2609/BetterSaaTaa`

本文记录当前情景算法中心的时间尺度（Horizon）设计、Realtime 与 LTCMA 的职责边界，以及基于当前项目数据对新增市场状态算法的深度研究和实际落地结果。

2026-09-16 审核修复：Historical Quality 的 `conditional_estimation.status` 保留兼容，仅代表完整 Episode 样本门槛，新增 `scope=episode_sample_sufficiency_only`、`ltcma_estimated=false`，不声明已完成预期收益、协方差或 LTCMA 分布估计。历史研究确认发布后，保存结果携带精确 `historical_reference` 三元组；下一步自动带入空白实时草稿，已有绑定须显式替换。

情景预览返回本轮 `publication_request_id`，客户端发布时原样回传。相同请求及说明重试幂等；同请求改说明、改有效期或改内容明确冲突。停用后重新预览生成新请求 ID，可以保留同一定义、路径和有效天数重新发布；原发布与预览制品不改写。旧客户端可省略该字段并沿用原幂等键。重复停用相同说明幂等，不同说明明确冲突。

## 1. 结论

1. **市场状态的 Horizon 不能由研究员填一个“长期/短期”标签决定。** 系统应从 Historical Reference 的独立完整 Episode 中直接测量状态持续期、转换频率和占用率。
2. **情景模拟/压力测试的 Horizon 是模型设计输入，但必须有证据支持。** `frequency × horizon` 只是路径长度；还必须证明该长度能覆盖冲击传播、风险实现或恢复/终点条件。
3. **Realtime Recognition 不再作为 LTCMA 输入。** Realtime 识别服务 TAA、产品 PIT 研究、监控和识别有效性验证；LTCMA 读取 Historical Reference / Historical Quality Evidence。
4. 在当前数据与算子下，新增并正式保存了一套强候选：**沪深300三状态波动率 HMM 事后参考 + 20日已实现波动率固定阈值实时识别**。
5. 该波动率状态虽然实时留出验证很强，但实测状态中位持续期只有约 36–52 个日历日、年化转换约 6.36 次，属于战术/中短周期市场环境，不应进入 LTCMA。这正说明 Horizon 必须由数据验证。

---

## 2. 市场状态 Horizon 如何定义

### 2.1 不采用用户声明作为事实

不使用：

```text
horizon = long_term
```

作为消费授权依据。名称、标签或 intended_use 最多是研究意图，不能覆盖数据事实。

### 2.2 Horizon Profile 使用完整独立 Episode 实测

Historical Reference 的每个状态输出：

- `independent_complete_episodes`
- 持续观测期数 P25 / Median / P75
- 持续日历天数 P25 / Median / P75
- classified occupancy
- 全局 transitions
- transitions per year
- observation frequency
- classified coverage

首端、尾端及由 unknown 隔开的 censored Episode 不进入持续期分位数。

新版报告声明 `calendar_boundary=observation_inclusive_next_observation_exclusive`：状态覆盖从首个观察日起，至下一状态首个观察日（不含）。2020-02-29 至 2020-03-31 的单次月频状态因此为 31 天；日/周频使用同一真实日期边界，不固定乘以 30 或 7。旧报告保持原值，页面明确说明其首末样本跨度口径；重新检查生成新报告。

系统实现：

`backend/historical_regimes/reliability/diagnostic_kernels.py::horizon_profile_kernel`

核心计算使用固定签名 NJIT，并进入启动预热。历史质量报告返回：

```text
horizon_profile.method = empirical_complete_episode_duration
horizon_profile.interpretation = measured_persistence_not_user_declared_horizon
```

### 2.3 为什么不用“平均持续时间”一个数

平均数容易被少数超长周期拉高。默认采用中位数和 P25/P75，同时结合 transitions/year 和 occupancy。

例如两个模型都可能平均持续 60 天，但：

- 模型A大多数状态持续50–70天；
- 模型B大多数状态持续5天，但偶尔一个状态持续500天。

它们显然不应有相同 Horizon 判断。

### 2.4 下游门禁

Horizon Profile 是**证据**，而不是系统自动贴的“长期/短期真理”。后续消费者按用途配置门禁：

- LTCMA：要求状态持续性、独立 Episode 数和跨时期稳定性达到 LTCMA 的长期条件研究要求；
- TAA：可以接受更短的 Episode，但必须有 Realtime/PIT 能力和识别有效性；
- Product Historical Research：可接受更宽范围，只需明确口径。

---

## 3. 情景模拟 / 压力测试 Horizon 如何定义

Scenario 的 Horizon 与 Historical Regime 不同。

Historical Regime 的 Horizon 可以从真实状态 Episode 中**测量**；Scenario 是假设未来路径，因此 Horizon 必然包含模型设计，但不能任意填写。

当前系统已有：

```text
frequency
horizon
method
usage_intent
```

现已保留这些字段，不新增一个可被人工随意指定的 long_term/tactical 标签；预览与发布结果增加 `horizon_evidence`。

当前实现使用固定签名 NJIT，分别记录末期增量、剩余模型响应、累计市场水平。预览与发布的 `horizon_evidence.method=explicit_path_horizon_evidence_v2`；旧版无证据字段仍能读取，不能默认恢复。系统明确标记：

```text
horizon_evidence.economic_horizon_status = not_empirically_established
```

这些字段描述显式路径及已发布模型在指定假设下的响应，不能证明真实经济冲击应持续 N 期。

### 3.1 Historical Replay

Horizon 应等于被引用历史事件/历史冲击的真实窗口长度，或其明确的缩放版本。

验证：

- exact historical window；
- source lineage；
- 如缩放，保留 scale rule；
- 不得把一周事件无解释地延长成十年结构情景。

### 3.2 Factor Path / Macro Scenario

`frequency × horizon` 定义未来路径长度，但还要验证：

- 冲击何时开始；
- 何时达到峰值/谷值；
- 传导是否在窗口内完成；
- 是否设定恢复、均值回归或新稳态；
- 终点仍有多大 residual shock。

末期输入和市场增量均归零时标记 `terminal_status=increments_zero`，否则为 `open_at_horizon`；这不表示累计财富恢复。`remaining_response_status` 在之后没有新增冲击的假设下，延长最多两段各 6 阶滞后所需的 12 期检查 `zero/pending`，对 lag=2 的 `[1,0]` 不会提前宣称结束。`cumulative_level_status` 按 return 单位复利、其他变化单位累计，区分 `recovered/not_recovered`；`[-10%,0%]` 尚未恢复。这里描述市场因子，产品财富仍需产品暴露与压测链计算。经济持续期保持 `not_empirically_established`，直到有独立证据。

### 3.3 Monte Carlo

Horizon 是模拟的未来时间网格。

验证：

- 模型参数频率与 simulation frequency 一致；
- calibration sample 足够；
- 相邻 Horizon（例如 3/6/12 月）敏感性；
- terminal distribution 不由数值边界或人为截断决定。

### 3.4 Regime-Conditioned Scenario

Horizon 需要和状态转移过程匹配：

- empirical episode duration；
- transition matrix persistence；
- simulated state occupancy；
- terminal state distribution。

不能在历史状态通常两个月切换一次的模型上，假设“未来10年一直保持当前状态”，除非这是另一个明确的结构情景，而不是从历史 Markov 过程自然推出的结论。

### 3.5 Market Shock / Stress

压力测试的 Horizon 由**风险实现和可平仓/对冲速度**决定，而不是越长越好。

Fed 的 Global Market Shock 直接按 risk-factor liquidity 设 calibration horizon：2026 方法中，政府证券、FX、公开股票等高流动风险因子使用更短（约1个月）窗口；低流动风险因子可使用更长（约3个月）窗口。宏观偿付能力压力测试则通常需要更长时间来让信用和资产负债表风险显现。

所以系统后续应验证：

```text
Scenario path length
+ shock mechanism
+ liquidity / realization horizon
+ propagation / decay completeness
+ terminal residual
+ neighboring-horizon sensitivity
```

而不是仅验证 `horizon > 0`。

---

## 4. 市场状态研究的用户流程融合

情景算法中心从四个并列页签收敛为三个一级模块：

```text
市场状态研究
全球历史事件库
情景模拟与压力测试
```

“市场状态研究”对用户表现为一个连续三步流程：

```text
1. 定义历史参考
      ↓
2. 建立实时识别
      ↓
3. 验证识别能力
```

第二步绑定第一步发布的精确 Historical Reference；第三步直接复用同一个实时草稿和绑定关系进入 Reliability / Calibration / Prospective Validation。底层仍保留独立不可变的 Historical Definition、Reference、Realtime Definition 和验证制品，不把事后标签与实时模型合并成同一个对象。

旧 `center=historical` / `center=realtime` 深链接继续兼容；新入口使用 `center=market-state&stage=...`。320 / 768 / 1440 三个真实浏览器视口已经通过流程、键盘导航、对比度与横向溢出验收。

---

## 5. Realtime Recognition 与 LTCMA 的最终边界

正式关系：

```text
Historical Reference
    -> Historical Quality / Horizon Evidence
    -> LTCMA conditional historical research

Realtime Recognition
    -> Recognition Evidence
    -> TAA / Product PIT / Monitoring
```

已移除新报告中的：

```text
cma_research_state_evidence
cma_research_ready
Base CMA fallback semantics
```

新报告使用：

```text
recognition_state_evidence
recognition_ready
unverified_state_policy = do_not_authorize_unverified_states
```

新只读接口：

```text
GET /api/historical-regimes/reliability/reports/{report_id}/recognition-evidence
```

旧 `/cma-evidence` 入口保留薄兼容路由，但明确返回 `REALTIME_CMA_EVIDENCE_REMOVED`，避免旧调用静默继续。

旧不可变报告仍可读取；兼容读取时把旧 `cma_research_ready` 投影成新的 `recognition_ready`，不改原 artifact。

---

## 6. 市场状态方法深度筛选

### 5.1 已有：主趋势 Bull / Sideways / Bear

Historical Reference：Pagan-Sossounov / Bry-Boschan 类峰谷与持续期规则。\
Realtime Recognition：闭合月末、9月 SMA、±4% 固定缓冲。

该状态描述较长的权益主趋势，适合：

- Product Historical Analysis；
- TAA / PIT（Realtime）；
- Historical Reference 可作为 LTCMA 的历史条件证据之一；
- Realtime Recognition 不进入 LTCMA。

### 5.2 新增：Volatility / Risk Regime —— 通过

理论基础：Markov-switching / HMM 对金融波动状态的建模是成熟做法；Hamilton & Susmel (1994) 是经典 regime-switching volatility 文献。

#### Historical Reference

```text
CSI300 Close
 -> Daily Return
 -> 20D Realized Volatility
 -> 3-State HMM
 -> High / Normal / Low Risk
```

现有模板：`historical_hmm_risk_v1`。

正式系统对象：

- Definition: `regime-c0f5d721809b4fa4931b1fa2419fc637` r1
- Run: `regime-run-31681ad1dcaf45eeadbe51c652278804`
- Publication: `publication-5abde0e470254e809547ac8b9cb09bf8`
- Quality report: `reference-quality-8ce7713fac06ac957f69fa4947098b72124ab1c6e576a70f45bc7b71a0a6c30f`

实测 Horizon：

| State | 完整 Episode | 中位观测期 | 中位日历天 | P25–P75 日历天 |
|---|---:|---:|---:|---:|
| High Risk | 26 | 29 | 44.5 | 30.25–61 |
| Normal Risk | 51 | 23 | 36 | 21–56 |
| Low Risk | 28 | 36.5 | 52 | 14.5–120.25 |

整体约 6.36 次状态转换/年。因此这是高质量的中短期风险环境状态，不是 LTCMA 长期结构状态。

#### Realtime Recognition

不直接采用“动态实时 HMM”作为正式 companion，因为历史可靠性回放的 expanding-refit 与生产实时 latent model 的冻结初始训练语义不同，容易让验证配方和实际部署模型错位。

最终采用更简单、可解释且生产语义一致的固定规则：

```text
20D realized volatility > 1.40% -> High Risk
20D realized volatility < 0.94% -> Low Risk
otherwise -> Normal Risk
```

阈值只用 2010–2018 的 HMM Historical Reference 选择；2019以后不参与阈值拟合。

结果：

| 时间 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| 2010–2018 training | 95.6% | 95.6% | 95.7% |
| 2019–2022 validation | 92.5% | 92.7% | 92.9% |
| 2023–2026 holdout | **93.7%** | **95.3%** | **92.4%** |

2023–2026 每状态：

| State | Precision | Recall | 完整 Reference Episode |
|---|---:|---:|---:|
| High Risk | 88.4% | 98.8% | 3 |
| Normal Risk | 81.9% | 94.3% | 8 |
| Low Risk | 99.8% | 92.8% | 7 |

正式模板：`csi300-volatility-hmm-reference-recognition-v1`。

正式对象：

- Definition: `regime-eb44ffe485ee44afa7d782eb7ac9d375` r1
- Run: `regime-run-36341ce3ac874f72b8d409602b54546c`
- Publication: `publication-fb9a4ad2275344839ecb79cbdd6ff3ff`
- Reliability report: `reliability-ee0ddd24587d9311271c411769db8ee15bd09e72fa165342c0a5b82ae0b60b8f`

报告状态：`verified` / `recognition_ready=true`，但 `production_eligible=false`：这是回顾留出验证，不是真实未来 Prospective Qualification。

### 6.3 GMM Volatility —— 不作为正式推荐

同一数据上 Historical vs Realtime agreement 约 35.3%，明显弱于 HMM Reference + 固定实时规则。保留为研究 benchmark，不作为首选市场状态。

### 6.4 Window Mean Change —— 淘汰正式候选

当前默认参数下约 4010/4049 观测都为 Stable，缺乏有效状态区分。

### 6.5 Trend Multi-Window Ensemble —— 不作为长期 Reference

大量观测未分类，典型 Episode 只有 1–2 个交易日。适合进一步做 tactical signal 实验，不适合作为中长期市场状态 Reference。

### 6.6 Size / Growth-Value Style Regime —— 本轮不进入正式状态库

本轮分别构造了严格的两状态事后 Reference：

- Size：中证1000 / 沪深300 的月频相对价格完整峰谷周期，Small / Large；
- Growth/Value：沪深300成长 / 沪深300价值的月频相对价格完整峰谷周期，Growth / Value。

再以只使用当时信息的 SMA / EMA / trailing momentum 规则识别这些 Reference。结果不够稳定：

- Size 最好方案的训练期 Macro-F1 约 0.68，2023+ 留出约 0.61–0.72；
- Growth/Value 最好方案训练期 Macro-F1 约 0.62，2019–2022 约 0.78，2023+ 约 0.82。

这说明相对价格的事后完整波段确实能描述 Style Cycle，但当前仅靠相对价格趋势无法跨时期稳定实时识别。按“Historical Reference + Realtime Recognition 必须成对可信”的原则，本轮**不新增正式 Style Cycle 模板**。

既有 `size-rotation-v2`、`growth-value-rotation-v2` 保留历史兼容，但从新建算法目录隐藏。未来若补充估值价差、信用条件或可用的宏观 PIT 数据，再重新研究。

### 6.7 Drawdown Cycle —— 已落研究模板，实时资格仍证据不足

最终采用月频局部回撤周期，而不是“历史最高点直到完全收复”的无限期状态：

```text
Normal
Stress
Recovery
```

Historical Reference：

- 月频局部峰谷；
- 完整 Peak → Trough 跌幅达到 12%：峰后至谷底为 Stress；
- 紧随其后的完整 Trough → Peak：谷底之后为 Recovery；
- 共享峰/谷归属于已完成的前一段：修复终点峰在后续完整下跌段出现后仍保留Recovery；谷底保留Stress；
- 其他完整区间为 Normal；
- 未完成首尾不回填。

Realtime Recognition：

```text
最近9个月高点回撤 >= 12% -> Stress
Stress 后从活动谷底反弹 >= 5% -> Recovery
Recovery 中剩余回撤 <= 8% -> Normal
Recovery 再创新低 -> Stress
```

缺失、非有限或非正价格以及含此类值的观察窗输出未分类并清除活动谷底。恢复有效观察后，只有达到压力进入阈值或正常退出阈值才重新分类；两阈值之间保持未知，不能默认Normal。首个合法观察窗的原始初始化规则保持。

正式研究模板已经进入系统：

- `csi300-drawdown-cycle-reference-v1`
- `csi300-drawdown-cycle-realtime-v1`

此前修正共享谷底边界语义后的正式系统运行记录：2023–2026 留出段 Accuracy **95.2%**、Balanced Accuracy **88.9%**、Macro-F1 **92.4%**；该段 Normal / Recovery / Stress 的独立完整 Episode 分别只有 **0 / 1 / 1**，结论为 `insufficient_evidence`，`recognition_ready=false`。这些数字早于本次共享峰点及缺口重入修复，不能作为最新代码精度验收；新修订需重新研究。既有记录不回写，模板仍不能包装成已经取得实时/TAA资格。

### 6.8 Macro Growth × Inflation —— Historical 可用，Realtime 阻断

当前 `merrill-clock-macro-v3` 在真实 PMI/CPI 数据上可以生成 2005–2026 Historical Reference：Recovery / Overheat / Stagflation / Recession。

但当前 PMI/CPI 的 `available_at` 全为空，`availability_status=release_date_unknown`。因此目前严谨结论是：

```text
Historical: available
Realtime/PIT: blocked by data lineage
```

不能把 observation_date 当 release date 来伪造实时能力。

### 6.9 Cross-Asset HMM —— 有研究价值，暂不落正式模板

以 CSI300 + 中证全债构造权益收益、债券收益、权益20日波动率三特征 HMM，历史状态能形成较清晰的 Stress / Transition / Risk-on 区间，但 naive realtime 对 ex-post Reference 一致率只有约 50.3%。继续作为 P2 研究，不进入正式推荐目录。

---

## 7. 当前状态库结论

### 正式优先维护

1. **Market Trend**：Bull / Sideways / Bear；
2. **Market Risk / Volatility**：High / Normal / Low Risk。

这两类已经形成较完整的 Historical Reference + Realtime Recognition + Reliability 流程。

### 研究候选

3. **Drawdown Cycle**：Normal / Stress / Recovery。模板已进入系统，但最终测试独立 Episode 不足，保持 `insufficient_evidence`；
4. **Macro Cycle**：Recovery / Overheat / Stagflation / Recession。Historical 可用，Realtime 等真实 PIT release data。

### 暂不推荐新建

- GMM Volatility；
- Window Mean Change；
- Trend Multi-Window Ensemble；
- Size Rotation；
- Growth/Value Rotation；
- Cross-Asset HMM 当前版本。

其中旧模板只做不可变历史兼容，不再作为默认作者化入口。不同状态族描述不同经济问题，不应强行统一成一套“唯一市场状态”。

---

## 8. 参考研究

- Hamilton, J. D. & Susmel, R. (1994), *Autoregressive Conditional Heteroskedasticity and Changes in Regime*, Journal of Econometrics.
- Pagan, A. & Sossounov, K. (2003), *A Simple Framework for Analysing Bull and Bear Markets*, Journal of Applied Econometrics.
- Chauvet, M. & Hamilton, J. D. (2005), *Dating Business Cycle Turning Points*, NBER Working Paper 11422.
- Guidolin, M. & Timmermann, A. (2007), *Asset Allocation under Multivariate Regime Switching*, Journal of Economic Dynamics and Control.
- Federal Reserve, *2026 Stress Test Scenarios* / Global Market Shock calibration-horizon methodology.

研究和数据核验日期：2026-09-16。
