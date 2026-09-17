# 情景算法中心：状态级证据与 CMA 契约验收

日期：2026-09-15。分支：`ISSUE2609/BetterSaaTaa`。本轮未提交、推送或合并。

## 1. 本轮目标

把“Rare Regime 样本不足”与“算法识别失败”彻底分开，并将 Historical Reference、Realtime Reliability、Prospective Qualification 统一为状态级证据语义。CMA 只消费稳定的 Evidence Contract，不读取情景计算图内部实现。

## 2. 最终业务契约

### Historical Reference

历史定义仍由原计算图产生，不改标签以适配实时模型。质量报告新增：

- 每状态观测数、状态区间数；
- `independent_complete_episodes`：只计两侧都有已知转折的完整独立区间；
- `conditional_estimation_status=ready|insufficient_evidence`；
- 总体 `conditional_estimation=ready|partially_ready|insufficient_evidence`。

默认每状态至少3个独立完整区间才达到基础条件估计门槛。该门槛只表示可以进入 CMA 的 shrinkage / 条件估计研究，不表示参数估计无误差。

### Realtime Reliability

最终独立评价块按校准门槛后的 accepted predictions 做逐状态验证。默认策略：

- 独立完整区间 ≥3；
- 高置信预测 ≥5；
- accepted prediction Precision ≥65%；
- confidence floor = 60%。

状态结论：

- `verified`：证据达到门槛且 Precision 达标；
- `insufficient_evidence`：独立区间或高置信预测不足；
- `failed`：证据已充足但 Precision 仍低于门槛。

整体允许 `partially_verified`。全局 Accuracy、Macro-F1 继续展示，但不再让一个 Rare Regime 自动拖累其他状态。

### CMA Evidence

新增只读接口：

`GET /api/historical-regimes/reliability/reports/{report_id}/cma-evidence`

只返回精确模型/参考/报告身份、逐状态证据、verified/fallback states、confidence floor 和限制说明。固定 fallback：

`base_cma_without_renormalizing_unverified_state_mass`

即未验证状态的概率质量进入 Base CMA，不能重新归一化到已验证状态。

### Prospective Qualification

沿用唯一 Prospective Journal，不创建第二套资格系统。真实未来证据也按状态取得资格：

- `qualified_states` 可先只有 Bull/Bear；
- Rare Sideways 可继续 `insufficient_evidence`；
- assessment 可为 `outcome=partially_qualified`；
- 消费端在每个决策点核对当前状态是否属于 `qualified_states`，否则返回 `calibration_state_not_qualified` 并回退。

总样本、捕获覆盖、完整时间块、逐块 Brier 改善、时点/期限仍是协议级硬门禁。全局参考一致率现在是诊断，不覆盖状态级资格。

## 3. 数值与实现约束

新增 `state_evidence_kernel` 使用 NJIT，加入固定签名启动预热与 runtime audit；核心循环未落回 Python。Historical Quality、Retrospective Reliability 与 Prospective Qualification 复用该状态证据定义。旧报告缺少新字段时仍可读，但不能自动生成新的 CMA Evidence，必须重新验证。

`source.constant` 保持前一轮修复后的语义：默认 `tunable`，只有显式 `parameter_role=structural` 才不参与稳定性扰动。识别数学和历史快照均未因本轮验证语义修改。

## 4. 真实沪深300验收

### 事后参考

固定算法：`market-trend-reference-csi300-v1`。

- 月末沪深300；
- 局部峰谷 + Pagan-Sossounov 式持续期/周期过滤；
- 完整波段 >15% Bull，<-15% Bear，其余 Sideways；
- 首尾未完成区间不外推。

当前全历史独立完整区间：

- Bull：6；
- Sideways：3；
- Bear：5。

因此三状态都达到默认3 Episode 的**基础 CMA 条件估计准备度**。Sideways 只有3个区间，CMA 仍应使用 shrinkage，不应把简单历史均值当高精度长期假设。

系统对象：

- Historical Definition：`regime-9a005c98d024497da3fea853aea8546d` r1；
- Quality Report：`reference-quality-397d581e2299f84dadb3f9bc9394af01b01ea60b2fe6f227596447c59b479255`。

### 实时识别候选

算法：9个月SMA + ±4%缓冲 + 60% class-frequency calibrated confidence。

它是因果、月末闭合、无未来峰谷、不回填历史。此前三段 rolling-origin 研究中高置信匹配率最低约73.68%，三个时间折 Brier 均优于类别基准；这些时间折参与过候选选择，因此只作为稳健性证据，不冒充独立最终验收。

最终独立 2023–2026 holdout 的状态证据：

- Bull：22个参考观测、1个完整独立区间、14个高置信预测、Precision 64.29% → `insufficient_evidence`；
- Sideways：7个参考观测、1个完整独立区间、0个高置信预测 → `insufficient_evidence`；
- Bear：12个参考观测、0个完整独立区间、6个高置信预测、Precision 83.33% → `insufficient_evidence`。

三个状态都因 Episode 不足而没有 Verified，不是 Failed。校准概率在该 holdout 仍优于类别基准，但这不能替代独立 Regime 数量。

因此当前 CMA Evidence：

- `status=insufficient_evidence`；
- `research_ready=false`；
- `production_eligible=false`；
- `verified_states=[]`；
- `fallback_states=[bull, sideways, bear]`。

最新完整月 2026-08-31 原始识别 Sideways，校准匹配概率约18.18%，低于60%门槛，同样明确回退 Base CMA。

系统对象：

- Realtime Definition：`regime-461c1de9243f427787be0c412bb97627` r3；
- Reliability Report：`reliability-36a8e68a1597457b72caa3286ab58f10c603b0223dd28b04674e0c70fbbcf850`；
- Formal Research Run：`regime-run-09dd77dc4b764d21a5e97f211a53e270`；
- Publication：`publication-16c681f1a21b43ba91c76096ba1ade7d`。

## 5. 测试结果

不同命令有覆盖重叠，不能相加为唯一测试总数。

- 新状态级 Reliability / Quality / Prospective / CSI300 核心组：**116 passed**。
- 历史情景图谱广泛回归：**527 passed**。`faulthandler_timeout=120` 仅打印已有指标批量预热的慢编译堆栈，测试随后全部通过。
- TAA / portfolio bridge / walk-forward 相关：**146 passed**。
- Prospective + TAA 状态资格回归：**79 passed**。
- 前端全量：**145 files / 1111 passed**。
- TypeScript、design check、i18n、production build：全部通过。
- Reliability 浏览器：**18 passed**。
- Prospective 浏览器：**7 passed**。
- 真实沪深300历史质量 + 实时状态级报告：320 / 768 / 1440 共 **6 passed**。

现有 React `act(...)`、Starlette/httpx、Vite chunk 警告未新增业务失败，也未通过降低门槛规避。

## 6. 最终结论

可信的“事后情景区分算法”已经具备：定义可解释、版本不可变、区间质量和独立 Episode 可审计。沪深300参考在全历史上满足当前基础条件估计 Episode 门槛。

可信的“实时情景验证算法”也已具备：因果回放、概率校准、最终时间块、逐状态 Episode/高置信 Precision、状态级前瞻资格和严格 fallback。但**当前 SMA9 实时模型在最终独立测试期证据不足，因此系统正确地不让它驱动实时条件 CMA**。

CMA Center 可以开始开发：历史 Regime 可用于条件统计和 shrinkage；Realtime Regime 通过 `regime_cma_evidence` 接入，只有未来 Verified/Prospective-qualified 状态才激活条件化，其余自动使用 Base CMA。
