# 投资目标与约束：精简联动设计

日期：2026-09-17

## 1. 目标

本模块只回答三件事：

1. 投资目标是什么；
2. 最多能承担哪一档风险；
3. 组合需要保留多少现金、未来现金流是什么。

风险等级配置中心提供冻结的 Reference CMA、参考有效前沿及 C1–C5；投资目标页在同一 Reference CMA 上叠加本次现金约束，重新计算“当前约束有效前沿”。正式 CMA 不在本页选择，留给后续 CMA/SAA 阶段。

## 2. 核心业务契约

### 2.1 风险标尺

- 用户选择已发布 RiskScaleVersion。
- PIT 开启时，目标研究日自动跟随当前 PIT，页面不可修改；只列出该研究日可用的 RiskScaleVersion。
- PIT 关闭时，用户可以手动选择目标研究日，再按该日期列出可用版本。
- RiskScaleVersion 的 C1–C5 永久冻结；本页重新计算有效前沿时不得重新分档。
- RiskScale 的 base currency 自动成为本目标计价币种，只读展示，不要求用户再次输入。

### 2.2 两条有效前沿

前端同时展示：

- Reference Frontier：RiskScaleVersion 原始参考有效前沿；
- Constrained Frontier：相同 Reference CMA + 本次现金约束重新求解得到的有效前沿。

C1–C5 仍使用 Reference Frontier 发布时冻结的波动率边界解释两条前沿。

### 2.3 投资目标

保留三种目标类型，但不再单独设置“成功标准”步骤：

- absolute_return：用户输入最低预期年收益；
- funding_goal：用户输入期末目标金额，可同时维护期间投入/支付；
- benchmark_relative：基准自动采用所选 RiskScale 风险等级的代表组合，默认填写目标年超额收益；也可记录合同基准原文，或在同一冻结资产轴上显式设置基准权重。合同原文只留痕，不参与数值计算。

正式 CMA 不在本页选择；历史 `cma_id` 请求继续兼容，但新 UI 不再暴露。

### 2.4 现金约束

用户只面对：

- 最低组合内现金占比；
- 可选现金流计划：总资金、期间投入/支付及其频率；
- funding_goal 时的期末目标。

通胀、额外费用、随机种子、路径数、不确定性惩罚、流动性窗口、投入压力比例、概率门槛不作为普通用户输入。现有数值模型仍复用，固定模型约定被冻结到预览/保存结果中，不能在运行时静默变化。

## 3. 前端设计

### 3.0 主页面：投资目标与约束列表

参考风险等级配置中心，`/pre-investment/objectives` 首先展示当前可用的已发布目标列表，而不是直接进入编辑器。列表展示名称、目标类型/目标值、研究日、风险等级、现金底线、发布时间及操作，并提供：

- **添加**：进入新的两步编辑器；
- **修改**：读取所选已发布版本形成可编辑副本，重新诊断并保存为新的不可变版本；新版本通过 `supersedes_mandate_id` 替代原版本进入可用列表，原版本不原地修改；
- **删除**：写入独立 retirement 审计记录，将版本移出可用列表，不物理删除历史成果；
- 已被新版本替代或删除的版本仍可按 ID 读取，保证历史 SAA/审计引用不失效；删除后不得自动恢复更早的被替代版本。

编辑工作台收敛为两步：第一步按研究信息、风险标尺、风险与现金、投资目标顺序填写，第二步诊断并确认。

### 01 目标与约束

显示：

- 目标名称；
- 目标研究日：PIT 开启只读，PIT 关闭可编辑；
- 投资期限；
- 政策复核日期：可选；
- 目标类型；
- 对应目标值：最低预期收益 / 期末目标 / 目标超额收益。

不显示计价币种输入；风险标尺选定后只读显示币种。

#### 同页风险与现金

显示：

- RiskScaleVersion 选择；
- 当前 RiskScale 的计价币种、研究日和版本；
- 最大风险等级 C1–C5；
- 该等级对应的冻结波动率上限；
- 最低现金占比；
- 可选现金流计划。

不再显示：

- manual/funding_suggestion/explicit_numeric 三模式；
- 手填 max_volatility；
- policy name/source/reviewed_on/valid_until；
- min_liquid_weight/max_illiquid_weight；
- TAA tracking error；
- risk_aversion；
- rebalance policy；
- asset/group limits；
- institutional context；
-正式 CMA 选择；
- simulation paths / seeds / uncertainty penalty。

### 02 结果与确认

主结果按顺序展示：

1. 目标是否在当前最大风险等级内可实现；
2. 最低已验证风险等级；
3. Reference Frontier 与 Constrained Frontier；
4. C1–C5 固定边界；
5. 本次现金约束：用户最低现金、现金流推导现金、最终生效现金下限；
6. funding_goal 时的资金测算摘要；
7. 高级证据折叠展示。

确认后保存不可变 MandateVersion；用户在主列表点击“修改”时生成新的不可变版本替代当前版本，历史版本保持只读审计记录。

### 3.4 前端规范

遵守 `docs/frontend-design-guidelines.md`：

- 复用 Card/Button/Badge/Field/NumberInput；
- 不增加页面背景、卡片套卡片和嵌套纵向滚动；
- 数值使用 `tabular-nums`；
- 字号不低于 12px；
- 浅底辅助文字使用 `slate-600`；
- 图表使用 ECharts，不用 div 拼图；
- 两条前沿除颜色外同时用实线/虚线区分；
- C1–C5 用垂直边界线表达，不重新计算；
- 加载、空、错误、禁用四态完整；
- 320/768/1440px 无页面级横向溢出。

## 4. 后端设计

### 4.1 RiskScale 研究日目录

新增只读接口：

`GET /api/strategic-allocation/risk-scales/study-options?as_of=YYYY-MM-DD`

返回在该研究日适用的 RiskScaleVersion。判断至少包括：

- `research_as_of <= as_of`；
- 未超过 `valid_until`；
- 在该研究日尚未退休；
- 冻结参考输入仍可校验；
- 风险口径为 `annualized-periodic-volatility-v1`。

该接口只做版本筛选，不重新拟合 Risk Scale。

### 4.2 Mandate 新输入语义

继续兼容 schema 1.0/现有 2.0 旧字段；新 UI 使用 2.0 精简输入。

新增/调整：

- `review_date`：2.0 可空，1.0 保持必填；
- `min_cash_weight`：用户明确的最低组合内现金占比；
- `target_excess_return`：benchmark_relative 的用户目标；
- `benchmark`：2.0 可由后端从 RiskScale 代表组合生成；
- `boundary_policy`：2.0 新 UI 可不提交，由服务端生成并冻结模型约定；
- `risk_authorization.source` 不再需要用户填写，UI 生成机器可读来源；
- 旧字段仍可读取，但新 UI 使用中性值，不形成额外硬约束。

### 4.3 自动基准

benchmark_relative 下：

1. 读取所选 RiskScaleVersion；
2. 读取用户选择的 C1–C5；
3. 取该等级 `representative_weights`；
4. 与 `ordered_asset_ids` 组成冻结 benchmark；
5. 保存 `source=risk_scale_reference`；
6. 用户只维护 `target_excess_return`。

若该等级无代表组合，诊断失败关闭，不猜权重。

### 4.4 现金模型约定

复用现有 `goal_kernels.py`，不复制现金流递推。普通 UI 不暴露模型参数；后端将当前模型约定冻结到解析后的 definition/assessment 中：

- 现金保障窗口：默认 12 个月，但不超过投资期限；
- 压力下认可投入比例：50%；
- funding 概率门槛：80%；
- 通胀与额外模型费用：新 UI 默认 0；
- 搜索/验证路径数与种子沿用固定默认并保存在 planning_settings。

这些是模型约定，不冒充用户风险偏好；结果页高级证据可查看。

### 4.5 约束有效前沿

复用 `frontier_moments.solve_frontier`：

1. 从 RiskScale frozen context 读取同一 `mu`、`covariance`、标准权重约束；
2. 计算现金流推导的 `funding_floor`；
3. `effective_cash_floor = max(min_cash_weight, funding_floor)`；
4. 把现金角色大类增加为 `weight >= effective_cash_floor`；
5. 重新求解有效前沿；
6. 仍用 RiskScale `applied_boundaries` 对结果解释 C1–C5。

不通过过滤原 Reference Frontier 伪装重新优化。

ReferenceDiagnosis 增加：

- `reference_frontier`；
- `constrained_frontier`；
- `risk_boundaries`；
- `cash_constraint`。

### 4.6 正式 CMA 边界

`MandateStudyRequest.cma_id` 保留兼容，但新 UI 固定为 null。Investment Objectives 只消费 RiskScale 的 Reference CMA。正式 CMA 由后续 CMA 模块产出，并在 SAA 时重新验证 Mandate。

### 4.7 已发布目标的修改与删除

不提供原地 UPDATE，也不物理删除冻结研究成果：

- `ConfirmMandateRequest.replaces_mandate_id`：修改已发布目标时指向当前活跃版本；服务端确认其仍为活跃版本后保存新 MandateVersion，并冻结 `supersedes_mandate_id`；
- `DELETE /api/strategic-allocation/mandates/{id}`：创建 `investment_mandate_retirement` 审计成果，接口保持幂等；
- `catalog()` 仅返回未被 supersede、未 retired 的目标；
- `get_mandate(id)` 继续允许读取历史版本，因此既有下游研究引用不会因主列表删除而损坏；
- 如果替代版本随后被删除，不恢复更早的 superseded 版本，避免“删除新版本导致旧政策意外重新生效”。

## 5. 兼容与清理

- 旧 MandateVersion 保持只读可读；
- 旧 API 字段不立即删除；
- 新 UI 不再产生无实际业务价值的输入；
- 被新 UI 完全替代且无外部调用的前端组件/import 在同一变更删除；
- 数值实现保持唯一：前沿、风险分类、现金流、Wilson 统计继续复用现有 NJIT 内核；
- 不新增 Python 数值回退。

## 6. 测试与验收

后端：

- PIT 日期 RiskScale 筛选：生效前、有效期内、过期、退休前后；
- review_date 2.0 可空 / 1.0 仍必填；
- 风险等级解析与自动 benchmark；
- 手工最低现金 + 现金流推导 floor 取最大值；
- 两条前沿同一 Reference CMA、C1–C5 不变；
- 目标收益不可达时明确 `needs_revision`；
- 旧 schema 回归；
- 固定签名 NJIT readiness 不回退。

前端：

- 主入口先展示现有目标列表，支持新增、修改、删除及删除确认；
- 修改进入可编辑副本，重新诊断后保存替代版本；删除后主列表即时移除；
- PIT 开启研究日只读并同步；PIT 关闭可手选；未知 PIT 阻止计算；
- RiskScale 只显示研究日适用版本；
- 选 RiskScale 后币种只读继承；
- 无正式 CMA 选择器；
- 三种目标类型仅显示必要输入；
- C1–C5 选择与冻结 cap 一致；
- 双有效前沿图、固定 C1–C5 边界、空/错/加载态；
- 保存后版本只读；后续修改必须从主列表重新形成新版本；
- design:check、i18n、tsc、build、Vitest 和相关 E2E 通过。

## 7. 实施与验收结果

已按上述设计完成前后端收敛：主入口先展示当前可用的投资目标与约束列表，并提供新增、修改、删除；编辑器保持“目标与约束 → 结果与确认”两步。修改已发布目标不会原地改写，而是重新诊断后保存新不可变版本并替代原版本；删除只退出可用列表并保留审计历史。PIT 开启时研究日锁定，关闭时可编辑；Risk Scale 按目标研究日筛选并冻结引用；正式 CMA 不再在本页选择；相对基准自动取所选 C 等级代表组合；现金约束重新求解同一 Reference CMA 下的有效前沿，C1–C5 仍使用原 Risk Scale 边界。旧资金计划迁移会保留本金、组合外储备、金额口径、通胀、费用与现金流事实；这些非日常字段收进“金额与费用口径”折叠区，避免既丢业务事实又增加主流程负担。

同时删除了已无生产调用的 `InstitutionalFields`、`InstitutionalResults`、`AssetAuthorizations` 以及旧的 Mandate 诊断整页实现；旧后端字段和不可变历史版本仍保留兼容读取，不复制数值算法。

以下为 2026-09-17 原实施阶段的历史验收记录，不代表本次提交候选的复验；当前结果见 [提交审核](risk-mandate-submit-review-2026-09-18.md)：

- 后端 Mandate / Risk Scale / SAA 相关回归：215 passed；
- 前端全量 Vitest：154 files / 1201 tests passed；
- 专项浏览器 E2E：桌面 1440、平板 768、手机 320 共 9 passed；
- TypeScript、production build、design:check、i18n、Mandate/Risk Scale 生成契约检查全部通过；
- AI Hermes routing validation 通过；
- 仅保留仓库原有的 React `act(...)`、Numba contiguous-array 性能提示、浏览器数据库陈旧及大 chunk 警告，无新增失败项。
