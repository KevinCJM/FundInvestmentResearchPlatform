# 指标中心 × 产品研究：AI 智能体详细设计

> - 日期：2026-09-18（审核修订：编译预览、会话草稿、上下文、确认提交与 UI 动作闭环）
> - 源码核对基线：`8a513575cda106ebdf433d6ac489b18def983fa1`；下文源码行号对应该基线，实现时须按符号重新核对。
> - 状态：**设计稿，尚未实现。** 本文描述目标实现与验收方式，不代表任何代码已落地。
> - 2026-09-20 Harness 修订：[Harness 无默认上限与无进展检测设计](ai-agent-harness-progress-design-2026-09-20.md)；运行与恢复协议以该文为准；核心链路已实现，自审核与测试见其第 18 节。
> - 实施级前后端设计：[ai-agent-indicator-product-research-implementation-design-2026-09-19.md](ai-agent-indicator-product-research-implementation-design-2026-09-19.md)。
> - 范围：指标计算中心（IndicatorStudio / `backend/custom_indicators`）+ 产品研究模块（ProductDetail 等页面 / `backend/services/product_analysis.py`、`backend/research_series` 等）+ AI Harness + 前端对话面板。
> - 关联约束：[AGENTS.md](../../AGENTS.md)（NJIT、算子治理、测试、AI Hermes 路由）、[docs/frontend/README.md](../frontend/README.md)（前端视觉与交互）、[branch_submission_rules.md](../governance/branch-submission-rules.md)（提交规则）。
> - 术语：本文中「指标」指自定义指标（custom indicator）；DSL 指 `backend/cal_indicators` 的类型化表达式；「落库」指写入 `data/custom_indicators.json` 形成不可变 revision。

---

## 0. 摘要与目标

### 0.1 目标

1. 用户用自然语言提出指标需求，智能体在指标中心允许的变量、算子与参数契约内产出可执行 DSL，供人类用 LaTeX、Excel 公式和真实数值三种材料复核；确认后落库为**不可变 revision**。
2. 产出的指标回写前端：指标库立即可见，产品研究页可在当前产品上计算与展示；修改指标产生新 revision，旧引用保持旧版本。
3. 一套 harness 支持两个作用域（`indicator-center`、`product-research`），单模块与跨模块功能共用同一个 agent 形态；跨模块优先使用受限工具，指标目录写入始终是显式人工动作。
4. 会话可恢复、有分层记忆、支持多人对话标注与冲突显式上报。
5. 智能体可直接操作前端界面（填写 LaTeX、设置筛选条件等），页面实时反显 AI 动作，动作可见、可撤销、可审计（见 §11）。

### 0.2 非目标（v1 明确不做）

- 不自动发布指标；不在产品研究侧写数据（评价方案、快照配置、展示偏好都只读）。
- 不生成新算子、不生成自由 Python/Numba、不改动任何既有计算路径。
- 不做多用户账号、跨用户编辑锁、跨设备同步；多人仅以 speaker 标签与审计呈现。会话内的短事务/revision 只防止丢失更新，不代表多人协作或身份隔离。
- 不做向量库/RAG，本期不做逐 token 输出；运行状态采用可恢复事件流，兼容整轮返回，不做多智能体集群。
- 不替代指标中心现有编辑器；agent 是并行入口，不是唯一入口。

### 0.3 设计原则

1. **唯一编译器**：校验、类型推断、可用性检查、数值预览、落库全部直调现有 service 层函数；AI 侧没有平行实现，AI 生成物与手工编辑物走同一编译与 NJIT 路径。
2. **唯一事实**：DSL 是计算真相；LaTeX（`display_latex`）与 Excel 均由服务端从 DSL 派生；对 AI 的「说明书」由注册表生成，不手写事实。
3. **失败关闭**：现有算子无法表达时明确回答「不可表达」，禁止近似、禁止静默替换；低置信问人。
4. **人类确认**：LLM 没有指标目录写工具；可以保存本会话草稿，指标目录只能经人类复核具体定义后触发的 `commit` 端点写入。`confirmed=true` 是请求校验，不是身份认证或人类来源证明。
5. **能力即权限（硬限制）**：AI 能改、能看什么，由代码中的工具白名单、服务端派发校验与唯一写路径决定；提示词只是第二道防线，不承担安全职责。四张清单见 §2.3，证明测试见 §7.1。

---

## 1. 现状事实（代码级）

本节是设计与验收的事实基础；实现完成后引用点若漂移，以代码为准并回改本节。

### 1.1 指标中心

- 前端 `frontend/src/pages/IndicatorStudio.tsx`：编辑器 + 变量/算子/指标目录弹窗；`useSearchParams` 支持 `kind / ids / period / as_of`（:1160-1181），带 `ids` 时默认进 preview 页签（:1491-1507）。
- 保存路径：`createCustomIndicator` / `updateCustomIndicator`（`frontend/src/services/customIndicators.ts:1236,1242`），更新时携带 `selectedIndicator.revision`（`IndicatorStudio.tsx:2114-2119`）。
- 后端路由 `backend/services/custom_indicator_routes.py`：`meta`(:426)、`validate`(:431)、`parameters/inspect`(:436)、`parameters/bind`(:441)、`compose`(:449)、`infer`(:454)、`graph/resolve`(:459)、`editor-state` GET/PUT(:464,469)、`derive-rolling-series`(:474)、`rolling-scalar-draft`(:486)、`variables/availability`(:499)、`prepare`(:512)、`evaluate`(:523)、`evaluate-series`(:540)、`export-excel`(:562)、`evaluate-portfolio`(:584)、`list`(:596)、`create`(:615，201)、`snapshot-config`、`get/update/delete`(:626-648)。
- 目录事实源：`service.py:1217 meta()` 返回 variables 与 operators（`signature`、`return_type`、`parameters[].allowed_shapes/default`、`latex_template`、`mathematical_essence`）；算子契约 `backend/cal_indicators/typed_operators.py`（`TypedOperatorSpec` / `OperatorSignature`，含 version/category/cost/`interval_policy`）；变量注册表 `backend/custom_indicators/variable_registry.py`（`VARIABLE_REGISTRY_VERSION 3.0.0`、`DATA_CONTRACT_VERSION tushare-eod-v2`、`CONTEXT_SCHEMA_VERSION typed-context-v2`，含语义角色、单位与观察/可用性规则）。
- 受限表达式：`backend/cal_indicators/latex_excutor.py` 的 `ExpressionPolicy`（`allowed_variables`、`function_arity`、`max_nodes=128`、`max_depth=20`）。
- LaTeX：`backend/cal_indicators/typed_latex.py`（`MATH_NOTATION_VERSION 1.5.0`）；`service.py:332 _typed_display_latex`；定义装饰与 `infer` 响应均含 `display_latex`（`service.py:1452,2040`）。
- 预览不写指标目录：未保存公式先调 `/validate`，由 `_validate_typed` 或序列校验路径编译、预热并返回 `compile_token`（`service.py:1903,2027,2063`，scope `current_process_warm_cache`），再携带同一完整定义和 token 调 `/evaluate` 或 `/evaluate-series`。`prepare_evaluation`（:2723）用于显式准备单产品批量计划；其 inline 分支要求已有 token，且不返回新 token，不能充当草稿首次编译入口。正式 evaluate 对缺失、过期或不匹配 token 失败关闭（:2797-2884）。现有前端 `IndicatorStudio.preview` 与 `customIndicators.evaluateCustomIndicators` 已按此分流；**agent 必须复用相同链路**。编译缓存可落盘，不能把“不写指标目录”表述为“没有任何文件副作用”。
- 版本化存储：`backend/custom_indicators/repository.py:101 IndicatorRepository`（`create`:168 / `update(expected_revision)`:185 / `delete`:215；旧版本进 `history`；内置只读），落盘 `data/custom_indicators.json`（`AtomicJsonStore` + 文件锁 + 原子替换 + fsync）。
- 创建/更新：`service.py:2338 create_indicator` / `:2351 update_indicator`：normalize → 编译校验 → 应用编译契约 → **NJIT 预热**（time_series → `series_service.warm`；single_product → `_warm_single_product_definition`）→ 写库 → 清缓存。
- 编辑器状态：`backend/custom_indicators/graph_service.py:197 save_state` 只为已保存的指标 revision 保存与其定义指纹一致的画布/布局，并校验 `expected_editor_revision`；它不是未保存公式的通用草稿仓库。AI 草稿保存在本会话 `state.draft`，发送到编辑器只更新页面内存；不改变现有 editor-state 契约。
- 引用保护：`delete_indicator` 中 `plans.references_indicator` / `snapshot_config.references_indicator`（`INDICATOR_IN_USE`）。

### 1.2 产品研究模块

- 页面：`frontend/src/pages/` 下 `ProductDetail`、`ProductResearch`、`ProductCompare`、`HoldingDiagnosis`、`EvaluationPlan`、`ProductPools`、`ProductPoolSelection`、`ProductPoolLifecycle`、`TimingResearch`。
- 目录加载：`ProductDetail.tsx:581` 调 `listCustomIndicators({contextKind:'single_product', productKind}) + getCustomIndicatorMeta()`。
- 上下文过滤：`indicatorsForContext(items, 'single_product')`（`customIndicators.ts:1206`）。
- 计算引用：标量走批量、序列走独立请求（`ProductDetail.tsx:483-490`）；请求体 `indicator_refs` 带 `indicator_id + indicator_revision + parameters`（:606-615），历史版本可复现。
- 展示选择：`frontend/src/components/metrics/useMetricDisplayPreference.ts`（localStorage `indicator-display:v{version}:{page}:{contextKind}`；`withSelectedIndicators` 在组件层增删）。
- 定义查看：`MetricDefinitionDrawer`（:2035）；深链回指标中心 `/settings/indicators-models?kind=&ids=`（:1515,1543）。
- 服务端守卫（cb4d3df）：`MAX_SERIES_INSTANCES=50`、`MAX_EVALUATION_INDICATORS=200`、`MAX_INDICATORS=10`（分别服务单次曲线数、单次 evaluate 指标数、预热批宽度与评价方案规模）；界面层条数上限已移除。
- LaTeX 展示依赖已存在：`katex ^0.16.47`，`MetricDisplay.tsx:379` 渲染 `display_latex`。
- 持仓诊断是组合计算域：`HoldingDiagnosis.tsx:41` 读取不可变 `run_id` 并加载 portfolio 指标；`evaluate-portfolio` 接受 run_id 与指标 id（最多 10 个），不接受单产品 targets/period。引擎支持经 validate 编译的 portfolio inline 标量，但当前 `IndicatorStudio` 的 `STUDIO_CONTEXT` 固定为 `single_product`，`evaluate-series` 也只有 ETF/基金目标。挂载 AI 面板不等于已有组合作者 UI、组合序列或组合 Excel 导出能力。

### 1.3 缺口清单

- 后端无任何 LLM 客户端（`chat/completions`、`deepseek`、`openai` 均无命中；`httpx>=0.24` 已在 `backend/requirements.txt`）。
- 无 agent/session/harness、无意图路由、无记忆文件、无 AI 目录快照、无前端对话面板。
- 无账号体系（无 auth/login）；多人会话 v1 以 `speaker` 标签与审计实现。

---

## 2. 总体架构

### 2.1 一个 Harness + 两个作用域

```
┌─ 前端 ──────────────────────────────────────────────────────┐
│ AgentPanel（IndicatorStudio / 产品研究各页挂载，右下角浮窗）    │
│  会话流 · 工具轨迹 · 草稿卡（DSL + LaTeX + 真实数值预览）      │
│  候选卡 · 提交条 · 作用域 chip · 记忆提案卡 · 冲突决策卡       │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST /api/agent/*
┌───────────────────────▼─────────────────────────────────────┐
│ backend/agent                                                │
│  routes ─→ harness（回合循环 / 无进展检测 / 压缩 / 恢复）           │
│            ├─ scopes    作用域清单（生成 + 校验）             │
│            ├─ router    两轴三层意图判定                      │
│            ├─ catalog   说明书（注册表生成，版本化）           │
│            ├─ tools     薄包装，直调现有 service 函数          │
│            ├─ sessions  SQLite 事务与运行事件                    │
│            ├─ memory    scope 记忆文件 + 审计                 │
│            ├─ commit    人类确认后的唯一落库路径               │
│            └─ llm       唯一供应商边界；fixture 可注入         │
└──────────┬──────────────────────────┬────────────────────────┘
           │ 直调（同进程）            │ 直调（同进程）
┌──────────▼────────────┐   ┌─────────▼──────────────────────┐
│ custom_indicators     │   │ 产品研究服务                    │
│ 编译 / 校验 / NJIT 预热│   │ product_analysis / research_   │
│ 版本化仓库 / 草稿      │   │ series / product_pools / plans │
└───────────────────────┘   └────────────────────────────────┘
```

两类智能体（模块内、全项目）在形式上完全一致，差异只存在于：
`scope`（当前模块）、`tools`（该模块工具集 + 跨模块只读）、`memory`（该模块记忆文件）、`rules`（提示词切片）。
v1 只实现两个模块作用域；全项目作用域是同一 harness 增加编排工具后的自然扩展，不在本次范围。

实现边界：harness、工具与 commit 必须与现有 service 同进程直调，不引入外部 agent 运行时或跨进程工具（MCP 等）。跨进程会破坏「同一编译与 NJIT 预热路径」「进程内缓存与文件锁」假设，并引入第二运行时、端口、认证与密钥面；若未来评估外部运行时，须单独设计评审。

### 2.2 改动面清单（文件级）

**新增（后端）**

| 文件 | 职责 |
| --- | --- |
| `backend/agent/__init__.py`、`__main__.py` | 包入口；CLI：`dump`（说明书/作用域快照）、`smoke`（fixture 冒烟） |
| `backend/agent/contracts.py` | Pydantic 契约：会话、消息、回合响应、草稿、候选、决策卡、提交、记忆提案、scope manifest |
| `backend/agent/catalog.py` | `build_agent_catalog()`：注册表 → AI 优化 JSON；`catalog_version` 计算 |
| `backend/agent/scopes.py` | 两个作用域 manifest 及其校验 |
| `backend/agent/router.py` | 两轴三层意图判定 |
| `backend/agent/tools/__init__.py`、`indicator_center.py`、`product_research.py` | 工具 spec 注册表与 handler |
| `backend/agent/llm.py` | `LLMClient` 协议、`HttpxClient`、`FixtureClient` |
| `backend/agent/harness.py` | 回合循环、无进展检测、恢复、压缩、上下文装配 |
| `backend/agent/sessions.py` | 会话/run、工具回执与事件（SQLite 短事务） |
| `backend/agent/memory.py` | 记忆文件读写与审计 |
| `backend/agent/ui.py` | UI 动作 schema、校验与 surface 快照契约（见 §11） |
| `backend/agent/commit.py` | 提交预检、确认快照、影响面投影与幂等提交记录；复用指标 service 写入 |
| `backend/agent/routes.py` | `APIRouter`（注册进 `app.py`） |
| `backend/agent/prompts/system.md`、`indicator_center.md`、`product_research.md`、`compaction.md` | 规则层提示词（随代码版本管理） |

**新增（前端）**

| 文件 | 职责 |
| --- | --- |
| `frontend/src/services/agent.ts` | API 客户端与类型 |
| `frontend/src/components/agent/AgentPanel.tsx` | 面板外壳（浮窗、作用域 chip、会话菜单） |
| `frontend/src/components/agent/AgentMessageList.tsx`、`AgentComposer.tsx` | 消息流与输入 |
| `frontend/src/components/agent/AgentDraftCard.tsx` | 草稿卡：DSL 代码块 + KaTeX 公式 + 参数表 + 数值预览 |
| `frontend/src/components/agent/AgentCandidateCard.tsx`、`AgentDecisionCard.tsx` | 候选与冲突决策 |
| `frontend/src/components/agent/AgentCommitBar.tsx` | 名称/描述/提交/修订信息 |
| `frontend/src/components/agent/AgentToolTrace.tsx`、`AgentMemoryProposal.tsx` | 工具轨迹（折叠）与记忆提案 |
| `frontend/src/components/agent/useAgentSession.ts` | 会话 hook（发送、轮询回合状态、恢复） |
| `frontend/src/components/agent/useAgentSurface.ts` | 页面 UI Surface 注册钩子（见 §11.3） |
| 同名 `*.test.tsx`、`agent.test.ts` | Vitest + RTL 测试 |

**新增（文档/事实）**

- `docs/agent/indicator_catalog.json`（说明书快照）、`docs/agent/scopes.json`（作用域清单快照）。
- 本文档；实现时按需产出实现说明或验收记录。

**修改**

- `backend/app.py`：`include_router(agent_router)`；启动状态增加非致命的 `agent.configured`。
- 前端挂载点：`IndicatorStudio.tsx`、`ProductDetail.tsx`、`ProductResearch.tsx`、`ProductCompare.tsx`、`HoldingDiagnosis.tsx`、`EvaluationPlan.tsx`（引入 launcher 与提交回调）。
- `locales/*.json`：`agent.*` 词条（中英齐备）。
- AI Hermes 路由记忆：`docs/repo_map.json`、`docs/task_routes.json`（新增 agent 模块）、必要时 `docs/pitfalls.json`；实现后运行 `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py`。

**不动**

- 全部既有计算路径、数据契约、指标 schema、产品研究请求/响应结构。agent 只复用，不改写。

### 2.3 硬性能力边界（代码强制，非提示词约定）

分层：**能力缺失 > 工具派发校验 > 服务层约束 > 审计留痕 > 提示词**。未注册的能力在物理上不可达；提示词只影响模型行为，不参与权限判定。

#### 2.3.1 AI 能改（写白名单，穷举且封闭）

| 能改 | 唯一通道 | 机械保证 |
| --- | --- | --- |
| 指标定义（新建/新 revision） | 人类调用 `/commit-preview` 复核，再触发 `POST /commit` | 两个端点均不注册为 LLM 工具或 UI 动作；确认绑定草稿哈希、目标与 revision；`expected_revision` 乐观锁及幂等记录见 §3.7 |
| 本会话草稿 | `metrics.draft_save` | 只更新本会话 `state.draft`；服务端绑定 session，校验 `expected_draft_revision`；不写 editor-state，不接受路径参数 |
| 会话事件 | 系统写本会话文件 | 目录固定、session_id 服务端生成；事件类型白名单（3.3） |
| 记忆文件 | 人类 accept 提案后追加 | LLM 仅有 propose；目标固定为 `data/agent_memory/*.md`；`audit.jsonl` 只增不改（4.5） |

除此之外没有第 5 类业务持久化能力。既有引擎的编译缓存按其原契约管理，不授予 agent 任意文件写权限；工具标记 read 表示不改业务对象，不承诺底层绝无缓存写入。

#### 2.3.2 AI 不能改（硬拒）

- 源码、配置、`.env`、token/凭据、prompt、scopes 清单：无文件读写工具，无任何 `path` 参数。
- 数据快照 / parquet / 存储迁移状态 / PIT 系统设置 / 数据下载配置：不在 handler 白名单。
- 评价方案、快照配置、展示偏好、产品池、发布中的情景与风险模型：对应模块写函数不在白名单（0.2 写边界）。
- 指标历史 revision 与已发布版本：commit 只追加新 revision，无覆盖语义。
- 其他 session 文件、shell、任意网络：无工具。
- 前端：只能产出 `UiAction[]`（§11）；不能导航、不能坐标点击、不能经 `ui.act` 触发 commit。
- agent 自身：工具集、提示词、执行策略、scope 清单不允许 LLM 自改；默认无调用次数/轮数上限。

#### 2.3.3 AI 能看（读白名单）

- catalog：变量/算子/签名/参数/LaTeX/`interval_policy`（`meta()` + 注册表生成，3.5）。
- 当前页面上下文：单产品 targets/period 或组合 run_id、indicator_id + revision、surface 快照（§3.2、§11.3）。
- 计算只读：validate、infer、availability(as_of)、preview、Excel 导出；组合页面额外可读当前 run 的冻结摘要与已有 portfolio 指标结果。
- 影响面只读：当前评价方案的引用 id 列表与快照配置是否引用（供 commit 前提示）；通过受限投影返回，不向 LLM 返回方案正文。
- 本会话事件与记忆文件。

数据最小化：工具结果截断，完整数值表不进 prompt；数量上限继承 `MAX_PLAN_TARGETS`、`MAX_SERIES_INSTANCES=50`、`MAX_EVALUATION_INDICATORS=200`。

#### 2.3.4 AI 不能看（硬拒）

- 凭据、`.env`、日志、其他 session、工作区外文件：无工具、无 `path` 参数。
- 页面上下文之外的产品全集：`products.search` 只做名称→id 解析且受页面上限约束。
- 未来数据：变量使用前必须过 `availability(as_of)`，`as_of` 由服务端解析；不可得即失败关闭。
- 评价方案/产品池正文、下载与存储内部件：不在读白名单。

#### 2.3.5 拒绝语义与实现位置

| 边界 | 实现位置 | 拒绝语义 |
| --- | --- | --- |
| 未注册工具 | `tools/__init__.py` 的 `execute()` 按名派发 | `AGENT_TOOL_NOT_ALLOWED`，记事件，不重试 |
| 参数越界/多余字段 | 工具 JSON Schema `additionalProperties: false` + Pydantic | 校验失败回喂稳定诊断；按无进展检测纠偏/暂停 |
| handler 越权 | 注册表构建期校验 handler 在 `(module, qualname)` 白名单 | 校验失败则 agent 禁用，不影响主应用启动 |
| 目录写路径 | `commit.py` 唯一调用指标 service 的 create/update；其他 handler 不可调用写方法，静态检查加运行期调用断言 | 测试失败；不能仅凭无 repository import 证明无间接写入 |
| 预览编译 | inline：`validate` → `compile_token` → 按结果类型 evaluate；已保存多标量引用：`prepare_evaluation` → 锁定 refs → evaluate | 无/错 token 或未预热计划直接报错 |
| UI 动作 | 服务端页面能力白名单与 surface 取交集；前端再次校验 page_instance/revision 与真实 setter | 非法/过期拒绝并按 §11 回执；客户端 surface 不授予新权限 |
| 无效循环与资源管理 | 默认无总调用/轮数配额；稳定指纹、进度检测、单操作超时、服务准入 | 无进展则有界纠偏并保留进度；见 [Harness 无默认上限与无进展检测设计](ai-agent-harness-progress-design-2026-09-20.md) 第 6–10 节 |

证明测试见 §7.1 `test_agent_boundaries.py`。

**无账号说明**：v1 的读隔离是进程级/单工作区边界，不是用户级隔离；多人共用时只能依赖部署边界（localhost/内网），引入认证前不得对外多用户开放。

---

## 3. 后端设计

### 3.1 模块结构

```
backend/agent/
├── __init__.py
├── __main__.py            # python -m backend.agent dump|smoke
├── contracts.py           # Pydantic 模型（见 3.2）
├── catalog.py             # 说明书生成（见 3.5）
├── scopes.py              # 作用域清单（见 3.6）
├── router.py              # 意图判定（见 3.6）
├── tools/
│   ├── __init__.py        # TOOL_REGISTRY：spec + handler
│   ├── indicator_center.py
│   └── product_research.py
├── llm.py                 # LLMClient 协议 + 实现（见 4.7）
├── harness.py             # 回合循环（见 4.3）
├── sessions.py            # 会话存储（见 3.3）
├── memory.py              # 记忆（见 4.5）
├── commit.py              # 落库（见 3.7）
├── routes.py              # API（见 3.8）
└── prompts/
    ├── system.md
    ├── indicator_center.md
    ├── product_research.md
    └── compaction.md
```

工具 handler 复用已注册路由共享的 indicator_service 与现有 service 函数（同进程），不创建第二个指标服务实例、不发起 HTTP 自调用；保留手工路径的编译、NJIT 缓存与数据上下文。

### 3.2 数据契约（Pydantic，`contracts.py`）

指标作者流程从对话开始，无需预选产品：澄清需求 → 生成与校验公式 → 人工保存。查看实际结果时再选择产品，或由 AI 从真实目录检索适当样例并检查数据可用性；样例不固化进通用指标定义。具体实现见实施设计 §3.11。

```python
class SingleProductContext(Contract):
    context_kind: Literal["single_product"]
    targets: list[EvaluationTarget] = Field(default_factory=list, max_length=10)  # 创作阶段可为空，仅试算需要产品
    period: str
    as_of: str | None = None              # 页面显式截止日；None 仍继承已有 PIT 口径

class PortfolioContext(Contract):
    context_kind: Literal["portfolio"]
    run_id: str                          # 必须为服务端可读取的不可变组合快照

class PageContext(Contract):
    module: Literal["indicator-center", "product-research"]
    page: Literal["indicator-studio", "product-detail", "product-research", "product-compare", "holding-diagnosis", "evaluation-plan"]
    page_instance_id: str           # 页签/挂载实例；跨页返回不能复用旧实例
    context_revision: int           # 目标、窗口、研究口径变化时递增
    view_state: Literal["unknown", "inherit", "explicit", "off"]
    calculation: SingleProductContext | PortfolioContext
    indicator_id: str | None = None
    indicator_revision: int | None = None

class AgentSessionCreate(Contract):
    scope_hint: Literal["indicator-center", "product-research"] | None = None
    page_context: PageContext
    speaker: str                   # v1 为人类可读标签；未来替换为认证身份
    surface: UiSurfaceSnapshot | None = None   # M5 启用；类型见 §11

class AgentMessageRequest(Contract):
    message_id: str                # 客户端生成；相同请求重试不再启动一轮
    expected_session_revision: int
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    speaker: str
    page_context: PageContext      # 每轮更新，不能只在创建会话时绑定页面
    surface: UiSurfaceSnapshot | None = None
    ui_results: list[UiBatchReceipt] = []       # 应用/拒绝/撤销回执，见 §11
    decision: DecisionResolution | None = None

class DecisionResolution(Contract):
    decision_id: str
    option_id: str                 # 必须属于服务端保存的待决卡；不靠自然语言猜选择

class DraftArtifact(Contract):
    draft_revision: int            # 服务端递增；与指标 revision、editor_revision 分开
    valid: bool                    # 服务端校验结果；失败草稿不进入提交条
    diagnostics: list[dict[str, Any]] = []
    definition: IndicatorDraft           # 直接复用现有 IndicatorDraft，不新造定义契约
    definition_hash: str           # 服务端规范化完整定义的哈希，含参数与展示元数据
    context_hash: str              # 本次解析后的计算上下文，见 §3.2.1
    catalog_version: str
    display_latex: str | None = None     # 服务端 infer 派生
    required_variables: list[str] = []
    channel_types: dict[str, Any] = {}
    preview: list[dict[str, Any]] | None = None   # 真实数值预览（截断版）

class CandidateOption(Contract):
    id: str
    label: str
    detail: str                    # 语义一句话
    definition: IndicatorDraft | None = None
    display_latex: str | None = None

class DecisionOption(Contract):
    id: str
    label: str
    detail: str
    effects: str | None = None

class DecisionCard(Contract):
    decision_id: str
    kind: Literal["scope_conflict", "param_conflict", "intent_ambiguous", "revision_conflict", "ui_action_rejected"]
    title: str
    question: str
    options: list[DecisionOption] = Field(min_length=2, max_length=4)
    default_option_id: str | None = None

class ToolTraceEntry(Contract):
    round: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str                    # 人类可读摘要，不含大块数据
    error_code: str | None = None

class MemoryProposal(Contract):
    proposal_id: str
    scope: str
    append_markdown: str
    reason: str

class ScopeChange(Contract):
    from_scope: Literal["indicator-center", "product-research"]
    to_scope: Literal["indicator-center", "product-research"]
    reason: str

class AgentTurnResponse(Contract):
    session_revision: int
    reply: str
    tool_trace: list[ToolTraceEntry] = []
    draft: DraftArtifact | None = None
    candidates: list[CandidateOption] = []
    decision_card: DecisionCard | None = None
    scope_change: ScopeChange | None = None
    memory_proposals: list[MemoryProposal] = []
    commit_suggestion: bool = False      # 是否已到"可提交"状态（仅提示，不自动提交）
    usage: dict[str, int] = {}
    ui_batch: UiActionBatch | None = None      # 服务端绑定批次与页面版本，见 §11

class CommitTarget(Contract):
    operation: Literal["create", "update"]
    indicator_id: str | None = None
    expected_revision: int | None = None

class CommitPreviewRequest(Contract):
    expected_draft_revision: int
    definition: IndicatorDraft
    target: CommitTarget
    page_context: PageContext

class CommitRequest(Contract):
    request_id: str                 # 客户端生成并在超时重试时复用
    confirmation_id: str            # 服务端保存的确认快照；不接收另一份自由 definition
    definition_hash: str
    expected_draft_revision: int
    page_context: PageContext
    confirmed: Literal[True]        # 人类显式确认；默认不成立
    acknowledge_no_preview: bool = False

class CommitImpact(Contract):
    evaluation_plans: list[str] = []       # 当前方案版本中引用该指标的 id；无正文
    snapshot_config: bool = False
    checked_at: str
    coverage: Literal["current_references"] = "current_references"
    historical_revisions_unchanged: bool = True

class CommitPreviewResponse(Contract):
    confirmation_id: str
    draft: DraftArtifact
    target: CommitTarget
    impact: CommitImpact
    preview_status: Literal["current", "not_run", "unavailable"]

class CommitResponse(Contract):
    request_id: str
    indicator_id: str
    revision: int
    name: str
    display_latex: str | None
    impact: CommitImpact
    refresh: dict[str, Any]                # 前端刷新指令，见 3.7

class MemoryResolution(Contract):
    proposal_id: str
    action: Literal["accept", "reject"]
    speaker: str
```

以上是结构示意；所有外部输入继承禁止额外字段的 `Contract`，并为列表、文本和动作数量设上限。消息须至少包含 text、decision 或 ui_results 之一；纯 UI 回执只记账并返回状态，不调用 LLM。`CommitTarget` 的 create 禁止 id/revision，update 必须同时提供 id 和正整数 revision，并与草稿锁定的来源一致；复制内置指标须显式选择 create。

`draft_revision`、哈希、编译结果、预览、影响面、usage、确认快照和 UI 批次标识均由服务端生成。模型只输出文字、候选定义、记忆/动作提案；不得直接使用模型生成的完整 `AgentTurnResponse` 作为可信响应。

模型响应使用独立的 `ModelTurnDraft`：仅允许 `reply`、`definition`、`candidates[{label, detail, definition}]`、`memory_suggestions[{scope, append_markdown, reason}]` 和 `ui_actions[{action, target, value, note}]`。嵌套字段也禁止额外属性；定义按现有 IndicatorDraft 校验，动作提案沿用 §11 的枚举/值约束。展示公式、标识、工具轨迹与所有数值证据由服务端重建，不能由模型填写“校验通过”或伪造预览。

### 3.2.1 计算上下文与页面接入边界

两个 scope 保持不变；`calculation.context_kind` 决定计算入口，scope 不替代计算域。联合类型以 context_kind 判别，并禁止混入另一域的字段。没有目标的指标中心可创作/校验草稿，预览必须补齐真实产品；目标数量取相关服务与变量可用性接口的共同上限，不照搬某一个最大值。

| 页面 | 计算域 | v1 接入与提交后处理 |
| --- | --- | --- |
| IndicatorStudio、ProductDetail、ProductResearch、ProductCompare | single_product | 创作、标量/序列预览、人工保存；只在兼容的当前产品上下文选中并计算 |
| EvaluationPlan | single_product | 只读已有指标与当前选中产品，支持创作；保存只刷新目录，不变更评价方案 |
| HoldingDiagnosis | portfolio | 解释已保存组合快照、按该 run 计算已有 portfolio 标量；不把单产品指标加入组合选择器 |

HoldingDiagnosis 保留 AI 入口，但 v1 不借此新增组合指标作者/序列编辑器。需要创作单产品指标时提示人类选择具体产品并进入已有单产品工作区，不能拿组合 run_id 冒充产品 id。portfolio inline 的引擎能力单独标为“已有 API、作者 UI 未接入”，不得先保存一个未验证草稿来绕过预览。后续若接入组合创作，须完整补齐编辑器、预览/导出与刷新验收，而非只放开一个工具参数。

研究时点由服务端绑定，LLM 没有修改它的工具参数：

1. 前端复用现有 API 客户端的 `x-pit-as-of / x-pit-run-mode / x-pit-release / x-pit-off` 请求头；`view_state=unknown` 允许交流、校验及人工保存定义，仅禁止发起产品数据计算，缺字段不能解释为关闭 PIT。`off` 必须与页面显式选择及请求头一致。
2. 单产品回合开始调用现有 `resolve_request_context`，保留“系统 < 页签 < 页面显式截止日”的优先级；冻结解析后的 as_of、run_mode、data_release_id 与实际数据 generation/指纹。调用 availability、validate 后预览、Excel 和提交预检时均用这一上下文。需要传递 contextvar 时复用现有 request-local override 并在 finally reset；`evaluate` 只接收其实际支持的参数，不能臆造 run_mode 等关键字，也不能只绕过路由直传 `as_of=None`。
3. 组合回合从服务端读取 run 的实际窗口、requested/effective_as_of、数据指纹和已记录的来源证据；页签时点只作为查看背景，不重算或改写历史快照。未记录的 run_mode/release 信息保留未知，不能从当前设置补造。
4. `context_hash` 记录上述有效上下文、目标、窗口及参数；每个预览另绑定完整定义哈希/锁定 refs。回合前后检查数据 generation；不一致即丢弃计算证据并要求重试，不能声称跨数据刷新仍是同一快照。
5. 页面换产品、run、窗口、研究口径或页签实例后立即作废未提交的预览/确认/UI 批次；晚到响应只进入原会话历史，不回填当前页面。恢复会话时重新比较上下文与 catalog，不能复用旧 compile_token 证明当前进程已预热。

### 3.3 会话存储（`sessions.py`）

会话、run、工具回执与事件采用标准库 SQLite 的短事务 checkpoint；完整协议和 JSON 迁移见 [Harness 无默认上限与无进展检测设计](ai-agent-harness-progress-design-2026-09-20.md) 第 9 节。网络/计算期间不持会话锁，用户消息接纳、工具回执、草稿和单调事件序号必须可恢复。

- 同 message_id 返回原 run/回执；同一会话只有一个活动 run。停止、读取状态和事件订阅不等待整轮执行。
- JSON 会话仅作迁移输入；切换后只有一个可写真相源。LLM 配置和经人类确认的长期记忆保留各自存储，不迁入工具上下文。
- 模型压缩不能删除原始审计或重置检测器；前端分页不改变事件序号。无法恢复的旧历史明确标记，不伪造。
- 指标仓库与 agent 状态不构成跨存储事务；提交写前记录 intent，结果不确定时禁止自动重放。

### 3.4 工具契约（`tools/`）

工具定义字段：

```python
@dataclass(frozen=True)
class AgentTool:
    name: str                      # 命名空间.动作，如 metrics.validate
    scope: str                     # 归属作用域
    kind: Literal["read", "write_local"]
    description: str               # 给 LLM 的一句话
    parameters: dict[str, Any]     # JSON Schema（由 Pydantic 模型导出）
    handler: Callable[..., Any]    # 直调 service 函数
    limits: dict[str, Any] = field(default_factory=dict)  # 结果截断上限等
```

**指标中心（`indicator_center.py`）**

| 工具 | 类型 | 直调 | 说明 |
| --- | --- | --- | --- |
| `metrics.lookup` | read | `catalog` 内存索引 | 按关键词检索算子/变量详情（签名、参数、形态、LaTeX 模板、interval_policy） |
| `metrics.validate` | read | `indicator_service` 校验路径 | 返回合法性与诊断（错误码 + 位置） |
| `metrics.infer` | read | `indicator_service` infer | 逐节点类型/形态、`display_latex`、`required_variables` |
| `metrics.availability` | read | `variables/availability` | 指定 `as_of` 下变量可得性（PIT 门禁） |
| `metrics.preview` | read | `validate` → 带 token 的 `evaluate` / `evaluate-series`（`inline_definition`） | 先绑定并校验完整定义、参数与页面上下文；按标量/序列分流；不写指标目录；仅用户要求试算时执行，重复无变化预览由进展守卫处理 |
| `metrics.excel` | read | 同一完整定义经 validate 后带 token 调 `export-excel` | 供人类逐单元格复核；绑定同一产品/窗口/口径，禁止把展示 LaTeX 作为导出公式 |
| `metrics.draft_save` | write_local | `sessions.save_draft`（新增薄编排） | 输入 `definition + expected_draft_revision`，校验后只保存本会话草稿；不调用 `IndicatorGraphService.save_state`，不产生指标 revision |

**产品研究（`product_research.py`，v1 全部只读）**

| 工具 | 类型 | 直调 | 说明 |
| --- | --- | --- | --- |
| `products.search` | read | 产品/池检索服务 | 解析产品名称/代码为 id；受页面上限守卫约束 |
| `products.eval` | read | 锁定 refs；多标量先 `prepare_evaluation`，inline 先 `validate`；再 `evaluate` | 复用同一预览适配器和进展守卫；工具别名不能绕过编译/上下文约束或重复检测 |
| `products.series` | read | 锁定 refs 或经 `validate` 的 inline → `evaluate-series` | 序列指标（曲线/表格），与 metrics.preview 共用计算准入与无进展守卫 |
| `products.plans` | read | 当前 `plans.list()` 筛选 + `snapshot_config.references_indicator()` | 只接收指标 id，只返回 `CommitImpact` 投影；不返回方案/快照配置正文，不提供任意方案读取 |
| `portfolios.context` | read | 当前 `run_id` 的组合快照读取 | 只返回冻结窗口、来源、持仓/净值摘要；不列举其他组合或返回完整数值矩阵 |
| `portfolios.eval` | read | `evaluate_portfolio` | 只计算当前 run 的已保存 portfolio 标量；上限 10 个，不接收产品 targets、任意 as_of 或序列定义 |

**明确不存在的工具**：`create / update / delete indicator`。落库只能经 `POST /sessions/{id}/commit`（人类触发），这是原则 0.3-4 的机械保证。

**派发硬校验**：`execute()` 只按注册表名派发，未注册即 `AGENT_TOOL_NOT_ALLOWED` 并记事件；参数 schema 一律 `additionalProperties: false`；handler 必须命中 `(module, qualname)` 白名单，启动期校验失败只禁用 agent、不影响主应用。见 §2.3.5。

**UI 动作工具**：`ui.inspect` 与 `ui.act` 不在上表，机制见 §11；`ui.act` 只能操作界面状态，不能触发指标落库。

`plans.references_indicator()` 只返回 bool，不能据此构造方案 id 列表。id 列表由薄适配器遍历当前 `plans.list()` 的 indicators 生成；不改变现有删除守卫，不声称扫描了全部历史方案版本。影响面是带检查时间的提示，不锁定其他模块，也不改写它们。

`portfolios.*` 仍归属 product-research scope，只在 portfolio 页面上下文启用。既有组合入口只有 indicator_ids，适配器须记录调用前预期 revision，并校验结果中的实际 revision；不匹配即拒绝展示，不能声称该入口已支持任意历史 refs。单产品创作/提交工具不会因切到 indicator-center scope 而绕过计算域限制。

单产品计算与 Excel 适配器必须从服务端校验结果取得真实依赖，并在冻结上下文下执行 availability；不能依赖模型是否先调用过该工具，也不能接受模型自行缩减的变量清单。部分目标不可得时保留明确 unavailable/null 与原因，按既有服务契约处理，不填零、不冒充全覆盖。字段可得性不证明历史数据版本完整，也不证明算法因果性；继续保留现有时点/因果门禁及未知状态。

工具结果一律有界：预览向模型提供至多 5 个关键点与状态/单位/证据引用；页面可展开受限结果。大结果复用既有结果句柄与分页能力，会话不嵌入完整数值表；句柄过期明确提示重新计算，不能用摘要假造全量结果。`metrics.draft_save` 不能覆盖编辑器未保存工作；显式“发送到编辑器”见 §5.6。

### 3.5 说明书（catalog）生成与保鲜（`catalog.py`）

- 来源（全部已在代码中）：`backend/cal_indicators/typed_operators.py` 的注册表、`variable_registry.py`、参数策略（`backend/cal_indicators/parameter_policy.py`）、LaTeX 别名（`typed_latex.py`）、`interval_policy`。
- 输出结构（示意）：

```json
{
  "catalog_version": "sha256:...",
  "dsl_version": "...",
  "variable_registry_version": "3.0.0",
  "data_contract_version": "tushare-eod-v2",
  "context_schema_version": "typed-context-v2",
  "math_notation_version": "1.5.0",
  "variables": [{"name": "...", "role": "...", "unit": "...", "contexts": ["single_product"], "latex": "...", "availability": "..."}],
  "operators": [{"id": "...", "version": "...", "category": "...", "signatures": [{"inputs": ["series<time>"], "output": "scalar", "shape_rule": "..."}], "parameters": [], "latex_template": "...", "description": "...", "interval_policy": null}],
  "shapes": {"scalar": "...", "series<time>": "...", "vector<asset>": "...", "matrix": "..."}
}
```

- 运行时由 `catalog.py` 内存构建，`catalog_version = sha256(规范 JSON)`；会话、回合与 commit 事件均记录该版本（满足版本对约束）。
- `python -m backend.agent dump --out docs/agent/` 生成 `indicator_catalog.json` 与 `scopes.json` 快照。
- **保鲜**：`backend/tests/test_agent_catalog.py` 重新生成并 diff 快照；注册表变更未同步快照即测试失败（沿用 Tushare 文档契约检查的既定模式）。
- 注入策略：从 M1 起只注入变量摘要、算子分类索引及有界示例，完整事实由 `metrics.lookup` 查询；M4 再依据实际目录规模完善分类加载，不能在早期默认塞入无界全文。

### 3.6 意图路由（`router.py`）

判定轴：**scope（哪个模块）× action（read / author / workflow）**。

- 输入：`page_context`、会话当前 `scope`、消息文本、最近工具轨迹。
- L0 规则（零成本，覆盖大多数）：
  - 页面默认 scope：`indicator-studio → indicator-center`；产品研究各页 → `product-research`。
  - triggers 表（per scope，示例）：`indicator-center`：指标、算子、公式、ddof、窗口、序列、口径；`product-research`：这只基金、产品、池、评价、持仓、比较。
  - 显式上下文：页面携带 targets、run_id 或 indicator_id 时优先视为该对象的请求，但不自动改变其计算域。
- L1 模型（L0 未命中或信号冲突时）：prompt 只含作用域清单（id + 一句话 + triggers），输出 `{scope, action, confidence, reason}`，**枚举约束**、不可发明新作用域。
- 决策表：

| 判定 | 行为 |
| --- | --- |
| `read`（含跨模块只读） | 仅放行当前计算域与页面白名单允许的只读工具；不切换作用域 |
| `author`（创作/修订） | 在兼容 single_product 上下文切到 indicator-center；响应携带 scope_change。组合页面按 §3.2.1 提示已有能力边界，不因关键词命中放开作者工具 |
| `workflow`（跨模块写编排） | 明确告知 v1 未开放，并给出分步建议 |
| 低置信 / 同会话指令冲突 | 返回 `decision_card`，默认选项保留当前 scope，禁止静默切换或静默取最后一个说话人 |

- 每次判定作为事件写入会话（审计与调参）。

### 3.7 提交与回写（`commit.py`）

- **确认前预检**：人类打开提交条或修改名称/描述后调用 `POST /sessions/{id}/commit-preview`。服务端校验草稿版本、目标与计算域，规范化完整定义并通过现有 validate 路径编译；名称等字段若发生变化，先更新会话草稿并递增 draft_revision，返回新的版本供确认。按 §3.4 取得影响面投影，把实际将保存的定义、哈希、catalog_version、上下文及预览状态保存为本会话确认快照，返回 confirmation_id。此操作不写指标目录，也不向 LLM 暴露为工具。
- **复核对象固定**：页面呈现确认快照中的公式、参数、名称、计算域和影响面。预览只有在定义哈希、有效参数、产品/组合、窗口、研究口径与数据指纹一致时才标为 `current`。允许无真实数值预览保存，但须明显显示 `not_run/unavailable`，人类另行勾选确认；这只表示保存定义，不代表数值或投资有效性验收。
- **唯一目录写入口**：`POST /sessions/{id}/commit` 只接受确认引用与哈希，不再接收可以偷换的完整 definition。服务端先处理幂等记录，再比较当前草稿版本、确认哈希、catalog_version、页面/研究上下文及目标。任一变化返回 `AGENT_CONFIRMATION_STALE`，必须重新预检与确认。
- **明确新建/更新**：依据确认快照的 `target.operation` 调 `indicator_service.create_indicator(definition)` 或 `update_indicator(indicator_id, expected_revision, definition)`；不再用 revision 缺失猜测新建。service 继续负责校验、预热和仓库乐观锁，agent 不直写 repository。内置只读、历史 revision、方案锁定引用等契约保持原样。
- **提交结果**：成功记录指标 id/revision、完整定义哈希、catalog_version、确认 id、request_id、speaker 与影响面检查时间，返回 `CommitResponse`。`REVISION_CONFLICT` 返回 409 和当前版本摘要，保留草稿；rebase 只能形成新草稿、重新预检，不能自动覆盖。

幂等与中断处理采用会话文件中的小型提交记录，不引入第二套指标仓库：

1. 同一会话文件锁下为 `request_id` 保存 `pending` 记录，再调用唯一 service 写入口；同 id 不同请求摘要返回 `AGENT_REQUEST_CONFLICT`。同一 draft_revision 只允许一次成功提交，换 request_id 或重复预检也不能重复创建；pending/unknown 时禁止生成新的确认快照和提交。明确另建副本须从新草稿版本重新确认。
2. 成功后在同一会话原子写入 `succeeded + response`，相同请求重试直接返回原响应。页面发送超时须复用 request_id，先查会话状态，不生成新请求重试。
3. 明确尚未写目录的校验/冲突错误可记为 `failed`；写调用结果不确定、进程在目录写入后但回执写入前中断时，记录保持 `pending/unknown`，返回 `AGENT_COMMIT_UNCERTAIN` 并阻止重放。只能核对实际目录与日志中的非敏感提交标识后人工处理，禁止按相似名称/哈希猜测成功或宣称自动 exactly-once 恢复。既有仓库随机分配 id，v1 不为此修改其 schema。

影响面在预检和实际提交前各读取一次，仅说明当时的当前引用；目录更新不改方案或快照配置，不能把这份提示当作跨模块锁。跨存储失败不得回滚或删除可能已成功写入的指标。
- `refresh` 指令（前端按能力处理）：

```json
{"kind": "indicator_created|indicator_updated", "indicator_id": "...", "revision": 3,
 "suggest": ["reload_catalog", "select_in_current_page"], "opened_from": {"module": "...", "page": "..."}}
```

- **回写范围**：指标本体（进 `data/custom_indicators.json` 目录）+ `display_latex`。
  不写：展示偏好（localStorage）、评价方案、快照配置、产品池。这些是其他模块的写边界，见 0.2。

### 3.8 API 端点（`routes.py`）

| 方法 | 路径 | 请求 | 响应 | 说明 |
| --- | --- | --- | --- | --- |
| GET | `/api/agent/meta` | — | `{configured, model, catalog_version, scopes}` | 未配置时前端只显示「未启用」，其余功能不受影响 |
| POST | `/api/agent/sessions` | `AgentSessionCreate` | 会话摘要 | 创建并推导初始 scope |
| GET | `/api/agent/sessions` | `?limit=` | 会话列表 | 恢复入口 |
| GET | `/api/agent/sessions/{id}` | `?event_from=` | `{meta, state, events}` | 事件增量拉取 |
| POST | `/api/agent/sessions/{id}/messages` | `AgentMessageRequest` | `AgentTurnResponse` | 兼容整轮返回；新增异步 run 与事件订阅见 Harness 设计第 12 节 |
| POST | `/api/agent/sessions/{id}/commit-preview` | `CommitPreviewRequest` | `CommitPreviewResponse` | 人类提交条预检，冻结待确认定义；不写指标目录 |
| POST | `/api/agent/sessions/{id}/commit` | `CommitRequest` | `CommitResponse` | 唯一落库路径 |
| POST | `/api/agent/sessions/{id}/memory` | `MemoryResolution` | 更新后的提案状态 | 接受即追加记忆文件 + 审计 |
| POST | `/api/agent/sessions/{id}/scope` | `{scope}` | 会话摘要 | 前端 chip 手动回退/切换 |

- 路由类沿用 `StableValidationRoute`（与 `custom_indicator_routes.py` 一致）；注册沿用 `app.include_router`（`app.py:325` 起）。
- 启动集成：`app.py` 启动阶段只读取配置并暴露状态；**不因 LLM 未配置而阻止启动**。
- `/messages` 的纯 `ui_results` 请求只验证已发出的批次并记录回执；`decision` 只处理已保存卡片。发送中断后先通过 GET session 的 message/request 状态恢复，不把轮询等同于重新执行。

### 3.9 错误语义

| 错误码 | HTTP | 语义 | 前端行为 |
| --- | --- | --- | --- |
| `AGENT_NOT_CONFIGURED` | 503 | 未配置模型 | 面板显示未启用与配置说明 |
| `AGENT_LLM_UNAVAILABLE` | 502 | 模型调用失败（已重试） | 就近错误态 + 重试按钮；草稿不丢 |
| `stop_reason=no_progress` | 200 / run.paused | 检测到停滞且有界纠偏未解决 | 保留产物与证据，解释阻碍并允许人类补充；非额度错误 |
| `AGENT_INTENT_AMBIGUOUS` | 200 | 非异常：`decision_card=intent_ambiguous` | 选项卡 |
| `AGENT_NOT_EXPRESSIBLE` | 200 | 非异常：现有算子无法表达 | 展示原因与最接近的可行表达（若有） |
| `REVISION_CONFLICT`（复用既有） | 409 | 指标已被他人更新 | 冲突决策卡（当前值 / 改动 / rebase 建议） |
| `AGENT_DRAFT_CONFLICT` / `AGENT_SESSION_BUSY` | 409 | 草稿版本已变化 / 已有回合执行 | 保留本地编辑，刷新状态后再操作 |
| `AGENT_CONFIRMATION_STALE` / `AGENT_CONTEXT_CHANGED` | 409 | 确认定义或研究上下文已失效 | 取消旧预览/动作/确认，重新复核 |
| `AGENT_REQUEST_CONFLICT` | 409 | 同一请求 id 对应不同内容 | 不重试写入，检查客户端状态 |
| `AGENT_COMMIT_UNCERTAIN` | 409 | 目录写入结果尚不能确定 | 显示待核对，禁止自动重放提交 |
| `VALIDATION_ERROR`（复用既有码） | 422 | 落库校验失败 | 错误定位到字段/节点；与现有 `ValidationError` 状态码一致 |

原则：LLM/工具失败不得触发指标目录写入；目录写入成功但响应丢失不等于写入失败，必须按提交记录返回或进入待核对状态，不能自动再建一个指标。

### 3.10 配置与安全

- 环境变量：`AGENT_LLM_BASE_URL`、`AGENT_LLM_API_KEY`、`AGENT_LLM_MODEL`、`AGENT_LLM_TIMEOUT_SECONDS`、`AGENT_LLM_MAX_RETRIES`、`AGENT_LLM_MODE=live|fixture`、`AGENT_LLM_FIXTURE_DIR`。
- 密钥只从环境变量读取，不入仓库、不进前端、不回显、不写日志（沿用仓库对敏感 token 的既有要求）。
- 工具白名单：无网络工具、无任意文件读写；`draft_save` 仅写当前会话草稿；`commit` 仅通过指标 service 写目录并记录会话审计。白名单由启动期校验与派发校验强制执行（§2.3.5），提示词不承担安全职责。
- 网络出口唯一：agent 模块仅 `llm.py` 允许使用 `httpx`；由静态 import 检查与 fixture 测试证明（§7.1）。
- 测试与 E2E 一律 `AGENT_LLM_MODE=fixture`，不调用网络（仓库测试规范：测试禁止网络调用）。

---

## 4. AI Harness 设计

### 4.1 组件与职责

完整架构、源码差距和文件职责见 [Harness 无默认上限与无进展检测设计](ai-agent-harness-progress-design-2026-09-20.md) 第 3–5 节。一个同进程 runner 复用现有工具和指标服务，默认不限制单次用户请求的工具调用数与模型往返数；进度账本、逐步 checkpoint、停止/恢复是同一执行链的组成部分。

### 4.2 上下文装配

固定保留系统边界、用户目标/口径、冻结研究上下文、当前与最佳草稿、真实进度与否定项、近期完整工具消息组。完整数值表与完整目录不进入 prompt；压缩不改变权限或计算事实。容量与失败降级见 Harness 设计第 11 节。

### 4.3 回合循环与无进展检测

调用前检查权限、参数、依赖版本与已知重复；调用后保存语义结果、草稿和进度。相同错误、重复结果、A/B 振荡、改变表达式但没有改善均纳入检测；revision、时间戳或新参数不是进展。

出现停滞时先给一次有界纠偏机会，仍无改善则禁用工具并做一次阶段总结，保留草稿后交还用户控制。正常长任务持续有进展即可继续；不采用固定 3 次修复或单回合 1 次预览配额。规则、阈值、批次配对和伪代码统一见 Harness 设计第 6–8、14 节，不在此维护第二套循环。

金融语义仍由现有 validate/infer/evaluate 契约决定；不猜测或自动修改口径。inline 数值预览必须经过精确定义的 compile_token，正式计算只进入已预热计划；忙碌、缺失、PIT 和组合快照边界不变。

### 4.4 压缩与恢复

压缩仅管理单次上下文容量，不是调用额度或任务进展。原转录先持久化，摘要保留来源与完整 call/result 配对；检测器、已失败策略与纠偏状态不清零。同一失败压缩不无限重试；具体可恢复事件与终态见 Harness 设计第 9–12 节。

### 4.5 记忆（`memory.py`）

三层：

| 层 | 载体 | 内容 | 写入者 |
| --- | --- | --- | --- |
| 会话记忆 | 会话/run 的权威 checkpoint | 草稿、待确认、用量 | 系统 |
| scope 记忆 | `data/agent_memory/<scope>.md` | 口径惯例、命名偏好、默认参数、禁用项 | 人类确认后系统追加 |
| 共享记忆 | `data/agent_memory/shared.md` | 跨模块约定（如「收益率一律复权净值口径」） | 同上 |

- 流程：LLM 在受限响应中提出记忆内容（不额外注册 `memory.propose` 工具）→ 服务端分配 proposal_id → 前端展示 diff → `POST /memory {accept|reject, speaker}` → 接受则追加文件并写审计（时间、作者、理由、append 摘要）。同一 proposal_id 重试不得重复追加。
- 记忆文件为 Markdown，人类可直接编辑；系统只追加，废止旧偏好以带引用的替代记录表达，不删除审计。接受前比较提案基于的记忆版本，人工编辑后须重新展示差异；记忆与摘要均不授予权限、不能覆盖工具契约。
- 禁止静默自动学习；每次会话最多 3 条提案（防止噪声）。

### 4.6 多人与冲突

- 每条消息带 `speaker`；上下文与产物卡标注归属（「由张三要求」「由李四否决」）。
- 参数/口径冲突：返回 `DecisionCard(kind=param_conflict)`，选项各带 LaTeX 与数值差异摘要；服务端不自动择一。
- 修订冲突：`REVISION_CONFLICT` → `DecisionCard(kind=revision_conflict)`（当前值 / 你的改动 / rebase 建议）。
- v1 无账号：`speaker` 为自由标签；若后续引入认证，将其替换为认证身份而不改会话结构。

### 4.7 模型管道（`llm.py`）

```python
class LLMClient(Protocol):
    def chat(self, messages, tools, response_schema) -> Completion: ...
    # Completion: {content, tool_calls, parsed, usage{prompt,completion}, model}
```

- `HttpxClient`：OpenAI 兼容接口；连接/读取超时；仅对超时、连接中断、429、5xx 做指数退避 + 抖动重试（有界）；鉴权/参数/契约错误不重试。
- `FixtureClient`：按 fixture 目录回放预置响应，用于全部后端测试与 E2E（无网络）。
- 供应商切换只影响本文件；harness 不感知具体供应商。

### 4.8 提示词结构（`prompts/`）

- `system.md`：角色与原则（§0.3）、能力边界、失败语义（不可表达 / 低置信问人）、工具纪律（先查目录再写表达式）、模型提案 schema、禁止事项（新算子、自由 Python、未来数据、自动写指标目录）。
- `indicator_center.md`：DSL 语法与形态规则、参数契约、常见写法示例（每个示例不超过 5 行）、PIT/可用性要求、`result_kind` 选择规则（标量 vs 序列与产品研究展示的关系）。
- `product_research.md`：产品上下文语义、只读边界、数值预览的时点与窗口说明。
- `compaction.md`：4.4 的摘要模板与禁忌（不得丢失归属与否定原因）。
- 全部提示词随代码版本管理；`catalog_version` 与 `dsl_version` 在装配时注入，禁止在提示词中写死算子清单。

---

## 5. 前端设计

遵循 [docs/frontend/README.md](../frontend/README.md)：令牌唯一来源、12px 字号下限、对比度、四态、可达性与性能预算；不新增颜色、不新增动画库。

> 智能体操作前端界面（填写 LaTeX、设置筛选条件并实时反显）的完整协议见 §11；本节的面板与页面通过 `useAgentSurface` 注册的 UI Surface 对接。

### 5.1 入口与挂载点

- 共享组件 `AgentPanel` + 页内 `AgentLauncher` 按钮（页头/工具条右侧，文案「AI 助手」），**不新增路由**。
- 挂载页：`IndicatorStudio`（`/settings/indicators-models`）与产品研究五个页面（ProductDetail、ProductResearch、ProductCompare、HoldingDiagnosis、EvaluationPlan）。
- 形态：右下角浮窗（非模态），桌面宽 420px，小屏保留至少 12px 边距；页面可同时滚动核对数值。浮窗是浮层：`shadow-xl`、z-index 走系统层级常量、仅消息区滚动，标题与输入区固定可见；Portal 挂载至 body，浮层高于导航。

### 5.2 组件树与职责

```
AgentLauncher（页内按钮，aria-expanded）
└── AgentPanel（浮窗外壳：标题、ScopeChip、会话菜单、关闭）
    ├── AgentMessageList（role=list，aria-live=polite 标记新消息）
    │   ├── 用户消息（speaker 标签 + 时间）
    │   ├── 助手消息（Markdown 文本；不渲染 HTML）
    │   ├── AgentToolTrace（<details> 折叠：轮次、工具、摘要、错误码）
    │   ├── AgentDraftCard
    │   │   ├── DSL 代码块（等宽、可复制）
    │   │   ├── KaTeX 公式（复用 katex ^0.16.47；throwOnError:false, trust:false, displayMode:true）
    │   │   ├── 参数表（名称/类型/默认值/含义；数字列 text-right tabular-nums）
    │   │   └── 数值预览表（真实结果关键点；注明产品、区间、as_of）
    │   ├── AgentCandidateCard（并排候选：一句话语义 + 公式 + 差异数值）
    │   ├── AgentDecisionCard（冲突/低置信选项，默认项标记）
    │   └── AgentMemoryProposal（diff + 接受/拒绝）
    ├── AgentCommitBar（名称、描述、目标 scope、影响面提示、提交按钮、修订号）
    └── AgentComposer（textarea + 发送；Enter 发送、Shift+Enter 换行）
```

- 组件优先使用 `components/ui.tsx` 的 `Card / Button / Badge / EmptyState`；卡片 `rounded-xl`、控件 `rounded-lg`、徽章 `rounded-full`。
- 主操作（发送、提交）用 `bg-accent-600`；成功/警告/危险只用 emerald/amber/rose；次要文字浅底 `text-slate-600`。
- 局部空态使用纯文字（`EmptyState mascot={false}`）；面板经常紧邻收益/净值数值，按 15.2 不出现吉祥物。

### 5.3 交互状态（四态齐备）

| 状态 | 表现 |
| --- | --- |
| 加载 | 与消息布局同形的骨架行（`motion-reduce` 下静态）；「正在生成…」为文本状态并置于 `aria-live` 区域 |
| 空 | 首次打开：说明面板用途 + 一个示例提问按钮（纯文字空态） |
| 错误 | 就近内联错误 + 重试按钮；模型未配置时显示「未启用」与配置说明，不出现重试 |
| 禁用 | 发送按钮在空输入/生成中禁用，并以可见文字说明原因（如「正在生成，请稍候」） |

- 按下反馈 `active:translate-y-px`；过渡仅 CSS，≤200ms；`prefers-reduced-motion` 下退化为静态。
- 键盘：`Escape` 关闭浮窗并把焦点交还 launcher；焦点样式 `focus-visible:ring-2 focus-visible:ring-accent-500`。

### 5.4 数据流

- `frontend/src/services/agent.ts`：封装 3.8 全部端点；类型从后端契约手写镜像（与 `customIndicators.ts` 现有做法一致）。
- `useAgentSession` hook：`send()`（POST 消息）、`commit()`、`resolveMemory()`、`switchScope()`、`resume(sessionId)`；维护事件列表与当前 `state`。
- 版本一致：先 `/commit-preview` 冻结完整定义及明确的创建/更新目标，再携确认 id、定义哈希和草稿版本提交；名称、描述、参数或上下文变化均需重新预检。

### 5.5 提交后的刷新协议（回写前端）

`CommitResponse.refresh` 由页面注册的处理器消费：

| 页面 | 处理 |
| --- | --- |
| IndicatorStudio | 刷新目录；用返回 `id/revision` 打开对应指标的编辑器（组件回调，不新增 URL 参数） |
| ProductDetail / ProductResearch / ProductCompare | 重新拉取兼容域目录；把新指标加入当前页临时选择并触发计算，不改持久化展示偏好 |
| HoldingDiagnosis | 保持当前不可变 run；仅刷新 portfolio 目录/结果，不把 single_product 新指标自动选入，不重写原快照 |
| EvaluationPlan | 刷新可用指标目录；不自动改写方案（写方案属产品研究侧，超出 v1 边界） |

- 自动选中使用页面临时状态；不能直接调用会写 localStorage 的偏好 setter，把“前端代写”当作绕过只读边界。用户随后显式保存偏好仍走原流程。
- 若指标处于「更新」路径：仅当前未锁定的编辑/展示可跟随新 revision；历史请求、方案和运行证据继续使用其锁定版本。页面上下文已变化时只提示保存成功，不替换当前选择。

### 5.6 指标中心编辑器联动

- `IndicatorStudio` 将 `AgentPanel` 挂载在编辑器工具条旁；提交成功后回调 `onIndicatorCommitted(id, revision)`，走页面既有的选择与编辑状态，不复制一套加载逻辑。
- AI 草稿归本会话所有，`metrics.draft_save` 不写 editor-state。人类点“发送到编辑器”后，页面适配器检查计算域、page_instance、surface revision 和未保存修改；有冲突先展示差异并确认，再一次性导入完整 draft（含可执行文本、参数和展示元数据）到现有 React 编辑状态。此人工动作不开放给 `ui.act` 自动覆盖。
- 导入不写指标目录，不调用 `IndicatorGraphService.save_state`；首次创建也不伪造 indicator_id/revision。提交成功后才按真实返回 id/revision 读取既有编辑状态；如需保存布局，仍使用既有定义指纹校验与 `expected_editor_revision`。不存在强制覆盖接口。

### 5.7 国际化与文案

- 全部界面文案走 `locales/`（中英齐备，`node scripts/check_i18n.mjs` 通过）；JSX 不落中文硬编码。
- 关键文案示例：`agent.panel.title`、`agent.composer.placeholder`、`agent.commit.confirm`、`agent.scope.indicatorCenter`、`agent.scope.productResearch`、`agent.error.notConfigured`、`agent.decision.title`。

### 5.8 可达性与性能

- 消息列表 `role="list"`；新消息区域 `aria-live="polite"`；工具轨迹用原生 `<details>/<summary>`；表格 `<th scope>`、`<caption>` 或 `aria-label`。
- 浮窗打开不锁定页面滚动（非模态）；焦点不被困住，但 `Escape` 可关闭。
- 不引入新依赖（KaTeX 已有）；不新增图表，数值预览复用表格与 `MetricDisplay` 的公式渲染路径。
- 响应式：320px 无横向溢出；窄屏浮窗保留边距、消息与卡片单列。

### 5.9 前端测试

- Vitest + RTL：`AgentPanel.test.tsx`（四态、发送、轮次渲染、决策卡选择）、`AgentCommitBar.test.tsx`（名称/描述校验、更新提示修订）、`AgentDraftCard.test.tsx`（公式/参数/预览渲染）、`agent.test.ts`（客户端请求体与错误映射）。
- 断言提交回调：`onIndicatorCommitted` 被调用且页面刷新函数被触发（mock 服务层）。
- E2E：`frontend/e2e/agent-panel.spec.ts`，以 `AGENT_LLM_MODE=fixture` 的后端运行，覆盖：产品详情页发起 → 草稿出现 → 提交 → 指标出现在选择器 → 公式与数值可见。

---

## 6. 端到端流程

### 6.1 场景 A：产品详情页发起创作（主链路）

1. 用户在 ProductDetail 打开「AI 助手」，输入「做一个衡量基金下跌后反弹力度的指标」。
2. 路由：页面默认 scope=`product-research`；triggers 命中「指标/反弹力度」→ `author`，目标 `indicator-center`；响应携带 `scope_change`，chip 显示「指标中心」。
3. 循环：`metrics.lookup`（找波动/回撤类算子）→ 产出 DSL → `metrics.validate`/`metrics.infer`（含 token、display_latex）→ `metrics.preview`（同一 inline_definition/token 在冻结的 targets、窗口和口径上算真实数值）。
4. 草稿卡呈报：DSL、公式、参数、最近 1 年数值关键点；人类质疑「窗口应该是 60 日」→ 参数调整 → 重算。
5. 人类填名称描述 → commit-preview → 查看实际待保存公式与影响面 → 人工确认 commit → 返回 `{id, revision:1}`；超时沿同一 request_id 查询/重试，不再建一个指标。
6. 页面刷新：目录出现新指标并被加入当前指标选择，图表/表格立即出数；`MetricDefinitionDrawer` 可查看定义。

### 6.2 场景 B：指标中心里的跨模块预览

1. 用户在 IndicatorStudio 编辑草稿，问「这只 ETF（页面已带 ids）最近一年表现如何」。
2. 路由：`read` 跨模块；直接用 `products.series`/`products.eval`，不切换 scope。
3. 结果以数值预览回到对话；继续编辑不受影响。

### 6.3 场景 C：修订既有指标（revision 冲突）

1. 用户在产品研究页说「把动量指标的年化改成 252 天」。
2. 路由 author；服务端读取并锁定当前 revision，产出草稿；commit-preview 明确 target.operation=update、indicator_id 与 expected_revision，最终提交引用该确认快照。
3. 若期间他人已更新 → 409 → 冲突决策卡：查看差异、以最新版本 rebase、或放弃。
4. 成功后未锁定的当前展示可跟随新 revision；历史请求/方案仍保留旧引用及结果。新建副本须显式选择 create，不把丢失 revision 当成新建指令。

### 6.4 场景 D：记忆提案

1. 会话末尾 LLM 提议：「记住：本工作区收益率口径为复权净值」。
2. 前端展示提案卡；用户接受 → 写入 `data/agent_memory/shared.md` 并记审计；拒绝则不落任何内容。

---

## 7. 测试与验收

本节列出实现阶段的验收要求，当前均不是已通过的证据。仅修改本设计时检查文档内部契约、示例、链接及引用路径，不运行未实现的 agent 测试，也不据此声称页面或计算已验收。

### 7.1 后端测试（`backend/tests/`，pytest，无网络）

| 文件 | 覆盖 |
| --- | --- |
| `test_agent_catalog.py` | 说明书生成与 `docs/agent/indicator_catalog.json` 快照一致；版本字段齐备 |
| `test_agent_scopes.py` | 工具名存在；scope 到真实页面/路由的显式映射正确（不要求 indicator-center 等于 processRegistry 的阶段 id）；计算域裁剪；LLM 无目录/外部模块写工具 |
| `test_agent_harness.py` | FixtureClient 驱动：无默认总次数上限、无进展检测与有界纠偏、并发消息、旧上下文拒绝、恢复令牌失效、压缩保留原始审计、模型伪造哈希/结果不得成为权威字段 |
| `test_agent_commit.py` | 真实临时 repository 的 create/update；预检哈希与目标绑定、名称变化重新确认、无预览显式确认、revision 冲突、重复请求不重复写；在目录写后/回执前注入中断，结果不确定时禁止重放 |
| `test_agent_router.py` | L0/L1 判定、低置信决策卡、冲突指令不自动择一 |
| `test_agent_memory.py` | 提案接受/拒绝、审计追加、人类可编辑文件不被静默覆盖 |
| `test_agent_api.py` | 3.8 端点状态码与响应结构；未配置时 503 仅限 agent 路由 |
| `test_agent_boundaries.py` | 业务写工具仅 metrics.draft_save 且只写当前会话；无新指标 id 时也可保存草稿；从不调用 editor-state 写方法；未知工具/越域目标/伪造 surface 拒绝；LLM 与 ui.act 均不可达 commit；静态 import 检查加 service 写调用 spy；fixture 不发网络；模型跳过 availability 或省略变量时仍由服务端检查；影响面只返回 id/bool、不泄露正文 |
| `test_agent_context.py` | 系统/页签/显式截止日优先级，unknown 与 off 区分，数据 generation 改变拒绝旧证据；portfolio 固定 run_id、实际 revision 核对，不改历史快照、不混用 single_product |
| `test_agent_ui.py` | 页面白名单交集、batch/action/receipt 关联、纯回执幂等、拒绝和撤销事件类型；未知/重放回执不触发模型或目录写入 |

### 7.2 NJIT 与数据真实性

- 预览、提交均走现有 service：`inline_definition` 预览使用真实执行引擎；`create_indicator` 触发既有 NJIT 预热（`series_service.warm` / `_warm_single_product_definition`）。
- 测试须断言：agent 提交的指标在预热完成后可直接计算，正式 evaluate 路径不编译新计划/签名且无 Python 回退。validate/prepare 是显式作者准备阶段，其编译不能与正式计算混为一谈。
- PIT：`metrics.availability` 覆盖；测试覆盖「不可得变量必须显式不可用」。
- 分别覆盖标量/序列的 validate → token → evaluate，以及已保存多标量的 prepare → 锁定 refs → evaluate；断言 prepare 不是 inline token 的生成入口。组合只走真实 run 的 evaluate-portfolio，不用单产品 fixture 冒充组合支持。

### 7.3 前端验证（按设计准则第 12/14 节）

```sh
npm run design:check --prefix frontend
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
node scripts/check_i18n.mjs
python skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```

浏览器验收覆盖：真正打开/关闭浮窗、生成中状态、错误重试、未配置态、决策卡选择、提交后页面刷新选中；320 / 768 / 1440 视口；浅底次要文字与 KaTeX 公式对比度。

### 7.4 E2E

`frontend/e2e/agent-panel.spec.ts`（fixture 模式）：产品详情页完整闭环 + 指标中心打开新 revision。测试夹具不写正式数据目录（沿用隔离夹具约定）。

### 7.5 验收清单（需求级）

- [ ] 自然语言 → 合法 DSL；不可表达时明确回答而不是近似。
- [ ] 人类可查：DSL、KaTeX 公式、Excel 导出、真实数值预览（带产品、区间、as-of）。
- [ ] 落库为不可变 revision；更新产生新 revision；冲突不覆盖。
- [ ] 提交后前端目录立即刷新，产品研究页可选中并计算展示。
- [ ] 历史 `indicator_revision` 引用不被改写（P12/P14 语义）。
- [ ] 跨模块只读不切换作用域且受计算域限制；指标目录写入只经人工确认的 commit，会话草稿单独保存。
- [ ] 会话可恢复；压缩后仍保留归属与否定原因；记忆变更有人工确认与审计。
- [ ] AI 能操作前端界面（写入 LaTeX、设置筛选条件并驱动真实查询），页面实时反显，动作可见、可撤销、可审计（见 §11.7）。
- [ ] 动作合法性由服务端白名单与 UI Surface 共同限定；同批多动作不误判过期，人工插入编辑停止剩余动作，撤销不覆盖后续修改。
- [ ] 硬限制由测试证明（§7.1）：未注册工具不可达、目录写路径只有 commit、草稿不写布局、无 path/网络工具、inline 预览必经正确 compile_token。
- [ ] 提交预检和实际保存是同一完整定义；重试不重复创建，中断不确定状态可见且禁止自动重放。
- [ ] 产品/组合计算域、时点与数据指纹可追溯；自动刷新/选中不改持久化偏好、方案或快照。
- [ ] 无 LLM 时系统其余功能正常。

---

## 8. 里程碑

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| **M1 指标中心单模块闭环** | harness v0、最小会话/草稿存储、正确编译预览、目录与边界工具、Panel、提交预检/幂等 create、人工导入编辑器；无长期记忆/压缩 | catalog/scopes/harness/commit/api/boundaries 的已实现项及 §7.3；浏览器完成「对话 → 草稿 → 预检确认 → 保存 → 编辑器打开」 |
| **M2 产品研究接入** | 五页按 §3.2.1 能力矩阵接入；产品跨模块创作、组合只读上下文、时点冻结、临时选中与影响面投影 | 场景 A/B、context 测试与 E2E；持仓诊断不误用单产品工具，不写偏好/方案 |
| **M3 会话、路由与记忆** | 会话恢复、router L0/L1、scope chip、记忆文件与提案、多人 speaker/决策卡、update 路径与冲突处理 | 场景 C/D 全通；`test_agent_harness/router/memory` 通过 |
| **M4 压缩与目录扩展** | 压缩与降级、catalog 分类按需加载、会话列表 UI、性能与用量记录 | 长会话压缩后继续可用；压缩摘要字段完整；目录快照契约通过 |
| **M5 UI 动作层（§11 基础动作）** | ui.inspect/ui.act、受限 surface、版本化批次、基础字段/条件/查询动作、回执与安全撤销 | §11.7 全部基础动作检查 + surface/ui 契约测试；图编辑器动作不在本里程碑 |

§11 基础 UI 动作独立到 M5，避免拖累 M1–M3 主链路；图编辑器动作（`insert_node / connect / set_arguments`）在此之后按需排期。

每个里程碑完成后按 `docs/governance/branch-submission-rules.md` 走分支、测试证据与合并流程；实现阶段同步更新 AI Hermes 路由记忆并运行校验脚本。

---

## 9. 风险与边界

| 风险 | 缓解 |
| --- | --- |
| LLM 产出看似合理但口径错误 | 编译器验证语法/类型/契约，不证明经济含义；用公式、Excel、真实数值与需求逐项人工复核，不可表达即拒绝 |
| 目录/提示词漂移 | 说明书由注册表生成 + 快照契约测试；提示词不写死算子清单 |
| 修订冲突覆盖他人工作 | `expected_revision` 乐观锁 + 冲突决策卡；无强制覆盖工具 |
| 上下文膨胀与成本 | 工具结果截断、预览关键点、压缩、目录按需加载、无进展检测与可停止运行 |
| 多人场景的隐性分歧 | 归属标注 + 冲突决策卡 + 审计；v1 明确不做并发编辑 |
| 供应商/密钥风险 | key 仅环境变量；`llm.py` 单一边界；fixture 模式覆盖全部测试 |
| 「AI 生成物」被误读为已验收模型 | 全部产物标注「AI 生成，待人工确认」；提交动作在 UI 上明确显示目标模块与影响面 |
| UI 动作误操作或与人类抢状态 | 动作可见、可撤销、带 surface revision 防过期；落库不经 UI 动作；非法动作由前端拒绝 |
| 性能：KaTeX/大消息流 | 公式渲染限草稿卡；消息虚拟化在 M4 视实测再引入 |
| 未保存公式预览触发 NJIT 编译占 CPU | 通过 validate 显式编译，计算准入、同依赖复用与无进展守卫；正式 evaluate 只用已预热计划；透传 busy |
| 提交成功但响应丢失 | 确认快照与 request_id/draft_revision 去重；跨存储不确定时失败关闭，禁止自动再建指标 |
| 无账号导致跨用户隔离不可执行 | v1 只做进程级/单工作区边界；多用户依赖部署边界（localhost/内网）；认证前不得对外多用户开放 |

---

## 10. 参考

- 指标中心：`frontend/src/pages/IndicatorStudio.tsx`、`frontend/src/services/customIndicators.ts`、`backend/services/custom_indicator_routes.py`、`backend/custom_indicators/service.py`、`repository.py`、`graph_service.py`、`variable_registry.py`、`backend/cal_indicators/typed_operators.py`、`typed_latex.py`、`latex_excutor.py`。
- 产品研究：`frontend/src/pages/ProductDetail.tsx` 等、`frontend/src/components/metrics/useMetricDisplayPreference.ts`、`MetricDisplay.tsx`、`backend/services/product_analysis.py`、`research_series_routes.py`、`backend/research_series/service.py`。
- 治理与规范：`AGENTS.md`、`docs/frontend/README.md`、`docs/governance/branch-submission-rules.md`、`docs/repo_map.json`、`docs/task_routes.json`、`docs/pitfalls.json`。
- 相关既有设计文档：[风险标尺](../pre-investment/risk-scale.md)（同类前后端一体设计写法）。

---

## 11. UI 动作层：智能体操作前端界面（追加需求）

### 11.1 结论：需要的不是「AI 的前端」，而是一层 UI 动作协议

- **否决像素级控制**（截图识别 + 模拟点击/键入）：不稳定、不可测试、绕过字段约束、无无障碍语义，且违反「唯一实现」原则。
- **正确形态**：页面注册 UI Surface，服务端把模型提出的动作校验为 `UiActionBatch`（§11.2），随会话响应返回；前端通过受控适配器更新真实 React 状态并回传结果，计算与目录保存仍走原服务。
- 智能体**不需要浏览器控制权，也不需要独立的「AI 前端服务器」**；它需要的接口就在现有前端里：页面注册表 + 动作 schema。后端只负责生成、校验与审计动作，不执行 UI。
- 三层分工保持不变：**后端工具**（计算/校验/数据）→ **UI 动作层**（页面状态补丁，本节）→ **人工确认**（筛选提交可自动，指标落库只走 §3.7 commit）。

### 11.2 UI 动作契约（`backend/agent/ui.py` + 前端 TS 镜像）

```python
class UiField(Contract):
    target: str                    # 稳定 id，如 productResearch.condition.found_date
    kind: Literal["text", "number", "date", "select", "multi_select",
                  "boolean", "condition_list", "code", "graph"]
    label: str
    value: Any                     # 当前值（前端上报）
    options: list[str] = []
    constraints: dict[str, Any] = {}   # min/max/pattern/enum/required/fields/operators

class UiSurfaceSnapshot(Contract):
    page: str                      # 与 PageContext.page 使用同一枚举
    page_instance_id: str           # 必须匹配本轮 PageContext
    revision: int                  # 前端每次状态变更自增，用于防过期应用
    fields: list[UiField] = []
    actions: list[UiCapability] = []  # target/action/参数约束；与服务端页面白名单取交集

class UiCapability(Contract):
    target: str
    action: str
    effect: Literal["local_state", "read_query"]

class UiActionSpec(Contract):
    action_id: str                 # 服务端生成；模型只提出 action/target/value/note
    action: Literal["set_field", "toggle_option", "add_condition", "remove_condition", "clear_conditions", "submit", "highlight"]
    target: str
    value: Any = None
    note: str | None = None        # LLM 对本次操作的一句话解释，显示在动作 chip 上

class UiActionBatch(Contract):
    batch_id: str
    page_instance_id: str
    context_hash: str
    base_revision: int
    actions: list[UiActionSpec]
    apply_mode: Literal["staged", "immediate"] = "staged"

class UiActionResult(Contract):
    action_id: str
    status: Literal["applied", "rejected", "not_executed", "undone"]
    before_revision: int
    after_revision: int
    old_value: Any = None
    new_value: Any = None
    reason_code: str | None = None

class UiBatchReceipt(Contract):
    receipt_id: str
    batch_id: str
    operation: Literal["apply", "undo"]
    results: list[UiActionResult]
```

| 动作 | 语义 | 适用 |
| --- | --- | --- |
| `set_field` | 写入单个字段 | 文本/数字/日期/下拉/复选框（LaTeX、参数、窗口、关键词） |
| `toggle_option` | 多选增删 | 分类筛选（管理人等） |
| `add_condition` | 追加条件 | 条件构建器（`{field, operator, value}`） |
| `remove_condition` / `clear_conditions` | 删除条件 | 同上 |
| `set_field`（关键词目标） | 设置搜索词 | 产品研究搜索框复用同一动作，不另造别名 |
| `submit` | 触发页面既有查询/推断 | 「查询」「校验公式」（不落库） |
| `highlight` | 仅视觉提示，不改状态 | 指向目标控件 |
| `insert_node` / `connect` / `set_arguments` | 图编辑器操作 | M5 之后按需设计与扩展动作枚举；当前 schema 不接受 |

- 前后端共享同一份动作 schema（后端 Pydantic 为源，前端 TS 镜像）；契约测试防漂移。
- 动作合法集合为**服务端页面能力白名单 ∩ 当前 surface 声明 ∩ 当前计算域允许项**；schema 一律禁止额外字段。surface 是客户端输入，不能授予新权限；前端还须校验真实控件的类型/枚举/范围/单位。服务端不得发明 target，也不得信任客户端声称某保存按钮是只读查询。
- `submit` 必须命中具体的只读动作 id（如产品查询、公式校验），不能是通用按钮点击或任意回调名；批次最多一个 submit 且放最后。指标保存、commit-preview/commit、editor-state 写入、方案/池/偏好写入均不得注册。工具 `ui.act` 只提出动作，最终批次由服务端验证、保存后再发给页面。

### 11.3 UI Surface 注册（前端）

- 每个挂载页使用 `useAgentSurface(pageId, { fields, actions })` 声明：target、类型、当前值、约束、setter 和具体只读查询绑定。setter 若会通过 effect 自动保存目录、布局或展示偏好，不能直接暴露；使用页面临时编辑状态，显式人类操作后再走原保存流程。
- M5 启用 UI 动作后，会话创建与**每条消息**都携带 UiSurfaceSnapshot；缺失时只允许对话/后端只读工具，不产生 UI 批次。人类可能手动改过页面，每轮必须基于最新快照决策。
- target 命名规则：`<page>.<区域>.<字段>`，全站唯一且稳定，可在测试中断言；禁止用 CSS 选择器或坐标。
- 批次开始核对 page_instance_id、context_hash 与 base_revision，任一不符则整批拒绝。批内维护当前 revision 游标：自己的成功修改更新游标，不能让第二个动作继续比较最初 base_revision；人类编辑、跨页或迟到请求造成的外部变化立即停止剩余动作，已执行前缀在回执中明确标识，不能报告整批成功。
- 会改变计算目标、窗口或研究口径的动作不得与依赖新上下文的计算放在同一批次；先回传变更，再按 §3.2.1 解析新的上下文。服务端系统 PIT 设置始终不属于可操作字段。不能通过 UI 设置一个新日期后继续使用旧预览 token/证据。

### 11.4 实时反显与「看着 AI 操作」

- 回合响应返回一个带版本的 `ui_batch`；不返回缺少批次身份的裸动作数组。前端先做整批预检，再按序应用，间隔 150–200ms；高亮复用 accent 令牌，不移动用户焦点。reduce 模式立即应用并取消动态高亮，文本回执保持完整。
- 文本/公式可在临时输入缓冲中分 2–5 段反显；完整值落定时才一次性提交到真实编辑状态，期间抑制自动 infer、查询和保存。遇到人类输入立即让出控制，不覆盖其文本。无论 staged/immediate，成功后的字段值和触发次数必须一致。
- 每个动作展示目标、新值、说明与状态；工具条提供“撤销本批 AI 操作”。撤销前先检查页面仍为该批次结束时的 revision，且受影响值仍是该批次写入值；若人类已编辑则拒绝自动回滚并提示冲突，不用旧值覆盖新工作。可撤销状态按逆序恢复，形成新的页面 revision；只读查询无法撤销已发生的网络请求，撤销时还原查询参数并清除/标记旧结果失效，不能假称恢复了服务端数据。
- 每次应用、部分执行、拒绝或撤销后，前端通过 `/messages` 提交无 text 的 `ui_results`、最新 surface 与 page_context；服务端核对 batch/action/receipt 标识、合法状态迁移后记事件，同一回执幂等。纯回执不调用模型；明确需要重新规划时由下一条用户消息或已展示决策卡选择触发，继续遵守无进展检测、单操作超时和计算准入。
- 客户端回执只证明客户端上报了什么，不替代服务端数值校验或人类确认。消息发送失败时保留待发送回执并重试，不重放已应用动作；刷新/恢复会话时仅显示历史批次，未经重新校验不得自动再执行。状态用 `aria-live="polite"` 宣告。

### 11.5 与既有页面机制的映射（需求示例）

**示例一：产品研究筛选「1990 年之前成立、规模大于 100 万」**

- 页面既有机制：`ProductCondition`（`ProductResearch.tsx:62-66`，operator `gte/lte/gt/lt/eq`）、条件字段来自后端 `condition_fields`（:109，缺省回退 `list_date/found_date`，:250-252）、`ProductConditionBuilder`（:784）、`setConditions`（:488-497），且 `conditions/filters/searchKeyword` 与 URL search params 双向同步（:240-247、:327-344）。
- “1990 年之前”按严格边界 `found_date < 1990-01-01`；若用户意指包含 1990 年则先澄清，不能自动改成 `<= 1990-12-31`。动作序列：`add_condition(found_date, lt, 1990-01-01)` → `add_condition(<规模字段>, gt, <按字段单位换算的100万元>)` → 注册的产品查询 submit。
- **合法性以 `response.condition_fields` 为准**：若页面没有「规模」字段或操作符不支持，前端拒绝并回喂「当前页面无该筛选字段」；agent 不得臆造字段（失败关闭）。「规模」字段名与单位必须在实现时按真实 `condition_fields` 核定，不得写死。
- 额外收益：该页状态本来就同步 URL，AI 操作结果天然可分享、可复现。

**示例二：指标中心把一套 LaTeX 写进编辑器**

- 页面既有机制：可编辑 LaTeX 是表达式来源（`editable_latex`，`IndicatorStudio.tsx:499-523`），KaTeX 渲染（`MathNotation`，:839-841），保存走 `createCustomIndicator/updateCustomIndicator`（:2114-2119）。
- 动作序列：`set_field(indicatorStudio.expression, latex_source)` → `submit`（触发既有 infer/validate 流程）。
- 约束：填写的是服务端校验通过的 `editable_latex`/可执行 expression，不能直接回填 `display_latex`。编辑器已有未保存修改时必须先按 §5.6 人工确认；普通 UI 动作不绕过此保护。页面照常显示诊断，指标目录写入仍需人类走提交条（§3.7）。

**示例三：图编辑器（后续）**

- `IndicatorGraphEditor` 已有节点/资源模型（`addResource` :67 起）；M5 之后可按需扩展 `insert_node/connect/set_arguments` 动作，与 §4.3 修复循环共用同一 UI 动作协议。

### 11.6 后端与 Harness 增量

- 工具新增：`ui.inspect`（读取本轮 surface 快照，无副作用）与 `ui.act`（产出待应用动作，**不直接改前端**）。
- `AgentTurnResponse.ui_batch`、`AgentMessageRequest.surface/ui_results/decision` 与 `DecisionCard.kind="ui_action_rejected"` 统一使用 §3.2/§11.2 的契约；不维护另一份只有文字描述的字段版本。
- 人机边界：界面操作可以自动应用，但必须**可见、可撤销、可审计**；`submit` 允许自动执行；**任何落库动作不得由 `ui.act` 触发**，仍只由 `commit` 端点完成。
- 审计：动作、应用结果、撤销分别进入 §3.3 的 `ui_action/ui_action_result/ui_action_undo`，含 speaker、批次/动作 id、old → new、前后 revision 与失败原因。
- UI 层无需额外端点：surface 随会话创建/消息携带，批次随回合响应返回，应用/撤销回执经同一个 `/messages` 的纯回执分支写入；这与 §3.8 的提交预检端点是不同用途。

### 11.7 约束与验收

- [ ] 每页 `surface.test.tsx`：清单中每个 target 能绑定到真实控件、setter 生效、非法值与过期 revision 被拒。
- [ ] harness 测试：动作生成、拒绝回喂、撤销回滚、旧快照防过期。
- [ ] 两个条件依次成功且只触发一次真实查询；自身 revision 递增不误伤后续动作，人类插入编辑会停止剩余动作。
- [ ] 拒绝伪造 surface 权限、保存按钮伪装 submit、未知回执/重放批次；纯回执不消耗模型轮次，丢回执不会再执行动作。
- [ ] 人类后续修改后撤销失败关闭；公式分段反显期间无计算、无自动保存；reduce 模式最终状态与普通模式一致。
- [ ] E2E（fixture 模式）：AI 设置两个条件 → 页面条件 chip 与结果表真实刷新；AI 写入 LaTeX → 公式渲染与诊断出现。
- [ ] 全部动作在对话流可见、可撤销、可审计；非法动作被拒并给出原因。
- [ ] 动效、高亮、焦点环、`aria-live` 宣告符合 `docs/frontend/README.md`（无新动画库，reduce 下降级）。
- [ ] 契约测试：后端动作 schema 与前端 TS 镜像一致；target 命名唯一且稳定。

---

## 附录 A：作用域清单（`docs/agent/scopes.json` 快照示意）

```json
{
  "scopes_version": "sha256:...",
  "shared_ui_tools": {"enabled_from": "M5", "names": ["ui.inspect", "ui.act"]},
  "scopes": [
    {
      "id": "indicator-center",
      "label": "指标中心",
      "pages": ["indicator-studio"],
      "authoring_contexts": ["single_product"],
      "rules": "backend/agent/prompts/indicator_center.md",
      "memory": "data/agent_memory/indicator-center.md",
      "triggers": ["指标", "算子", "公式", "ddof", "窗口", "序列", "口径"],
      "tools": ["metrics.lookup", "metrics.validate", "metrics.infer", "metrics.availability",
                "metrics.preview", "metrics.excel", "metrics.draft_save"],
      "cross_module_read": ["products.search", "products.eval", "products.series", "products.plans"],
      "agent_writes": ["session:draft"],
      "human_actions": ["commit-preview:indicator", "commit:indicator"]
    },
    {
      "id": "product-research",
      "label": "产品研究",
      "pages": ["product-detail", "product-research", "product-compare", "holding-diagnosis", "evaluation-plan"],
      "rules": "backend/agent/prompts/product_research.md",
      "memory": "data/agent_memory/product-research.md",
      "triggers": ["产品", "基金", "池", "评价", "持仓", "比较", "这只"],
      "tools": ["products.search", "products.eval", "products.series", "products.plans", "portfolios.context", "portfolios.eval"],
      "tool_contexts": {
        "products.search": ["single_product"],
        "products.eval": ["single_product"],
        "products.series": ["single_product"],
        "products.plans": ["single_product"],
        "portfolios.context": ["portfolio"],
        "portfolios.eval": ["portfolio"],
        "metrics.lookup": ["single_product", "portfolio"],
        "metrics.infer": ["single_product"],
        "metrics.availability": ["single_product"]
      },
      "cross_module_read": ["metrics.lookup", "metrics.infer", "metrics.availability"],
      "agent_writes": [],
      "human_actions": []
    }
  ]
}
```

## 附录 B：会话与运行结构

存储结构和可恢复事件协议统一见 [Harness 设计](ai-agent-harness-progress-design-2026-09-20.md) 第 5、9、12 节。session、run、工具回执与事件分开表达；不再用单个 `<id>.json` 的示意结构指导新实现。旧文件导入保留实际已有信息，不能补造已被裁剪的历史。

## 附录 C：系统提示骨架（`prompts/system.md` 结构）

```
# 角色
你是投研工作台内的指标研究助手，服务两个模块：指标中心与产品研究。

# 原则（不可协商）
1. 只能用目录中声明的变量与算子；不能发明算子、不能生成自由代码。
2. 表达不了就明确说「现有算子无法表达」，并说明最接近的可行方案；禁止近似。
3. 公式真相是 DSL；你在对话里引用的公式必须与提交的 DSL 完全一致。
4. 你没有指标目录写工具；仅能保存本会话草稿，指标提交由人类复核确认快照后完成。
5. 使用服务端绑定的计算域与时点；单产品变量先查 availability，组合只读当前不可变 run，禁止使用未来数据。

# 可用工具
<tool schemas>

# 输出
按 ModelTurnDraft schema 提出文字、定义、候选与动作；服务端校验后生成 AgentTurnResponse。
不得伪造哈希、token、校验状态、预览数值、工具轨迹或人类确认。

# 版本
catalog_version={{catalog_version}} dsl_version={{dsl_version}}
```
