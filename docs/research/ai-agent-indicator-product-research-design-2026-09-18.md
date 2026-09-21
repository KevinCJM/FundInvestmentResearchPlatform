# 指标中心 × 产品研究：AI 智能体详细设计

> - 日期：2026-09-18（追加 §11 UI 动作层；补充 §2.3 硬性能力边界与 preview `compile_token` 契约）
> - 基线：HEAD `e4f85b1`（2026-09-17，含 `cb4d3df` 产品研究展示条数上限调整）
> - 状态：**设计稿，尚未实现。** 本文描述目标实现与验收方式，不代表任何代码已落地。
> - 范围：指标计算中心（IndicatorStudio / `backend/custom_indicators`）+ 产品研究模块（ProductDetail 等页面 / `backend/product_analysis`、`research_series` 等）+ AI Harness + 前端对话面板。
> - 关联约束：[AGENTS.md](../../AGENTS.md)（NJIT、算子治理、测试、AI Hermes 路由）、[docs/frontend/README.md](../frontend/README.md)（前端视觉与交互）、[docs/governance/branch-submission-rules.md](../governance/branch-submission-rules.md)（提交规则）。
> - 术语：本文中「指标」指自定义指标（custom indicator）；DSL 指 `backend/cal_indicators` 的类型化表达式；「落库」指写入 `data/custom_indicators.json` 形成不可变 revision。

---

## 0. 摘要与目标

### 0.1 目标

1. 用户用自然语言提出指标需求，智能体在指标中心允许的变量、算子与参数契约内产出可执行 DSL，供人类用 LaTeX、Excel 公式和真实数值三种材料复核；确认后落库为**不可变 revision**。
2. 产出的指标回写前端：指标库立即可见，产品研究页可在当前产品上计算与展示；修改指标产生新 revision，旧引用保持旧版本。
3. 一套 harness 支持两个作用域（`indicator-center`、`product-research`），单模块与跨模块功能共用同一个 agent 形态；跨模块优先降级为工具调用，写入始终是显式人工动作。
4. 会话可恢复、有分层记忆、支持多人对话标注与冲突显式上报。
5. 智能体可直接操作前端界面（填写 LaTeX、设置筛选条件等），页面实时反显 AI 动作，动作可见、可撤销、可审计（见 §11）。

### 0.2 非目标（v1 明确不做）

- 不自动发布指标；不在产品研究侧写数据（评价方案、快照配置、展示偏好都只读）。
- 不生成新算子、不生成自由 Python/Numba、不改动任何既有计算路径。
- 不做多用户账号、并发编辑锁、跨设备同步；多人仅以 `speaker` 标签与审计呈现。
- 不做向量库/RAG，不做 token 流式输出（先整轮返回），不做多智能体集群。
- 不替代指标中心现有编辑器；agent 是并行入口，不是唯一入口。

### 0.3 设计原则

1. **唯一编译器**：校验、类型推断、可用性检查、数值预览、落库全部直调现有 service 层函数；AI 侧没有平行实现，AI 生成物与手工编辑物走同一编译与 NJIT 路径。
2. **唯一事实**：DSL 是计算真相；LaTeX（`display_latex`）与 Excel 均由服务端从 DSL 派生；对 AI 的「说明书」由注册表生成，不手写事实。
3. **失败关闭**：现有算子无法表达时明确回答「不可表达」，禁止近似、禁止静默替换；低置信问人。
4. **人类确认**：LLM 不具备任何落库工具；只能产出草稿与建议，写目录只能经人类触发的 `commit` 端点。
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
- 预览不落库（两阶段契约）：`/prepare`、`/evaluate`、`/evaluate-series` 均接受 `inline_definition: IndicatorDraft`（`custom_indicator_routes.py:293,300,333`）。未保存公式必须先经 `/prepare` 显式编译并返回 `compile_token`（`service.py:2027,2063`，scope `current_process_warm_cache`）；`evaluate` 携带的 token 与当前公式或已预热计划不匹配即拒绝（`service.py:2797-2884`）。**agent 预览必须沿此契约，不得绕过。**
- 版本化存储：`backend/custom_indicators/repository.py:101 IndicatorRepository`（`create`:168 / `update(expected_revision)`:185 / `delete`:215；旧版本进 `history`；内置只读），落盘 `data/custom_indicators.json`（`AtomicJsonStore` + 文件锁 + 原子替换 + fsync）。
- 创建/更新：`service.py:2338 create_indicator` / `:2351 update_indicator`：normalize → 编译校验 → 应用编译契约 → **NJIT 预热**（time_series → `series_service.warm`；single_product → `_warm_single_product_definition`）→ 写库 → 清缓存。
- 草稿：`backend/custom_indicators/graph_service.py:107` 使用 `AtomicJsonStore` 持久化 `indicator_editor_states.json`。
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

### 1.3 缺口清单

- 后端无任何 LLM 客户端（`chat/completions`、`deepseek`、`openai` 均无命中；`httpx>=0.24` 已在 `backend/requirements.txt`）。
- 无 agent/session/harness、无意图路由、无记忆文件、无 AI 目录快照、无前端对话面板。
- 无账号体系（无 auth/login）；多人会话 v1 以 `speaker` 标签与审计实现。

---

## 2. 总体架构

### 2.1 一个 Harness + 两个作用域

```
┌─ 前端 ──────────────────────────────────────────────────────┐
│ AgentPanel（IndicatorStudio / 产品研究各页挂载，右侧抽屉）    │
│  会话流 · 工具轨迹 · 草稿卡（DSL + LaTeX + 真实数值预览）      │
│  候选卡 · 提交条 · 作用域 chip · 记忆提案卡 · 冲突决策卡       │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST /api/agent/*
┌───────────────────────▼─────────────────────────────────────┐
│ backend/agent                                                │
│  routes ─→ harness（回合循环 / 修复 / 压缩 / 预算）           │
│            ├─ scopes    作用域清单（生成 + 校验）             │
│            ├─ router    两轴三层意图判定                      │
│            ├─ catalog   说明书（注册表生成，版本化）           │
│            ├─ tools     薄包装，直调现有 service 函数          │
│            ├─ sessions  原子 JSON 会话存储                    │
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
| `backend/agent/harness.py` | 回合循环、修复、预算、压缩、上下文装配 |
| `backend/agent/sessions.py` | 会话存储（AtomicJsonStore 模式） |
| `backend/agent/memory.py` | 记忆文件读写与审计 |
| `backend/agent/ui.py` | UI 动作 schema、校验与 surface 快照契约（见 §11） |
| `backend/agent/commit.py` | 人类确认后的落库与影响面检查 |
| `backend/agent/routes.py` | `APIRouter`（注册进 `app.py`） |
| `backend/agent/prompts/system.md`、`indicator_center.md`、`product_research.md`、`compaction.md` | 规则层提示词（随代码版本管理） |

**新增（前端）**

| 文件 | 职责 |
| --- | --- |
| `frontend/src/services/agent.ts` | API 客户端与类型 |
| `frontend/src/components/agent/AgentPanel.tsx` | 面板外壳（抽屉、作用域 chip、会话菜单） |
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
| 指标定义（新建/新 revision） | `POST /commit`，人类在 UI 触发 | 注册表无 create/update/delete 工具；`CommitRequest.confirmed: Literal[True]`；`expected_revision` 乐观锁（`repository.py:185`） |
| 编辑器草稿 | `metrics.draft_save` | 只写 `indicator_editor_states.json`（`graph_service.py:107` 同模式）；键绑定当前页 indicator_id/revision；无路径参数 |
| 会话事件 | 系统写本会话文件 | 目录固定、session_id 服务端生成；事件类型白名单（3.3） |
| 记忆文件 | 人类 accept 提案后追加 | LLM 仅有 propose；目标固定为 `data/agent_memory/*.md`；`audit.jsonl` 只增不改（4.5） |

除此之外没有第 5 项。

#### 2.3.2 AI 不能改（硬拒）

- 源码、配置、`.env`、token/凭据、prompt、scopes 清单：无文件读写工具，无任何 `path` 参数。
- 数据快照 / parquet / 存储迁移状态 / PIT 系统设置 / 数据下载配置：不在 handler 白名单。
- 评价方案、快照配置、展示偏好、产品池、发布中的情景与风险模型：对应模块写函数不在白名单（0.2 写边界）。
- 指标历史 revision 与已发布版本：commit 只追加新 revision，无覆盖语义。
- 其他 session 文件、shell、任意网络：无工具。
- 前端：只能产出 `UiAction[]`（§11）；不能导航、不能坐标点击、不能经 `ui.act` 触发 commit。
- agent 自身：工具集、提示词、预算、scope 清单均为代码常量，无自改工具。

#### 2.3.3 AI 能看（读白名单）

- catalog：变量/算子/签名/参数/LaTeX/`interval_policy`（`meta()` + 注册表生成，3.5）。
- 当前页面上下文：product_ids/product_kind、indicator_id + revision、surface 快照（§11.3）。
- 计算只读：validate、infer、availability(as_of)、preview、Excel 导出。
- 影响面只读：引用某指标的方案 id 列表（供 commit 前提示）。
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
| 参数越界/多余字段 | 工具 JSON Schema `additionalProperties: false` + Pydantic | 校验失败回喂诊断（≤3 次） |
| handler 越权 | 注册表构建期校验 handler 在 `(module, qualname)` 白名单 | 校验失败则 agent 禁用，不影响主应用启动 |
| 落库路径 | `commit.py` 唯一写点；agent 模块禁止 import repository 写函数（静态检查） | 测试失败 |
| 预览编译 | 必须 `prepare_evaluation` → `compile_token` → `evaluate`（3.4） | 无/错 token 直接报错 |
| UI 动作 | 服务端校验 target 在最新 surface；前端校验 revision | 非法/过期拒绝并回喂 |
| 资源滥用 | 每会话预算、≤6 工具轮、≤3 修复、每轮 ≤1 预览、工具超时 | `AGENT_BUDGET_EXCEEDED` 429 |

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

工具 handler 直接 import 并调用现有 service 函数（同进程），不发起 HTTP 自调用；这保证与手工路径同一实现、同一编译与 NJIT 预热路径。

### 3.2 数据契约（Pydantic，`contracts.py`）

```python
class PageContext(Contract):
    module: Literal["indicator-center", "product-research"]
    page: str                      # "indicator-studio" | "product-detail" | ...
    product_kind: Literal["etf", "fund"] | None = None
    product_ids: list[str] = []    # 上限与服务端守卫一致
    indicator_id: str | None = None
    indicator_revision: int | None = None

class AgentSessionCreate(Contract):
    scope_hint: Literal["indicator-center", "product-research"] | None = None
    page_context: PageContext
    speaker: str                   # v1 为人类可读标签；未来替换为认证身份

class AgentMessageRequest(Contract):
    text: str = Field(min_length=1, max_length=4000)
    speaker: str

class DraftArtifact(Contract):
    definition: IndicatorDraft           # 直接复用现有 IndicatorDraft，不新造定义契约
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
    kind: Literal["scope_conflict", "param_conflict", "intent_ambiguous", "revision_conflict"]
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
    reply: str
    tool_trace: list[ToolTraceEntry] = []
    draft: DraftArtifact | None = None
    candidates: list[CandidateOption] = []
    decision_card: DecisionCard | None = None
    scope_change: ScopeChange | None = None
    memory_proposals: list[MemoryProposal] = []
    commit_suggestion: bool = False      # 是否已到"可提交"状态（仅提示，不自动提交）
    usage: dict[str, int] = {}

class CommitRequest(Contract):
    definition: IndicatorDraft
    expected_revision: int | None = None   # 更新既有指标时必填
    confirmed: Literal[True]               # 人类显式确认；默认不成立

class CommitImpact(Contract):
    evaluation_plans: list[str] = []       # 引用该指标的方案
    snapshot_config: bool = False
    historical_revisions_unchanged: bool = True

class CommitResponse(Contract):
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

### 3.3 会话存储（`sessions.py`）

目录与文件：

```
data/agent_sessions/
├── index.json            # 会话索引：id/title/scope/speaker/updated_at/status
├── <session_id>.json     # 单文件原子写：{meta, state, events[]}
data/agent_memory/
├── shared.md
├── indicator-center.md
├── product-research.md
└── audit.jsonl           # 记忆变更审计（时间/作者/理由/diff 摘要）
```

- 写入方式：`AtomicJsonStore`（与 `indicator_editor_states.json` 同一模式）逐事件重写；单会话在一次人工交互内的事件量有限，压缩（4.4）保证文件有界。
  `# ponytail: 单文件重写；若会话事件数超过数千条，升级为 JSONL 追加 + 周期快照。`
- 事件类型：`user / assistant / tool_call / tool_result / summary / scope_change / commit / memory_proposal / error`；每条含 `seq、ts、speaker、scope、payload`。
- 恢复：读取 `<id>.json` 即得 `state`（当前 scope、draft、待确认项、用量、summary 指针）；`events` 供压缩与审计。无隐藏状态。
- 索引：`index.json` 更新标题（首条用户消息截断）与 `updated_at`。

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
| `metrics.preview` | read | `prepare_evaluation` → `evaluate` / `evaluate-series`（`inline_definition`） | 先取 `compile_token` 再计算；当前页产品上下文；不落库；每轮最多一次 |
| `metrics.excel` | read | `export-excel` | 供人类逐单元格复核 |
| `metrics.draft_save` | write_local | `editor-state` PUT | 仅保存编辑草稿；不产生指标 revision |

**产品研究（`product_research.py`，v1 全部只读）**

| 工具 | 类型 | 直调 | 说明 |
| --- | --- | --- | --- |
| `products.search` | read | 产品/池检索服务 | 解析产品名称/代码为 id；受页面上限守卫约束 |
| `products.eval` | read | `evaluate`（`indicator_refs` 或 `inline_definition`） | 在页面上限内计算标量指标 |
| `products.series` | read | `evaluate-series` | 序列指标（曲线/表格） |
| `products.plans` | read | 评价方案/快照配置读取 | 落库前影响面检查所需 |

**明确不存在的工具**：`create / update / delete indicator`。落库只能经 `POST /sessions/{id}/commit`（人类触发），这是原则 0.3-4 的机械保证。

**派发硬校验**：`execute()` 只按注册表名派发，未注册即 `AGENT_TOOL_NOT_ALLOWED` 并记事件；参数 schema 一律 `additionalProperties: false`；handler 必须命中 `(module, qualname)` 白名单，启动期校验失败只禁用 agent、不影响主应用。见 §2.3.5。

**UI 动作工具**：`ui.inspect` 与 `ui.act` 不在上表，机制见 §11；`ui.act` 只能操作界面状态，不能触发指标落库。

工具结果一律截断：预览带 5 个关键点 + 完整结果写会话文件、供 UI 展开；长表格不以原文进入 prompt。

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
- 注入策略：默认把「变量摘要 + 算子分类索引 + 高频算子全文」放进系统提示；完整目录由 `metrics.lookup` 按需检索（目录增大后的扩展方式；M4 起启用分类加载）。

### 3.6 意图路由（`router.py`）

判定轴：**scope（哪个模块）× action（read / author / workflow）**。

- 输入：`page_context`、会话当前 `scope`、消息文本、最近工具轨迹。
- L0 规则（零成本，覆盖大多数）：
  - 页面默认 scope：`indicator-studio → indicator-center`；产品研究各页 → `product-research`。
  - triggers 表（per scope，示例）：`indicator-center`：指标、算子、公式、ddof、窗口、序列、口径；`product-research`：这只基金、产品、池、评价、持仓、比较。
  - 显式上下文：页面携带 `product_ids` / `indicator_id` 时优先视为该上下文的请求。
- L1 模型（L0 未命中或信号冲突时）：prompt 只含作用域清单（id + 一句话 + triggers），输出 `{scope, action, confidence, reason}`，**枚举约束**、不可发明新作用域。
- 决策表：

| 判定 | 行为 |
| --- | --- |
| `read`（含跨模块只读） | 放行；跨模块只读工具直接可用，不切换作用域 |
| `author`（创作/修订） | 切换到目标 scope 的 authoring 工具集；响应携带 `scope_change`，前端显示 chip（可点按回退） |
| `workflow`（跨模块写编排） | 明确告知 v1 未开放，并给出分步建议 |
| 低置信 / 同会话指令冲突 | 返回 `decision_card`，默认选项保留当前 scope，禁止静默切换或静默取最后一个说话人 |

- 每次判定作为事件写入会话（审计与调参）。

### 3.7 提交与回写（`commit.py`）

- 入口：`POST /api/agent/sessions/{id}/commit`，请求见 3.2 `CommitRequest`。
- 服务端步骤：
  1. `expected_revision is None` → `indicator_service.create_indicator(definition)`；否则 `update_indicator(id, expected_revision, definition)`。这是 agent 域内唯一写函数入口；agent 模块禁止直接 import `repository` 写方法（静态 import 检查兜底）。
  2. 影响面：调 `plans.references_indicator` 与 `snapshot_config.references_indicator`，组装 `CommitImpact`（历史引用保持旧 revision 是既有语义，不在这里改变）。
  3. 失败处理：`REVISION_CONFLICT` → 409 + 当前 revision 与摘要，前端出冲突决策卡；校验失败 → 透传既有错误码，草稿保留在会话。
  4. 成功：记录 `commit` 事件（indicators id/revision、`catalog_version`、DSL 哈希、speaker），返回 `CommitResponse`。
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
| POST | `/api/agent/sessions/{id}/messages` | `AgentMessageRequest` | `AgentTurnResponse` | 整轮返回；MVP 不做流式 |
| POST | `/api/agent/sessions/{id}/commit` | `CommitRequest` | `CommitResponse` | 唯一落库路径 |
| POST | `/api/agent/sessions/{id}/memory` | `MemoryResolution` | 更新后的提案状态 | 接受即追加记忆文件 + 审计 |
| POST | `/api/agent/sessions/{id}/scope` | `{scope}` | 会话摘要 | 前端 chip 手动回退/切换 |

- 路由类沿用 `StableValidationRoute`（与 `custom_indicator_routes.py` 一致）；注册沿用 `app.include_router`（`app.py:325` 起）。
- 启动集成：`app.py` 启动阶段只读取配置并暴露状态；**不因 LLM 未配置而阻止启动**。

### 3.9 错误语义

| 错误码 | HTTP | 语义 | 前端行为 |
| --- | --- | --- | --- |
| `AGENT_NOT_CONFIGURED` | 503 | 未配置模型 | 面板显示未启用与配置说明 |
| `AGENT_LLM_UNAVAILABLE` | 502 | 模型调用失败（已重试） | 就近错误态 + 重试按钮；草稿不丢 |
| `AGENT_BUDGET_EXCEEDED` | 429 | 会话预算耗尽 | 提示开启新会话或压缩后继续 |
| `AGENT_INTENT_AMBIGUOUS` | 200 | 非异常：`decision_card=intent_ambiguous` | 选项卡 |
| `AGENT_NOT_EXPRESSIBLE` | 200 | 非异常：现有算子无法表达 | 展示原因与最接近的可行表达（若有） |
| `REVISION_CONFLICT`（复用既有） | 409 | 指标已被他人更新 | 冲突决策卡（当前值 / 改动 / rebase 建议） |
| `VALIDATION_ERROR`（复用既有码） | 400 | 落库校验失败 | 错误定位到字段/节点 |

原则：任何 LLM/网络失败不产生半成品目录写入；失败只污染会话内草稿。

### 3.10 配置与安全

- 环境变量：`AGENT_LLM_BASE_URL`、`AGENT_LLM_API_KEY`、`AGENT_LLM_MODEL`、`AGENT_LLM_TIMEOUT_SECONDS`、`AGENT_LLM_MAX_RETRIES`、`AGENT_LLM_MODE=live|fixture`、`AGENT_LLM_FIXTURE_DIR`。
- 密钥只从环境变量读取，不入仓库、不进前端、不回显、不写日志（沿用仓库对敏感 token 的既有要求）。
- 工具白名单：无网络工具、无任意文件读写；`draft_save` 仅写 editor-state 白名单路径；`commit` 仅写指标库。白名单由启动期校验与派发校验强制执行（§2.3.5），提示词不承担安全职责。
- 网络出口唯一：agent 模块仅 `llm.py` 允许使用 `httpx`；由静态 import 检查与 fixture 测试证明（§7.1）。
- 测试与 E2E 一律 `AGENT_LLM_MODE=fixture`，不调用网络（仓库测试规范：测试禁止网络调用）。

---

## 4. AI Harness 设计

### 4.1 组件与职责

| 组件 | 职责 | 关键约束 |
| --- | --- | --- |
| 上下文装配 | 按固定顺序组装本轮输入 | 草稿永远 pin；目录摘要化 |
| 回合循环 | LLM ↔ 工具直至产出最终响应 | 最大工具轮数 6 |
| 修复循环 | 校验/推断失败时回喂诊断重试 | 最大 3 次；仍失败则降级为「仅解释，无草稿」 |
| 压缩 | 超预算时摘要早期事件 | 摘要必须保留「谁在何时要求改了什么」 |
| 转录 | 全事件落盘、可恢复 | 无隐藏状态 |
| 记忆 | scope/共享记忆文件 + 人工确认 | LLM 只能提案 |
| 预算 | 每会话 token/轮数记账 | 超限返回 `AGENT_BUDGET_EXCEEDED` |

### 4.2 上下文装配（固定顺序）

```
1. system 规则层（prompts/system.md，静态）
2. scope 规则切片（prompts/<scope>.md）
3. catalog 摘要（变量摘要 + 算子分类索引 + 高频算子全文；含 catalog_version）
4. 记忆文件：shared.md + <scope>.md
5. 会话摘要（如有压缩）
6. 最近 K 条消息（默认 12）
7. Pinned artifacts：
   - 当前指标草稿（definition + display_latex + required_variables）
   - 最近一次数值预览摘要（3-5 个关键点）
   - 待确认问题/决策卡
8. 本轮工具轨迹（循环内累积；结果已截断）
```

- 完整数值表、完整目录、Excel 导出内容**不进 prompt**；需要时由工具重新取。
- 每轮装配前重新读取记忆文件（人类可能在会话进行中编辑）。

### 4.3 回合循环与修复（`harness.py`）

```
def run_turn(session, message):
    append(user_event)
    scope, action, decision = router.decide(session, message)
    if decision:  # 意图模糊 → 直接返回决策卡，不启动工具循环
        return respond(decision_card)
    tools = tools_for(scope, cross_module_read=True)
    ctx, usage = assemble(session, message, tools)
    final = None
    for round in 1..MAX_TOOL_ROUNDS:                 # 6
        completion = llm.chat(ctx, tools=tool_schemas(tools),
                              response_schema=AgentTurnResponse)
        if completion.tool_calls:
            for call in completion.tool_calls:
                result = execute(tools[call.name], call.arguments)   # 直调 service
                ctx += trace(call, result)                # 截断后追加
            continue
        final = completion.parsed
        break
    if final is None:
        return error(AGENT_BUDGET_EXCEEDED)
    if final.draft:
        final.draft = server_validate_and_decorate(final.draft)   # validate + infer
        for repair in 1..MAX_REPAIRS:                 # 3
            if final.draft.valid: break
            ctx += repair_prompt(final.draft.diagnostics)
            final = llm.chat(...)                     # 仅重出草稿
            final.draft = server_validate_and_decorate(final.draft)
    append(assistant_event(final))
    return final
```

- `server_validate_and_decorate` 只做两件事：调校验/推断、把 `display_latex` 与 `required_variables` 填进草稿。**不修补、不猜测、不自动改口径。**
- 数值预览（`metrics.preview`）由 LLM 决定是否调用，每轮最多一次，且必须走 §3.4 的 `prepare → compile_token → evaluate` 契约；服务端在「人类要求提交」之前不强制预览，但提交条会显示「尚无真实数值预览」提示。`INDICATOR_ENGINE_BUSY` 按既有 503 + `Retry-After` 透传，不静默重试。
- 工具执行异常按错误码记入轨迹并回喂；连续相同错误码 2 次即停止该工具重试。

### 4.4 压缩（`compaction.md` 驱动）

- 触发：估算上下文 ≥ 预算的 0.7（预算按模型窗口配置）。
- 保留：system、scope 规则、目录摘要、记忆文件、pinned artifacts、最近 6 条消息。
- 摘要模板（强制字段）：
  1. 会话目标与当前状态（草稿在第几版、是否可提交）；
  2. 已确认口径与参数（含由谁确认）；
  3. 被否定的方案及原因（防止重提）；
  4. 未决问题与决策卡；
  5. 产物版本（indicators DSL 哈希、catalog_version）。
- 摘要以 `summary` 事件写回，覆盖范围 `covers:[seq...]`；后续装配使用摘要 + 覆盖范围之后的事件。
- 摘要生成失败：截断最老事件并把会话标记 `degraded`，在 UI 提示；不阻断继续对话。

### 4.5 记忆（`memory.py`）

三层：

| 层 | 载体 | 内容 | 写入者 |
| --- | --- | --- | --- |
| 会话记忆 | `<id>.json` 的 state | 草稿、待确认、用量 | 系统 |
| scope 记忆 | `data/agent_memory/<scope>.md` | 口径惯例、命名偏好、默认参数、禁用项 | 人类确认后系统追加 |
| 共享记忆 | `data/agent_memory/shared.md` | 跨模块约定（如「收益率一律复权净值口径」） | 同上 |

- 流程：LLM 调 `memory.propose`（产出 `MemoryProposal`）→ 前端展示 diff → `POST /memory {accept|reject, speaker}` → 接受则追加文件并写 `audit.jsonl`（时间、作者、理由、append 摘要）。
- 记忆文件为 Markdown，人类可直接编辑；系统只追加（删除也走提案）。
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

- `system.md`：角色与原则（0.3 四条）、能力边界、失败语义（不可表达 / 低置信问人）、工具纪律（先查目录再写表达式）、输出 schema 约束、禁止事项（新算子、自由 Python、未来数据、自动落库）。
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
- 形态：右侧抽屉（非模态），桌面宽 420px，`<768px` 全宽；页面可同时滚动核对数值。抽屉是浮层：`shadow-xl`、z-index 走系统层级常量、自身是唯一滚动容器（不产生页面嵌套滚动）。

### 5.2 组件树与职责

```
AgentLauncher（页内按钮，aria-expanded）
└── AgentPanel（抽屉外壳：标题、ScopeChip、会话菜单、关闭）
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
- 键盘：`Escape` 关闭抽屉并把焦点交还 launcher；焦点样式 `focus-visible:ring-2 focus-visible:ring-accent-500`。

### 5.4 数据流

- `frontend/src/services/agent.ts`：封装 3.8 全部端点；类型从后端契约手写镜像（与 `customIndicators.ts` 现有做法一致）。
- `useAgentSession` hook：`send()`（POST 消息）、`commit()`、`resolveMemory()`、`switchScope()`、`resume(sessionId)`；维护事件列表与当前 `state`。
- 版本一致：提交时携带草稿的 `definition` 与（更新时）`expected_revision`；服务端二次校验后落库。

### 5.5 提交后的刷新协议（回写前端）

`CommitResponse.refresh` 由页面注册的处理器消费：

| 页面 | 处理 |
| --- | --- |
| IndicatorStudio | 刷新目录；用返回 `id/revision` 打开对应指标的编辑器（组件回调，不新增 URL 参数） |
| ProductDetail / ProductResearch / ProductCompare / HoldingDiagnosis | 重新拉取 `listCustomIndicators + getCustomIndicatorMeta`；通过 `withSelectedIndicators` 把新指标加入当前页指标选择并触发计算 |
| EvaluationPlan | 刷新可用指标目录；不自动改写方案（写方案属产品研究侧，超出 v1 边界） |

- 展示偏好（localStorage）由前端在提交成功后写入；agent 不触碰。
- 若指标处于「更新」路径：页面刷新后展示新 revision；历史引用（带 `indicator_revision` 的旧请求/方案）不被改写。

### 5.6 指标中心编辑器联动

- `IndicatorStudio` 将 `AgentPanel` 挂载在编辑器工具条旁；提交成功后回调 `onIndicatorCommitted(id, revision)`，走页面既有的选择与编辑状态，不复制一套加载逻辑。
- 草稿与编辑器草稿互不覆盖：agent 只有在用户点「发送到编辑器」或提交后才写 `editor-state`；编辑器已有未保存修改时先提示覆盖确认。

### 5.7 国际化与文案

- 全部界面文案走 `locales/`（中英齐备，`node scripts/check_i18n.mjs` 通过）；JSX 不落中文硬编码。
- 关键文案示例：`agent.panel.title`、`agent.composer.placeholder`、`agent.commit.confirm`、`agent.scope.indicatorCenter`、`agent.scope.productResearch`、`agent.error.notConfigured`、`agent.decision.title`。

### 5.8 可达性与性能

- 消息列表 `role="list"`；新消息区域 `aria-live="polite"`；工具轨迹用原生 `<details>/<summary>`；表格 `<th scope>`、`<caption>` 或 `aria-label`。
- 抽屉打开不锁定页面滚动（非模态）；焦点不被困住，但 `Escape` 可关闭。
- 不引入新依赖（KaTeX 已有）；不新增图表，数值预览复用表格与 `MetricDisplay` 的公式渲染路径。
- 响应式：320px 无横向溢出；窄屏抽屉全宽、消息与卡片单列。

### 5.9 前端测试

- Vitest + RTL：`AgentPanel.test.tsx`（四态、发送、轮次渲染、决策卡选择）、`AgentCommitBar.test.tsx`（名称/描述校验、更新提示修订）、`AgentDraftCard.test.tsx`（公式/参数/预览渲染）、`agent.test.ts`（客户端请求体与错误映射）。
- 断言提交回调：`onIndicatorCommitted` 被调用且页面刷新函数被触发（mock 服务层）。
- E2E：`frontend/e2e/agent-panel.spec.ts`，以 `AGENT_LLM_MODE=fixture` 的后端运行，覆盖：产品详情页发起 → 草稿出现 → 提交 → 指标出现在选择器 → 公式与数值可见。

---

## 6. 端到端流程

### 6.1 场景 A：产品详情页发起创作（主链路）

1. 用户在 ProductDetail 打开「AI 助手」，输入「做一个衡量基金下跌后反弹力度的指标」。
2. 路由：页面默认 scope=`product-research`；triggers 命中「指标/反弹力度」→ `author`，目标 `indicator-center`；响应携带 `scope_change`，chip 显示「指标中心」。
3. 循环：`metrics.lookup`（找波动/回撤类算子）→ 产出 DSL → `metrics.validate`/`metrics.infer`（含 `display_latex`）→ `metrics.preview`（`inline_definition` 在 `product_ids` 上算真实数值）。
4. 草稿卡呈报：DSL、公式、参数、最近 1 年数值关键点；人类质疑「窗口应该是 60 日」→ 参数调整 → 重算。
5. 人类点「提交」：填名称描述、查看影响面（无方案引用）→ `commit` → 返回 `{id, revision:1}`。
6. 页面刷新：目录出现新指标并被加入当前指标选择，图表/表格立即出数；`MetricDefinitionDrawer` 可查看定义。

### 6.2 场景 B：指标中心里的跨模块预览

1. 用户在 IndicatorStudio 编辑草稿，问「这只 ETF（页面已带 ids）最近一年表现如何」。
2. 路由：`read` 跨模块；直接用 `products.series`/`products.eval`，不切换 scope。
3. 结果以数值预览回到对话；继续编辑不受影响。

### 6.3 场景 C：修订既有指标（revision 冲突）

1. 用户在产品研究页说「把动量指标的年化改成 252 天」。
2. 路由 `author`；服务端读取当前 revision，产出草稿；提交时携带 `expected_revision`。
3. 若期间他人已更新 → 409 → 冲突决策卡：查看差异、以最新版本 rebase、或放弃。
4. 成功后旧展示在下次刷新时跟随当前 revision；带旧 revision 的历史请求保持旧结果。

### 6.4 场景 D：记忆提案

1. 会话末尾 LLM 提议：「记住：本工作区收益率口径为复权净值」。
2. 前端展示提案卡；用户接受 → 写入 `data/agent_memory/shared.md` 并记审计；拒绝则不落任何内容。

---

## 7. 测试与验收

### 7.1 后端测试（`backend/tests/`，pytest，无网络）

| 文件 | 覆盖 |
| --- | --- |
| `test_agent_catalog.py` | 说明书生成与 `docs/agent/indicator_catalog.json` 快照一致；版本字段齐备 |
| `test_agent_scopes.py` | 作用域清单引用的工具名存在；scope id 与 `processRegistry`/模块清单一致；无写工具暴露给 LLM |
| `test_agent_harness.py` | FixtureClient 驱动：工具循环、修复循环（≤3）、轮数上限、预算超限、压缩触发与摘要必填字段、恢复重建 |
| `test_agent_commit.py` | create/update 往返（真实 repository + 临时目录）、`REVISION_CONFLICT`、影响面、无半成品写入 |
| `test_agent_router.py` | L0/L1 判定、低置信决策卡、冲突指令不自动择一 |
| `test_agent_memory.py` | 提案接受/拒绝、审计追加、人类可编辑文件不被静默覆盖 |
| `test_agent_api.py` | 3.8 端点状态码与响应结构；未配置时 503 仅限 agent 路由 |
| `test_agent_boundaries.py` | 工具集快照（写工具仅 `draft_save`）；未注册工具拒绝；schema 无 path/file/url 参数；`backend/agent/*` import 白名单（无 subprocess、无 repository 写函数，`httpx` 仅 `llm.py`）；`confirmed=true` 门禁；`draft_save` 路径与键校验；fixture 下 httpx 传输层未被触发；`availability` 不可得时失败关闭 |

### 7.2 NJIT 与数据真实性

- 预览、提交均走现有 service：`inline_definition` 预览使用真实执行引擎；`create_indicator` 触发既有 NJIT 预热（`series_service.warm` / `_warm_single_product_definition`）。
- 测试须断言：agent 提交的指标在预热完成后可直接计算，无请求期临时编译、无 Python 回退（沿用既有预热测试思路）。
- PIT：`metrics.availability` 覆盖；测试覆盖「不可得变量必须显式不可用」。

### 7.3 前端验证（按设计准则第 12/14 节）

```sh
npm run design:check --prefix frontend
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
node scripts/check_i18n.mjs
python skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```

浏览器验收覆盖：真正打开/关闭抽屉、生成中状态、错误重试、未配置态、决策卡选择、提交后页面刷新选中；320 / 768 / 1440 视口；浅底次要文字与 KaTeX 公式对比度。

### 7.4 E2E

`frontend/e2e/agent-panel.spec.ts`（fixture 模式）：产品详情页完整闭环 + 指标中心打开新 revision。测试夹具不写正式数据目录（沿用隔离夹具约定）。

### 7.5 验收清单（需求级）

- [ ] 自然语言 → 合法 DSL；不可表达时明确回答而不是近似。
- [ ] 人类可查：DSL、KaTeX 公式、Excel 导出、真实数值预览（带产品、区间、as-of）。
- [ ] 落库为不可变 revision；更新产生新 revision；冲突不覆盖。
- [ ] 提交后前端目录立即刷新，产品研究页可选中并计算展示。
- [ ] 历史 `indicator_revision` 引用不被改写（P12/P14 语义）。
- [ ] 跨模块只读不切换作用域；写入只经 commit；作用域切换可见可回退。
- [ ] 会话可恢复；压缩后仍保留归属与否定原因；记忆变更有人工确认与审计。
- [ ] AI 能操作前端界面（写入 LaTeX、设置筛选条件并驱动真实查询），页面实时反显，动作可见、可撤销、可审计（见 §11.7）。
- [ ] 动作合法性以页面上报的 UI Surface 为准；非法或过期动作被前端拒绝并回喂 agent。
- [ ] 硬限制由测试证明（§7.1 `test_agent_boundaries.py`）：未注册工具不可达、写路径只有 commit、无 path/网络工具、预览必经 `compile_token`。
- [ ] 无 LLM 时系统其余功能正常。

---

## 8. 里程碑

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| **M1 指标中心单模块闭环** | harness v0（无压缩/无记忆/单 scope）、指标中心工具、API、Panel、commit(create)、编辑器联动；catalog 快照与契约测试 | 测试 7.1 前四项 + 7.3；浏览器完成「对话 → 草稿 → 提交 → 编辑器打开新 revision」 |
| **M2 产品研究接入** | 产品研究只读工具、五个页面挂载、跨模块创作闭环、提交后刷新选中、影响面检查 | 场景 A/B 全通；E2E 主链路通过 |
| **M3 会话、路由与记忆** | 会话恢复、router L0/L1、scope chip、记忆文件与提案、多人 speaker/决策卡、update 路径与冲突处理 | 场景 C/D 全通；`test_agent_harness/router/memory` 通过 |
| **M4 压缩与目录扩展** | 压缩与降级、catalog 分类按需加载、会话列表 UI、性能与预算仪表 | 长会话压缩后继续可用；压缩摘要字段完整；目录快照契约通过 |
| **M5 UI 动作层（§11 基础动作）** | `ui.inspect` / `ui.act`、页面 surface 注册、`set_field / toggle_option / add_condition / remove_condition / submit / highlight`、撤销与审计 | §11.7 清单前四项 + surface 契约测试；图编辑器动作不在本里程碑 |

§11 基础 UI 动作独立到 M5，避免拖累 M1–M3 主链路；图编辑器动作（`insert_node / connect / set_arguments`）在此之后按需排期。

每个里程碑完成后按 `docs/governance/branch-submission-rules.md` 走分支、测试证据与合并流程；实现阶段同步更新 AI Hermes 路由记忆并运行校验脚本。

---

## 9. 风险与边界

| 风险 | 缓解 |
| --- | --- |
| LLM 产出看似合理但口径错误 | 服务端校验为唯一裁判；KaTeX/Excel/真实数值三重人工材料；不可表达即拒绝 |
| 目录/提示词漂移 | 说明书由注册表生成 + 快照契约测试；提示词不写死算子清单 |
| 修订冲突覆盖他人工作 | `expected_revision` 乐观锁 + 冲突决策卡；无强制覆盖工具 |
| 上下文膨胀与成本 | 工具结果截断、预览关键点、压缩、目录按需加载、会话预算 |
| 多人场景的隐性分歧 | 归属标注 + 冲突决策卡 + 审计；v1 明确不做并发编辑 |
| 供应商/密钥风险 | key 仅环境变量；`llm.py` 单一边界；fixture 模式覆盖全部测试 |
| 「AI 生成物」被误读为已验收模型 | 全部产物标注「AI 生成，待人工确认」；提交动作在 UI 上明确显示目标模块与影响面 |
| UI 动作误操作或与人类抢状态 | 动作可见、可撤销、带 surface revision 防过期；落库不经 UI 动作；非法动作由前端拒绝 |
| 性能：KaTeX/大消息流 | 公式渲染限草稿卡；消息虚拟化在 M4 视实测再引入 |
| 未保存公式预览触发 NJIT 编译占 CPU | `prepare → compile_token` 单次授权；每轮 ≤1 次预览；透传 `INDICATOR_ENGINE_BUSY` 503 + Retry-After |
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
- **正确形态**：前端页面注册一份「可被 AI 操作的面」（UI Surface，§11.3）；智能体产出**类型化 UI 动作**（`UiAction[]`，§11.2），随会话响应返回；前端把动作应用到**真实 React 状态**（复用页面现有 setter），界面自然实时反显；提交与校验仍走页面既有逻辑。
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
    page: str                      # "product-research.products" | "indicator-studio.editor" | ...
    revision: int                  # 前端每次状态变更自增，用于防过期应用
    fields: list[UiField] = []

class UiActionSpec(Contract):
    action: str
    target: str
    value: Any = None
    note: str | None = None        # LLM 对本次操作的一句话解释，显示在动作 chip 上
```

| 动作 | 语义 | 适用 |
| --- | --- | --- |
| `set_field` | 写入单个字段 | 文本/数字/日期/下拉/复选框（LaTeX、参数、窗口、关键词） |
| `toggle_option` | 多选增删 | 分类筛选（管理人等） |
| `add_condition` | 追加条件 | 条件构建器（`{field, operator, value}`） |
| `remove_condition` / `clear_conditions` | 删除条件 | 同上 |
| `set_keyword` | 设置搜索词 | 产品研究搜索框 |
| `submit` | 触发页面既有查询/推断 | 「查询」「校验公式」（不落库） |
| `highlight` | 仅视觉提示，不改状态 | 指向目标控件 |
| `insert_node` / `connect` / `set_arguments` | 图编辑器操作 | 指标中心图形编辑器（M4 起，按需） |

- 前后端共享同一份动作 schema（后端 Pydantic 为源，前端 TS 镜像）；契约测试防漂移。
- 动作由服务端做形状校验（target 是否在当前 surface、类型/枚举/范围是否合法），**合法性以页面上报的 surface 为准**，服务端不得发明 target。

### 11.3 UI Surface 注册（前端）

- 每个挂载页使用 `useAgentSurface(pageId, { fields, actions })` 声明：target、类型、当前值、约束、setter 绑定。
- 会话创建与**每条消息**都携带 `UiSurfaceSnapshot`：人类可能手动改过页面，agent 每轮基于最新快照决策（observe → act → verify 闭环）。
- target 命名规则：`<page>.<区域>.<字段>`，全站唯一且稳定，可在测试中断言；禁止用 CSS 选择器或坐标。
- 快照的 `revision` 用于防过期：前端应用动作前核对 `revision`；不一致则拒绝并把新快照回喂，agent 重新规划（不静默套用旧动作）。

### 11.4 实时反显与「看着 AI 操作」

- 回合响应新增 `ui_actions: list[UiActionSpec]` 与 `ui_apply: "staged" | "immediate"`。
- 默认 **staged apply**：动作按序应用，间隔 150–200ms，目标控件加 `ring-2 ring-accent-500` 短时高亮；`prefers-reduced-motion: reduce` 下立即应用且不高亮，符合设计准则对动效与焦点环的限制（不新增动画库）。
- 文本/公式字段（LaTeX）**分段写入**：把值切成 2–5 段依次追加，最后落完整值；不在中途触发校验，落定后由页面既有 `infer/validate` 显示诊断与 KaTeX 渲染。
- 对话流中每个动作是一条 chip：`目标标签 + 新值 + 一句话解释 + 撤销`；工具条提供「撤销全部 AI 操作」（按逆序回滚 old value）。
- 应用结果（成功 / 被拒 + 原因）作为事件回喂 agent；界面状态变化通过 `aria-live="polite"` 宣告（如「已设置：成立日期 ≤ 1990-12-31」）。

### 11.5 与既有页面机制的映射（需求示例）

**示例一：产品研究筛选「1990 年之前成立、规模大于 100 万」**

- 页面既有机制：`ProductCondition`（`ProductResearch.tsx:62-66`，operator `gte/lte/gt/lt/eq`）、条件字段来自后端 `condition_fields`（:109，缺省回退 `list_date/found_date`，:250-252）、`ProductConditionBuilder`（:784）、`setConditions`（:488-497），且 `conditions/filters/searchKeyword` 与 URL search params 双向同步（:240-247、:327-344）。
- 动作序列：`add_condition(found_date, lte, 1990-12-31)` → `add_condition(<规模字段>, gt, 1000000)` → `submit`。
- **合法性以 `response.condition_fields` 为准**：若页面没有「规模」字段或操作符不支持，前端拒绝并回喂「当前页面无该筛选字段」；agent 不得臆造字段（失败关闭）。「规模」字段名与单位必须在实现时按真实 `condition_fields` 核定，不得写死。
- 额外收益：该页状态本来就同步 URL，AI 操作结果天然可分享、可复现。

**示例二：指标中心把一套 LaTeX 写进编辑器**

- 页面既有机制：可编辑 LaTeX 是表达式来源（`editable_latex`，`IndicatorStudio.tsx:499-523`），KaTeX 渲染（`MathNotation`，:839-841），保存走 `createCustomIndicator/updateCustomIndicator`（:2114-2119）。
- 动作序列：`set_field(indicatorStudio.expression, latex_source)` → `submit`（触发既有 infer/validate 流程）。
- 约束：LaTeX 由后端 DSL/`typed_latex` 派生或经 `metrics.validate` 校验；页面照常显示诊断，AI 不获得任何格式豁免；落库仍需人类走提交条（§3.7）。

**示例三：图编辑器（后续）**

- `IndicatorGraphEditor` 已有节点/资源模型（`addResource` :67 起）；M5 之后可按需扩展 `insert_node/connect/set_arguments` 动作，与 §4.3 修复循环共用同一 UI 动作协议。

### 11.6 后端与 Harness 增量

- 工具新增：`ui.inspect`（读取本轮 surface 快照，无副作用）与 `ui.act`（产出待应用动作，**不直接改前端**）。
- `AgentTurnResponse` 增加 `ui_actions` 与 `ui_apply`；`DecisionCard` 增加 `kind="ui_action_rejected"`（非法/过期动作的回喂与再决策）。
- 人机边界：界面操作可以自动应用，但必须**可见、可撤销、可审计**；`submit` 允许自动执行；**任何落库动作不得由 `ui.act` 触发**，仍只由 `commit` 端点完成。
- 审计：动作、应用结果、撤销均写入会话事件（含 `speaker`、`target`、`old → new`、`surface.revision`）。
- API 无新增端点：surface 快照随 `POST /sessions` 与 `POST /sessions/{id}/messages` 携带，动作随回合响应返回。

### 11.7 约束与验收

- [ ] 每页 `surface.test.tsx`：清单中每个 target 能绑定到真实控件、setter 生效、非法值与过期 revision 被拒。
- [ ] harness 测试：动作生成、拒绝回喂、撤销回滚、旧快照防过期。
- [ ] E2E（fixture 模式）：AI 设置两个条件 → 页面条件 chip 与结果表真实刷新；AI 写入 LaTeX → 公式渲染与诊断出现。
- [ ] 全部动作在对话流可见、可撤销、可审计；非法动作被拒并给出原因。
- [ ] 动效、高亮、焦点环、`aria-live` 宣告符合 `docs/frontend/README.md`（无新动画库，reduce 下降级）。
- [ ] 契约测试：后端动作 schema 与前端 TS 镜像一致；target 命名唯一且稳定。

---

## 附录 A：作用域清单（`docs/agent/scopes.json` 快照示意）

```json
{
  "scopes_version": "sha256:...",
  "scopes": [
    {
      "id": "indicator-center",
      "label": "指标中心",
      "rules": "backend/agent/prompts/indicator_center.md",
      "memory": "data/agent_memory/indicator-center.md",
      "triggers": ["指标", "算子", "公式", "ddof", "窗口", "序列", "口径"],
      "tools": ["metrics.lookup", "metrics.validate", "metrics.infer", "metrics.availability",
                "metrics.preview", "metrics.excel", "metrics.draft_save"],
      "cross_module_read": ["products.search", "products.eval", "products.series"],
      "writes": ["commit:indicator"]
    },
    {
      "id": "product-research",
      "label": "产品研究",
      "rules": "backend/agent/prompts/product_research.md",
      "memory": "data/agent_memory/product-research.md",
      "triggers": ["产品", "基金", "池", "评价", "持仓", "比较", "这只"],
      "tools": ["products.search", "products.eval", "products.series", "products.plans"],
      "cross_module_read": ["metrics.lookup", "metrics.infer", "metrics.availability"],
      "writes": []
    }
  ]
}
```

## 附录 B：会话文件结构（`data/agent_sessions/<id>.json` 示意）

```json
{
  "meta": {"id": "session-...", "title": "反弹力度指标", "created_at": "...", "updated_at": "...",
           "created_by": "张三", "catalog_version": "sha256:..."},
  "state": {
    "scope": "indicator-center",
    "draft": {"definition": {"name": "...", "expression": "...", "result_kind": "time_series"},
              "display_latex": "...", "required_variables": ["adjusted_nav"]},
    "pending_decision": null,
    "memory_proposals": [],
    "summary_covers": [1, 24],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "tool_rounds": 0}
  },
  "events": [
    {"seq": 25, "ts": "...", "type": "user", "speaker": "张三", "scope": "product-research", "payload": {"text": "..."}},
    {"seq": 26, "ts": "...", "type": "tool_call", "scope": "indicator-center", "payload": {"tool": "metrics.validate", "arguments": {}}},
    {"seq": 27, "ts": "...", "type": "summary", "payload": {"covers": [1, 24], "text": "..."}},
    {"seq": 28, "ts": "...", "type": "commit", "speaker": "张三", "payload": {"indicator_id": "...", "revision": 1}}
  ]
}
```

## 附录 C：系统提示骨架（`prompts/system.md` 结构）

```
# 角色
你是投研工作台内的指标研究助手，服务两个模块：指标中心与产品研究。

# 原则（不可协商）
1. 只能用目录中声明的变量与算子；不能发明算子、不能生成自由代码。
2. 表达不了就明确说「现有算子无法表达」，并说明最接近的可行方案；禁止近似。
3. 公式真相是 DSL；你在对话里引用的公式必须与提交的 DSL 完全一致。
4. 你没有落库工具；提交由人类在界面上完成。
5. 时点规则：使用变量前先查 availability；禁止使用未来数据。

# 可用工具
<tool schemas>

# 输出
按 AgentTurnResponse schema 输出；draft 必须能被 validate/infer 通过。

# 版本
catalog_version={{catalog_version}} dsl_version={{dsl_version}}
```
