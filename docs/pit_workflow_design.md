# PIT（时点可得）全流程设计

## 0. 问题陈述

一条数据同时挂在三条时间轴上，混用即穿越：

| 轴 | 含义 | 本项目字段 |
|---|---|---|
| 事件时间 event / valid time | 事情发生在哪天 | `nav_date` / `trade_date` / 报告期 |
| **可得时间 knowledge time** | **最早哪天能看见它** | `ann_date` / `available_at` / `created_at` |
| 版本时间 vintage / revision | 看到的是第几版 | `data_release_id` |

**PIT 研究 = 只允许「可得时间 ≤ 研究日」的数据进入计算，并锁死用的是哪个数据版本。**

### 开工前实测（`data/etf_daily_df.parquet`，1,547,292 行 / 1,723 只产品）

| 观测 | 数值 |
|---|---|
| `ann_date` 缺失率 | **0.0%** |
| 公告日晚于净值日的行占比 | **98.0%** |
| 滞后中位数 | 1 天 |
| 滞后尾部 | 20~25 天，14,084 行 |
| `date` 列与 `nav_date` 的关系 | **完全相等** |

`fit.py:_load_adj_nav` 只加载 `["ts_code","name","date","adj_nav"]`——`ann_date` **从不加载**。
该函数是大类拟合、滚动相关、类内一致性、自动构建大类、风险平价的**唯一取数入口**，
所以全平台的净值研究稳定穿越 1 天，尾部穿越 25 天。这是本次要修的核心缺陷。

---

## 1. 数据集 PIT 能力分级

分级是**声明 + 实测**的结合，不是拍脑袋：

| 等级 | 判定条件 | 含义 |
|---|---|---|
| **A 严格 PIT** | 声明了可得时间列，实测覆盖率 ≥ 99.5%，且数据不可回溯修订 | 可用于正式回测与 TAA |
| **B 近似 PIT** | 无可得时间列但不可回溯修订（用「事件日 + 声明滞后」作保守可得时间）；或有可得时间列但覆盖率不足 | 可用于研究，正式结论需标注 |
| **C 无 PIT** | 会被回溯修订且无可得时间列 | **严格 PIT 模式下禁止使用** |

声明表在 `backend/pit/catalog.py`，实测在 `backend/pit/audit.py`。

## 2. 研究上下文（全局唯一入参）

```
研究上下文 = { as_of: 研究日, run_mode: RESEARCH | STRICT_PIT, data_release_id: 数据版本 }
```

- 常驻顶栏，所有下游页面只读。
- `RESEARCH`：允许 B/C 级数据，穿越以警告形式返回，不阻断。
- `STRICT_PIT`：引用任何 C 级数据集 → **fail-closed 报错**；A/B 级按可得时间硬截断。
- `as_of` 为空表示「用全部已有数据」，等价于旧行为，但会明确标注 `as_of_applied: false`。

## 3. 数据版本封版（data_release）

每次封版记录一条不可变版本：

```
{ id, created_at, parent_release_id, name, note,
  tables: [{ dataset_id, file, rows, event_range, availability_range, fingerprint }],
  release_fingerprint }
```

`fingerprint` 是**元数据指纹**（文件字节数 + 行数 + 事件时间区间 + 可得时间区间的 SHA-256），
不是逐行内容摘要——对 150 万行做全量哈希的代价不值得，而 parquet 任何一次重写都会改变字节数。
字段名如实叫 `fingerprint`，不叫 `content_hash`，避免过度承诺。

## 4. 全链路

```
① 数据下载 ──→ ② PIT 能力体检 ──→ ③ 数据版本封版
                     │                    │
                     └────────┬───────────┘
                              ↓
                    ④ 研究上下文 as_of / run_mode / release
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
  ⑤ 产品池版本          ⑥ 可投域快照           ⑦ 指标 / 因子
    时点有效性            + fingerprint          as_of 截断（已实现）
        └─────────────────────┼─────────────────────┘
                              ↓
                    ⑧ 大类构建 / 拟合 / 回测
                       按 ann_date 截断
                              ↓
                    ⑨ 保存产物：回写全链路引用
```

## 5. 后端设计

### 5.1 新模块 `backend/pit/`

| 文件 | 职责 |
|---|---|
| `catalog.py` | 数据集 PIT 声明表 + 分级规则。纯函数，无 IO |
| `audit.py` | 扫描 parquet，计算覆盖率、滞后分位、等级、可得区间。带文件指纹缓存 |
| `release.py` | 数据版本仓库（复用 `AtomicJsonStore` 的落盘语义） |
| `context.py` | 研究上下文校验；`STRICT_PIT` 下的 fail-closed 闸门 |

### 5.2 路由 `backend/services/pit_routes.py`

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/pit/audit` | PIT 能力体检表（含汇总 KPI） |
| GET | `/api/pit/releases` | 数据版本列表 |
| POST | `/api/pit/releases` | 封版 |
| GET | `/api/pit/releases/{id}` | 单个版本明细 |
| POST | `/api/pit/context/resolve` | 校验研究上下文，返回被禁用的数据集 |

### 5.3 取数层改造 `backend/fit.py`

```python
def load_adj_nav_pit(data_dir, codes, names, *, as_of=None, run_mode="RESEARCH") -> NavLoad
```

- 额外加载 `ann_date`，派生 `available_date`
- `as_of` 存在时按 `available_date <= as_of` 截断
- 缺 `ann_date`：`RESEARCH` 降级到 `nav_date` 并告警；`STRICT_PIT` 直接报错
- 返回 `NavLoad(frame, lineage)`，`lineage` 记录截断前后行数、可得字段、是否降级

`_load_adj_nav` 保留为薄封装（老调用点与既有测试不受影响）。

### 5.4 血缘回写

`asset_alloc_info.parquet` 增加 4 列：
`universe_snapshot_id` / `data_release_id` / `as_of` / `run_mode`。
存量 47 行没有这些列，读取时按缺列补空处理，不做破坏性迁移。

`repository.create_universe_snapshot` 补 `content_hash`（`membership.py` 早就在读，此前恒为 `None`）。

## 6. 前端设计

### 6.1 全局研究上下文条

紧贴顶栏下方，sticky，40px 单行，密度优先：

```
研究日 [2026-09-07]  模式 [研究模式 ▾]  数据版本 [rel-3f2a ▾]  PIT  A3 B2 C1 ›
```

- 状态存 React Context + `localStorage`，刷新不丢
- `严格 PIT` 模式整条变琥珀色，提示当前处于受限口径
- 右侧红绿灯点击跳 `/settings/pit-snapshots`

### 6.2 `/settings/pit-snapshots`（原型占位 → 真实页面）

信息密度优先，数据在 120px 内出现：

```
┌ KPI 条（单行，6 个数）────────────────────────────────────┐
│ 数据集 9 │ A 级 2 │ B 级 5 │ C 级 2 │ 最新可得 2026-09-05 │ 版本 4 │
└──────────────────────────────────────────────────────────┘
┌ 数据集 PIT 能力（主表，可点选）──────────────────────────┐
│ 数据集 │ 事件时间 │ 可得时间 │ 覆盖率 │ P50 │ P95 │ 等级 │
│ ETF净值 │ nav_date │ ann_date │ 100.0% │  1  │  2  │  A  │
└──────────────────────────────────────────────────────────┘
┌ 选中数据集：滞后分布 ┐┌ 当前 as_of 下的可见性 ┐
└──────────────────────┘└───────────────────────┘
┌ 数据版本（data_release）+ 封版按钮 ───────────────────────┐
```

数字全部 `tabular-nums` 右对齐；等级用色块而非纯文字。

## 7. 自审要点

- NJIT 规则不涉及（本次无新增数值内核）
- 所有新测试用 `tmp_path` 构造夹具，无网络调用
- `STRICT_PIT` 必须 fail-closed，不得静默降级
- 存量 parquet 缺列必须能读，不得因为新列崩溃

---

## 8. 实现记录（开发完成后回填）

### 落地清单

| 文件 | 状态 | 内容 |
|---|---|---|
| `backend/pit/catalog.py` | 新增 | 9 张表的 PIT 声明 + 分级规则（纯函数） |
| `backend/pit/audit.py` | 新增 | 扫描实测；只读 footer 与两列日期，带文件指纹缓存 |
| `backend/pit/context.py` | 新增 | 研究上下文校验；`STRICT_PIT` fail-closed 闸门 |
| `backend/pit/release.py` | 新增 | 数据版本仓库（原子写 + 文件锁 + 单调 sequence） |
| `backend/pit/__init__.py` | 新增 | **双命名别名**，见下 |
| `backend/services/pit_routes.py` | 新增 | 6 条路由 |
| `backend/fit.py` | 改造 | `load_adj_nav_pit`，按 `ann_date` 截断；`_load_adj_nav` 保留为薄封装 |
| `backend/auto_asset_class.py` | 改造 | spec 新增 `as_of` / `run_mode` / `data_release_id`，接入闸门与告警 |
| `backend/app.py` | 改造 | 保存血缘 4 列 + `/api/allocation-lineage` + 启动预热 PIT 体检 |
| `backend/product_pools/repository.py` | 改造 | 快照补 `content_hash` |
| `frontend/src/app/ResearchContext.tsx` | 新增 | 全局上下文 + `localStorage` 持久化 |
| `frontend/src/components/ResearchContextBar.tsx` | 新增 | 常驻顶栏 |
| `frontend/src/pages/PitSnapshots.tsx` | 新增 | 替换原型占位 |
| `frontend/src/services/pit.ts` | 新增 | API 客户端 |
| `frontend/src/utils/apiError.ts` | 新增 | 统一错误文案（修 `[object Object]`） |

### 实测分级结果（本机数据）

```
A  ETF / 场内基金净值   1,620,672 行  覆盖 99.99%  P50=1 P95=1
A  场外公募基金净值    33,733,507 行  覆盖 99.98%  P50=1 P95=2
A  大类配置净值            74,206 行  覆盖 100%（自算序列，滞后按构造很大）
B  ETF 二级市场行情     1,579,113 行  无可得列，按收盘当日可得
B  交易日历                 6,091 行  提前发布
C  ETF 合同与分类信息       1,792 行  最新态维表，就地覆盖
C  指数基础信息             9,643 行  同上
C  ETF 跟踪指数映射           560 行  同上
C  股票基础信息             5,556 行  同上
汇总：A 3 / B 2 / C 4；A/B 数据可得至 2026-09-01
```

### 自审中发现并修复的四个问题

1. **schema 探测读了整表** —— 用 `pd.read_parquet(path).head(0).columns` 取列名，把 3370 万行拉进内存再丢掉。单产品取数 8.26 秒。改为 `pq.read_schema(path)` 只读 footer → **0.49 秒**（改造前基线 0.28 秒，差值就是 `ann_date` 这一列本身的代价）。
2. **封版指纹漏掉字节数** —— 用 `data_dir / file` 取文件大小，而真实文件可能位于 manifest 指向的快照子目录，`size` 恒为 `None`。改为与体检同一套 `resolve_market_data_file`。
3. **数据版本排序依赖时钟精度** —— `created_at` 只到秒，同秒内封两版时排序不确定，进而把 `parent_release_id` 挂到错误的父版本。改为单调 `sequence`，排序不再依赖时钟（也能扛 NTP 回拨）。
4. **`pit` 包被加载成两份** —— `pit.audit` 与 `backend.pit.audit` 各有一份 `_CACHE`，启动预热的是前者，`context/resolve` 走的是后者 → 首次请求 10.2 秒；更严重的是 `PitContextError` 成了两个不同的类，`except` 会静默漏掉。在 `__init__.py` 中把包与全部子模块在两个命名下互相别名，保证进程内只有一份实例。修复后三条路由均为毫秒级。

### 性能

| 场景 | 耗时 |
|---|---|
| 冷体检（读 3700 万行的两列日期） | 11.0 s |
| 热体检（文件指纹命中） | 0.002 s |
| 启动预热后首次 `/api/pit/audit` | 0.010 s |
| `/api/pit/context/resolve` | 0.003 s |
| 单产品取数（无 as_of / 有 as_of） | 0.49 s / 0.46 s |

冷扫描成本挪到启动预热，且**不 fail-closed**——数据问题应该由体检报告出来，而不是让服务起不来。

### 测试

- `backend/tests/test_pit_workflow.py` 28 条：分级规则、实测边界（缺列/缺文件/负滞后/覆盖率不足）、缓存跟文件而非跟时钟、`STRICT_PIT` fail-closed、**按公告日截断而非净值日**、修订版本取"当时可知的最新版"、封版链路与拒绝条件、6 条路由。
- `frontend/src/pages/PitSnapshots.test.tsx` 8 条：体检表渲染、滞后分布切换、封版并接管上下文、严格模式禁用提示、无研究日的警示、严格模式不发无效请求、跨挂载记忆。
- `frontend/src/utils/apiError.test.ts` 4 条：422 数组 detail 不再渲染成 `[object Object]`。
- 全量：后端 **1449 passed**；前端 **458 passed**，另有 2 条既有失败（`ClassAllocation` 一条与本次改动无关，已用"仅回退本次改动"对比确认；`FactorResearchCenter` 一条在单跑与两文件组合下均通过，仅在全量并行下偶发，3 次全量中 2 次出现）。

---

## 9. 修订：PIT 改为系统级设置（第二轮）

### 起因

第一版把研究上下文做成常驻顶栏 + `localStorage`。两个问题：

1. 一个一周改一次的设置，不该在每个页面占 40px。
2. **`localStorage` 是每浏览器一份** —— 两个分析师打开同一页面，看到不同口径的数字，双方都不知道。「系统级」必须落服务端。这是真 bug。

### 保留的分歧

`数据版本` 是系统级；`研究日 as_of` **不能纯系统级** —— 回测天生要扫 as_of。
解法：**数据版本本身带 as_of 天花板**（`summary.available_through`），所以

```
系统级设定 = 应用哪个数据版本 + 运行模式
as_of      = 默认取该版本的 available_through，请求可显式覆盖
```

一个旋钮，回测仍能扫。

### 改动

| 文件 | 改动 |
|---|---|
| `backend/pit/settings.py` | **新增**。`data/pit_settings.json`，服务端持有；`describe()` 一次性给出 settings / effective / release / available_releases |
| `backend/pit/context.py` | 新增 `resolve_request_context()`：**请求什么都不说就继承系统设置**，说了就以请求为准 |
| `backend/services/pit_routes.py` | 新增 `GET/PUT /api/pit/settings` |
| `backend/services/auto_class_routes.py` | 服务端解析口径；`asOf/runMode/dataReleaseId` 默认全为 `None`（"未声明"必须区别于"显式选择"） |
| `backend/app.py` + `analytics_routes.py` | `/api/fit-classes`、`/api/rolling-corr`、`/api/rolling-corr-classes`、`/api/risk-parity/solve`、`/api/save-allocation` 全部继承系统口径 |
| `frontend/src/app/ResearchContext.tsx` | 改为**服务端读取、只读**；删掉 `localStorage` |
| `frontend/src/components/ResearchContextBar.tsx` | **删除** |
| `frontend/src/components/PitBadge.tsx` | 新增：顶栏只读 chip，借用已有导航行，不新增高度 |
| `frontend/src/components/PitProvenance.tsx` | 新增：结果口径脚注；无 PIT 的结果**明确打标** |
| `frontend/src/pages/PitSnapshots.tsx` | 新增「应用口径（系统级）」区块置于页首；封版不再自动改口径 |

### 关键设计决策

**后端兜底 > 前端穿参。** 请求未声明口径时由后端从系统设置解析。任何页面忘记传也不会错，前端也不必到处穿参数。第一版只有 `AutoAssetClassification` 一个页面传，其余页面全是漏的。

**封版 ≠ 应用。** 封版只记录一个 vintage；要不要让全平台按它展示是另一个动作。封完只预选、不静默切换口径。

**版本消失要大声退回。** 应用的版本被删/被手改后，`describe()` 退回无 PIT 并给出 `release_error`，绝不保留一个悬空的 as_of；同时保留 `settings.active_release_id` 让用户看见原本配置了什么。

**无版本不能开严格 PIT。** 没有截止日可执行，等同研究模式；后端 400 拒绝而不是静默降级。

### 自审发现并修复

1. `runMode` 默认值是 `"RESEARCH"`，导致"未声明"与"显式选择研究模式"无法区分，系统设置永远不会生效 —— 改为 `Optional[...] = None`。
2. `compute_rolling_corr` 调用里关键字参数放到了位置参数前面，`SyntaxError`。
3. JSX 里写了 `**系统级**`，会原样渲染出星号 —— 改 `<strong>`。
4. `PitBadge` 的 `hidden lg:flex` 与父容器 `hidden xl:flex` 打架 —— 父容器已控制可见性，改为 `flex`。
5. `test_analytics_routes` 的替身 `lambda *_` 不接受新的 kwargs —— 改 `lambda *_a, **_k`。

### 端到端验证（真实数据，`/api/fit-classes` 完全不传任何 PIT 参数）

```
无 PIT 口径         -> 最后净值日 2026-09-03
应用口径 as_of=2026-09-01 -> 最后净值日 2026-09-01
口径确实生效: True
```

一个从不发送 PIT 参数的页面，其结果随系统设置改变 —— 「各个页面的展示逻辑都基于这个版本的 PIT」是可验证的事实，而非声明。

`resolve_request_context` 实测 **0.08 ms/次**，因此每请求实时解析，不做缓存（避免读到过期口径）。

### 测试

- 后端 `test_pit_workflow.py` **38 条**（新增 10 条：默认无 PIT、应用版本推导 as_of、无版本禁开严格、清除回退、版本消失大声退回、请求继承系统设置、显式 as_of 覆盖、应用后真的截断、settings 路由 apply/clear）
- 前端 `PitSnapshots.test.tsx` **13 条**（体检 2 + 系统级口径 5 + 顶栏标签 2 + 结果脚注 4）
- 全量：后端 **1486 passed**；前端 **472 passed**，3 条既有失败（`ClassAllocation` 确定性、与本次无关；`FactorResearchCenter` 与 `DataHealthRefreshPanel` 单跑均通过，仅全量并行下偶发）

## 10. 修订：系统级口径是默认，不是强制（第三轮）

### 起因

第二轮把口径做成系统级设置之后，它变成了唯一口径：各功能页只能按平台设定的版本看数，想换个 vintage 看一眼、或者临时关掉 PIT 看全部磁盘数据，都必须去 PIT 页面改系统设置 —— 而那会改变**所有人**看到的数字。

**默认 ≠ 唯一。** 系统级设定该管的是"打开页面时按哪个版本算"，不该管"我现在能不能换个版本看看"。

### 三层口径，弱到强

```
系统级设置（服务端，全平台默认）
  ← 本标签页的临时查看口径（请求头，只影响自己）
      ← 请求显式声明的值（回测扫 as_of 等）
```

`resolve_request_context()` 就是这三层的合并点，没有第二个地方能决定口径。

### 为什么走请求头，而不是给每个接口加参数

实际读数的接口分布在 `app.py`（4 处 `_system_pit()`）、`analytics_routes`（2 处）、`auto_class_routes`、`save-allocation`……给每个请求体加 3 个字段，是更大的 diff，而且**下一个新接口忘记穿参就静默按错口径算** —— 正是本功能要防的事。请求头在传输层注入一次，所有现有和未来的接口自动生效。

### 改动

| 文件 | 改动 |
|---|---|
| `backend/pit/context.py` | 新增 `_VIEW_OVERRIDE` ContextVar + `parse_view_override()`；`resolve_request_context()` 的兜底基准从"系统设置"改为"临时口径 → 系统设置" |
| `backend/pit/settings.py` | 新增 `release_as_of()`：版本自带研究日，标签页不必同时携带日期，避免两者漂移 |
| `backend/services/pit_routes.py` | 新增 `PitViewOverrideMiddleware`（纯 ASGI，非 `BaseHTTPMiddleware`：ContextVar 必须设在跑 endpoint 的同一个 task 里） |
| `backend/app.py` | 装载中间件，且**装在 CORS 之前**，让 CORS 保持最外层 —— 否则被拒的请求头在跨域前端里会变成看不懂的网络错误 |
| `frontend/src/services/pitOverride.ts` | **新增**。`sessionStorage` 持有本标签页选择 + `installPitOverrideFetch()` 在传输层注入请求头 |
| `frontend/src/app/ResearchContext.tsx` | 新增 `override` / `overrideRelease` / `applyOverride`；`label`/`asOf`/`runMode`/`noPit` 改为合并后的生效值 |
| `frontend/src/components/PitBadge.tsx` | 只读 chip → chip + 切换面板（跟随系统 / 按版本查看（研究/严格）/ 关闭 PIT / 去管理版本） |
| `frontend/src/components/Header.tsx` | 移动端菜单里也放一个，窄屏下不至于没有入口 |
| `frontend/src/pages/PitSnapshots.tsx` | 本标签页存在临时口径时给出提示 + 一键恢复，否则本页显示系统口径、别处显示临时口径，数字看起来无法解释 |

### 关键设计决策

**"关闭 PIT" 必须是显式信号。** 什么都不说 = 继承系统设置，所以"我要看全部磁盘数据"不能用"不传值"表达 —— 用 `X-Pit-Off: 1`。这与第二轮 `Optional[...] = None` 是同一个教训的另一面。

**临时口径存 `sessionStorage`，而系统设置绝不能。** 系统设置进 `localStorage` 是第二轮修掉的真 bug（两人看到不同数字且互不知情）；而临时口径本来就该是"我这个标签页、这一会儿"，关掉即失效，正好是 `sessionStorage` 的语义。

**只换版本不写模式时，模式沿用系统设置。** 否则在严格 PIT 的平台口径下换个版本会静默掉回研究模式 —— 用户以为在看严格结果。

**切换口径后整页 reload。** 屏幕上已有的数字是按旧口径算的；同屏混两个 vintage 正是本功能要消灭的困惑。reload 是唯一便宜且可靠的失效方式。

**前端豁免 `/api/pit/*`。** 一个失效的临时口径（版本被删）会让每个数据请求 400 —— 如果连读取/清除口径的接口也带上这个头，标签页就把自己锁死在改不回来的状态里。同时 `ResearchContext` 发现所选版本已不在 `available_releases` 里时自动退回系统默认。

**坏请求头大声 400，不静默忽略。** 静默忽略用户选定的口径，等于让人按不知道的 vintage 读数。

### 端到端验证（真实数据，系统设置全程保持"无 PIT"）

```
系统设置                                  无 PIT 口径 · 使用全部磁盘数据
不带请求头                                -> 最后净值日 2026-09-03
X-Pit-Off: 1                              -> 最后净值日 2026-09-03
X-Pit-Release: <封版 available_through=2026-09-01>
                                          -> 最后净值日 2026-09-01
  + X-Pit-Run-Mode: STRICT_PIT            -> 最后净值日 2026-09-01
  + X-Pit-As-Of: 2026-06-30（显式值更强）  -> 最后净值日 2026-06-30
X-Pit-Release: release-nope               -> 400 临时 PIT 口径无效：未找到数据版本 release-nope。
验证后系统设置                             无 PIT 口径 · 使用全部磁盘数据（未被任何查看动作改动）
```

验证用的临时封版已删除，`data/data_releases.json` 与 `data/pit_settings.json` 已还原。

### 测试

- 后端 `test_pit_workflow.py` **45 条**（新增 7 条：无头不动系统设置、`X-Pit-Off` 与"什么都不说"可区分、换版本沿用严格模式、未知版本拒绝而非忽略、显式值压过临时口径且不跨请求泄漏、请求头到达**同步** endpoint（证明 ContextVar 过线程池）、坏头 400）
- 前端 `pitOverride.test.ts` **7 条**（请求头映射、`sessionStorage` 往返、`/api/pit/*` 豁免、不覆盖调用方请求头）；`PitSnapshots.test.tsx` **17 条**（新增 4 条：临时换版本不改系统设置、关闭 PIT、临时标记 + 一键恢复、被删版本不锁死标签页）
- 全量：后端 **1541 passed**；前端 **519 passed / 1 failed**（`ClassAllocation` 既有确定性失败，与本次无关）


## 11. 第四轮设计：从「取数时点化」到「决策时点化」

### 11.0 起因

前三轮解决的是**取数**：`load_adj_nav_pit` 按公告日切行、封版锁 vintage、口径三层可控。这层做得扎实。

但投前研究的穿越大头不在取数，在**决策**：

- 产品池是**今天**选出来的，却拿去跑 2018 年起的回测 —— 名单里没有任何一只后来清盘的基金
- `asset_nv.parquet` 的大类净值是**今天**算的（`creat_time`），却当作 2018 年的回测输入
- 可投资域来自最新态维表，`etf_info_df.parquet` 整表覆盖写，没有历史版本

一句话概括缺口：

> PIT 现在能回答「T 日能看到哪些**行**」，还不能回答「T 日有哪些**产品可选**」和「T 日会做出**什么决策**」。

这三件事都属于后视镜效应，而且它们比公式层的未来函数更难发现——公式可以逐节点探测，产品池不能。

---

### 11.1 现状盘点

**已建成，不重复造：**

| 能力 | 位置 | 评价 |
|---|---|---|
| 数据集时点声明（事件/可得/修订）与 A/B/C 分级 | `pit/catalog.py` | 完整 |
| 研究上下文 + 三层口径合并 | `pit/context.py` | 完整，严格模式 fail-closed |
| 数据封版 | `pit/release.py` | **仅元数据指纹，不能还原历史 vintage** |
| 系统级口径设置 | `pit/settings.py` | 完整 |
| 净值按公告日切割 + 修订感知去重 | `fit.py:135 load_adj_nav_pit` | **全仓唯一真正的 PIT 取数，质量高，是后续泛化的模板** |
| 回测按调仓日切拟合窗 | `backtest_engine.py:44 slice_fit_data` | 事件时间上因果正确 |
| 情景模型实时/回顾模式分离 | `historical_regimes/algorithms.py:571` | `realtime` 只用前 N 期训练并样本外分类；`retrospective` 用 smoothed 后验（含未来）且已明确标注 —— 设计正确 |
| Tushare 物理快照目录 + 原子切换 | `market_data.py:119 activate_tushare_snapshot` | 存在，但**封版没有引用它** |

**缺口，按危害排序：**

#### G1 维表无历史版本 → 可投资域不可还原（幸存者偏差）

代码里已有自述：

- `pit/catalog.py:117`（etf_info）：「维表按最新状态整体覆盖写入，没有生效时间也没有历史版本；用它做历史分类会带入今天的信息。」
- `pit/catalog.py:146`（stock_basic）：「含退市状态的最新态维表；缺历史版本时无法还原当时的可选股票域。」

严格模式下这两张表被判 C 级、直接 block（`context.py:237 require_usable`）。这是「我知道自己不行所以罢工」，不是「我能给出 T 日正确的名单」。

#### G2 回测完全不接 PIT

`backtest_engine.py` / `optimizer.py` / `strategy.py` 三个文件里 PIT 相关标识符**零出现**。
`backtest_engine.py:415 load_nav_wide_from_parquet` 直接 `pd.read_parquet`，不过滤可得时间、不做修订去重。

讽刺的是 `context.py:170` 的注释已经预留了这条路：「a stated value always wins, which is what lets a backtest sweep `as_of`」——设计时想到了，回测那头没接线。

#### G3 `asset_nv.parquet` 双重穿越

`pit/catalog.py:157` 自述：「平台自算序列，创建时间即可得时间。滞后天数按构造就很大——2026 年算出的配置净值，2018 年确实不可得。」

而 `backtest_engine.py:416` 读它时**不过滤 `creat_time`**。两层问题：

1. 行层面：所有行的 `creat_time` 都是今天，严格按可得时间过滤会把整段清空
2. 定义层面：「哪些 ETF 代理哪个大类」这个映射本身是今天定的

SAA 回测的输入是它，所以这是 SAA 链路的根节点问题。

#### G4 产品池冻结的是「名单」不是「规则」

`product_pools/membership.py:69` 只用 `research_date` 校验成员是否过期，没有按回测日期回放的能力。

好消息：`run_plan(plan_id, as_of)`（`custom_indicators/service.py:6382`）**已经接受 as_of**，评价方案本身是可回放的。缺的只是把回放能力接到池子版本上。

#### G5 封版不能还原

`release.py:96 _table_fingerprint` 是刻意的元数据摘要（注释解释了原因：行级 hash 太贵且收益低）。结论正确，但后果是：**封版能检测漂移，不能恢复 vintage**。而 `market_data.py` 的物理快照目录机制已经存在，只是封版记录里没有指针指过去。

#### G6 PIT 取数只覆盖净值

只有 NAV 走 `load_adj_nav_pit`。指标、评价方案、因子、情景特征全部各自 `read_parquet`。每加一个新读数点，就多一个忘记切时点的机会。

---

### 11.2 核心概念：第四个时钟

前三轮定义了三条时间轴（见 §0）。第四轮要加第四条：

| 轴 | 含义 | 谁在用 |
|---|---|---|
| 事件时间 | 事情发生在哪天 | 全部 |
| 可得时间 | 最早哪天能看见 | `load_adj_nav_pit` |
| 版本时间 | 看到的是第几版 | `data_release_id` |
| **决策时间 decision time** | **这次计算站在哪天做决定** | **回测 / SAA 再优化 / TAA 信号** |

关键区分：

```
单点决策时钟 —— 产品研究看某只基金、产品池评审定稿
              现有 ResearchContext 已经够用

序列决策时钟 —— 回测、滚动优化、TAA 信号生成
              需要新东西：一次运行扫过多个 as_of
```

`ResearchContext` 是单点的，这没错。要加的不是替换它，而是一个能生成一串 `ResearchContext` 的东西：

```python
# pit/clock.py（新增，约 40 行）
@dataclass(frozen=True)
class DecisionClock:
    """一次研究要依次站上的时点序列。"""
    dates: tuple[str, ...]          # 调仓日 / 信号日
    base: ResearchContext           # 版本与运行模式在整个扫描期内固定

    def at(self, date: str) -> ResearchContext:
        return replace(self.base, as_of=date)

    @classmethod
    def from_rebalance_dates(cls, dates, base) -> "DecisionClock": ...
```

版本（vintage）在扫描期内**必须固定** —— 扫的是 as_of，不是 release。混扫两者结果不可解释。

---

### 11.3 五层设计

```
L4 证据层   pit/guard.py         断言 + 血缘扩展
L3 决策层   pit/clock.py         DecisionClock 扫描
L2 域层     pit/universe.py      investable_universe(ctx) -> 可投资域
L1 取数层   pit/frame.py         read_pit(dataset_id, ctx) 统一入口
L0 数据层   维表 append-only 快照   地基
```

自下而上说明。

---

#### L0 · 维表历史化（append-only 快照）

**问题**：`T01_get_data.py:2235` 用 `save_dataframe` 整表覆盖写 `etf_info_df.parquet`。上一版没了。

**方案选型**：

| 方案 | 存储 | 改造量 | 取舍 |
|---|---|---|---|
| SCD-2（valid_from / valid_to，只记变化） | 最小 | 大：要做 diff、合并、闭区间维护 | 正统，但现在不值 |
| **每次刷新追加整表 + `snapshot_date` 列** | 每天一份全量 | **一个 helper + 4 个调用点** | **选它** |

选后者。ETF 信息表数千行，日频快照一年约 100 万行，parquet 压缩后几十 MB。存储不是瓶颈，改造量才是。

```python
# T01_get_data.py，在 save_dataframe 旁边
def save_dataframe_with_history(df, path, *, snapshot_date=None):
    """照旧覆盖写最新态，同时把这一版追加进 <stem>_history.parquet。

    ponytail: 整表追加而非 SCD-2 diff。表涨到读取变慢时再改成只记变化，
    届时 history 表本身就是重建 SCD-2 的原料。
    """
    save_dataframe(df, path)
    stamped = df.copy()
    stamped["snapshot_date"] = snapshot_date or date.today().isoformat()
    history = path.parent / "pit_dim" / f"{path.stem}_history.parquet"
    ...  # 已存在则 concat 后原子写；只在当天尚无快照时追加
```

接入四张表：`etf_info_df` / `index_info` / `stock_basic` / `etf_index`（即 catalog 里 `revisable=True` 且无 availability 列的四张）。

同时 `pit/catalog.py` 给这四条声明补 `history_file` 字段，让分级能感知：**有历史快照 → 从 C 升到 B**（不是 A，因为 `snapshot_date` 是我们抓取的日子，不是数据商发布的日子）。

**必须诚实说明的边界**：历史维表版本**已经丢失，找不回来**。这套机制只能从上线那天开始积累。所以要落一个字段：

```
universe_history_begins_at = min(snapshot_date)
```

回测起始日早于它 → 该段的可投资域不可信 → L4 报警。这一条不能藏。

---

#### L1 · 取数层统一

`fit.py:135 load_adj_nav_pit` 已经把该做的都做对了：可得时间切割、修订感知去重（按 `available_date` 排序后 `keep="last"`）、严格模式拒绝无公告日的文件、全过程血缘。

问题只是它写死在 `fit.py` 里、只服务净值。把它的逻辑提到 catalog 驱动的通用入口：

```python
# pit/frame.py（新增）
def read_pit(dataset_id: str, ctx: ResearchContext, data_dir: Path,
             *, columns=None, **filters) -> PitFrame:
    """按 catalog 声明自动选择可得时间列并切割，返回 frame + lineage。"""
```

`declaration.availability_field` 为 None 时用 `event_field + declared_lag_days` 作保守估计（catalog 里这个字段已经存在但目前没人消费）。

`load_adj_nav_pit` 改成 `read_pit("etf_nav" | "fund_nav", ...)` 的薄封装，**保持现有签名与返回值不变**，不破坏 `app.py:387` 和 `auto_asset_class.py:670` 两个调用点。

收益：以后新增读数点只有一个入口，忘记切时点从"容易"变成"要绕路才能做到"。

---

#### L2 · 可投资域

```python
# pit/universe.py（新增）
@dataclass(frozen=True)
class UniverseView:
    as_of: str
    codes: tuple[str, ...]
    detail: pd.DataFrame          # code / name / list_date / delist_date / status
    history_begins_at: str | None
    coverage: str                 # "REPLAYED" | "LATEST_ONLY" | "PARTIAL"
    warnings: tuple[str, ...]

def investable_universe(data_dir: Path, ctx: ResearchContext,
                        *, kind: str = "fund") -> UniverseView:
    """as_of 那天真实存在、未清盘、可申赎的产品集合。"""
```

实现：读 L0 的 history 表，取 `snapshot_date <= as_of` 的最后一版，再按 `list_date <= as_of < delist_date` 过滤。

- history 表覆盖不到 as_of → `coverage="LATEST_ONLY"`，退回最新态并置 warning
- 部分覆盖 → `"PARTIAL"`

**所有模块从此只从这里拿域**，不再各自 `read_parquet` 后过滤。这是把 G1 一次修在共用点上，而不是在每个调用方各补一个 guard。

---

#### L3 · 决策时钟扫描

`backtest_engine.py:44 slice_fit_data(nav, up_to, ...)` 已经在按调仓日切窗——但切的是**一份今天取出来的宽表**（`load_nav_wide_from_parquet` pivot 后 `available_date` 已经丢了）。

改造分两级，内层完全不动：

```python
# 外层（新增）：每个决策日重取
for t in clock.dates:
    ctx_t = clock.at(t)
    nav_t = pit_nav.view(t)          # 按 available_date <= t 切片后 pivot
    universe_t = investable_universe(data_dir, ctx_t).codes
    nav_t = nav_t[[c for c in nav_t.columns if c in universe_t]]
    # 内层（不动）：
    nav_fit = slice_fit_data(nav_t, t, window_mode, data_len)
```

**性能**：不真的每个调仓日重读 parquet。一次读长表（保留 `available_date`），在内存里按 t 做 mask + pivot。

进一步的懒优化——**先测有没有修订**：

```python
if (nav_long["available_date"] <= nav_long[NAV_EVENT_FIELD]).all():
    # 无修订：available_date 切割等价于 event_date 切割，走现有快路径
```

ETF 日频行情绝大多数当天可得（§0 实测：滞后中位数 1 天，尾部 25 天），所以这条快路径命中率高，但**不能默认成立** —— 必须实测后再走。

同时 `run_from_payload`（`backtest_engine.py:424`）新增可选的 `clock` 参数，不传时行为与今天完全一致。旧调用不破。

---

#### L4 · 证据与断言

血缘扩展：沿用 `fit.py` 里已有的 lineage 结构，补三个字段。

```python
lineage["universe"] = {
    "source": "REPLAYED" | "FROZEN_LIST" | "MANUAL",
    "history_begins_at": "2026-09-09",
    "coverage": "PARTIAL",
}
lineage["decision_clock"] = {"swept": True, "dates": 96, "first": "...", "last": "..."}
```

一条硬断言：

```python
# pit/guard.py（新增）
def assert_no_universe_lookahead(ctx: ResearchContext, lineage: dict) -> None:
    """回测区间早于可投资域历史起点：严格模式 raise，研究模式 warn。

    研究模式不 raise 是有意的 —— 现在全部历史回测都会命中这条，
    一刀切禁止等于禁用整个回测功能。分档标注，让人知情。
    """
```

前端在结果脚注（已有的 `PitProvenance.tsx`）上加一档标记：

```
✓ 域可回放      universe 来自历史快照，回测区间被完整覆盖
⚠ 域仅最新态    universe 用了今天的名单，存在幸存者偏差
⚠ 人工池        成员由人工判断，无法回放
```

---

### 11.4 分模块落地

#### 产品研究（`/product-research/*`）

| 节点 | 改动 |
|---|---|
| 基金市场概览 panorama | 域改走 L2。现在读最新态维表，展示的是"活到今天的基金"，规模分布天然上偏 |
| 产品与管理人研究 products | 取数改走 L1 |
| 产品评价与分类 evaluation | `run_plan` 已有 as_of；补：run 结果落库时记录 `universe_ref`（用了哪个 as_of 的域） |
| 产品与信号回测 product-backtest | 接 L3 扫描 |

单点时钟就够，这一层不需要 DecisionClock（除了 product-backtest）。

#### 产品池（`/product-research/pools`、`pool-lifecycle`）

核心改动：版本除了 `members` 名单，再存 `rule_spec` + `replayable` 标志。

```json
{
  "members": [...],                    // 保持不变，是 as_of 当天的物化结果
  "rule_spec": {                       // 新增
    "plan_ids": ["plan-a", "plan-b"],
    "max_rank": 50, "min_score": 60,
    "manual_overrides": [...]          // 人工增删，天然不可回放
  },
  "replayable": true,                  // rule_spec 完整且无人工覆盖时为 true
  "universe_history_begins_at": "2026-09-09"
}
```

回测里：

```python
pool_at_t = version.replay(as_of=t) if version["replayable"] else version["members"]
```

**必须承认的边界**：纯人工挑选的池子**无法回放**。这不是 bug，是事实——人的判断没有可重放的输入。设计上的处理是**分档标注**（`universe.source = MANUAL`）而不是一刀切禁止。假装能解决它才是错的。

前端：「生成锁定快照」按钮旁加可回放性徽章，让人在封版那一刻就知道这个池子将来能不能用于历史回测。

#### SAA（`/pre-investment/saa/*`）

这是改动最重的一块，因为 G3 在根上。

| 节点 | 改动 |
|---|---|
| 大类资产构建 asset-classes | 代理产品的候选从 L2 取域；已有 `load_adj_nav_pit`，取数层已经对了 |
| 自动构建大类 auto-classification | 已接 PIT（`auto_asset_class.py:651`），是全平台接得最好的一个，可作模板 |
| 大类资产配置与回测 allocation-lab | 接 L3 扫描；**并修 `asset_nv` 穿越** |

`asset_nv` 的修法，两个选项：

1. **回测里不读缓存，按 as_of 重算大类净值**。正确，但每个调仓日重算一次拟合，慢。
2. **给 `asset_nv` 加 `as_of` 维度**：`compute_classes_nav` 在指定 as_of 下算出的结果带上 `as_of` 列落库，回测按 `as_of <= t` 取。等于把重算结果缓存起来。

选 2。理由：`compute_classes_nav` 已经接受 `as_of` 参数（`analytics_routes.py:76`），能力已经在了；加一列比加一套重算调度便宜得多。第一次跑某个 as_of 时算并缓存，之后命中。

现在的 `creat_time` 保留不动（它记录的是"这行什么时候写进磁盘"，是另一回事）。

#### TAA（`/pre-investment/taa`）

**基础最好的一块**，改动最小。

已经对的地方：`historical_regimes/algorithms.py:571 _run_latent_model` 的 `realtime` 模式只用前 `initial_train_size` 期训练、标准化参数也只从训练窗算（`:590`），随后样本外分类；`retrospective` 模式用 smoothed 后验（确实含未来）但明确标注，且 TAA 节点描述里写明「引用已发布的**实时因果**情景版本」。这是正确的设计，不要动。

要补的两点：

1. **训练窗固定不刷新**。realtime 模式训练一次就再不重拟合。这是因果的（不含未来），但模型会陈旧。接入 DecisionClock 后可选 expanding-window 重拟合——**每个决策日只用截至当日的数据重训**。这是增强，不是修 bug。
2. **情景版本发布时要记录 `fit_as_of`**，与 `applied_at` 分离。现在版本元数据里没有"这个模型是用截至哪天的数据训的"，导致一个 2026 年训的模型可以被静默用在 2018 年的回测上。

第 2 点是真缺口，第 1 点是可选项。

---

### 11.5 诚实边界

设计里有三件事**做不到**，写在这里免得后面有人以为是 bug：

1. **历史维表版本找不回来。** L0 只能从上线日开始积累。在那之前的回测，可投资域永远只能标 `LATEST_ONLY`。唯一的替代是买数据商的历史成分/退市数据。
2. **人工判断不可回放。** 人工池、人工调整的大类划分，都只能标注不能重建。
3. **封版仍不能还原 vintage。** 除非把 `release` 指向 `market_data.py` 的物理快照目录（G5），那需要磁盘空间与保留策略，属于另一个决策，不在本轮。

---

### 11.6 优先级与工作量

| 序 | 事项 | 层 | 为什么这个次序 |
|---|---|---|---|
| **P0-1** | 维表 append-only 快照 + `universe_history_begins_at` | L0 | **今天不开始积累，一年后还是没有历史。** 这一条越早越好，与其他所有项无依赖 |
| **P0-2** | `asset_nv` 加 `as_of` 维度 | SAA | SAA 回测的根节点，不修则 SAA 全链路结论不可用 |
| P1-1 | `pit/frame.py` 取数层泛化 | L1 | 后续所有接入的公共前置 |
| P1-2 | `pit/universe.py` + `pit/guard.py` | L2/L4 | 让 G1 可被观测（即使暂时只能标 `LATEST_ONLY`） |
| P1-3 | `pit/clock.py` + 回测接扫描 | L3 | 依赖 L1/L2 |
| P2-1 | 产品池 `rule_spec` 回放 | 产品池 | 依赖 L3 才有消费方 |
| P2-2 | 情景版本记录 `fit_as_of` | TAA | 独立小改动 |
| P2-3 | 封版引用物理快照目录 | L5 | 需要先定保留策略 |

P0 两项互不依赖，可并行。

### 11.7 验收口径

每项落地必须能给出一条可跑的证明，格式沿用第二、三轮：

- L0：连续两天刷新后 history 表有两个 `snapshot_date`，且中间被删除的产品在旧快照里仍在
- L2：`investable_universe(as_of=T)` 包含一只在 T 之后清盘的产品
- L3：同一回测在 `swept=False` 与 `swept=True` 下结果**不同**（相同说明扫描没生效）
- SAA：回测读到的 `asset_nv` 行数随 as_of 单调变化
- 产品池：`replayable=True` 的版本 `replay(as_of=T)` 与当初 T 日发布的名单一致

---

## 12. 第四轮实现记录：设计与落点的差异

第 11 节是设计，本节是**实际落地的代码**，以及实现过程中和设计不一致的地方——设计文档如果和代码对不上，下一个人会两边都不信。

### 12.1 代码落点

| 层 | 文件 | 内容 |
|---|---|---|
| L0 | `T01_get_data.py:961 append_dimension_snapshot` | 维表整表追加快照，写 `data/pit_dim/<stem>_history.parquet`；`save_dataframe(..., history=True)` 挂在 4 个维表落盘点上（etf_info / stock_basic / index_info / etf_index） |
| L0 | `backend/pit/catalog.py` | 声明新增 `history_file`、`key_fields`；`SNAPSHOT_FIELD = "pit_snapshot_date"`；`grade()` 新增 `history_snapshots` 参数——有快照的可修订维表从 C 升到 B |
| L0 | `backend/pit/audit.py:history_path/_history_profile` | 体检表新增 `history` 块：快照数、历史起点、最新快照 |
| L1 | `backend/pit/frame.py:read_pit` | 目录驱动的通用 PIT 取数：维表快照回放 → 可得时间解析（公告列 → 事件日+声明滞后 → 无）→ 按研究日截断 → 血缘 |
| L2 | `backend/pit/universe.py:universe_as_of` | 产品域出口，返回 `UniverseView`；覆盖度三档 `REPLAYED / INTERVAL / LATEST_ONLY` |
| L3 | `backend/pit/clock.py` | `DecisionClock`（研究日序列，不越过自身上下文）、`visible_at`（决策时点切窗）、`availability_from_rows` |
| L3 | `backend/backtest_engine.py` | `slice_fit_data` / `ensure_valid_rebalance_window` / `backtest_portfolio` 新增 `available_at`；新增 `load_allocation_nav` 作为 `asset_nv` 的唯一读入口 |
| L3 | `backend/fit.py:last_nav_availability` | 每个观测日的可得时间（取成分中最晚公告者） |
| L4 | `backend/pit/guard.py` | `check_universe` / `assert_no_universe_lookahead` / `universe_lineage`；研究模式记录，严格模式抛错 |

### 12.2 接入点

- **SAA**：`app.py:save_allocation` 给 `asset_nv` 每行盖 `as_of`、`run_mode`、`available_at` 三列；`services/strategy_routes.py` 的四个端点收敛到一个 `_load_alloc_nav`，回测/权重/调仓表/默认起点全部走同一口径；`services/analytics_routes.py` 有效前沿同样接入。调仓表缓存键加入 PIT 口径，否则换研究日会命中旧权重。
- **产品池**：`product_pools/service.py:replay_version` + `GET /api/product-pool-versions/{id}/replay?as_of=`，用版本自带的评价方案绑定重跑到目标研究日，输出 kept / added / removed / manual_only；`GET /api/product-pool-versions/{id}` 响应新增 `pit` 块。
- **产品研究**：评价方案运行结果新增 `universe` 块，标注候选名单是人工选定的、选定日期、以及是否晚于研究日。
- **TAA**：`historical_regimes/service.py` 与 `v2_service.py` 的发布记录新增 `fit_as_of` / `fit_mode`，与 `published_at` 并列。
- **前端**：`GET /api/pit/universe` + PIT 设置页「站在某日的可选产品域」面板；`PitDecisionNotice` 组件挂在大类配置回测结果上。

### 12.3 与设计不同的三处

1. **发现了一条不需要历史快照的回放路径。** 设计假设维表没有历史就只能是 `LATEST_ONLY`。实际上 `etf_info_df.parquet` 带 `list_date` / `delist_date`，`index_info.parquet` 带 `list_date` / `exp_date`——用区间就能还原当时的成分，而且对全部历史立即生效，不用等快照积累。这是新增的 `INTERVAL` 档：成分对、但属性值（名称、分类、管理人）仍是今天的。实测 2019-06-28 的基金域是 224 只，今天的表是 1792 只，1568 只是幸存者偏差。`stock_basic.parquet` 没有退市日期列，仍然只能 `LATEST_ONLY`。
2. **产品池做了真回放，不只是标注。** 设计把「池子按规则回放」放在 P2。实际上 `run_plan(plan_id, as_of)` 和版本快照里的 `evaluation_plans` 绑定已经齐了，组合起来就是回放，所以直接做了。
3. **`asset_nv` 的 `as_of` 维度按设计做了，但决策时钟的主战场是 `available_at`。** 逐个调仓日重算大类净值代价太大；实际做法是给每行存可得时间，回测在每个调仓日按可得时间而不是净值日期切拟合窗。差异是可测的——`test_publication_lag_changes_the_backtest_result` 断言两种口径结果**必须不同**。

### 12.4 仍然做不到的

- 历史维表版本从今天起才开始积累，`history_begins_at` 之前的研究日只能靠 `INTERVAL` 或退回 `LATEST_ONLY`。
- 人工判断（approved / excluded）没有可重放输入，回放结果里单列 `manual_only`，不假装能还原。
- 封版仍然只是元数据指纹，不能物理还原 vintage。

---

## 13. 前端重构：把「站在哪一天」变成一个可见的控件

### 13.1 观察到的失败

一位用户想研究「截止 2009-12-31 的数据」，实际操作是：在**封版**表单里填了版本名「2010研究」、备注「基于2010年1月1日之前的数据研究」，然后点封版。

备注是自由文本，不会产生任何效果。用户不是操作错了——他在屏幕上找不到别的地方可填。

### 13.2 根因：两个问题共用一个旋钮

| | 回答的问题 | 答案形式 | 改版前 |
|---|---|---|---|
| 研究日 `as_of` | 我**站在哪一天**看 | 一个日期 | **无 UI**，由封版的 `available_through` 推导 |
| 数据版本 `release` | 我读的是**哪一次**的历史 | 不可变文件指纹 | 唯一可见控件 |

`settings.py` 原注释把这写成一句设计取舍：「A sealed release already knows the last date it can honestly answer for, so pinning a release fixes `as_of` too. That keeps one knob where users expect one knob.」

一个旋钮的假设错了。它成立的前提是「用户只想看最新数据的最新一天」，而 PIT 的**全部意义**就是站到过去某一天。这条取舍恰好挡住了唯一的核心用例。

三处具体缺陷：

1. **主控件缺失。** 研究日不可输入，且不封版就永远不能启用严格 PIT——一个新用户被两道门同时锁在外面。
2. **术语不落地。** 「封版」「口径」「研究日」是实现词汇；用户想的是「我要看某年某月」。
3. **顺序倒置。** 页面先要求封版（一个还不知道为什么要做的动作），才谈应用。用户的决策顺序是：先站到哪天 → 再用哪批数据 → 再定严不严格。

### 13.3 新布局：三问，按人的决策顺序

```
研究口径（系统级 · 全平台生效）
当前：站在 2009-12-31 · 最新数据（未封版）· 研究模式

① 站在哪一天看？        [2009-12-31] [数据最新一天] [不设研究日]
   只使用该日当时已经公开的数据。之后才公布的净值、之后才上市的产品，一律不可见。

② 用哪一批数据？        [最新数据（未封版） ▾]
   和研究日无关：研究日决定看到哪一天为止，数据版本决定读的是哪一次的历史。

③ 严不严格？            (研究模式) (严格 PIT)

将要生效：站在 2009-12-31 · 最新数据（未封版）· 研究模式
         届时可选基金 224 只，比今天的表少 1,568 只     [应用到全平台]
```

三个设计判断：

- **①②③ 编号并各自带一句「它管什么」。** 分不清两个旋钮是这次失败的根源，标题里就要把区别说完。
- **预览是数字不是文案。** 「站在 2009-12-31」很抽象；「届时可选基金 224 只，比今天少 1,568 只」不抽象。复用 §12 的 `/api/pit/universe`，在**点应用之前**就把代价摆出来。
- **封版区改口。** 第一句改成「封版不是用来选日期的——要"只看某天为止"，请用上面的 ① 研究日」，把用户从错误的控件上引开。

### 13.4 约束

两个旋钮独立，只有一条单向约束：**研究日不能晚于所选数据版本的可得截止日**——版本回答不了它不包含的日子。日期框的 `max` 跟着版本走，后端也拦一道。

严格 PIT 的前置条件从「必须有封版」改成「必须有研究日」。封版仍然强烈建议（否则下次刷数据结论会变），但不再是硬门槛——它挡住的正是本节这位用户。

### 13.5 代码

| 文件 | 改动 |
|---|---|
| `backend/pit/settings.py` | `as_of` 成为独立存储的设置；`describe()` 增 `as_of_source`（explicit / release）；`no_pit` 改为两者皆空；`update()` 增 `as_of` 参数与跨版本校验；标签统一为「站在 X · 版本 · 模式」 |
| `backend/services/pit_routes.py` | `SettingsRequest.asOf` |
| `frontend/src/services/pit.ts` | `effective.as_of_source`、`settings.as_of`、`applyPitSettings({asOf})` |
| `frontend/src/services/pitOverride.ts` | 每标签页 override 增 `asOf`，可单独成立（不需要版本）；发 `X-Pit-As-Of` 头 |
| `frontend/src/pages/PitSnapshots.tsx` | 三问卡片 + 域预览；封版区改口 |
| `frontend/src/components/PitBadge.tsx` | 顶栏弹层增「只看某一天为止」日期框——之前只能换版本，不能换日子 |
| `frontend/src/app/ResearchContext.tsx` | 处理只有研究日的临时口径 |

### 13.6 兼容

已应用版本但没设研究日的老配置，`as_of` 仍从版本推导，口径不变，只是 `as_of_source` 现在会说明这个日期是用户选的还是版本带出来的。
