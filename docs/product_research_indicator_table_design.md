# 产品研究页 · 研究指标区块重构详细设计（卡片栅格 → 指标表）

适用范围：`/product-research/products/:productId` 「走势与指标」页签里的 **研究指标** 区块
（`frontend/src/components/metrics/ResearchIndicatorPanel.tsx` + `MetricDisplay.tsx` 的 `MetricResultCard`）。

前序设计：`docs/product_research_indicator_panel_design.md`（本次接手其 §3.1 的信息结构）、
`docs/product_trend_chart_design.md`（时序指标已迁走，本区块只剩标量）。
设计准则：`docs/frontend-design-guidelines.md`。契约来源：`backend/custom_indicators/periods.py`、
`backend/services/custom_indicator_routes.py`、`MetricPresentation`。

---

## 1. 现状诊断

当前是 `md:grid-cols-2 xl:grid-cols-3` 的等权卡片栅格，每张卡的结构是
`类别·来源版本 / 名称 / 数值 / 口径行 / 计算区间下拉 / 参数区 / 查看定义`。

实测一屏（4 个指标）的问题，按损害排序：

| # | 问题 | 证据 | 违反 |
| --- | --- | --- | --- |
| 1 | **每张卡一个「计算区间」下拉**，4 张卡 4 个一模一样的「近 1 年（1Y）」 | `MetricResultCard` 的 `onPeriodChange` 块，`min-h-10 w-full` 的 `<select>` 是卡里第二大的元素 | 重复控件；区间在 99% 的使用里是全局意图 |
| 2 | **「此版本未开放可变参数，按固定公式计算。」逐卡重复** | 同上，`schema.length` 为 0 时的兜底文案 | 零信息量文案占了一整行 |
| 3 | **口径行逐卡重复且换行**：`1Y · 2018-12-28 至 2019-12-30 · 246 个观察值 · 数据截至 2026-09-03` | 同一区间同一产品，四行内容完全相同 | 该信息是**列的属性**，不是单元格的属性 |
| 4 | **4 个数字占满一屏**，第二行只有 1 张卡，2/3 空栅格 | 截图 | 准则 2「`VISUAL_DENSITY: 7`」 |
| 5 | **数值一律 `text-accent-700` 蓝** | `MetricResultCard` 的 `text-2xl font-semibold text-accent-700` | 准则 3.1 表：`accent` 语义是「链接与选中」，读者会把数字当链接 |
| 6 | **`direction` 契约没进界面** | `presentation.direction` 已下发；`MetricMatrix` 用它标最佳/最弱，本卡完全没用 | 已有契约闲置 |
| 7 | **`category_label` 只当作卡头的小字，没有用来分组** | 「收益型指标 / 风险型指标 / 收益风险性价比指标」全在卡内 | 准则 6.5「分组优先用 `divide-y`」 |
| 8 | 加载态是一行文字「正在基于真实数据批量计算…」 | `ResearchIndicatorPanel` | 准则 8.1 要求骨架屏 |

一句话：**它把「行」画成了「卡」。**

---

## 2. 调研：主流的指标展示逻辑

### 2.1 消费级 dashboard 的 KPI 卡

Stripe / Linear / Vercel 一类的顶部指标条是 4-6 张卡，每张卡三件套：
**大数值 + 变化量（带箭头的 delta）+ 迷你趋势线**，再加可选的目标进度。
业界给的空间分配是 40/30/20/10：最重要的一个指标 40%，2-3 个次级 30%，趋势上下文 20%，筛选 10%。
初始视图不超过 5-6 张卡。

### 2.2 专业投研终端

Koyfin 的 Risk Statistics 把指标**分成四张表**，每张表的列是 `1Y / 3Y / 5Y / 10Y / 15Y`；
Trailing Returns 的列是 `1M / 6M / YTD` 加上年化列。Morningstar 的基金页同构。
也就是说：**行是指标，列是区间**，一屏给出的是「期限结构」，不是一个孤立的数。

### 2.3 卡还是表

- 「如果你会自然地把这批数据叫做 rows，它要的是表；叫做 items，它才要卡。」
- 卡「适合少量重点数据，易扫读，但**难比较**」；表「适合扫读与比较大量数据，帮助发现模式」。
- 「KPI 卡负责概览，表负责细节，两者是搭档不是对手。」
- 「密度是功能，不是默认值」：紧凑/舒适/宽松应当是真选项。

### 2.4 让一个数字有上下文

bullet graph 的定义要求三样东西同时存在：**实际值 + 目标/基准值 + 定性区间边界**。
只有实际值时，进度条、分位条、子弹图都退化成装饰。

### 2.5 对照本项目逐条判定

| 主流做法 | 本项目适用性 | 判定依据 |
| --- | --- | --- |
| 行=指标、列=区间的表 | **采用** | `periods.py` 有 18 个区间且带 `group` 字段；`EvaluateRequest.period` 是单值，多列 = 多次请求，无需改后端 |
| 指标分组成几张表 | **采用**（改为一张表内的分组表头） | `presentation.category_label` 已下发；准则 6.5 禁止再套一层卡 |
| 大数值 + delta + sparkline | **不采用 sparkline** | 标量指标没有滚动序列孪生体。内置时序指标全库只有 `builtin-rolling-5d-annualized-sharpe-series` 一个，其余指标画不出线。列即期限结构，已经承担了「形状」这件事 |
| 目标进度 / 分位条 / bullet graph | **不采用** | 产品记录里没有基准指数绑定，也没有同类样本；缺 §2.4 的后两样，画出来就是假上下文 |
| KPI 卡 + 表并存 | **不采用** | 规模只有 4-8 项，表本身就是概览。再加大卡等于同一份数据两种语法，那正是上一轮重构删掉的毛病 |
| 密度三档 | **不采用**（先给一档紧凑） | 没有实际的宽松需求信号，YAGNI |
| Gen-AI「智能解读」文案 | **不采用** | 准则要求数据可追溯；自动生成的结论没有口径来源 |

**Sources:**
[Koyfin Risk Statistics](https://www.koyfin.com/help/release-notes/v3-92/) ·
[Koyfin Mutual Fund Data](https://www.koyfin.com/help/mutual-fund-data/) ·
[Cards vs. Lists vs. Tables vs. Data Grids](https://smart-interface-design-patterns.com/articles/cards-vs-lists-vs-tables-vs-data-grids/) ·
[NN/g · Data Tables: Four Major User Tasks](https://www.nngroup.com/articles/data-tables/) ·
[Data table UI design reference 2026](https://www.setproduct.com/blog/data-table-ui-design) ·
[Dashboard Design Patterns 2026](https://artofstyleframe.com/blog/dashboard-design-patterns-web-apps/) ·
[Dashboard Design Best Practices 2026](https://5of10.com/articles/dashboard-design-best-practices/) ·
[Domo · What is a bullet graph](https://www.domo.com/learn/charts/bullet-graphs) ·
[Viz Advantage · Use Bullet Graphs to Provide Context](https://www.vizadvantage.com/the-benefits-of-using-a-bullet-graph.html)

> `design-taste-frontend` skill 第 13 节自述不适用于 dashboards 与 data tables，
> 准则第 11 节已登记本项目对它的显式覆盖。本设计以准则为准。

---

## 3. 设计

### 3.1 信息结构

区间从「每张卡一个下拉」上移为 **区块级的列**：

```
研究指标                                                      [在指标中心分析]
用指标中心已保存的版本计算，每个指标在所选区间内给出一个数。
──────────────────────────────────────────────────────────────────────────────
[选择指标 · 已选 4/8 ▾]   区间  [近1月][近3月][近1年 ✓][成立以来 ✓] [+]
截止日 [ 2019-12-31 ]  未填则跟随平台研究日 2026-09-14
──────────────────────────────────────────────────────────────────────────────
                                     近 1 年              成立以来
指标                          方向    2018-12-28 至        2005-06-30 至
                                     2019-12-30           2019-12-30
                                     246 个观察值          3542 个观察值
── 收益型指标 ──────────────────────────────────────────────────────────────
  累计收益率                   高优         37.51%             412.83%     [⋯]
  年化收益率                   高优         38.58%              11.27%     [⋯]
── 风险型指标 ──────────────────────────────────────────────────────────────
  年化波动率                   低优         19.75%              24.06%     [⋯]
── 收益风险性价比指标 ──────────────────────────────────────────────────────
  年化夏普比率                 高优          1.676               0.468     [⋯]
──────────────────────────────────────────────────────────────────────────────
全部按复权口径计算 · 数据截至 2026-09-03
```

一屏从 4 个数变成 8 个数，纵向占用从约 900px 降到约 320px，控件从 4 个下拉降到 1 组分段按钮。

### 3.2 层级与元素

只有两层：区块外壳（唯一卡片）→ 表。分组用 `divide-y` 的分组行，不再有第三层边框（准则 6.5）。

| 元素 | 内容 | 来源字段 |
| --- | --- | --- |
| 行首表头 `<th scope="row">` | 指标名（按钮，点开定义抽屉）+ 下一行小字「内置 v1」与生效参数 | `presentation.name` / `source` / `revision` / `result.parameters` |
| 方向列 | `高优` / `低优` / `仅展示` 文字 | `presentation.direction` |
| 数值单元格 | `text-right tabular-nums text-slate-900`，异常时数值位置显示 `—` 并加状态徽标 | `result.value` + `formatMetricValue(presentation)` |
| 列头 `<th scope="col">` | 区间中文名 + 该列共同窗口 + 观察值数 | `indicatorPeriodLabel(period)` + `result.window` |
| 行尾 `[⋯]` | 展开行：参数输入、不可计算原因、移除 | `parameter_schema` / `MetricUnavailableReason` |
| 表尾 | 口径声明 + `data_latest_date` | `result.window.data_latest_date` |

**列头口径的判定规则（确定性，不靠猜）**：该列所有单元格的 `window.start_date/end_date/observation_count`
完全一致时，列头显示这份窗口；只要有一行不同，列头只显示区间名，差异留在该行的展开区里说明。
理由：同一区间下不同指标的 `minimum_observations` 和输入可得性不同，窗口**可能**不同，
把众数写进列头会骗人。

### 3.3 颜色与状态

- **数值不着色**，统一 `text-slate-900`。`accent` 在本项目的语义是链接与选中（准则 3.1 表），
  现在的全蓝数字是误用。
- 颜色只承载状态：`正常` 不画徽标，`有警告 / 样本不足 / 不可计算` 用 `amber`，`计算失败` 用 `rose`，
  沿用既有 `MetricStatus` 的 `statusCopy`，不新增色相。
- **不做跨列的最佳/最弱高亮。** `MetricMatrix` 的绿/红是同一指标跨产品比较，成立；
  近 1 月收益和近 1 年收益是不同期限的数，比大小没有意义。
- 方向用文字（`高优` / `低优`），不用箭头单独承载（准则 10.3）。

### 3.4 四态

| 状态 | 做法 |
| --- | --- |
| 加载 | 骨架行：行数 = 已选指标数，列数 = 已选区间数，灰条宽度固定，布局不跳（准则 8.1；当前全仓 `Skeleton` 0 处，这里补上第一处） |
| 空 | 未选指标时一段说明 + 两条出路（打开选择器 / 去指标中心），沿用现有文案 |
| 错误 | 整块失败就近 `role="alert"`；单格失败降级为单元格徽标，不牵连同列其它指标 |
| 禁用 | 计算中参数输入禁用并说明原因 |

### 3.5 响应式

- `>= 1024px`：完整表格。
- `< 1024px`：表外 `overflow-x-auto`，指标名列 `sticky left-0 bg-white`，
  与 `MetricMatrix` 已有做法一致（复用，不新写一套）。
- `< 640px`：区间分段控件换行为两行，`min-w-[560px]` 保证列不被压扁；页面本身不横向滚动。

### 3.6 请求模型

后端不改。列数就是请求数：

```ts
Promise.all(periodColumns.map((period) => evaluateCustomIndicators({
  indicator_refs: selected.map(...),   // 与现在完全相同，含 revision 与参数
  targets: [{ kind, product_id }],
  period,
  as_of,
})))
```

- 结果键从 `indicator_id` 改为 `${indicator_id}:${result.period}`。`EvaluationResult.period` 已随结果下发。
- **上限钉死**：区间列 ≤ 6，指标 ≤ 8（`MAX_RESEARCH_INDICATORS`，后端 `indicator_refs` 上限是 10），
  最坏 6 次请求 × 8 指标。
- **增量请求（二期）**：按 `${indicatorId}:${period}:${parameterHash}:${asOf}` 缓存，加一列只发一列。
  一期仍是列变化后重发全部列，靠后端快照兜住重复。
- 后端的快照命中键已含 `parameter_hash`，重复列命中缓存，不会重算。

### 3.7 组件改动

| 文件 | 改动 |
| --- | --- |
| `components/metrics/MetricPeriodTable.tsx` | **新增**。行=指标、列=区间的表，含分组表头、展开行、骨架行 |
| `components/metrics/MetricDisplay.tsx` | **删除 `MetricResultCard`**。它的唯一调用方是本区块（已核对：仅 `ResearchIndicatorPanel` 与其自身单测引用），替换后必须一并删除（AGENTS.md 存量代码治理） |
| `components/metrics/ResearchIndicatorPanel.tsx` | 顶部工具条加区间分段控件；主体换成 `MetricPeriodTable` |
| `components/metrics/useMetricDisplayPreference.ts` | **新增** `useMetricPeriodColumns`，独立存储键 `indicator-period-columns:v1:{page}:{kind}`。原 `useMetricDisplayPreference` 与 `periodsByIndicator` **不动**：`ProductCompare` / `ProductResearch` / `HoldingDiagnosis` 仍按「每个指标一个区间」工作，把两种形态塞进同一个存储结构会给三个页面塞一个它们永远不读的字段 |
| `pages/ProductDetail.tsx` | 请求循环从「按指标分组的区间」改为「按列的区间」；`researchResults` 按 `id:period` 存放 |
| `components/metrics/MetricDisplay.test.tsx` | 删除 `MetricResultCard` 用例，新增表格用例 |

`MetricMatrix` **不动**。它的列是产品、且带最佳/最弱语义，和本表只是外形像；
现在就抽公共壳会得到一个带五个开关的壳。出现第三个调用方时再抽。

### 3.8 需要确认的取舍

现在每个指标可以各自选一个区间（`periodsByIndicator`）。改成共享列之后，
**所有指标共用同一组列**，不能再出现「累计收益率看 1Y、波动率看 3Y」这种组合。

- 信息量是净增的：原来一个指标一个数，现在一个指标一行数。
- 真正丢掉的只有「我只想看这两个特定配对」。
- 迁移：首次读取新键时若为空，回落读旧 `periodsByIndicator` 的**去重值集合**作为初始列。
  全部卡都在 1Y 的用户，升级后看到一列 1Y，行为不变。
- 影响范围只有产品详情页；对比页与诊断页的每指标区间不受影响。

### 3.9 遵循的准则条款

- 3.1：只用 `accent` / `slate` / `emerald` / `amber` / `rose`；焦点环统一 `focus-visible:ring-2 focus-visible:ring-accent-500`。
- 3.3 / 3.4：最小 `text-xs`；卡片 `rounded-xl`，控件 `rounded-lg`，徽标 `rounded-full`，不新增第四档圆角。
- 5.5：分段按钮 `min-h-10`，日期输入 `min-h-11`。
- 6.5：分组用 `divide-y`，不加第三层卡片。
- 6.6：数字列 `text-right tabular-nums`；`<thead>` 内 `scope="col"`、行首 `scope="row"`；表格带 `aria-label`；行间只用一条 `border-b`。
- 6.9：三个断点的坍缩显式声明（§3.5）。
- 第 4 节棘轮：不新增 hex 字面量、不新增按钮色相、不新增焦点环色相、`<th>` 全部带 `scope`，
  `unscoped-tables`、`tiny-font`、`card-radius-variants` 预算不受影响。

---

## 4. 明确不做

- **不做 sparkline。** 标量指标没有滚动序列契约，全库内置时序指标只有一个。要做得先在指标中心建配对的时序指标，那是另一个需求。
- **不做基准/同类分位条、子弹图。** 产品记录里没有基准指数绑定。真要做的话后端已具备条件（`EvaluateRequest.targets` 上限 50，可同时算产品与基准），缺的是「产品 → 基准」这条契约，属新需求。
- **不做置顶大 KPI 卡。** 见 §2.5。
- **不做跨区间高亮、不做排名。**
- **不加新接口、不改后端。**
- **不做深色模式**（准则列为 P2，须先完成令牌收敛）。
- **不做密度切换、不做列拖拽排序。**

---

## 5. 分期与验证

| 期 | 范围 | 判据 |
| --- | --- | --- |
| 一期 | 工具条区间分段控件 + 单列表格 + 删除 `MetricResultCard` | 一屏容纳 8 个指标；重复控件从 N 个降到 1 个 |
| 二期 | 多区间列 + 增量请求缓存 + 骨架行 | 加一列只发一次请求；列口径判定按 §3.2 |

验证：

- 前端单测：`MetricDisplay.test.tsx`、`ProductDetail.test.tsx`（新增列增删、缓存不重发、列头口径一致/不一致两条分支、`scope` 与 `tabular-nums`）。
- 端到端：`metric-display.spec.ts`、`contrast.spec.ts`。
- 设计检查：`npm run design:check --prefix frontend` 无回归；`node scripts/check_i18n.mjs` errors 为空。
- 浏览器验收：1440 / 1024 / 768 / 320 四个宽度，加载/空/错误/禁用四态，键盘可达表格与展开行。
- 列内指标窗口不一致时，单元格必须展示各自起止日期和观察值数，不能仅以“口径随指标不同”替代可追溯日期。浏览器回归使用区间列的添加/移除控件及指标名称入口，不再定位已删除的卡片控件。
- 不可计算诊断仅在状态、警告、输入需求和数据日期说明完全相同时跨列合并；不同区间的样本不足或输入缺失原因分别展示，并标明对应区间。

2026-09-17 PR #24 集成复核：前端 147 文件 / 1148 用例通过，构建、TypeScript、design:check、i18n 检查通过。更新后的 `metric-display.spec.ts` 在 320/768/1024/1440 四档通过，覆盖多区间添加/移除、最后一列禁用删除、定义抽屉、页面横向溢出和文字对比度；使用离线 API 夹具，不代表生产数据验收。列内窗口不同、跨区间诊断不同的展示由组件回归覆盖。
- 路由记忆：`python3 skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --diff-range <range>`。
