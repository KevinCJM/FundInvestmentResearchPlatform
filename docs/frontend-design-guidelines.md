# 前端设计准则

本文是本项目前端视觉与交互的规范来源。机械可查的部分由 `scripts/check_frontend_design.mjs` 执行（`npm run design:check --prefix frontend`），共享令牌由 `frontend/tailwind.config.js` 承载。三者必须一致：改规则要同时改检查器，改检查器要同时改本文。

方法论参照 `design-taste-frontend` skill。该 skill 把 dashboards、dense product UI、data tables 明确列为适用范围之外，而本项目 122 条路由中有 121 条正是这个形态，因此第 11 节逐条记录了本项目对它的显式覆盖及理由。未记录的覆盖不成立。

## 1. Design Read 与适用范围

- **工作台（121 条路由）**：内部机构级投研工作台，面向量化研究员与投资运营人员，语言是可追溯优先、克制。
- **首页（`/` 一条路由）**：产品落地页，面向内部用户与演示场景。

| 表面 | 适用章节 | 不适用 |
| --- | --- | --- |
| 工作台 | 第 3-6、8-10 节 | 落地页构图规则（hero 栈、eyebrow 配额、section layout family） |
| 首页 | 全部，含第 7 节 | 第 6 节的表格与数字规则 |

两个表面共用第 3 节令牌层。这是它们唯一必须共享的东西；构图规则不互相套用。

## 2. 三个 Dial

| Dial | 工作台 | 首页 | 直接后果 |
| --- | --- | --- | --- |
| `DESIGN_VARIANCE` | 3 | 6 | 工作台用对称栅格、等距留白；不做非对称构图、不做负 margin 叠压 |
| `MOTION_INTENSITY` | 2 | 3 | **不引入 Motion / GSAP / Three.js**。只允许 CSS `:hover` / `:active` / `transition`，单条 ≤ 200ms |
| `VISUAL_DENSITY` | 7 | 4 | 工作台所有数字必须 `tabular-nums`；分组优先用 `divide-y` 与留白，而不是再套一层卡片 |

`MOTION_INTENSITY: 2` 是有意压低的：本项目的动画预算应该花在 ECharts 重绘与大表滚动上，不是页面装饰。当前 `package.json` 无任何动画库依赖，保持这个状态。

## 3. 令牌层（唯一真相源）

所有跨页面共享的值只能来自 `frontend/tailwind.config.js`。页面内不得再出现同义的字面量。

### 3.1 颜色

`accent` 是围绕首页 `--home-blue #1662f5` 展开的完整色阶（50-950），定义在 `frontend/tailwind.config.js`。

| 语义 | 令牌 | 用途 |
| --- | --- | --- |
| 主操作 | `bg-accent-600` / `hover:bg-accent-700` | 提交、保存、运行、发布；与首页 `.home-button-blue` 同色 |
| 链接与选中 | `text-accent-700` / `bg-accent-50` | 链接、选中态、当前步骤、图标底 |
| 成功 | `emerald-*` | 已实现、校验通过、运行成功 |
| 警告 | `amber-*` | 部分实现、口径提醒、待确认 |
| 危险 | `rose-*` | 删除、冲销、强制覆盖、请求失败 |
| 中性 | `slate-*` | 唯一中性色阶 |

规则：

1. **全站唯一强调色是 `accent`。** 阶段身份靠色调（下方 3.1.1）与序号表达，不靠换强调色。
2. **`gray-*` 禁用。** 中性色只用 `slate-*`。同一文件混用两套色阶是回归项（`neutral-ramp-mix`）。
3. **`indigo` / `violet` / `purple` / `fuchsia` / `sky` / `cyan` / `teal` / `blue` 一律不得直接出现在类名里。** 装饰性蓝紫是 skill 点名的 AI 默认色，全部走 `accent-*`。例外为 3.1.1 的阶段身份色和第 10 条有明确分类语义的图表色。
4. **实心按钮底色只允许 5 种语义色相**：`accent`（主操作）、`slate`（中性）、`emerald`（成功）、`amber`（警告）、`rose`（危险）。金融界面需要成功与警告两种状态底色，压到三种会把状态挤回纯文字。
5. **焦点环全站一种**：`focus-visible:ring-2 focus-visible:ring-accent-500`。未被 Tailwind 覆盖的元素由 `index.css` 的全局 `:focus-visible` 兜底。
6. **页面级背景只能声明一处**：`App.tsx` 根 div 的 `bg-slate-100`。
7. **次要文字分浅底、深底两套令牌，不能只有一套。**
   - **浅色表面（白 / `slate-50` / `slate-100` / `accent-50`）用 `text-slate-600`，不用 `slate-500`。** `slate-500` 在纯白上是 4.76:1 过线，但落在 `slate-50` / `slate-100` / `accent-50` 上只有 4.3-4.5，而调用处根本看不出自己压在哪层底上。一个只在三种底色中的一种上安全的令牌不是令牌。`slate-600` 在三种底色上分别是 7.58 / 7.24 / 7.0。
   - **深色表面（`slate-800` 及更深、深色渐变横幅）用 `text-slate-200`。** 在 `slate-900` 上 14.48:1，在 `slate-950` 上 16.36:1，在 `accent-950` 上 12.13:1。
   - 只写"次要文字一律 slate-600"会把横幅里的说明文字压到 1.8-2.4:1。这条正是 2026-09-12 复核查出的回归：换色脚本按行判断深浅底，而底色写在父元素那一行。**改次要文字颜色前先确认所在表面**，判断不了就跑 `frontend/e2e/contrast.spec.ts`。
8. **文字不叠透明度，深底浅底都一样。** 深色文字：`text-amber-700/70` 压在 `bg-amber-50/60` 上只有 2.89:1，两个透明度分开写，谁也算不出结果。浅色文字同样不行：`text-white/90` 压在 `accent-600` 上 4.45:1、压在 `orange-700` 上 4.48:1，都差一点到 4.5；`bg-white/10` 还会把底色提亮，`text-white/75` 压上去只剩 3.76:1。要弱化层级用字号和字重，不要用透明度。
9. **占位符按正文门槛处理。** Tailwind preflight 的 `input::placeholder` 默认 `gray-400`，白底 2.54:1。`index.css` 全局改为 `slate-500`（白底 4.76:1）。占位符是输入格式提示，不是装饰。注意 preflight 的选择器是 `input::placeholder, textarea::placeholder`，盖它需要同等特异性。
10. **分类色不是界面色。** 图表、堆叠条、图例里的颜色承载"是哪一类"，必须单独定义（如 `app/portfolioSolutionDemoData.ts` 的 `ASSET_CLASS_COLORS`），不得跟着强调色一起改，同一组里也不得有两类同色。另外，**分段边界不能只靠色差**：堆叠条要有分隔线，图例要有色块。

### 3.1.1 阶段色调

九个阶段的色调只在 `app/processRegistry.ts` 的 `TONE_CLASSES` 定义一次，首页 `homepage/catalog.ts` 从这里读取，两处不再各写一份。

| 阶段 | tone | 类名 |
| --- | --- | --- |
| 产品研究 / 组合中心 / 方案展示 / 设置 | `blue` | `bg-accent-50` `text-accent-700` |
| 投前决策 / 基金会计 | `teal` | `bg-teal-50` `text-teal-700` |
| 投中执行 | `orange` | `bg-orange-50` `text-orange-700` |
| 投后管理 | `purple` | `bg-violet-50` `text-violet-700` |
| 反馈与迭代 | `pink` | `bg-pink-50` `text-pink-700` |

1. **色调只标身份，不标交互。** 它只能出现在图标底、区块眉标和阶段工作台横幅上。链接、按钮、焦点环一律 `accent`。
2. **徽章保持中性 `bg-slate-100`。** 原来九个阶段各有一种徽章与主按钮颜色，颜色因此不再表示「这是主操作」。
3. **不得把阶段色调选成 `emerald` / `amber` / `rose`。** 那会让「投后管理」的标签和「运行成功」的状态长得一样。这正是改造前的实际状态。
4. 文字色一律取 700 档：五种色调在白底上都不低于 4.7:1。

### 3.2 字体

```
font-sans: Avenir Next, -apple-system, BlinkMacSystemFont, Segoe UI,
           PingFang SC, Microsoft YaHei, Noto Sans SC, sans-serif
```

1. **不自托管 webfont。** 一份 CJK 字体 3-8MB，直接毁掉 LCP 预算。中文用系统字体栈是本项目的正确解，不是妥协。
2. **不用衬线体做显示字体。** 投研工作台没有编辑/奢侈品语境，标题用同一 sans 的更重字重。
3. **字重只用 400 / 500 / 600 / 700。** 系统 CJK 字体不是可变字体，`450 / 550 / 650 / 710 / 750 / 760` 会被浏览器吸附到相邻档位，写了也不生效，只会让代码说假话。
4. **强调词用同族的 italic 或 bold，不混排第二种字族。**

### 3.3 字号

| 用途 | 类 | 尺寸 |
| --- | --- | --- |
| 页面标题 | `text-2xl sm:text-3xl font-bold` | 24 / 30px |
| 区块标题 | `text-lg font-semibold` | 18px |
| 正文与表格 | `text-sm` | 14px |
| 辅助说明、表头、徽标 | `text-xs` | 12px |

**12px 是硬下限。** 任何 `text-[9px]` / `[10px]` / `[11px]` 都是回归项（`tiny-font`）。中文在 12px 以下没有可靠的屏显字形，Windows 上尤其糊。首页作用域 CSS 同样受此约束（`homepage-subpixel-font`），`5px / 6px / 7.3px / 8px / 9px / 10px / 10.5px` 全部要抬到 ≥ 12px。

### 3.4 圆角与阴影

- **卡片一律 `rounded-xl`（12px）。** 改造前 `rounded-2xl` / `xl` / `lg` / `md` / `full` 五档混用，现已收敛到一档。
- 交互控件 `rounded-lg`，胶囊徽标 `rounded-full`。这是文档化的三档规则，不是自由混用。
- **阴影只用 `shadow-sm`。** 悬浮层（下拉、弹窗）用 `shadow-xl`。中间档不用。不用纯黑投影。

### 3.5 共用原语

`frontend/src/components/ui.tsx`。新代码优先用这里的组件，不要再复制一串 className。

| 组件 | 用途 | 约束 |
| --- | --- | --- |
| `Card` | 卡片外壳 | 圆角、描边、底色不接受调用方覆盖 |
| `Button` | 按钮 | `tone` 只有 `primary` / `secondary` / `danger`；`min-h-10` 是触控下限 |
| `Badge` | 状态徽章 | `tone` 只有 `neutral` / `success` / `warning` / `danger`；**颜色只表示状态，分类靠文字** |
| `SectionHeader` | 区块标题 | 眉标不加 `uppercase` |
| `EmptyState` | 空态 | 必须同时说明「为什么空」和「下一步做什么」；默认带吉祥物 |

1. **只写「暂无数据」不算空态。** 空态要回答用户下一步该做什么。
2. **`EmptyState` 默认渲染吉祥物，一屏只能有一个。** 页面的主空态用它；局部面板的空态传 `mascot={false}` 或直接用纯文字。禁区见第 15.2 节，`mascot-in-forbidden-zone` 会连同 `EmptyState` 的间接引用一起拦截。
3. 旧页面顺手替换，不做大爆炸重写。334 处卡片外壳靠复制粘贴维持一致性，必然漂移。

## 4. 硬规则与棘轮阈值

`npm run design:check --prefix frontend` 的每条规则都是机械计数。语义是棘轮：`budget` 是基线实测值，只允许下调；`current > budget` 即退出码 1。`target` 是本准则要求的终值。

| 规则 | 含义 | 首测 | 现值 = budget | target |
| --- | --- | --- | --- | --- |
| `tiny-font` | 换算后小于 12px 的字号 | 299 | **0** | 0 |
| `low-contrast-text` | 浅色底上的 `text-slate/gray-300/400` | 224 | **0** | 0 |
| `neutral-ramp-mix` | 同时含 slate 与 gray 的文件数 | 5 | **0** | 0 |
| `card-radius-variants` | 卡片外壳圆角档数 | 5 | **1** | 1 |
| `focus-ring-hues` | 焦点环色相数 | 10 | **1** | 1 |
| `solid-button-hues` | 实心按钮底色相数 | 13 | **5** | 5 |
| `page-bg-declarations` | 页面级背景声明处数 | 2 | **1** | 1 |
| `english-em-dash` | 英文界面文案里的破折号 | 56 | **0** | 0 |
| `uppercase-on-cjk` | 作用在中文上的 `uppercase` 行数 | 48 | **11** | 0 |
| `homepage-subpixel-font` | 首页 CSS 中 < 12px 的字号数 | 27 | **0** | 0 |
| `homepage-nonstandard-weight` | 首页 CSS 中非 400/500/600/700 的字重数 | 13 | **0** | 0 |
| `unscoped-tables` | 没有 `scope` 的 `<th>` 个数 | 439 | **0** | 0 |
| `bare-chart-hex` | 散落的十六进制色字面量种数 | 57 | 57 | 0 |
| `homepage-accent-variants` | 首页饱和蓝变体数（令牌 3 档） | 3 | 3 | 3 |
| `mascot-in-forbidden-zone` | 吉祥物出现在禁区文件中（含 `EmptyState` 间接引用） | 0 | 0 | 0 |
| `same-element-contrast` | 同一元素/同一三元分支内底色与文字色不足 4.5:1 | 37 | **0** | 0 |
| `duplicate-category-color` | 同一数组里重复的分类色 | 3 | **0** | 0 |

以下预算保留明确边界，`target` 与预算的含义不能混淆：

- `solid-button-hues` **5**：见 3.1 第 4 条。
- `uppercase-on-cjk` 预算 **11**、目标 **0**：剩下的是英文眉标（`Data governance`、`Immutable run` 等），只因同一行上另有中文才被计入，属规则的行级近似。

**静态检查够不到的部分由浏览器量。** 祖先的底色常写在三元里（`selected ? 'bg-accent-700' : 'bg-accent-50'`），
纯文本扫描无法判定子元素到底压在哪一层上；渐变底色写在 `background-image` 上，连 `backgroundColor` 都读不到。
`frontend/e2e/helpers/contrast.ts` 在渲染结果上估算纯色、渐变色标、文本与祖先分组透明度的对比度，并检查可见的表单值与占位符。
`contrast.spec.ts` 从 `processRegistry` 派生阶段与节点路由，严格核对最终路径，使用固定的 API 离线错误态检查外壳。
`indicator-studio.spec.ts` 使用已加载的固定夹具覆盖指标库、编辑和预览切换；历史情景工作台也验证模板载入后的文字。
可见的 `aria-hidden` 信息照常检查；失效控件、完全透明或未显示的文字不计。颜色过渡结束后再断言稳定状态。
算法自检包含可见 `aria-hidden`、半透明深色组、已填写输入框和占位符，不能用“用例通过”代替完整无障碍验收。
图片、视频、滤镜、遮挡和未覆盖的交互状态仍需人工验证。

阈值只降不升。需要上调时说明理由并同步本表，不得静默改脚本。

## 5. 颜色与对比度

WCAG AA：正文 ≥ 4.5:1，大号文字（≥ 18.66px 且加粗，或 ≥ 24px）≥ 3:1。本项目背景有 `#ffffff` / `slate-50` / `slate-100` 三层，实测：

| token | 白底 | slate-50 | slate-100 |
| --- | --- | --- | --- |
| `slate-500` `#64748b` | 4.76 ✅ | 4.55 ✅ | **4.34 ❌** |
| `slate-400` `#94a3b8` | **2.56 ❌** | 2.45 ❌ | 2.33 ❌ |
| `slate-300` `#cbd5e1` | **1.48 ❌** | 1.42 ❌ | 1.35 ❌ |
| `slate-600` `#475569` | 7.58 ✅ | 7.24 ✅ | 6.92 ✅ |

规则：

1. **`slate-300` / `slate-400` 只能用于浅色底上的边框与分隔线。** 压在深色面板上时它们是正确的文字色，那是另一回事。
2. **文字色必须与表面一起选择。** 浅底次要文字用 `slate-600`，深底用 `slate-200`；不能全局统一压深。占位符遵守相同的表面规则。
3. 辅助文字的字号下限是 `text-xs`；颜色按浅底 `slate-600`、深底 `slate-200` 选择。改造前有 120 处「≤ 11px + slate-300/400/500」的叠加，是最直接的可读性缺陷，现已清零。
4. **按钮文字必须与按钮底色对比达标。** 透明按钮压在图片或深色面板上时必须带描边或遮罩。
5. **每个按钮的有效点击区 ≥ 40px 高**（`min-h-10`）。当前 `min-h-8`（12 处）不合格。

## 6. 工作台构图

1. **任务页优先展示当前任务。** `StageLayout` 仅在阶段总览显示阶段大标题、眉标和描述；子页面保留紧凑面包屑、当前上下文和必要的配置旅程。因子参考证据默认收起，业务口径与错误状态仍由工作区直接提示。
2. **阶段内节点导航响应式展示。** ≥1280px 使用常驻左侧栏；窄屏用展开按钮，Escape 关闭并把焦点交回按钮，切换路由后收起。历史情景全幅画布保留独立布局。指标与历史情景工作台在 <1280px 通过区域页签切换，避免平板双栏内再挤三列输入。**待实施**：当前 1440px 仍是下拉，见第 13 节；这是要求，不是已验收状态。
3. **一个页面一个滚动容器。** 不做嵌套滚动区域。
4. **容器宽度 `max-w-[1600px]`**，画布类全幅页面除外。首页用 1280/1440，两者不互相套用。
5. **分组优先用 `divide-y` 与留白。** 只有当层级真的需要抬升时才用卡片。`VISUAL_DENSITY: 7` 下卡片套卡片是禁止的。
6. **表格**：数字列 `text-right tabular-nums`，`<thead>` 内表头 `scope="col"`、`<tbody>` 内行首表头 `scope="row"`，`<caption>` 或 `aria-label` 必填；只在行间用一条 `border-b`，不要同时加 `border-t`。`index.css` 已对 `td` / `th` 全局开启 `tabular-nums`。
7. **中文标签不用 `uppercase`。** 对中文是空操作，只留下被拉开的字距。eyebrow 类微标签在工作台每页最多 1 个。
8. **超过 5 项的列表换组件**，不要加长 `<ul>`：两列分组、卡片栅格、标签页、横向 scroll-snap 任选。
9. **响应式坍缩逐区块显式声明**，不依赖「Tailwind 会处理」。

## 7. 首页构图（仅 `/`）

按 skill 第 14 节 Pre-Flight 对当前实现的核对结果：

**通过**：eyebrow 2 个 / 5 区块（配额 2）；使用真实 WebP 图像而非 div 假截图；`prefers-reduced-motion: reduce` 分支完整（`homepage.css:277`）；hero 用 `min-height` 而非 `100vh`；hero 图 `fetchpriority=high` + `srcset`，地球图 `loading=lazy`；`useEffect` 键盘监听有清理；搜索空态 / 历史空态 / 存储失败三态齐备；`role="tablist"` 带方向键导航；无滚动提示；无城市/天气/时间装饰条；无装饰状态点；无 marquee；导航单行且 78px ≤ 80px；hero 标题 2 行、`padding-top: 72px` < `pt-24`、CTA 首屏可见；无 AI 紫渐变；深色数据面板属于 skill 允许的「每页一次的 Color Block」。

**不通过，需修**（按修复收益排序）：

1. ~~**`home-button-blue` 白字对比度 4.43:1，不合格。**~~ **已修**：改用 `--home-blue: #1662f5`，白字 5.12:1；hover 改为更深的 `--home-blue-hover: #0f52d6`，6.57:1（hover 提升对比度，方向正确）。
2. ~~**同一强调色有 10 种取值。**~~ **已修**：`#1263f7 / #216ffc / #286bf6 / #2374ff / #1966fa / #367bff / #3b83ff / #8eaded` 全部改为 `var(--home-blue*)`，16 处引用。最终 3 档饱和蓝：`--home-blue: #1662f5`（链接、实心按钮、焦点、选中）、`--home-blue-hover: #0f52d6`、`--home-blue-on-dark: #65a4ff`（深色面板上 7.48:1），另加非饱和的 `--home-blue-soft: #e5efff`。由 `homepage-accent-variants` 规则锁定。
3. ~~**圆角 10 档。**~~ **已修**：收敛为三档 —— 卡片/面板/弹窗 `12px`（8 处）、控件/输入/列表项/徽标 `8px`（11 处）、圆形 `50%`（2 处）。
4. ~~**两个 CTA 同一意图两种文案。**~~ **已修**：两处统一为 `landing.tourButton`「查看平台导览」，`landing.exploreWorkflow` 键已删除。skill 允许同一文案出现在多个位置（nav / hero / footer），禁止的是同一意图两种说法。`ProcessHome.test.tsx` 已加断言锁定。
5. **Hero 文本元素 6 个，上限 4 个。** eyebrow + 标题 + 描述 + 2 个 CTA + 4 项 `home-facts` 数据条 + `home-mascot-caption` 角注。数据条下移为独立区块，角注删除。
6. **`.home-mascot-caption` 删除。** 文案「认真研究，也保持好奇。」同时命中两条 AI-tell：图片上的装饰性图注、表演式匠人腔标签。
7. **同族 layout 重复 3 次。** 5 列卡片栅格 → 4 列卡片栅格 → 5 列卡片栅格。同一 layout family 每页只能出现一次。
8. **「核心能力」4 张卡全部 `background: #fff` 且只有文字。** Bento Background Diversity 要求多格栅格里至少 2-3 格有真实视觉变化。这 4 格是全页最适合放吉祥物姿势图的位置（见第 15 节）。
9. **引用署名不合格。** 当前只有品牌名 `app.title`，应为「姓名 + 角色」，或删掉该引用区块。自我引用对内部工具没有说服力。
10. **已删除导航里不表达真实状态的产品标签。** 保留搜索、语言与工作区入口；英文导航须验证 1250/1251px 和 1400/1401px 两侧，不能靠缩小字号或隐藏页面溢出解决。
11. **`landing.workflowNote` 位置错误。** 内容是实质的（产品研究持续运行、会计贯穿投中投后），但它作为栅格下方的脚注是「eyebrow 下的微型元句」形态。移到 `h2` 下方的区块描述里。
12. **已完成**：首页原有 27 条 <12px 字号声明、13 条非标准字重已清零，对应预算降为 0。
13. **无 `prefers-color-scheme` 分支**，按第 9 节第 4 条（P2）处理。`prefers-reduced-motion` 已有，不需补。
14. **两段相邻区块用同一修辞句式。**「把工具交给平台，把精力留给研究」与「少一点寻找，多一点专注」。一页一种文案语域，改掉其中一句。

首页的数值是从 2994×4198 的 2 倍参考图逐像素还原、按一半读数得到的（见 `design-qa.md`），`7.3px`、`10.5px` 是这个过程的产物。**它是一次测量结果，不是设计系统**，其具体数值不得外推到任何其他页面。

## 8. 交互状态

每个数据区块必须四态齐备。当前 `Skeleton` 0 处、`animate-pulse` 7 处、`aria-live` 仅 21 个文件覆盖，缺口明确。

1. **加载态**：骨架屏，形状与最终布局一致。不用居中转圈。
2. **空态**：说明为什么空，并给出填充它的入口。不伪造记录。
3. **错误态**：表单内联在字段下方；请求失败就近展示并提供重试。不要只 toast 了就算完。
4. **禁用态**：说明为什么禁用。当前 1021 处 `disabled` 中大部分没有解释。
5. **按下反馈**：`active:translate-y-px`。不用缩放。
6. **表单**：label 在输入框上方，帮助文字可选但要在 DOM 里，错误文字在下方。**禁止用 placeholder 当 label。**

## 9. 可达性与性能

1. **焦点可见**：全站 `focus-visible:ring-2 focus-visible:ring-accent`，不得 `outline: none` 而不补替代。
2. **表格语义**：`<th scope="col">` 必填。`unscoped-tables` 逐个 `<th>` 计数（按文件计数会让同一张表里漏标的表头永远查不出来），当前 0 处。
3. **`prefers-reduced-motion`**：`MOTION_INTENSITY` 虽为 2，所有 `transition` 与 `transform` 仍需在 reduce 下退化为静态。
4. **深色模式**：`darkMode: 'class'` 已在配置中声明。**必须先完成第 3 节令牌收敛再实现**，否则 9672 处 `className` 要改两遍。这是 P2，不是 P0。
5. **只动画 `transform` 与 `opacity`。** 不动画 `width` / `height` / `top` / `left`。
6. **禁止 `window.addEventListener('scroll')`。** 用 IntersectionObserver 或 CSS 滚动驱动动画。
7. **z-index 只用于系统层**（sticky 顶栏、弹窗、遮罩），在常量里集中声明，不散写 `z-50`。
8. LCP < 2.5s、INP < 200ms、CLS < 0.1。首屏图片显式声明尺寸。

## 10. 图表（ECharts）

1. **单一主题**。用一次 `echarts.registerTheme` 定义调色板、字号、网格线、tooltip，页面不再传 `color: [...]`。当前 57 种散落的十六进制字面量收敛到这里。
2. 调色板从 `accent` 与 slate 派生。不使用高饱和霓虹色。
3. **涨跌色不靠红绿单独承载语义**，必须同时有符号或箭头（色盲可读）。
4. 坐标轴与图例字号跟随 3.3 的 12px 下限。
5. **不用 div 拼柱状图。** `PrototypeWorkspace` 的 `SampleChart` 当前用 `<div>` 高度模拟柱图，ECharts 已是既有依赖，直接用它；34 条原型路由共享这个组件，改一处即可。

## 11. 本项目对 skill 的显式覆盖

| skill 规则 | 本项目做法 | 理由 |
| --- | --- | --- |
| S13 适用范围 | 121/122 路由只取排版、颜色、交互状态、可达性、性能章节 | skill 自述不适用于 dense product UI |
| S2.A 选用官方设计系统 | 不引入 Carbon / Fluent / Radix Themes | 已有 9672 处 Tailwind 类、108 个业务组件；换基座等于重写。skill 的诚实原则反对「引入 token 再覆盖 90%」 |
| S3.A Next.js + RSC | React 18 + Vite 5 + react-router 7 | 既有栈，后端静态托管 `frontend/dist` |
| S3.A Tailwind v4 | Tailwind v3.4.4 | 既有依赖，升级与本准则无关 |
| S3.A 自托管 webfont | 系统 CJK 字体栈 | 一份 CJK webfont 3-8MB，与 LCP 预算冲突 |
| S3.C 图标用 Phosphor/HugeIcons/Radix/Tabler | 继续用 `@heroicons/react` | 已安装且是单一家族，满足 skill 的「一个项目一个图标家族」；替换 100+ 处调用是纯churn |
| S4.1 显示字体 | 同族 sans 的重字重 | 投研工作台无编辑/奢侈品语境 |
| S9.G 破折号零容忍 | **只作用于英文文案**；中文 `——` 与表格空值 `'—'` 合法 | `——` 是 GB/T 15834 规定的中文标点；`'—'` 是数据表空值的行业惯例（194 处）。skill 的禁令针对英文 LLM 文风，机械照搬会破坏中文排版与数据语义 |
| S6.C 深色模式 mandatory | 列为 P2 目标 | 本项目是内部工具而非 consumer-facing；但研究员长时间盯屏，深色是真实需求，不是可以永久跳过的项 |
| S4.7 eyebrow 配额 | 首页按 skill 执行；工作台改为「每页 ≤ 1 个且不作用于中文」 | 工作台不是落地页，但 48 处中文 `uppercase` 确实无效 |
| S4.2 一页一个强调色 | 强调色全站唯一是 `accent`；另有九个阶段色调只作用于图标底与眉标 | skill 的 COLOR CONSISTENCY LOCK 约束的是「一页之内」。九个阶段是九个页面组，页内仍然只有一种色调，跨页的色调差异是导航所需的身份信号，首页 `home-tone-*` 本来就是这么做的 |
| S9.G 破折号计数口径 | 区间分隔符（`{start} — {end}`）与代码注释不计入 | 连接号是区间的标准写法，用连字符反而与 ISO 日期里的连字符混淆；代码注释不是界面文案。按这个口径重测，全仓英文文案的破折号是 0 |

## 12. 新增或修改页面的检查清单

提 PR 前逐条确认，任何一条不通过就是没做完：

- [ ] 颜色、字体、圆角全部来自第 3 节令牌，无同义字面量
- [ ] 最小字号 `text-xs`（12px），无 `text-[Npx]`
- [ ] 次要文字按表面选择：浅底 `slate-600`、深底 `slate-200`，文字不叠加透明度
- [ ] 主操作用 `bg-accent-600`，状态色只用 emerald / amber / rose，无第六种色相
- [ ] 卡片、按钮、徽章、区块标题、空态优先用 `components/ui.tsx` 的原语，不复制 className 串
- [ ] 焦点环是 `focus-visible:ring-2 focus-visible:ring-accent-500`
- [ ] 卡片 `rounded-xl`、控件 `rounded-lg`、徽标 `rounded-full`，无第四档
- [ ] 加载 / 空 / 错误 / 禁用四态齐备，禁用态说明原因
- [ ] 每个 `<th>` 都有 `scope`（`thead` 内 col，`tbody` 内 row），数字列 `text-right tabular-nums`
- [ ] 中文标签无 `uppercase`；eyebrow 每页 ≤ 1
- [ ] 图表走统一主题，无内联 `color: [...]`；分类色一组之内不重复，分段有分隔线、图例有色块
- [ ] 响应式坍缩显式声明，320px 无横向溢出
- [ ] 英文文案无破折号；中文 `——` 与表格 `'—'` 不受限
- [ ] `npm run design:check --prefix frontend` 无回归
- [ ] 第 14 节的单测、类型、构建、语言检查通过；相关浏览器回归按覆盖状态留存结果

## 13. 收敛路线

**已完成（2026-09-11，首页）**

- 令牌层 `frontend/tailwind.config.js`：`darkMode: 'class'`、CJK 字体栈、`accent` 色阶。
- 首页第 7 节第 1-4 项：按钮对比度、强调色收敛、圆角三档、CTA 去重。
- 吉祥物：5 个 WebP 资产 + `components/Mascot.tsx` + `mascot-in-forbidden-zone` 规则。

**已完成（2026-09-12，功能页对齐首页）**

改动集中在 4 个共享文件 + 一次全仓机械改写，没有逐页重写。

- **令牌**：`accent` 展开成 50-950 完整色阶；删掉无人使用的 `ink`；`index.css` 只留全局焦点环与表格 `tabular-nums`，页面底色移交 `App.tsx`。
- **色相收敛**：`indigo` / `violet` / `purple` / `fuchsia` / `sky` / `cyan` / `teal` / `blue` 共 1275 处装饰性用色 → `accent`；`gray-*` 300 处 → `slate-*`；焦点环 178 处 26 种色相 → 1 种。
- **阶段色调**：`processRegistry` 的九色彩虹改为 `TONE_CLASSES` 五色调，只作用于图标底与眉标；`homepage/catalog.ts` 改为从注册表读取，两处色调不再各写一份。原来的 emerald / rose / amber 三种阶段色与成功 / 错误 / 警告状态撞色，已消除。
- **对比度与字号**：204 处 `text-slate-300/400` → `slate-500`（深色面板上的 20 处保留）；299 处 `text-[9/10/11px]` → `text-xs`。
- **圆角**：793 处收敛到卡片 `rounded-xl` / 控件 `rounded-lg` / 胶囊 `rounded-full` 三档。
- **表格可达性**：52 个文件、466 处 `<th>` 补上 `scope`（`thead` 内 col，`tbody` 内 row）。
- **原语**：`components/ui.tsx` 的 `Card` / `Button` / `Badge` / `SectionHeader` / `EmptyState`。
- **空态**：6 处整页主空态改用 `EmptyState`，带吉祥物；`EmptyState` 的间接引用也纳入禁区检查。
- **中文排版**：31 + 5 处作用在中文上的 `uppercase` 与宽字距移除。

**已完成（2026-09-12，第二轮：三类 P2 问题）**

- **深色区域文字**：上一轮的"提高对比度"改写是按行判定的，而横幅的渐变底色写在父元素那一行，
  导致 20 处浅色眉标（`text-cyan-200` 一类）被当成白底文字压深到 `accent-600`，横幅上只剩 1.0-2.1:1。
  已按原始档位逐处还原；PitBadge 的 PIT 提示同因同修。
- **分类色**：`bg-indigo-500` 与 `bg-fuchsia-500` 双双落到 `bg-accent-500`，堆叠条上固收 52% 与权益 28% 连成一段。
  已改为 `ASSET_CLASS_COLORS` 独立定义，并给堆叠条加分隔线、给图例加色块 —— 边界不再依赖色差。
- **检查器**：`tiny-font` 原来只枚举 9/10/11px，`text-[8px]` 能过；`unscoped-tables` 原来按文件计数，
  文件里有一处 `scope` 就全放过。两条都按实际语义重写，当场查出 2 个漏标的 `<th>`。
  新增 `same-element-contrast`（查出 37 处真实不达标）与 `duplicate-category-color`，
  并新增浏览器实测的 `e2e/contrast.spec.ts` 覆盖静态分析够不到的祖先底色与渐变。

**P1（剩余）**

1. **已完成**：阶段导航在桌面常驻，子页面移除重复阶段介绍，参考证据默认收起。财务报表优先显示主体选择、报表类型和数据，账簿及详细核算边界可展开；手机表格保留行名并提供横向滚动提示。
2. `PrototypeWorkspace.SampleChart` 换成 ECharts。
3. ECharts 统一主题，收敛 57 种散落色值（`bare-chart-hex`）。**本轮刻意没做**：轴线与网格的 slate 灰本来就是一致的，需要重配的是多序列的区分色，而后端未启动时看不到真实图表，盲改会破坏序列间的可辨识度。做这件事需要能跑起来的后端。
4. 其余 141 处局部空态逐步改用 `EmptyState`（吉祥物一屏只能有一个，局部空态保持纯文字）。

**P2**

5. 首页第 7 节其余构图事项按独立任务推进；小字号和非标准字重已完成。
6. 四态补齐：骨架屏与 `aria-live`。
7. 深色模式 —— 必须在令牌收敛完成之后。

## 14. 基线与可复现命令

budget 实测于 2026-09-12，功能页对齐首页令牌之后（`main @ bc72526` + 未提交改动）。前端依赖未安装时 `design:check` 仍可运行（纯 Node，无第三方依赖），但 `build` / `test` / `tsc` 需先 `npm install --prefix frontend`。

```sh
npm run design:check --prefix frontend
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
node scripts/check_i18n.mjs
python skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
npm run test:e2e --prefix frontend -- contrast.spec.ts indicator-studio.spec.ts historical-regime-workbench.spec.ts process-framework.spec.ts homepage.spec.ts --workers=2
```

浏览器测试需要本机 Chrome；流程夹具需要带 uvicorn 的 Python，可用 `INDICATOR_TEST_PYTHON` 指定。测试使用隔离夹具，不向真实业务后端提交操作。

## 15. 吉祥物（卡通牛）使用规范

小牛是本项目的品牌形象，卫衣上的柱状图标与 `brand.svg` 同源。源文件在仓库根目录 `images/`（已 git 跟踪）：`0001.png` 是首页 hero 场景、`0002.png` 是地球场景，两者已出货为 `hero-bull-*.webp` / `data-globe-*.webp`；`0003`-`0011.png` 是 9 个 1254×1254 透明背景姿势。

### 15.1 姿势与状态的一对一契约

形象出现必须承载一个状态。**一个状态一个姿势，姿势不得复用到第二个语义上**，否则它退化成随机装饰。语义重叠的姿势不出货。

资产宽度是显示宽度的 2 倍，用于高密度屏，因此不需要 `srcset`。

| `state` | 姿势 | 显示 | 资产文件 | 说明 |
| --- | --- | --- | --- | --- |
| `empty` | `0010` 站立侧指 | 120px | `mascot-empty-240.webp` 8.7K | 手指向创建/导入入口 |
| `noresult` | `0011` 抓头疑惑 | 120px | `mascot-noresult-240.webp` 7.7K | 唯一带「?」的姿势 |
| `welcome` | `0005` 举手打招呼 | 120px | `mascot-welcome-240.webp` 8.3K | 只在首次会话出现 |
| `working` | `0004` 侧背身行走 | 80px | `mascot-working-160.webp` 3.6K | 与骨架屏并用，不替代骨架屏 |
| `success` | `0008` 双手举高 | 48px | `mascot-success-96.webp` 3.1K | 内联于确认条 |

已接入：首页搜索无结果（`ProcessHome` 弹窗，`noresult`）、因子研究中心「尚未运行检验」（`FactorResearchCenter`，`empty`）。`welcome` / `working` / `success` 资产已就绪，尚未接入。

`0003`（正面垂手）、`0006`（前伸行走）、`0007`（跳跃张手）、`0009`（欢呼）与上表语义重叠或过于接近，**不转 WebP、不出货**，留在 `images/` 作为后续素材。

### 15.2 禁止出现的位置

这一节优先于任何视觉偏好。`AGENTS.md` 规定平台不输出投资建议、不提供下单能力；卡通形象紧邻收益数字会把研究材料读成营销材料，紧邻核算差错会直接损失可信度。

**不得出现在：**

1. 任何数值旁边：净值、收益率、回撤、权重、绩效、归因、风险指标。
2. 风险提示、PIT 口径警告（`PitBadge` / `PitDecisionNotice`）、因果性审计告警。
3. 基金会计的对账差异、关账错误、凭证错误、双账勾稽差额。
4. 表格内部、图表内部、图表 tooltip、画布节点。
5. 交易计划、前置检查、订单分配相关的任何界面。
6. `VISUAL_DENSITY: 7` 的工作区正文区域。

**允许出现在：** 空态、无结果、首次进入、成功确认、404 / 错误边界、首页品牌区块、以及 34 条 `prototypePage` 的 `StaticDemoBanner`（明确标注为未实现原型的页面）。

共同点是：**这些位置都没有数据。** 有数据的地方交给数据。

### 15.3 技术契约

1. **资产管线沿用既有约定**：`images/NNNN.png`（源，已 git 跟踪）→ `frontend/public/homepage/images/mascot-<state>-<资产宽度>.webp`。**禁止在代码里引用 `images/*.png` 原图**，单个 1MB。新增姿势时复跑：

```sh
for map in "0010:empty:240" "0011:noresult:240" "0005:welcome:240" "0004:working:160" "0008:success:96"; do
  IFS=: read pose state w <<< "$map"
  magick "images/$pose.png" -resize ${w}x${w} -quality 82 -define webp:alpha-quality=90 \
    "frontend/public/homepage/images/mascot-$state-$w.webp"
done
```
2. **体积预算**：5 个资产实测合计 31.3KB，`public/` 从 404KB 增至 444KB。每个状态只出一档宽度，不做 `srcset`。
3. **装饰语义**：`alt=""` + `aria-hidden="true"`。状态含义由旁边的文字承载，屏幕阅读器不需要「一头牛在欢呼」。
4. **不做动画**。`MOTION_INTENSITY: 2` 下不做 CSS 弹跳、不做 Lottie、不做逐帧序列。
5. **单页最多 1 个实例**。同一页两处出现即为装饰滥用。
6. **显示尺寸上限**：空态 120px 宽，内联确认 48px 宽（资产各为 2 倍）。不做全屏、不做背景水印。
7. **首屏之外的实例必须 `loading="lazy"`**，并显式声明 `width` / `height` 以保住 CLS。
8. **单一组件承载**，不逐页复制 `<img>`：`components/Mascot.tsx`，props 仅 `state` 与可选 `className`，由 15.1 的表驱动。姿势与状态的映射只存在一处。禁区由 `mascot-in-forbidden-zone` 规则按文件名机械拦截（`Accounting|Financial|Ledger|Booking|Reconcil|Statement|Trade|Risk|Pit|Performance|Attribution|Valuation|Allocation|Portfolio|Dashboard`）。

### 15.4 首页的具体机会

第 7 节第 8 条指出「核心能力」4 张卡全白、只有文字，违反 Bento Background Diversity。这是首页唯一适合加形象的位置：给其中 1 格换成带姿势图的图文格，既修掉那条不通过项，又不增加新区块。**hero 已有场景图，不要再叠一个姿势图**；第 7 节第 6 条要删的是 hero 的文字角注，不是 hero 的形象。
