# 工程、指标与界面：验收纪要

本页保留已完成整改的关键决策、历史验证和剩余边界；设计规范分别由 [前端设计准则](../frontend/README.md)、[行情指标中心](../indicators/README.md) 和 [滚动区间图](../indicators/rolling-intervals.md) 维护。所有测试数字只代表原记录日期，不是本次文档整理重新执行的结果。

## 关键整改及其原因

| 问题 | 已确认的处理与反例 |
| --- | --- |
| A01 缺失值压缩时间轴 | 保留日期和缺失位置；`[10, NaN, 30, 40]` 两期均值为 `[空, 空, 空, 35]`；缺失净值后一笔收益仍为空。 |
| A02 并发 PIT 污染 | 移除“上一次取数”全局状态；ClassNavResult 显式携带本次 lineage/available_at，两线程交错与保存流程分别验证。 |
| A03 合法滚动被拒 | 共享作用域因果校验；窗口归约允许，`rolling_apply(mean(a),3)+mean(a)` 的窗口外全样本统计仍拒绝。 |
| A04 跨中心指标缺失 | 情景/择时锁定精确 revision、真实共同日期轴和年度参数，共用 Typed DAG/NJIT；旧引用不会静默换新版。 |
| A05 收益窗口多丢一期 | 区分净值行数、收益观察数和基准前置行；20 期 Calmar 可读 21 行净值，但窗口仍为 20 期。 |
| A06 错误参数变500 | 无效日期、NaT、空 classes 在输入边界返回明确400；不吞任意运行异常。 |
| A07 导航语言缺失 | 补齐风险模型中心语言项并检查导航覆盖。 |
| 图编辑与参数 | 公式和真实可编辑图双向转换，Scalar/Series 来源独立；参数绑定节点名、定义 revision 和运行实例，不靠显示名称猜测。 |
| 异步及响应式 | 注册表迟到响应不覆盖用户输入，未就绪禁用保存/校验；窄屏表单纵向布局；ResizeObserver 调整真实 ECharts 画布。 |
| i18n 矩阵 | 动态语言、冻结代码列、键盘/TSV/批量保存；单次最多1000项，非法占位符、越界等导致本次粘贴整体拒绝。 |

## 历史证据及适用范围

| 阶段 | 证据 |
| --- | --- |
| 09-07 指标画布 | 相关后端325、前端80、真实API浏览器15；中文补充前端95、浏览器12。全仓 TypeScript 当时仍有既有错误，不能把局部通过改写为全仓通过。 |
| i18n 矩阵 | 后端70、前端39、真实API浏览器15，320/768/1440。全前端当时563/565通过，另有并行改动引起的旧断言失败；不是全仓验收。 |
| 指标运行参数 | 后端198，真实后端浏览器2（1440/320）；全量TypeScript当时33项其他文件诊断。 |
| 09-09 指标原语 | 专项508后端、147前端、浏览器9；提交整理时392后端与56因果通过、全后端2195仅收集；前端662通过/2失败。后续全项目修复另有记录，不能回写成此阶段全部通过。 |
| 09-11 A01–A07 | 全后端3129，0失败/跳过；全前端810/115文件；浏览器253通过、5项原配置跳过，320/768/1440；TypeScript、构建、i18n通过。 |
| 09-11 提交前修复 | 全后端3176，0失败/错误/跳过；全前端851/118文件；最终完整浏览器253通过、0 flaky、5项原跳过。首次浏览器250通过/3失败及编排器指纹变化未隐去，修复后另完整执行。 |
| 首页/全站视觉收口 | 前端898/122文件、相关后端51；完整浏览器第二轮272通过/3失败/7跳过；画布请求清理后30项两轮复验、首页10通过/2跳过。没有一次最终全量浏览器全绿记录，不表述为有。 |

A01–A07 全量浏览器的5项跳过：三个宽度的真实工作区写入需显式启用，另两项历史版本抽屉仅桌面运行；不是为消除失败新增的 skip。真实数据、线上服务与发布不在离线验收范围。

## 性能与证据边界

09-11 通用滚动 v3，4096时点、预热后9次中位数：布林带0.3677ms、价格均线0.1292ms、KDJ0.5030ms、5期年化夏普0.0615ms、成交量均线0.0513ms。每项额外40次只读计算净存活NRT分配增量0；不含编译/取数，不代表服务吞吐或所有窗口为O(T)。当前协议见滚动契约，不恢复旧数值实现。

旧日志路径可能位于本地 `.tmp_*`、`.pytest_cache` 或仓库外，不能作为其他机器必然可访问的交付物。原始日期、命令和全文可按 Git 历史定位；新的验收应保存对应源码指纹、退出码、范围和跳过原因。

## 保留事项

- 构建大 chunk、依赖弃用和部分 React act 警告为工程维护项，历史记录未给出全部清零证据。
- 当时路由未跟踪文件与发布未完成属于交付状态，不等于数学测试失败，也不能据历史路由通过宣称当前可发布。
- 浏览器协议夹具、真实隔离后端与实际生产数据验收必须区分；静态设计检查和对比度脚本不等于完整视觉/无障碍认证。

2026-09-17 走势图多实例交付：相关后端87、全前端1156/147文件；真实后端浏览器覆盖1440/768/320的参数独立、并图/拆图、口径不兼容、禁用原因和无横向溢出。图例颜色及窄屏控件裁切同时修复。50条分批及加载期间复制属于后续契约，不能用此早期记录代替后续验收。

## 复权价格迁移的有效证据（2026-09-16）

历史快照 `tushare_snapshot_20260910_merrill_macro01` 的1,579,113行/1778只ETF：1776只由pre_close推导、2只用供应商因子；340只存在复权事件。原13列未变、新增6列，推导收益与 close/pre_close-1 最大偏差4.44e-16；两种因子的6542行可比样本最大相对差2.27e-03，已披露，不能声称两种来源完全一致。

重建指标快照30,954行/8指标、error=0后才重新激活。重建必须显式传 workspace_data_dir；遗漏会走旧内置口径，曾被max_drawdown符号复算拦截。该记录不代表今天仍激活同一快照。

专项后端810，参数/导出139（不据此推算去重总数），全前端1031/136文件，类型/构建通过。历史迁移中的v4指标和用户授权删除记录由Git追溯；当前种子定义只维护唯一版本，旧测试数量不是当前指标总数。

因果审核初始基线：104算子中62 CAUSAL、42 WINDOW_CONSUMING、0 LEAK/UNKNOWN，7项预热敏感；专项56通过，全后端1908通过/1既有失败。该历史目录数量不应覆盖后来的注册表。尾部扰动、截断重算和预热敏感分别验收；公式全样本均值广播到历史点的两个泄露反例必须保持可检测。

## 保留的视觉证据

首页数值来自2994×4198双倍参考图的半尺寸测量，7.3px、10.5px等只属于该页，不能推广为全站设计令牌。独立设计规则仍以设计准则为准。历史截图：[移动报表](../images/frontend-design-20260912/report-mobile.png)、[桌面报表](../images/frontend-design-20260912/report-desktop.png)、[平板指标](../images/frontend-design-20260912/indicator-tablet.png)、[英文首页](../images/frontend-design-20260912/home-en-1251.png)、[产品象限](../images/frontend-design-20260912/product-quadrant-tooltip.png)、[披露不可用状态](../images/frontend-design-20260912/disclosure-unavailable-mobile.png)。它们记录当时界面，不能替代后续改动的浏览器验收。


## 文档工程升级（2026-09-20）

范围：在已有历史文档精简的基础上迁移 45 份文档，根目录仅留 README/AGENTS；建立主题入口、专业契约、研究草稿及历史证据的索引。未修改业务功能，也未将历史投资验证更新为当前资格。

源码基线为 `a9dd01bd143e7634a1a6436d69be8d25f2744f85` 上的未提交文档候选。检查器 SHA-256：`e74f2994ae8bbf160d21c8484d04d75a6d2f60e344db703ffe90975ea53870fd`；检查器测试 SHA-256：`61dc3aafb5617e58b72eae80d46173162aa01395d5d499cfa3418e49372f1225`。以下均为本轮重新执行的本地结果，不能外推为全仓业务回归或远端发布结果。

| 检查 | 结果及范围 |
| --- | --- |
| `python3 -m pytest scripts/tests/test_documentation.py -q` | 31 通过；含相对/引用链接、锚点、目录漂移、计划证据、符号链接、暂存与工作区隔离、PR merge-base/HEAD、改名、审核回执失效及只读检查 |
| `PYTHONPATH=.:backend python3 -m pytest backend/tests/test_tushare_data_script.py -k document -q` | 3 通过、90 未选择；只验证下载文档契约及迁移后的路径，无真实下载 |
| `python3 scripts/check_documentation.py` 及 `--staged` | 64 份受管文档的目录、链接、锚点、计划字段和索引通过；暂存测试使用隔离临时 index/object 目录，未改变实际暂存区 |
| Hermes validate/evolve、R01/R03/R04 路由 | 路径及模块/坑点引用通过，任务范围无未覆盖的现存文件；稳定路径可复现检查在上述隔离候选通过，实际工作区仍有未提交/未跟踪文件 |
| 规范迁移与差异自审 | 原算子 14 条、计算 18 条、窗口 4 条、采集 13 条要求保留；只调整跨文件指向。研究草稿仅修复引用，业务待办未被标为已交付 |
| `git diff --cached --check` | 隔离候选通过；真实 index 哈希未改变 |
| Documentation 工作流 | YAML 与本地调用入口验证通过，尚未提交或取得远端 PR 运行证据 |

完整目录由 `repo_map.json` 单独维护，索引生成区接受一致性检查。AGENTS 的收尾协议要求按实际差异更新文档、结项计划和沉淀坑点；结构检查不会替代语义审核。PR 工作流未配置为远端 required check，未安装客户端原生 hook。后续上线状态由[文档维护计划](../governance/documentation.md#计划与证据生命周期)继续记录。


同日 AI Hermes 自演进复核：`route_task.py` 按 `operational_list_resolution.apply_to_fields` 合并并输出必读列表，原四处 `read_before_edit` 虽被覆盖/路径检查接受，却没有进入实际 context。已将这些现有要求移入 `first_read_files`；R20、R70、R80 的修复前输出缺少要求，修复后分别包含计算、算子及采集规范。P02 记录防复发检查。另移除 80 项正式源码、测试、配置和文档的未跟踪豁免，仅保留 7 类本地诊断/缓存产物。

本轮重新执行的结果：任务范围 evolve 与 routing-only 均无遗漏；R01/R02/R03/R04 及受影响路由读取检查通过，检查器回归 31 项通过。实际工作区严格 validate 返回 1，全部 64 项均为未跟踪的稳定引用（60 份文档、3 个脚本/测试/依赖文件、1 个工作流）；这些文件仍须随已获授权的文档提交纳入 Git。跳过可复现性后的结构检查及隔离临时候选完整校验通过，不能将其表述为真实暂存区已通过。未改变实际 index、HEAD 或其他业务改动，也未验证远端 CI/required check。

同日获准提交后的复验：上述 64 个稳定文件已纳入真实暂存区，严格 Hermes validate 通过，原未跟踪引用问题已解除。对真实暂存内容导出的隔离副本重新运行检查器测试（31 通过）、Tushare 文档契约测试（3 通过、90 未选择）、完整提交范围 evolve 及 R01/R02/R03/R04/R20/R70/R80 路由检查，均通过；暂存候选的 64 份文档、343 处本地链接及差异格式检查通过。提交范围只包含文档整理、路由索引、检查工具与路径引用修正，其他业务改动保留在原工作区。远端 PR 工作流运行及 required check 配置仍未验证，DOC-02 保持待验证状态。

### PR 远端检查验证（2026-09-20）

[PR #45 的 Documentation 运行](https://github.com/KevinCJM/FundInvestmentResearchPlatform/actions/runs/35506124121)已在提交 `2fb26760d91f7f341b48fa2f03500f5cd9d0e028` 上成功完成。GitHub 返回 `pull_request` 事件、完整 HEAD SHA 和 `success` 结论；检查器测试、PR 候选检查与报告上传步骤均成功。这补齐 DOC-02 的首次远端运行证据，该项计划据此结项。

证据仅适用于上述提交及工作流运行，后续 PR 提交仍须重新检查。未修改远端 required check、分支保护或 Bot 审核规则；工作流成功不等于 Bot 审核或合并通过。

### PR 候选辅助代码隔离修复（2026-09-20）

Bot 在 PR #45 指出检查器从工作区导入 Hermes 辅助模块，可能使暂存或指定提交的检查混入候选外逻辑。新增 5 个反例在修复前全部失败，分别覆盖暂存/指定提交不受外部辅助代码影响、未暂存修复不能掩盖候选辅助模块失败，以及辅助代码版本参与回执指纹。

修复后通过 Candidate 从同一来源读取并加载辅助模块，按候选缓存，缺失或加载失败直接报错；回执指纹同时绑定该辅助代码。检查器完整回归为 36 项通过。此前 31 项和旧脚本哈希仍属于前述历史提交的验证记录，不替代本次修复证据。

### 工作台全站检索与加载骨架（2026-09-21）

UI 调研定位到三处可直接修复的缺陷：九十多条路由只有十个一级入口、工作台内没有检索入口（检索此前只存在于首页 `/` 的对话框）；加载态用「正在读取…」这类文字行替换内容，不保留布局；`tracking-wide r` 这个无效类名被复制到两个文件共 6 处表头。

`CommandPalette` 挂在 `Header` 上，条目由 `processRegistry` 派生，不另维护清单；⌘K / Ctrl+K 唤起，弹层用 `createPortal` 挂到 `body`，避开顶栏的层叠上下文。键盘实现按 APG：combobox 的 DOM 焦点留在输入框、`aria-activedescendant` 表示选中项，modal dialog 负责 Esc 关闭、Tab 循环和焦点归还触发按钮。会计工作区在两个阶段各挂一次、路径相同名称不同，列表键按「阶段 + 路径」取，否则 React 键重复会让过滤结果残留旧行——这一点由空结果用例覆盖。

`ui.tsx` 新增 `Skeleton`，数据质量明细表的加载行换成保留六列形状的骨架，配 `sr-only` 的 `role="status"`；其余加载态仍是文字行，未改。另修正 5 处中文省略号写成三个点。

已验证：`tsc --noEmit` 0 项；Vitest 全量 162 文件 / 1306 用例通过（含新增 5 个检索用例，覆盖快捷键唤起、过滤、方向键与 Enter 跳转、Esc 归还焦点、空结果文案）；`check_frontend_design.mjs` 无回归，`uppercase-on-cjk` 11 与 `bare-chart-hex` 57 两项仍高于目标，本次未触及；`check_i18n.mjs` errors 为空，新增 6 个 `navigation.search*` 词条双语齐备。

未验证：浏览器视觉与交互验收未做，屏幕阅读器未实测，Playwright e2e 未运行。z-index 仍是散写字面量，全局检索占 `z-[300]`，常量表未建立。表格共用原语、ECharts 单主题与暗色模式均未开始。

### 表格共用原语与表名棘轮（2026-09-21）

164 张表分散在各页手写，`scope`、`caption`、数字列对齐、空态与加载态全靠每页自觉；`tracking-wide r` 这个无效类名被抄到 6 处表头就是这么来的。`ui.tsx` 新增 `DataTable`：调用方只描述列，`caption` 与 `empty` 必填，`scope="col"`、数字列右对齐、可聚焦的横向滚动区、加载骨架都是默认值。`numeric` 只负责右对齐，等宽数字由 `index.css` 对 `td` / `th` 全局给出，不在原语里重复。

已迁三张表：方案展示中心的分区间收益与规则链表（原先 `overflow-hidden` 配 `min-w-[640px]`/`min-w-[720px]`，窄屏是裁切而不是滚动，且没有表名和数字列对齐），以及数据质量明细表（顺带把上一轮的临时骨架行收进原语）。其余 161 张表未动。

新增棘轮规则 `unnamed-tables`：没有 `<caption>` 也没有 `aria-label` 的表，实测 48，预算锁 48，目标 0。准则 6.6 一直要求表名必填，此前没有任何检查拦过。`<caption>` 按 HTML 规范是表的第一个子元素，规则只往后看到 `<tbody>`，避免把下一张表的 caption 算成自己的；规则自检加了 4 条反例，含"注释里写出 table 标签也会被计入"这一已知近似——本轮就踩到过，`ui.tsx` 的注释因此改掉了尖括号。

已验证：`tsc --noEmit` 0 项；`check_frontend_design.mjs` 无回归，新规则自检通过；Vitest 全量 163 文件 / 1311 用例通过，新增 `ui.test.tsx` 5 例覆盖表名与列头语义、数字列对齐、空态文案、加载时表头保留并宣告、滚动区可聚焦。

测试稳定性：默认 5 秒超时下的全量并发跑，`HistoricalRegimeWorkbench`、`ProductCompare` 等重文件偶发超时（4 次全量里 2 次），单独重跑稳定通过，`--testTimeout=20000` 全量通过。属并发负载下的超时，不是断言失败。

未验证：浏览器视觉与交互验收仍未做，屏幕阅读器未实测，e2e 未运行。剩余 161 张表、ECharts 单主题、暗色模式、z-index 常量表均未开始。

### 投前流程条与侧栏同源（2026-09-21）

投前研究页顶部的 6 步流程条与左侧 9 项节点导航是两份各写各的定义：侧栏来自 `processRegistry` 的 `preInvestment.nodes`、文案按路由查 `locales`；流程条是 `StageLayout` 里就地硬编码的 6 个中文串。于是同一页出现「侧栏 02 选择研究路径与范围 / 顶部 1. 产品范围」，另有「战略资产配置（SAA）/ 长期配置 SAA」「产品配置与择时 / 产品配置」等三处一步两名，且流程条不随语言切换。

改法：流程条只保留闸门（ready / done / current），步骤名与编号改从注册表推导——落在流程节点上的步骤沿用侧栏序号（01 / 02 / 03 / 04 / 05 / 06），落在工具页上的「大类资产构建」与侧栏「现有工具」一样不编号，因此顶部编号既不重号也不倒序。流程条也会出现在 `/product-research/pools`，那里的 `stage` 不是投前决策，所以步骤名单独取 `useLocalizedStage('pre-investment')`。窄屏缩写与 Strategy first 的「产品映射」不在注册表里，新增 `navigation.allocationFlow`、`navigation.journey*` 共 6 个双语词条；LTCMA / SAA / TAA 两种语言写法相同，仍是字面量。

未改行为：原有六步的顺序、跳转目标与闸门条件一字未动。`04 战略资产配置（SAA）` 取节点名，链接仍指向工具页 `/saa/policy`（侧栏 04 指向节点页 `/pre-investment/saa`），两者名字一致、落点不同。

已验证：`tsc --noEmit` 0 项；Vitest 全量 164 文件 / 1331 用例通过；`StageLayout.test.tsx` 去掉了假造 `nodes: []` 的 stage mock，改用真实注册表，新增 1 例锁住六步名单与「顶部编号 = 侧栏编号」；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空。

补做（同日，用户指出顶部缺 `01 投资目标与约束`）：流程条首位补上 `01 投资目标与约束`，`ready` 恒为真（它是链路起点），`done` 取 `journey.mandateId`；`AllocationJourneyStep` 增加 `objectives` 成员、`allocationJourneyPath` 增加 `/pre-investment/objectives`（该页不读 `?mandate=`，因此不带查询身份）。步数由 6 增至 7，栅格随之改为 `grid-cols-4 lg:grid-cols-7`，全名推迟到 `xl`（1280px）才显示，更窄时用缩写，新增 `navigation.journeyObjectivesShort` 一个词条（累计 7 个）。原有六步的顺序、跳转目标与闸门条件仍未改动。

### 投前流程条按上游状态放行下游（2026-09-21）

流程条此前只有 4/7 步真正绑定状态：`01` 的选定结果不喂给 `02`（`pool.ready` 恒为真），`03 LTCMA` 的 `ready: true` / `done: false` 两个都写死（`AllocationJourney` 根本没有 LTCMA 字段），`06` 的 `done` 恒为假且亮着也可能被 `allocationJourneyPath` 送回 TAA。

改法：`AllocationJourney` 新增 `ltcmaId`（本次研究选定的主 LTCMA），由 `StrategicAllocationWorkspace` 选定/采纳时写入、清空选择时清除，`LtcmaWorkspace` 在 `?mandate=` 与当前研究一致时发布即写入。级联失效补两条：目标、产品范围、大类方案、战略范围或实施映射任一变化即作废 `ltcmaId`（LTCMA 绑定这些身份，见后端 `SAA_MANDATE_CMA_BASIS` / `SAA_CMA_SOURCE_CHANGED`）；`ltcmaId` 变化即作废 `baselineId` 与 `taaRunId`。闸门相应改为：`02` 由 `mandateId` 放行，`03` 的 `done` 取 `ltcmaId`，`06` 的 `ready` 与 `done` 统一取 `productsHandoffReady`（与 `allocationJourneyPath` 同一判断，杜绝"亮着却被送回 TAA"）。落点带身份：`03` 有选定版本时直接回到 `/pre-investment/ltcma/<id>`，`04` 带 `?cma=`（该页本就读 `params.get('cma')`）。未就绪原因分三种文案，六条状态文案一并迁入 `locales/system.json` 的 `navigation.journey*`。

未改行为：`04 SAA` 的 `ready` 仍是 `saaScopeReady`，**没有**挂到 `ltcmaId` 上——LTCMA 的选用就发生在 SAA 页内（它的第二步即「选择 LTCMA」），挂上去会死锁。该闸门的权威实现在后端 `preview_policy`（`SAA_MANDATE_CMA_BASIS`、`SAA_MANDATE_EXPIRED`、`SAA_CMA_SOURCE_CHANGED`）与 `policy_gate.require_policy_application`，已是硬校验，前端不重复造。步骤顺序、各步落点与其余闸门条件未动。

后端未改：本次需要的准入判断后端已具备且为权威；`confirm_universe` 不绑定投资目标是既有数据契约（战略范围为 `research_only`，目标绑定发生在政策层），未获授权不改。

已验证：`tsc --noEmit` 0 项；Vitest 全量 164 文件 / 1338 用例通过（新增 7 例：流程条四条闸门与 `allocationJourney` 三条级联/落点）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空。满负载并发下 `CommandPalette` / `InvestmentObjectivesWorkspace` / `HistoricalRegimeWorkbench` 各出现过一次超时抖动，单独重跑与 `--maxWorkers=4` 全量均通过，与本次改动无关。

未验证：浏览器验收未做，闸门的真实观感（尤其新用户首次进入时 02–06 全灰）没有在浏览器里看过。已知未处理：`07–09`（研究包与风险汇总 / 研究验证 / 研究定稿）在流程条里没有对应步，且这四页被 `isPackageWorkspace` 整条隐藏流程条；属改流程契约，未获授权前不动。

### 投前页面上下文改由地址栏唯一决定（2026-09-21）

此前约十处页面读身份时写成 `params.get(x) ?? journey.x ?? ''`：从侧栏裸路径进入时，页面会悄悄认领上次研究的目标、可投资域、战略范围、大类方案、基线或 TAA 版本，用户看不出这些选择从哪来，同一个 URL 在不同浏览器/标签页还会渲染成不同结果。`01` 的流程条落点又只到 `/pre-investment/objectives` 列表页，从别的步骤点回去看不到已选目标的详情。

操作链路改成三条规则。一、**导航不写状态**：点导航永远不改 `allocationJourney`（`updateAllocationJourney` 带级联作废，导航顺手写状态会把下游一起清掉）。二、**地址栏是页面上下文的唯一事实来源**：删掉全部身份回落，`ProductPoolSelection`（`mandate`、编辑器自动展开的 `resumeScope`、`restoreId` 里的 `universeId`）、`ClassAllocation`（`alloc`/`universe`）、`StrategicAllocationWorkspace`（`alloc`/`mandate`/`strategic_universe`）、`StrategicScopeWorkspace`（`strategic_universe`/`universe`/`mapping`）、`TacticalAllocationWorkspace`（`decision`/`baseline`，并删掉裸路径时把 `journey.taaRunId` 写回地址栏的那个 effect）、`PortfolioConstruction`（草稿 scope 与 `universeId`）、`ManualConstruction`（`universe`）现在一律按地址栏渲染，缺身份就是未选择。三、**续接改成显式动作**：新增 `frontend/src/components/ResumeResearch.tsx`，只在落点 pathname 等于当前页、且地址栏缺少某个身份时显示一行（列出上次研究还带着哪些身份 + 一个跳到 `allocationJourneyPath(...)` 规范 URL 的链接）。它由 `StageLayout` 按当前步统一挂一次（覆盖 02/03/04/05/06 与产品优先的大类步），`allocation-lab` 不是流程步，由 `ClassAllocation` 自己挂一次。另外 `allocationJourneyPath('objectives')` 在已有目标时改落到 `/pre-investment/objectives/new?view=<id>` 详情页（该路由与 `?view=` 已存在，`InvestmentObjectivesCenter` 的详情链接就是这个形状）。

保留未动的三类回落：日期默认值（`ProductPoolSelection` 的 `researchDate`、`ClassAllocation` 的 `endDate`）、显式交接负载（`portfolioResearchImport`）、保存时的写入保值（`StrategicScopeWorkspace` 保存战略范围时把未变的 `mandateId` 一并写回，直接剥掉会写成 `mandateId: undefined` 并级联清空 `baselineId`/`taaRunId`）。流程条的 `selected`/`done` 标记仍读 `allocationJourney`——那是状态标注，不是页面选择。

后端未改：`allocationJourney` 只存在于浏览器，后端没有任何耦合（`grep allocation_journey backend/` 无命中）；本次进入地址栏的身份此前就已由 `allocationJourneyPath` 带出，读取端点都已存在，新增的 `?view=` 也复用既有的 `GET /mandates/{identifier}`（`backend/strategic_allocation/routes.py:88`）。权威准入仍在 `preview_policy` 与 `policy_gate.require_policy_application`。

已验证：`tsc --noEmit` 0 项；Vitest 全量 165 文件 / 1359 用例通过（新增 8 例：`ResumeResearch` 五条可见性与词条断言、`StageLayout` 两条续接条挂载、`allocationJourney` 一条 01 落点）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空，system 词条 1999 条。`ClassAllocation.test.tsx` 有一条夹具过期（只在 journey 里写 `universe` 而不写地址栏，保存基线时会被当成"换了范围"而级联清空 `researchDate`），已改成在 `initialEntries` 里带 `?universe=range-test`，断言本身未动。

e2e 有一条断言写的就是旧契约（`mandate-boundaries.spec.ts` 里“SAA 地址栏故意不带 mandate，期望下拉框自动选中上次目标”），改成断言下拉框为空、且续接条的 href 带着该目标 id。后端未改，本轮没有重跑 4547 例的 pytest 全量。

未验证：Playwright e2e 本轮未跑（需前后端服务与夹具），上述 e2e 改动只做了静态核对。浏览器验收未做，续接条的真实观感（尤其它与页内已有提示是否重复）没有在浏览器里看过。已知未处理：`validate_ai_routing.py` 仍因未跟踪路径报红，本次又新增 `ResumeResearch.tsx` / `ResumeResearch.test.tsx` 两条，清红需要 `git add`，未获授权不做。`07–09` 在流程条里没有对应步的问题依旧未动。

同一天补上了流程条这一半：页面改成只认地址栏之后，`StageLayout` 的 `ready` / `done` 仍读 `allocationJourney`，于是出现「01 页面一个目标都没选、顶部 02 却写着继续研究」。现在 `StageLayout` 先按书签算出一份 `bookmarkSteps`，若当前落在某个流程步上而地址栏一个身份参数都没带（`RESUME_IDENTITY_KEYS`，LTCMA 的身份在 `/ltcma/:id` 路径上），就判定 `detached`：整条按未开始渲染（只剩 01 与 LTCMA 可点）、流程条链接改用空 journey 构造、「当前研究方案」一并显示为未选择，续接仍由 `ResumeResearch` 发起。判定限定在 `stageId === 'pre-investment'` 且确实站在流程步上：阶段总览、工具页与产品研究的产品池页不受影响，那里的流程条本来就是书签旁注。`StageLayout.test.tsx` 有六条夹具只把身份写进 `allocationJourney`、地址栏留裸（`/pre-investment/taa`、`/pre-investment/ltcma`、`?scope=strategic`），按新契约改成在 `initialEntries` 里带上该步的规范身份，断言本身未动；另加两条新用例覆盖「裸路径进 01 时 02 不可点」与「地址栏带 `view=m1` 后 02 解锁并带出目标」。两处 e2e 流程条断言的落点本来就带 `mandate` / `baseline` / `decision`，不受影响。前端全量 165 文件 / 1361 用例通过，`tsc --noEmit` 0 项，design 无回归，i18n errors 为空。

后端全量 pytest 这次跑完了：4545 通过、2 失败（`tests/test_investment_mandate.py::test_benchmark_and_mandate_bounds_are_consumed_not_just_stored`、`tests/test_mandate_reference_diagnosis.py::test_frozen_scale_constrained_frontier_then_independent_validation`）。两条都报 `MANDATE_NAME_CONFLICT`，来自工作区里尚未提交的 `backend/strategic_allocation/service.py` 新增的目标名称唯一性校验（`_require_unique_mandate_name`），而这两条用例仍在同一用例内用同名连存两版目标。与本次前端改动无关，未处理，留给该后端变更的作者。

### 产品研究与投前决策解耦（2026-09-21）

`/product-research/pools`（产品池构建）此前被 `StageLayout` 的 `isAllocationJourney` 特判识别为投前决策 Product first 路径的「02」步（见前一条「投前流程条与侧栏同源」），复用同一条流程条并按 `allocationJourney`（session/localStorage）渲染书签状态。用户从「产品研究」顶部导航直接进入该页时会看到与「产品研究」无关的「01 投资目标与约束…06 产品配置与择时」流程条，还可能带着上一次投前研究残留的目标/范围状态；用户判定两个模块不该混用，要求去掉。

改法：`isAllocationJourney` 去掉 `location.pathname === '/product-research/pools'` 特判，只在 `stageId === 'pre-investment'` 时渲染流程条；随之删掉专为该特判开的 `journeyStage = useLocalizedStage('pre-investment')` 独立取数，`named()` 改回直接用当前页的 `stage`（两者现在数据源已一致）；`pool` 步的 `current` 判断去掉 `\/pools$/` 这条只为匹配该页而加的正则分支。`/product-research/pools` 页面本身与跨模块的普通超链接（如产品范围页在没有已发布产品池时给出的「去创建或发布产品池」链接）未动——要去掉的是流程条混用，不是切断两个模块间的正常跳转。

未改行为：投前决策自己页面上的流程条闸门、书签与 `detached` 逻辑一字未动。

已验证：`tsc --noEmit` 0 项；Vitest 全量 165 文件 / 1378 用例通过；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空。

未验证：浏览器验收未做，没有实际打开「产品研究 → 产品池构建」核实流程条确实消失。

### 02 选择研究路径与范围拆成列表页与独立编辑页（2026-09-21）

`ProductPoolSelection.tsx`（02）此前把「已保存研究范围」列表和编辑器（Product first 的内部 `ProductPoolWorkspace` / Strategy first 的 `StrategicScopeWorkspace`）挤在同一页面：点「新建研究范围」「编辑」「复制新建」只是往地址栏加查询参数，在列表下方就地展开编辑器，配 `revealScope`/`workspaceRef` 负责滚动定焦。用户要求对齐 01 投资目标与约束（`InvestmentObjectivesCenter` 列表页 + `InvestmentObjectivesWorkspace` 独立编辑页，`/objectives` → `/objectives/new?fresh=1|view=<id>|editFrom=<id>`）：点「新建」要进独立页面，从零开始配置。

改法：新增 `frontend/src/pages/ProductPoolWorkspacePage.tsx`（路由 `/pre-investment/product-pool/new`，紧邻 `App.tsx` 里 `product-pool` 路由注册，与 `objectives`/`objectives/new` 相邻写法一致），把原先内嵌在 `ProductPoolSelection.tsx` 里的 `ProductPoolWorkspace` 函数整段搬过去（模块私有，不导出，唯一调用方就在同一文件），`StrategicScopeWorkspace`（已经是独立导出组件，原样复用）按 `?scope=strategic` 分流渲染。编辑页自己独立发起 `getStrategicCatalog()` 与 `listInvestableUniverseSnapshots()` 两个请求（不管哪种模式都要两个都拉，因为同名预检要跨产品/战略两边查重），照抄列表页原有的 `mandateBlockedReason` 判定梯子与 `existingScopeNames` 拼接逻辑。列表页瘦身：删掉 `revealScope`/`workspaceRef` 滚动定焦、`showEditor`/`deepLinked`/`scopesKnown`、`paramsRef` 及 `removeScope` 里判断"当前地址栏是否正指向被删项"的那段（列表页地址栏以后只会有 `mandate`/`scope`，不会再有 `universe`/`strategic_universe`/`edit`/`copy`，那段判断永远不成立），整个 `ProductPoolWorkspace` 函数体搬走不留副本。「名称」「继续研究」「编辑」「复制新建」的 `<Link>` base path 从 `/pre-investment/product-pool` 改成 `/pre-investment/product-pool/new`；「新建研究范围」从 `<Button onClick={startNewScope}>` 改成 `<Link>`（复用 `InvestmentObjectivesCenter.tsx` 已有的 Link+onClick 副作用写法），`to` 里的 `new=<Date.now()>` 每次渲染都重算，不写死成常量，避免两次点「新建」撞上同一个草稿 key。编辑页顶部加一个「← 返回研究范围库」，带回 `mandate`/`scope`，不让用户回去后重选模式和目标。

关联文件同步改：`allocationJourney.ts` 的 `allocationJourneyPath('pool', ...)` 目标路径改成三元——已选定 `universeId`/`strategicUniverseId` 时指向 `/pre-investment/product-pool/new`（回到那个具体范围），否则仍指向列表页，和上面 `objectives` 那一行的写法（`journey.mandateId ? '.../new' : '...'`）同构。`StrategicScopeWorkspace.tsx` 内部「编辑此范围」链接、`ProductPools.tsx`「使用已发布版本开展配置研究 →」链接 base path 同步改为 `/new`。

已知、可接受的副作用（"点新建要跳新页面"这个要求本身的直接代价，不额外处理）：进入编辑页后不能像以前那样就地切换 Product first / Strategy first 或切换投资目标，要切换得先回列表页——不丢数据（草稿 key 不含 mandateId），只是少了原地切换的即时性，和 01 的编辑页一样没有这个能力。裸地址栏 `/pre-investment/product-pool`（没带任何身份参数）且书签里已有 `universeId`/`strategicUniverseId` 时，`ResumeResearch` 续接条不会再出现在列表页（续接目标现在是 `/new` 子页，pathname 对不上）——这和 01 现状一致（`InvestmentObjectivesCenter` 列表页今天就没有这条续接条），列表页本来就有的完整表格承担"继续研究"入口，不是回归。

已验证：`tsc --noEmit` 0 项；Vitest 全量 166 文件 / 1376 用例通过（`ProductPoolSelection.test.tsx` 大改为列表页专属用例，编辑器相关用例迁到新增的 `ProductPoolWorkspacePage.test.tsx`；`StrategicScopeWorkspace.test.tsx`、`allocationJourney.test.tsx`、`StageLayout.test.tsx`、`TacticalAllocationWorkspace.test.tsx`、`ProductPools.test.tsx` 同步改了受影响断言）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空；Hermes evolve 覆盖新文件后 14/14。`docs/repo_map.json` 把 `ProductPoolWorkspacePage.tsx`/`.test.tsx` 加进了原先 `ProductPoolSelection` 所在的 `frontend_etf_research_pages`/`frontend_allocation_workflows` 两个模块条目。

未验证：浏览器验收未做，没有实际走一遍 Product first 与 Strategy first 各自的"新建→保存→返回列表→继续研究→编辑→复制新建"。e2e 只改了一处安全的机械替换：`frontend/e2e/bettersaataa-m1.spec.ts` 里 `strategic_universe=` 深链的 base path 改到 `/new`（纯路径替换，不改流程结构）。`frontend/e2e/research-scope-ui.spec.ts` 本轮未动——它测的正是这次拆分影响最大的交互：「新建研究范围」原先是同页随时可点的按钮，现在变成必须先回列表页才能点到的链接；「PIT 不一致只提醒」这条断言原先蹭的是列表页与工作区同屏共存的便利，现在要挪到列表页单独断言。这类流程级改写没有能跑的 Playwright 环境核实，本轮不做盲改，标注清楚留给下次连上前后端服务再一起改和跑。`validate_ai_routing.py` 因新文件未 `git add`（未跟踪路径）报红，和既有的几条未跟踪路径红一样，清红需要暂存，未获授权不做。



### 战略范围编辑页去掉两处无用展示（2026-09-21）

用户走查「02 选择研究路径与范围」的战略范围编辑页（`/pre-investment/product-pool/new?scope=strategic`）截图反馈两处：顶部「参考：因子证据」面板对这一步没有消费场景；`UniverseFields.tsx` 里紧跟表单字段的「当前战略范围概览」（战略资产数/经济角色分布/流动性构成）只是把下方表单里已经填了的内容又摆了一遍，是纯展示、没有交互，无用。

改法：`StageLayout.tsx` 的 `factorContext` 判定原本只排除裸路径 `^/pre-investment/(objectives|product-pool)/?$`（01/02 的列表页），02 拆成独立编辑页（`/product-pool/new`）后子路由不匹配这条正则，面板又在编辑页上冒出来——这正是拆分那次改动就记录过的「已知可接受副作用」，现在用户确认它没用，就把正则扩到 `(/new)?`，01/02 两个列表页与各自的编辑页统一不显示该面板；`FactorEvidencePanel` 组件本身及其在其他阶段（SAA/TAA/组合中心/投后/产品研究/情景设置）的用法未动——那些页面因子证据有真实消费场景，用户只针对 01/02 这两步表态，不做超出授权范围的全局删除。`UniverseFields.tsx` 删掉 `roleSummary`/`liquiditySummary` 两个仅供该展示块使用的派生量与整个「当前战略范围概览」`<dl>` block；确认过它是纯 UI 展示、不参与 `issue` 校验或 `previewUniverse`/`confirmUniverse` 提交（那两处直接读 `draft.assets` 原始数组），删除不影响任何契约。`StrategicScopeWorkspace.tsx` 里已保存只读视图自己的摘要行（`只读战略范围：{name} · {as_of}` 那一段，`UniverseFields` 之外的另一处代码）不在此次改动范围内——那是保存后的摘要收据，不是用户截图指的编辑态展示块，未询问用户前不当作同一件事处理。

文档核对：`check_documentation.py` 把这两个文件关联到的十几篇 docs 标了 `needs_review`（模块级粗粒度关联），逐篇核对后均不需要改——`docs/pre-investment/README.md`「战略范围」一节本就没有把「当前战略范围概览」写成规格描述，`docs/factor-research.md`「各投研页面提供紧凑的因子证据入口」是概括性陈述，01/02 本来就不在其列举范围内，仍然成立。

已验证：`tsc --noEmit` 0 项；Vitest 全量 166 文件 / 1377 用例通过（`StageLayout.test.tsx` 新增一条覆盖编辑页不显示因子证据面板，其余用例断言未动）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空；Hermes evolve 覆盖变更文件 3/3。

未验证：浏览器验收未做，没有实际打开该页面确认面板消失、表单区视觉不留空洞。

### 大类资产构建：产品选择器改为内嵌批量选择（2026-09-21）

用户反馈「大类资产构建」（`ManualConstruction.tsx`，路由 `/pre-investment/saa/asset-classes`）里给大类挑代理产品的界面令人费解，参考风险等级配置中心 `SourcePicker`/`ReferenceEditor` 的做法给出改造建议：内嵌展开、搜索、多选清单、一次确认。用户确认只重做选择器交互本身，代理来源仍是当前已锁定产品池筛出的 ETF/基金，不新增"原始指数"代理类型，不碰 `ETFItem`/`AssetClass` 数据模型和 `/api/save-allocation`、`/api/fit-classes`、`/api/risk-parity/solve` 等后端接口，不涉及数值计算契约变更。

改法：原先点「+ 添加新的产品」弹出一个 `fixed inset-0` 全屏模态，逐个产品用裸 `<button>` 罗列、无防抖搜索（`searchQuery` 每次按键都直接发请求）；现在改成挂在触发它的那个大类卡片、紧跟在按钮下方的内嵌 `<section>`（不再挡住其余大类卡片与已有权重输入），产品列表换成 `DataTable`（补上 `caption`/`scope`/`tabular-nums` 等既有页面缺的无障碍语义），选择方式从整行按钮点击换成复选框 + 可访问名（`aria-label`），已属于其他大类的产品禁用勾选并保留原有的「已属于XX大类」提示；搜索请求加 180ms 防抖（照抄 `SourcePicker.tsx` 的节奏），新增 `productSearchBusy` 驱动 `DataTable` 的骨架态，不再无提示地转圈。确认/取消按钮换成 `components/ui.tsx` 的 `Button`（`tone="primary"` 与默认 tone），排序、每页条数、翻页三个控制原样保留，未删减任何既有筛选能力。状态仍全部留在父组件（`searchOpen`/`searchQuery`/`sortBy`/`sortDir`/`page`/`pageSize`/`selectedProducts`），只是把渲染出口从页面级全屏模态改成按 `classId` 匹配后传给对应 `AssetClassCard` 的 `picker` 节点，未新增状态管理层。

未改动：`addProductsToClass` 的写回逻辑、类内权重模式（自定义/等权重/风险平价）、NJIT 校验、拟合与相关性分析、保存/导入大类配置的两个模态、锁定产品池后才能选产品的前提、未锁定产品池时点「添加」直接跳转丢草稿的既有行为——这些都不在本次授权范围内，用户明确选择的是"只重做选择器交互"这一档，没有涉及它们。三处名字不统一（侧栏"大类资产构建"、`App.tsx`/`processRegistry.ts` 注册表描述、页面自己的 `<h1>`"资产大类构建模块"）与"未锁定产品池时跳转丢草稿"是走查时另外发现的两处可读性问题，本次未处理，留作后续单独的小改动。

文档核对：`check_documentation.py` 把改动关联到 `docs/pre-investment/README.md`、`docs/pre-investment/asset-classification.md`、`docs/verification/allocation.md` 标了 `needs_review`；逐篇核对后均不需要改——`asset-classification.md` 实际记录的是 Product-first 自动聚类算法（`/api/asset-classes/auto/preview`，对应 `AutoAssetClassification.tsx`），不是本次改的手动构建页面的选择器细节；另外两篇没有描述过这个弹窗/选择器的具体交互，无内容需要修正。

已验证：`tsc --noEmit` 0 项；Vitest 全量 166 文件 / 1377 用例通过（`ManualConstruction.test.tsx` 3 条用例的交互断言从「点击带产品代码文字的按钮」改成「勾选带 `aria-label` 的复选框」，行为覆盖范围未变）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空；Hermes evolve 覆盖变更文件 2/2。

未验证：浏览器验收未做，没有实际打开该页面核实内嵌选择器在真实窗口宽度下的观感、防抖后的搜索手感，以及禁用行的视觉状态。

### 参考因子证据面板：确认无真实消费后全部移除（2026-09-21）

用户在看到 02 战略范围编辑页上的「参考：因子证据」面板后追问：其他还在显示这个面板的页面（产品研究、SAA/TAA/其他配置页、组合中心、投后管理、情景设置）是不是也没有真的在用。先用一次只读排查确认：这个面板不是摆设——它能浏览已发布因子研究成果，还能把某次发布登记为当前研究的引用依据（`factorApi.bind`），并把已登记的引用读出来（`factorApi.bindings`）。用户看完排查结果后仍判断"目前没有真实在用"，明确要求整体去除。

改法：`StageLayout.tsx` 删掉 `factorContext` 判定与 `{factorContext && <details>…<FactorEvidencePanel/></details>}` 渲染块，以及两个只为它而加的 import（`FactorEvidencePanel`、`type ContextType`）；`isPackageWorkspace` 因为还被别处用到，保留。删除 `frontend/src/components/FactorEvidencePanel.tsx` 整个文件——排查已确认它只被 `StageLayout.tsx` 引用，删除后无孤立引用。`services/factorResearch.ts` 里的 `factorApi.bind`/`factorApi.bindings` 两个方法只被这个面板调用，随之一并删除，避免留下没有调用方的死代码；其余方法（`releases`、`getRun`、`monitor` 等）原样保留。

明确不动的部分：独立的因子研究中心页面（`FactorResearchCenter.tsx`）、`ReleaseWorkbench.tsx`、后端 `backend/factor_research/service.py` 的绑定存储与 `service.get_bindings()`、`/bindings` 两条路由——这些不是本次讨论的对象，且 `ReleaseWorkbench.tsx` 的「检查更新与漂移」仍在读同一份绑定数据（`monitor().bindings`）展示"已登记 N 个投研引用"，删除面板不影响这条只读展示路径，只是从此没有任何入口能再登记新引用。

已知、如实告知的后果：以后任何页面都不能再新建"因子发布 → 研究场景"的引用记录；已有的引用记录仍然可以在因子研究中心某个发布的「检查更新与漂移」里看到。如果之后需要恢复登记能力，需要新建一个 UI 入口（可以是重新做一个精简版面板，或者把登记功能并入因子研究中心自己的发布详情页），不是简单地撤销本次改动就能找回——因为本次一并清理了前端调用方法，回滚需要重新决定新入口长在哪个页面上。

关联清理：`docs/repo_map.json`（两处，`entry_files`/`key_symbols`）、`docs/task_routes.json`（关键词列表）、`docs/pitfalls.json`（P19 的 `related_paths`）里指向 `FactorEvidencePanel.tsx`/符号名的引用一并删除，避免路由记忆指向不存在的文件；P19 本身覆盖的因子时序语义、口径混用等问题依旧成立，未删除该条目，只删了失效的一个路径。`docs/factor-research.md`「发布与应用」一节的"各投研页面提供紧凑的因子证据入口"改写为准确描述当前能力（可停用发布、可查看已登记引用，登记入口已随面板移除、注明日期）。

本次改动过程中注意到仓库工作区里还有大量与本任务无关、早于本次对话已存在的未提交改动（`frontend/src/components/strategic-scope/UniverseFields.tsx` 等文件在本任务执行期间又发生了進一步变化，切换成了基于 `ReferenceEditor` 的指数/产品代理选择）——这些不是本次改动的一部分，本次未touch、未依赖、未验证其正确性，只是如实记录观察到的现象。

已验证：`tsc --noEmit` 0 项；Vitest 全量 166 文件 / 1378 用例通过（`StageLayout.test.tsx` 删掉了对该面板的 mock 与全部可见性断言，包括一条整体只为验证该面板而写的用例）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空；`evolve_ai_routing.py` 覆盖变更文件 7/7；`validate_ai_routing.py` 未新增失败项（既有的"未跟踪路径"红仍是此前记录过的那几条，与本次改动无关）。

未验证：浏览器验收未做，没有实际打开 SAA/TAA/组合中心/投后管理/产品研究/情景设置几个页面核实面板确实消失、页面顶部留白是否正常。

### 02 大类资产配置界面改为密集清单 + 常用大类快速添加（2026-09-22）

用户的反馈是：上一轮只在大类名称输入框上叠了一个「常用大类」下拉，页面本身没有变化，仍然信息分散、密度低、操作不友好，要求重新设计整页并说明常用大类如何与页面融合。

先做了一次外部 UI 调研（`$ui-design-research`）：Ant Design Table 的 editable rows / expandable rows（已查文档，每行一套编辑控件、操作列在最右、"Add a row" 追加新行）、PatternFly Inline edit 设计指引（已查文档，"all editable elements can be viewed within the row or expanded row" 才用行内编辑，复杂内容放展开区或模态）、Pencil & Paper 的企业数据表模式分析（已查文档，展开行是展示行内详情最直观的方式；行间用分隔线而非斑马纹；数字右对齐并用等宽数字；状态用字重/色标承载而非只靠底色）。Carbon 的 Data table usage 页两次抓取都只返回截断内容，未取到，这条来源留空。结论是把"一个大类一段纵向表单"改成"一个大类一行、明细按需展开"，并把复用入口从行内下拉提升为清单上方的快速添加区。

改法（只动 02，08 与 LTCMA 不变）：新增 `frontend/src/components/strategic-scope/CategoryEditor.tsx`，`UniverseFields` 改为调用它，上一轮加在共享组件 `risk-scales/ReferenceEditor.tsx` 上的 `categoryReuse` 入参、`risk-scales/CategoryNameField.tsx` 整个文件、`risk-scales/shared.tsx` 里的 `useReuseCategorySuppression` 全部回退删除，`ReferenceEditor` 恢复到本次改动前的状态，08 风险等级配置中心与 LTCMA 的 `fixedAssets` 用法逐字节不变。`UniverseFields` 里 `ReferenceInputRequest` ↔ `StrategicAsset` 的适配逻辑（含现金/非现金切换时的 role、liquidity 契约）原样保留，本次只换渲染层，不碰映射规则。

具体信息架构：一行一个大类（大类名称 / 资产类型 / 再平衡或现金预期年化收益率 / 研究代理与状态 / 操作），代理成分、权重与「备注（选填）」收进该行展开区（`hidden` 属性切换，默认收起，新增的大类自动展开）；宽屏用一行表头承载列名，窄屏在每个控件上方显示字段名，控件本身始终带 `aria-label`；每行给出状态徽章（已配代理 / 待选代理 / 待填收益率 / 权重合计 xx%），标题行给出「共 N 个大类 · M 个可用 · K 个待完成」；原来逐个大类重复的口径说明合并到清单上方只出现一次；百分比输入补可见的 `%` 单位；保存操作条改为卡片底部常驻（`sticky bottom-0`）；空清单改用 `Empty` 说明下一步。`ScopeMandateSummary` 的目标名称、期限与三项指标由上下堆叠改为同一行，页面顶部省下约一屏高的四分之一。

常用大类的融合方式：候选来自已保存的其他战略范围（`getStrategicCatalog()` 已有数据，无新增后端接口），在清单上方渲染成一排可点标签，点一下直接追加一个已配好代理的大类；已在清单中的名称、以及已有现金大类时的现金候选置灰并说明原因；标签上的「不再推荐」只影响建议列表，隐藏后可「恢复已隐藏的 N 个」整组恢复。该隐藏名单存在本机 `localStorage`，不是后端字段，跨浏览器/设备不同步——这是明确的简化，不是已实现的账号级偏好。

验证：`tsc --noEmit` 0 项；`vitest` 全量 166 文件 1414 用例，1413 通过；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空（新增 19 条 `riskScales.*`，删除 3 条随组件一起废弃的键）。浏览器验收用 `e2e/research-scope-ui.spec.ts` 在 mobile-320 / tablet-768 / desktop-1440 三个视口实际跑通，新增用例「常用大类可一键补齐、隐藏与恢复建议」覆盖标签点击带入代理、已添加置灰、隐藏与恢复，并在三个视口通过 `expectNoOverflow` 与 `auditTextContrast`；夹具里的已保存战略范围补了 `research_proxy`，否则建议区没有候选可测。

未通过、且确认与本次改动无关的既有红灯（都在仓库里早于本次的未提交改动范围内，本次未修）：`StrategicScopeWorkspace.test.tsx` 的「独立战略进入 LTCMA 只发战略 ID」——该用例直接渲染 `LtcmaWorkspace`，不经过 02 的编辑器；`research-scope-ui.spec.ts` 的「范围库在两条路径可搜索…」「范围库多行时…把工作区带入视口并聚焦」「产品范围可编辑保存、同名阻止并删除」三条，分别卡在列表页的投资目标下拉、编辑页拆分后已不存在的聚焦行为、以及 Product first 工作区折叠的研究名称输入上，均属 02 拆页那轮遗留的待更新用例。

未验证：真人手动点选未做，只有上述固定夹具下的自动化浏览器验收。

### 代理来源候选清单限高内滚（2026-09-22）

用户反馈：搜索一个指数或产品时，结果列表长度不受限，整块把下面的其他大类配置顶出屏幕。

改法：`components/ui.tsx` 的 `DataTable` 新增可选 `maxHeight`，传入时容器改为 `overflow-auto` 并加 `style.maxHeight`，表头 `<th>` 加 `sticky top-0 z-10 bg-slate-50`（底色必须加在 `th` 上，`thead` 的底色不为 sticky 单元格绘制，行会从透明表头下面透出来）；不传时行为与原来完全一致（仅 `overflow-x-auto`，表头不吸顶）。`risk-scales/SourcePicker.tsx` 传 `maxHeight="24rem"`。该选择器由 02 战略范围与 08 风险等级配置中心共用，属同一处根因，两个入口同时生效，未再复制第二份实现。这是准则 6.3「一个页面一个滚动容器」的例外，已在该条款写明适用边界（只给展开在表单行内部的候选清单用，不得给页面主表格套固定高）。

验证：`tsc --noEmit` 0 项；`ui.test.tsx` 6 用例通过（新增一条断言限高、`overflow-auto` 与表头吸顶带底色）；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空（无新增文案）。浏览器验收用 `e2e/research-scope-ui.spec.ts` 新增用例「代理候选清单限高内滚，表头吸顶，不把后面的大类顶出页面」：该用例在共享夹具之上单独覆盖目录接口、返回 20 条结果（共享夹具本身未改，只回 1 条指数，看不出清单会不会顶开页面），断言清单可视高度 ≤ 384px 且 `scrollHeight` 更大（即在清单内滚动）、滚到底后表头仍贴在清单顶部、第二个大类与清单底部的距离 < 400px，三个视口（mobile-320 / tablet-768 / desktop-1440）全部通过，并留存 `scope-source-picker-capped.png`。08 的选择器由既有用例「代理来源批量选择跨搜索、翻页与类型保留，确认与取消可控」在同一轮中跑通（`/settings/risk-scales/new`），未回归。

同一份 spec 的 24 条通过、9 条失败仍是上一条目记录的三条既有红灯（× 3 视口），与本次改动无关，本次未修。

未验证：真人手动滚动未做，只有固定夹具下的自动化浏览器验收。

### 大类与研究代理编辑器合并为共享组件（2026-09-22）

用户要求：把 02 那套大类配置界面的优化同样用到 08 风险等级配置中心的大类选择与映射配置上，并排查其他需要「选大类 + 给大类挂产品/指数」的地方是否要同步；这个前端模块应当作为共享模块。

排查结果（按「是否同一件事」分档）：

- 同一个组件、同一份数据契约（`ReferenceInputRequest`），本次全部合并：02 研究范围（`strategic-scope/UniverseFields`）、08 风险标尺第 2 步「参考资产与代理」（`pages/RiskScaleWorkspace`）、05 LTCMA 统计模型的「研究代理」（`ltcma/LtcmaStatisticsFields`）。
- 同一类交互、不同数据源，本次只做限高：06 手动构建「从可投资域添加产品」（`pages/ManualConstruction`）是在大类行内展开的产品检索表，来源是已锁定产品域而不是指数/ETF 目录，`DataTable` 加 `maxHeight="24rem"`，其余不动。
- 形态相近但不是一回事，本次未动并记录原因：`strategic-scope/ImplementationMappingEditor`（大类→真实代理产品，候选来自已锁定产品域的大类方案下拉，没有目录检索）；`pages/PortfolioConstruction` 的「产品搜索结果」是一个未限高的 `<ul>`（20 条/页）；`pages/AutoAssetClassification`（产品归类到大类，方向相反，清单本来就有 `max-h-*`）；`pages/ClassAllocation`（用产品组装大类的研究工具，数据模型是 `strategyWeights`）。这几处要不要跟进由人类决定，未擅自改。

改法：`strategic-scope/CategoryEditor.tsx` 移到 `risk-scales/CategoryEditor.tsx`（与它依赖的 `SourcePicker`、`shared` 同目录，三个调用方都从这里引），并补齐原 `ReferenceEditor` 独有的能力——`fixedAssets`（大类由上游冻结：名称只读、不能增删、无模板与「常用大类」建议区）、`cashEligibleIds`（只有指定大类可切成现金）、不传 `renderAssetDetails` 时展开区兜底渲染「经济定义与依据（可选）」、展开区补一行当前再平衡规则的口径说明。同一变更中删除被取代的 `risk-scales/ReferenceEditor.tsx`，不保留新旧双实现；`ReferenceEditor.test.tsx` 随之改名为 `CategoryEditor.test.tsx`，三条原有契约（现金只填收益率、切非现金默认每日再平衡、批量选择跨搜索保留且原子追加）按行式构图改写后保留，另加一条覆盖 `fixedAssets` + `cashEligibleIds`。`docs/repo_map.json` 的模块说明、related_tests 与最小回归命令同步改名。

08 与 LTCMA 的行为差异（用户已授权的界面变化，不是静默删能力）：每个大类的代理明细、权重与经济定义从「一直摊开」改为收在该行展开区，行内改为显示状态徽章与代理摘要；新增的大类自动展开，所以「添加大类→填名称→挑代理」一路不用多点。逐个大类重复的现金/非现金口径说明合并到清单上方一次。数据契约、校验、冻结与发布链路一律未动。

顺带修正：`e2e/risk-scales.spec.ts` 里挑代理那步还停在 `SourcePicker` 改批量选择之前的写法（点一个不存在的「选择」按钮），本次改为勾选复选框后点「确认选择」。

验证：`tsc --noEmit` 0 项；`vitest` 全量 166 文件 1419 用例、1418 通过，唯一失败是既有红灯 `StrategicScopeWorkspace.test.tsx`「独立战略进入 LTCMA 只发战略 ID」（直接渲染 `LtcmaWorkspace`，卡在人工假设输入，不经过本组件）；`CategoryEditor.test.tsx` 4 条全过；`check_frontend_design.mjs` 无回归；`check_i18n.mjs` errors 为空（无新增文案，复用既有 `riskScales.*` 键）。浏览器验收：`e2e/risk-scales.spec.ts` 用本地夹具后端（`backend/tests/risk_scale_app.py`）跑通 08 全链路——新建标尺 → 逐个添加大类与代理 → 检查并确认参考输入 → 计算前沿与 C1–C5 → 发布不可变版本，1 passed；02 的 `e2e/research-scope-ui.spec.ts` 复跑确认无回归。

未验证：05 LTCMA 的研究代理区只有组件级用例（`fixedAssets` / `cashEligibleIds`），没有浏览器链路覆盖——既有 LTCMA e2e 本来就不走这一段；真人手动点选未做。

### 情景算法中心改为「先看已保存，再进编辑器」（2026-09-22）

用户反馈：13 情景算法中心没有展示已保存的情景，一点进去就是算法编辑页面；历史识别算法、实时识别算法、情景模拟压力测试都有这个问题，希望参考 01 投资目标与约束的「标题 + 说明 + 已保存列表 + 右上角新建」。

排查结论（三档）：

- **落地页就是编辑器，本次改**：市场状态研究的三个步骤（`pages/MarketStateResearchCenter`）此前每个步骤直接挂 `HistoricalRegimeWorkbench`，已保存研究只出现在工作台左侧「情景算法库」面板和底部「历史版本与旧版迁移目录」抽屉里。
- **已经有清单但埋得深，本次只调入口**：情景模拟与压测（`pages/PublishedScenarioCenter`）默认停在「已发布情景」，但「构建情景」和它并列成第三层页签，「新建情景」只切本地 state。
- **本次未动并记录原因**：全球历史事件库的「人工历史事件」子工作区（`regime-workbench/GlobalEventCenter` 的 `event_view=manual`）同属一类问题，但用户未点名，且事件库本身已是清单，留待人类决定。`pages/HistoricalRegimeCenter.tsx`（1386 行）除自己的测试外没有任何引用、没挂路由，属存量死代码，按 AGENTS.md 应删，但不在本次授权范围，已单独提请人类确认。

后端不用改：`GET /api/historical-regimes/v2/definitions`（含 `study.purpose`、`revision`、`created_at`/`updated_at`）、`/runs`、`/references`、`/reliability/catalog` 已提供清单所需的全部字段。已知代价：`list_regime_graph_definitions`（`backend/services/historical_regime_routes.py:306`）会对每条定义跑一次 `infer()` 拼 `temporal_capabilities`，清单页按 `study.purpose` 归档并不需要它；定义数量变大后应改为按需（`?capabilities=1`）或缓存。本次未改后端。

改法：新增 `regime-workbench/RegimeStudyList.tsx`，三步共用一个组件——`SectionHeader`（步骤标题 / 说明 / 右上角新建）+ `DataTable`（名称、状态徽章、绑定的历史参考或研究结构、精确版本、正式运行次数、最近更新、继续研究）。状态徽章按步骤取值：历史参考为已确认参考 / 已运行未确认 / 尚未运行，实时模型为已发布 / 已运行未发布 / 尚未运行，验证步骤为已验证可识别 / 已出报告未达可识别 / 尚未验证。清单与工作台共用同一条路由，靠 `definition`（+`revision`）、`template`、`new` 区分，工作台顶部给「返回…清单」；这沿用投前各页「地址栏确定页面身份」的既有约定，20 多条带精确版本的深链接语义不变。归档判定抽成 `regimeStudy.studyBelongsToPurpose`，清单与工作台侧栏共用，不写两份过滤。

刻意保留的行为：两步之间的历史参考交接原来靠 `MarketStateResearchCenter` 的 React state（`onReferenceReady` → `incomingReference`）。拆出清单后工作台改为隐藏而不卸载，「下一步：建立实时识别」跳 `stage=realtime&new=1` 直接打开空白实时模型，交接提示和两侧草稿保留与原来一致。跨历史/实时仍清掉另一侧的 `definition/revision/template`，并额外清 `new`。

验证：`tsc --noEmit` 0 项；`check_frontend_design.mjs` 无回归（`unnamed-tables` 仍 48）；`check_i18n.mjs` errors 为空；`RegimeStudyList.test.tsx` 4 条、`ScenarioCenters.test.tsx` 7 条、`PublishedRiskFlow.test.tsx` 21 条、`HistoricalRegimeWorkbench.test.tsx` 27 条、`HistoricalRegimeDirectory.test.tsx` 4 条、`regimeStudy.test.ts` 5 条通过。浏览器：`e2e/market-state-research-center.spec.ts` 6/6（320/768/1440 三视口，含对比度与无横向溢出断言，覆盖三步清单空态、新建进工作台、返回清单）；`e2e/regime-reference-confidence.spec.ts` 需先 `npm run build` 并以 `REGIME_OFFLINE_BUILD=dist` 运行，15/18 通过。

既有红灯（非本次引入）：`regime-reference-confidence.spec.ts`「无参考与目录错误都有下一步」在三个视口断言未绑定参考时「验证识别能力」按钮 disabled，但工作台的参考优先门禁在该状态下整块隐藏模型配置（页面显示「先完成历史参考选择与状态对应，再开始模型配置。」）。`git show HEAD` 确认门禁文案与该断言在 HEAD 上已并存，与本次改动无关，未擅自改断言。

未验证：`market-state-workflow.spec.ts`、`regime-full-stack.spec.ts`、`csi300-reference-study.spec.ts`、`scenario-remediation.spec.ts` 这几条需要真实后端或专用夹具服务的链路本轮没跑，其中 `market-state-workflow.spec.ts` 两处空白实时模型入口已按新 URL 补 `&new=1`，但未实测；情景模拟压测的「返回情景库」只有组件级覆盖，没有浏览器链路。


## 风险标尺读取优化（2026-09-22）

作用域为风险标尺编辑页的初始化、展示名称补全，以及共用指数/ETF/基金目录查询。详细设计与接口边界见[风险标尺读取设计](../pre-investment/risk-scale.md#编辑页读取与名称查询设计)。本次未修改计算算法、来源资格或历史冻结数据。

- 旧标尺含 13 个不同指数。优化前编辑页约 29.8 秒才显示表单；优化后在 1440/768/320px 分别为 1.086/0.799/0.784 秒可编辑，名称完成为 1.324/0.873/1.822 秒（含导航和页面加载）。每次仅一个批量名称请求，零逐指数目录查询；这是本机开发服务实测，不是固定响应承诺。
- 批量名称 API 实测 0.079–0.127 秒，全部 13 个精确来源均找到；单指数目录查询稳定重复测量为 0.323–0.383 秒，优化前约 1.36–1.41 秒。服务刚重启的首次目录请求仍可能包含文件校验成本，实测 2.40 秒；没有引入目录缓存或省略计算时校验。
- 后端 `test_risk_scale_service.py`、`test_research_series_routes.py`、`test_regime_product_sources.py`、`test_risk_scale_reference.py` 合计 108 passed。覆盖轻量元数据读取、精确来源/重复 ID、请求边界、快照切换、离线失败、过滤/状态/分页等价和文件变化后指纹更新。
- 前端相关四文件 `RiskScaleFlow.test.tsx`、`RiskScaleStandalone.test.tsx`、`riskScales.test.ts`、`RiskScaleInputs.test.tsx` 共 50 条通过；涵盖编辑/复制/草稿、名称等待中编辑保存、失败/缺失重试、超时/取消、跨方案晚到响应拒绝及冻结身份错误阻断。全量结果为 1509 passed / 1 failed：`StrategicAllocationModels.test.tsx` 仍用 `selectOptions` 操作已改为按钮菜单的 LTCMA 生成方法控件，本次未改该页面或测试。
- 真实浏览器三视口分别验证正常加载、名称请求延迟与失败后重试，共 6 个场景通过；输入修改保留、无页面横向溢出，复用现有文字对比度检查通过，并人工检查桌面与窄屏截图。
- TypeScript、Vite build、设计检查、语言检查、风险标尺生成契约检查通过。服务重启后已复核前端页面、API 代理、完整 NJIT/worker 预热及数据目录在线；启动脚本末次状态探测曾短暂失败，随后独立 GET 与 `status` 均通过。
- 文档结构/链接及本任务 Hermes 路由覆盖通过。全工作区 Hermes validate 仍受已有未跟踪源码引用及并行新增 P28 的无效模块名影响；本任务仅为原 P07 补充名称读取坑点，并更新既有模块说明，没有为通过检查改动这些其他任务内容。

### 投前 01–09 的等待态统一为 `LoadingPanel`（2026-09-22）

起因是用户在投前 03「新建 LTCMA」看到一行「正在读取 LTCMA…」加两块灰色占位，指出等待中应该出现走路的小牛形象，并要求把它做成公共能力，覆盖整个投前研究的加载界面。

排查到的现状：`h-12` + `h-24` 两条 `animate-pulse` 灰色块在 01/03/06–09 逐页抄了 6 份（`InvestmentObjectivesCenter`、`InvestmentObjectivesWorkspace` ×2、`LtcmaCenter`、`LtcmaWorkspace`、`LtcmaVersionView`、`PreInvestmentImplementation`），04/05 则退化成一行纯文字（`StrategicAllocationWorkspace` ×2、`TacticalAllocationWorkspace`、`PolicyFrontier` 用 `Skeleton`），02 的范围初筛上一轮已单独处理过。

改法：在 `frontend/src/components/ui.tsx` 新增 `LoadingPanel({ text, className, mascot })`——预留高度内居中放 `working` 姿势加一行说明，不铺灰底色块；11 处整块等待态改为调用它，被取代的灰条与那份 `Skeleton` 在同一变更里删除，不保留旧写法。设计准则 8.1 第 1 条据此改写为「整块区域在等数据、且区域内此刻没有别的内容用 `LoadingPanel`，其余仍是骨架屏或纯文字」。

未接入的等待态及理由：`DataTable` 的 `loading` 骨架行（15.2 第 4 条表格内部禁区，骨架本来就占住行形状）；`MandateRiskFields`、`LtcmaScenarioFields`、`LtcmaNiwStrengths`、`BaselineSetup` 的字段内联等待（旁边就是已渲染的表单控件）；`StrategicScopeWorkspace`、`ImplementationMappingEditor` 的「正在读取…」行（下方表单已有内容）；`AutoAssetClassification` 第 ⑤ 节、`ClassAllocation` 方案列表、`TimingResearch` 运行进度、`ManualConstruction` 的全屏遮罩（同屏已有业务数值，15.2 第 1 条）。`InvestmentObjectivesWorkspace` 的诊断计算中只在尚无上一版结果时用 `LoadingPanel`，重算时退回纯文字，同样是为了不让形象挨着上一版数字。

一并修掉的检查漏洞：`mascot-in-forbidden-zone` 按文件名机械拦截，只认 `<Mascot` 与 `<EmptyState`。形象一旦经 `LoadingPanel` 间接引用，`StrategicAllocationWorkspace.tsx`、`TacticalAllocationWorkspace.tsx` 这类命中禁区正则的文件就能绕过拦截而检查仍报 0。`scripts/check_frontend_design.mjs` 改为把 `<LoadingPanel`（未传 `mascot={false}`）一并算作使用处，并用 `MASCOT_LOADING_EXCEPTIONS` 逐个登记允许的等待态例外；豁免只对 `LoadingPanel` 生效，同一文件里直接用 `Mascot` 或带形象的 `EmptyState` 仍然算违规。例外与准则 15.3 第 8 条同步，坑点记为 `docs/pitfalls.json` P29。

已验证：`tsc --noEmit` 0 项；Vitest 全量 174 文件 / 1513 用例，1 条失败与本次无关——`StrategicAllocationModels.test.tsx:86` 对 `生成方法` 调 `user.selectOptions`，而该控件已被另一会话未提交的新文件 `frontend/src/components/ltcma/LtcmaMethodSelect.tsx`（`git status` 显示 `??`）换成按钮式下拉，该测试文件当前 diff 未覆盖这一行。`ui.test.tsx` 新增 2 例覆盖 `LoadingPanel` 的居中、预留高度、无灰块与 `mascot={false}`；`ScopeFeasibility.test.tsx` 6 条通过。`check_frontend_design.mjs` 无回归，`mascot-in-forbidden-zone` 仍 0 且使用处列表如实列出 21 个文件；`check_i18n.mjs` errors 为空（未新增文案）；`evolve_ai_routing.py` 覆盖变更文件 16/16；`validate_ai_routing.py` 未新增失败项（既有"未跟踪路径"红来自另一会话的新文件）。

浏览器验收：临时用例把所有 `/api/**` 响应延迟 8 秒，在 1440 与 320 两个视口实拍 01 目标清单、03 LTCMA 列表、03 新建 LTCMA 三处等待态，6/6 通过——同屏恰好 1 个形象，宽 80px，水平中心与等待区中心误差 ≤1px，页面内 `animate-pulse` 计数为 0，无页面级横向溢出，对比度审计只剩既有的禁用态「重试读取范围目录」2.64:1（`actionClass` 的 `disabled:opacity-50`，共享原语的既有问题，本次未改）。临时 spec 与临时 config 已删除。

补充（同日）：LTCMA 提交计算时只有一行「正在核验来源并计算，请勿重复提交。」，看不出系统在算。`LtcmaWorkspace` 的该提示按步骤分流——输入步骤（`step === 0`，屏幕上只有用户填写的表单控件）改用 `LoadingPanel`；结果步骤上预览数值已经渲染，按 15.2 第 1 条仍只留纯文字。草稿保存共用同一分支，文案未改。这是 15.2 第 1 条的一条具名例外，已写入设计准则 15.2 的「允许出现在」段落，并在 15.1 的已接入清单里标明适用步骤。检查脚本无需新增豁免：`LtcmaWorkspace.tsx` 不命中 `mascot-in-forbidden-zone` 的文件名正则。

该补充的验证：`tsc --noEmit` 0 项；`StrategicAllocationModels.test.tsx` 6 条中 5 条通过（`discards late model preview when clock changes` 内新增一行断言，挂起预览期间形象可见），唯一红灯仍是上文那条与本次无关的 `生成方法` 用例；`check_frontend_design.mjs` 无回归、`mascot-in-forbidden-zone` 仍 0；`check_i18n.mjs` errors 为空。浏览器：临时用例把 `/cma/preview` 延迟 6 秒，在 1440 与 320 实拍输入步骤计算中，2/2 通过——同屏 1 个形象、80px、相对状态区居中误差 ≤1px、页面内 `animate-pulse` 计数 0、无横向溢出、对比度审计为空（禁用中的按钮被审计器按 `[disabled]` 跳过），预览返回后形象随分支消失。临时 spec 与临时 config 已删除。

未验证：04 SAA、05 TAA、06 择时、06–09 实施收尾四处的等待态没有实拍，只有静态检查与类型检查覆盖；它们与已实拍的三处走同一个 `LoadingPanel`，但页面各自的留白与相邻区块未经浏览器核对。
## LTCMA 与情景列表读取优化（2026-09-22）

作用域：LTCMA 首屏、方法资料加载、共享情景运行及参考目录。基于本轮共享工作区代码和在线 SSD 数据；同文件并行的等待态插画修改不计入本任务。

- 原因：首屏等待完整情景资格列表；一次选项请求重复解析约 18 MB 的运行 JSON 12 次，情景相关处理约占九成耗时。
- 实现：`study-options` 按 `base/priors/regimes/scenarios` 分组读取，原无分组 API 保持兼容。方法资料独立取消、局部重试，日期或页面身份变化后忽略旧响应，保留用户编辑。运行目录只在单次只读操作内复用 JSON 和 ID 索引；下一请求重新读取与校验。前端共用列表显式请求 `summary=true`，投影先于深拷贝，完整详情仍按 ID 读取。
- 同一完整研究选项响应逐字段相等：本地只读调用约 2.35 秒降到 0.81 秒，运行 JSON 读取 12 次降到 1 次。
- 重启后实测：基础选项 0.188 秒、3,964 字节；情景选项 0.672 秒。运行列表摘要 21 条、321,716 字节、0.171 秒；原完整列表同批读取为 11,350,683 字节、1.112 秒，此前实测 1.729 秒。未清除操作系统缓存，这不是磁盘裸速测试，也不表示所有页面均已优化。
- 真实浏览器 1440／768／320 首屏表单分别为 577／523／528 毫秒，此前两次导航为 3,773／3,028 毫秒。每次进入仅请求基础选项；切换 NIW、长期情景后才读取对应资料。人工延迟及 503 故障验证编辑、草稿按钮、计算阻断、重试和输入保留；三视口文字对比度、无横向溢出检查通过。没有向真实业务数据保存测试研究。
- 后端 89 项回归通过，覆盖 V1/V2 摘要、完整详情、按需分组、PIT、冻结校验、单次读取、下一请求发现篡改、线程隔离和异常释放。前端全量 174 文件、1,516 项通过；TypeScript、生产构建、设计与 i18n 检查通过。同步修正了受影响测试的查询参数，以及生成方法由原生选择框变为菜单后遗留的旧交互断言。
- 服务按 8001／5175 重启，Numba 与 worker 完整预热后就绪。没有新增跨请求资格缓存、数据库依赖或重写冻结产物；计算和发布仍执行原完整校验。
- Hermes 本任务路径覆盖无遗漏；全仓路由校验仍受其他工作区未跟踪稳定文件影响，未放宽规则或擅自暂存文件。

本机临时证据：`/tmp/ltcma-loading-browser.json`、`/tmp/ltcma-read-counts-after.log`、`/tmp/ltcma-load-api-after.json`、`/tmp/ltcma-load-browser-results.json`、`/tmp/ltcma-load-backend-test2.log`、`/tmp/ltcma-load-frontend-full-final.log`。


### 01 「添加投资目标与约束」改为每次从空白开始（2026-09-23）

用户反馈：在 01 投资目标与约束点「添加投资目标与约束」，进去的表单里已经带着上一次的目标名称、投资期限、收益下限和现金下限，不是空白页。

根因不在页面的初始值分支，而在草稿 hook 的读取顺序。`InvestmentObjectivesWorkspace` 用 `useAllocationDraft('mandate-study:editor', ...)`，scope 是常量，只在初始值函数里用 `!editFrom && !fresh` 表达「新建要空白」。但 `useAllocationDraft` 的实现是 `decode(stored(key), fresh())`——先读 localStorage，初始值只是读不到时的兜底。同一个 key 上只要留着上一次没填完的草稿，初始值函数返回什么都不会上屏。hook 自己的注释「scope must include the explicit identity」说的就是这件事：身份要写进 key，不能写进初始值。02 的 `ProductPoolWorkspace` 与 `StrategicScopeWorkspace` 早就按这个写法把 `new=<令牌>` 拼进草稿 key，只有 01 没有。

改法（前端三个文件，不动后端与数据契约）：`InvestmentObjectivesCenter.tsx` 的「添加」链接（表格上方与空态两处共用同一个 `addHref`）改成 `objectives/new?fresh=${crypto.randomUUID().slice(0, 8)}`，每次渲染重取；`InvestmentObjectivesWorkspace.tsx` 把 `fresh` 由布尔改成令牌字符串，草稿 key 随之变成 `mandate-study:editor:new:<令牌>`，并删掉原先挂载后把 `fresh` 从地址栏抹掉的 effect（key 依赖它，抹掉就会在下一次渲染滑回固定 key 并把旧草稿捞回来；留着它刷新才不丢正在填的内容）；`save()` 在切地址栏到 `?view=<id>` 之前先 `writeAllocationDraft('mandate-study:editor', draft)`，让 key 滑回固定值时读到的就是刚保存的这一版。固定 key 继续承担「裸 `/objectives/new` 续接上次没填完的草稿」，新建不清它。

令牌用 `crypto.randomUUID()`（仓库既有写法，`indicatorGraphAdapter.ts`、`risk-scales/editor.ts` 等都直接用）而不是 02 的 `Date.now()`：浏览器验收里 `page.clock.setFixedTime` 会把 `Date.now()` 一起冻住，两次点「新建」拿到同一个时间戳，首轮实测就是这么红的。02 现有写法不在本次授权范围，未改。

已验证：`tsc --noEmit` 0 项；`InvestmentObjectivesWorkspace.test.tsx`（新增一条「新建入口带 fresh 令牌时是空白表单，上一次没填完的草稿留在固定 key 上」）、`InvestmentObjectivesCenter.test.tsx`（「添加」href 断言改成令牌正则）、`InvestmentObjectivesBoundaries.test.tsx`、`allocationJourney.test.tsx` 合计 58 项通过；`check_frontend_design.mjs`、`check_i18n.mjs` 无回归。浏览器实测用临时用例跑 `playwright.mandate.config.ts` 的夹具服务（1440 与 320 各一轮，2/2 通过）：点「添加」进来表单为空 → 填入名称和期限 → 刷新内容仍在 → 「返回投资目标列表」→ 再点「添加」，令牌与上一次不同、名称为空、投资期限回到默认 10、现金下限 0.00%、无横向溢出；1440 全页截图已复核。临时用例与临时配置已删除。

未验证：保存后再回填的完整链路（新建→诊断→确认→`?view=<id>`）只有单测覆盖，浏览器未走；`fresh` 令牌在同一毫秒内两次点击的兼容性问题只在 02 的 `Date.now()` 写法上存在，本次未改 02。

### 读取失败态统一为 `ErrorPanel`，形象改用带问号的姿势（2026-09-23）

用户反馈（三张截图：01 投资目标与约束、02 选择研究路径与范围、03 LTCMA 中心）：后端不可达时页面只有一行「Failed to fetch」红字，主体反而给出空态文案——01 显示「还没有投资目标与约束」，02 的表格空行写「范围库读取失败；请重试。」。要求读取失败时在页面中间展示带问号的卡通牛，并且做成前端公共能力推给各页面。

问题不是某一页写错，而是缺一个共用原语：8.1 的整块等待态在 2026-09-22 已经收敛到 `ui.tsx` 的 `LoadingPanel`，与它成对的失败态却仍由每页自行拼「一行红字 + 重试按钮」，读取失败与「本来就没有数据」在界面上分不开。因此在 `ui.tsx` 加 `ErrorPanel`（`message` / 可选 `title` / `action` / `className` / `mascot`），占 `LoadingPanel` 同一块 `min-h-56` 预留高度、互斥出现，`role="alert"`，形象居中、下面是失败原因和重试入口。判定统一成「读取失败且该区域此刻没有内容」，各页用 `loadFailed` / `catalogFailed` 局部变量表达，同一条失败原因只出现在一处，不与顶部提示重复。

形象按用户明确要求改用带「?」的姿势：`0011` 抓头疑惑是 11 张源图里唯一带问号的，原先 `error` 借用的 `0003` 正面垂手没有问号。按 15.3 的既有管线重新生成同名资产 `mascot-error-240.webp`（8.4K → 7.7K，六个资产合计 40.8K → 40.1K），状态键不变，代码无需改 `Mascot.tsx` 的映射表。由此 `error` 与 `noresult` 共用同一姿势，是设计准则 15.1「一个状态一个姿势」的唯一例外，已在该节写明理由（两者语义相邻、同一位置互斥出现、含义由相邻文字承载）与解除条件（补到真正的「出错」姿势后替换同名资产即可）。`check_frontend_design.mjs` 同步把 `ErrorPanel` 与 `LoadingPanel` 一并算作形象使用处，禁区登记表更名 `MASCOT_PANEL_EXCEPTIONS`，成员不变。

接入 12 处：`InvestmentObjectivesCenter`、`InvestmentObjectivesWorkspace`（版本不可读）、`ProductPoolSelection`（范围库，同时删掉已经走不到的 `rows={scopesError ? [] : ...}` 与空态失败文案）、`ScopeFeasibility`、`LtcmaCenter`、`LtcmaWorkspace`（初始化）、`LtcmaVersionView`、`StrategicAllocationWorkspace`（政策版本、目录，原 `Empty title="目录暂时不可用"` 一并替换，`Empty` 导入随之删除）、`PolicyFrontier`、`TacticalAllocationWorkspace`（目录与基准读取失败，并补上该页原先没有的重试：新增一个 `reload` 计数，两个读取的 effect 都跟着它重跑）、`TimingResearch`、`PreInvestmentImplementation`。`ProductTrendChart` 自己手写的那份「走势数据没能读出来」面板同时换成 `ErrorPanel`，同一功能不留两套实现。

刻意没换的位置：保存、删除、重算、交接这类操作失败时业务内容还在屏幕上，按 15.2 第 1 条仍是 `Feedback` 纯文字；字段级校验、`DataTable` 内部、02 顶部「投资目标与约束」下拉旁的目录读取失败（旁边就是表单控件）同样保持纯文字。

已验证：`tsc --noEmit` 0 项；前端全量单测 174 文件 / 1521 项通过（`ui.test.tsx` 新增 2 条覆盖居中、`mascot-error` 资产与 `mascot={false}`；`InvestmentObjectivesWorkspace.test.tsx` 两条失败用例的断言由「页面上唯一那条 alert」改为按文案定位，因为读取失败现在落在中间的错误态里、PIT 提示是另一条 alert，被断言的行为未变）；`check_frontend_design.mjs` 无回归（`mascot-in-forbidden-zone` 0、`same-element-contrast` 0）；`check_i18n.mjs` errors 空（未新增文案，失败原因用后端返回的消息）。浏览器实测用临时用例跑 `playwright.mandate.config.ts` 的夹具服务，把 `**/api/**` 全部 abort 模拟后端不可达，desktop-1440 与 mobile-320 各跑 01、02、03 三页，4/4 通过：形象水平居中误差 ≤1px、显示宽度 120px、同屏只有 1 个形象、无页面级横向溢出、`auditTextContrast` 为空；点「重试」恢复后形象消失、清单正常渲染。1440 与 320 的截图已人工复核。临时用例与临时配置已删除。

未验证：04 SAA、05 TAA、06 产品配置与走势图面板的失败态只有单测与静态检查覆盖，未实拍；`LtcmaWorkspace` 只在「初始化」分支换面板，输入步骤计算失败仍是纯文字；`TacticalAllocationWorkspace` 新增的重试计数只在单测与类型层面验证，未在浏览器里触发过目录读取失败后的重试。
