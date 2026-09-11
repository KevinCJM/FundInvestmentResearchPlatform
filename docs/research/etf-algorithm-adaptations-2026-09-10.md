# 14 个实验思想的 ETF 原生改编

## 需求与身份

用户在 2026-09-10 明确选择“ETF 改编版，保留算法思想”。因此交付是 `etf-v1` 模板，不是股票母池逐笔复现，不继承股票实验胜率、样本数或评分。原始依据来自 WF 总登记表、正式记录及 `backend/scripts/experiments/` 的对应入口；WF 文件与实验登记本次只读。

沿用 System Architect / Developer 的设计、实现及验证顺序。CodeGraph 用于既有注册表导航；当前索引未覆盖新择时文件时直接核对源码。未知的外部股票母池不进入生产调用链。

## 模板覆盖与具体改编

| 原实验 | ETF 模板保留的核心 | 明确改变的内容 |
| --- | --- | --- |
| A2552 | 首阳、T-1 收益/回撤、对数 Alpha-Beta、斜率转正或加速创新、收盘位置；108 组规则 | ETF 成熟交易置信下界代替原70/30训练验证加权目标；保留完整日期轴 |
| A160S-106 | 前60日最高价回撤、前20日区间位置、连续确认、开盘、MFI/CMF；108 组 | ETF 篮子横截面 MA250 广度代替全股票宽度；删除查看整个测试期无信号后回退的未来依赖 |
| A160-MOMO-141 | 两个动量专家与反转专家、市场状态、15交易日隔离 | 原股票专家及Beta收缩效用改为明确ETF专家和共享训练目标；不足证据留现金 |
| A2536-Momo-095 | 原动量候选上作高置信筛选 | 不冒充股票横截面Top-Quota；以ETF时间序列趋势/量价/同类相对动量质量筛选改编 |
| A2536-Momo-096 | 095基础上按环境状态学习准入 | 市场/行业股票状态改为ETF篮子状态；门槛与效用在训练期冻结 |
| A2536-Momo-053 | 市场—类别—产品状态，protect/neutral/downweight/gate | 明确改编为补充候选/保留/降频/空仓动作；降频不是资金仓位缩放 |
| ActionLearner-001 | 动量、反转、修复专家和组合/空仓之间选择 | RD专家替换为ETF规则型修复专家；不使用原40自然日代理标签成熟度 |
| A2067 | clean主信号优先、月内有限补位 | 单ETF候选配额替代股票跨产品配额；仅累计当时已发信号 |
| A2074 | 历史同月成熟交易决定单一来源 | ETF候选/置信效用替代股票固定高胜率门槛；样本外整体冻结 |
| A2076 | 历史同季度成熟交易决定单一来源 | 同季度分组，不声称滚动季度重训；无合格候选空仓 |
| A2140 | 独立市场状态、趋势、压力与修复门控 | 显式ETF篮子广度与低滞后代理，不冒称原全市场LLT/HMM状态 |
| A2143 | 风险期关闭主信号，只允许clean修复重入 | ETF原生确认/风险/重入步骤，分别可编辑 |
| A2276 | 风险确认、核心优先、有限低重叠补位 | ETF量价/路径/压力代理，不复用A959股票母池或横截面分位结果 |
| A2296 | 强来源合并去重、负证据否决 | ETF独立风险否决；压力释放本身不能产生买入候选 |

实际每个模板包含 `adaptation.source_experiments/preserved/changed`。用户可在页面查看差异，保存时原样冻结；执行真相在真实节点和训练配置，不在表格或前端文案。

### 原始依据中的关键校正

- A2552 的 108 组为3组联动 Alpha/Beta ×3个前期收益门槛 ×3个回撤门槛 ×2种确认 ×2个收盘位置门槛。不能解释为108套独立发现器。
- A160S-106 的市场广度是横截面概念，不能用单只ETF站上均线的历史时间比例代替。其原测试期零信号回退不迁移。
- 095/096 入口 `A2536_momo_094_096_high_confidence_gate_router.py`，与摘要中其他后代特征族筛选不是同一算法。这里只迁移核实后的高置信筛选/状态准入思想。
- 原学习器中代理标签成熟度、完整15日路径统计与实际退出日不一定等价。本实现只读取已完成交易的净收益、退出原因及日期，不用未来MAE/MFE拟合。

## 数据与执行链

`目录模板 → 可编辑 Definition → 显式 prepare → 冻结候选计划 → ETF及环境篮子读取 → 因果特征/条件图 → 训练成熟交易 → 冻结动作 → 样本外信号 → 既有T+1执行 → 不可变结果与审计`。

定义增加可选 `adaptation`、`training`，既有无训练定义仍走唯一固定规则实现。新字段不重写历史快照。128个可见步骤、512个内部数学步骤；最多8种动作、108个非现金训练候选。搜索空间可联动多个参数，不能修改数据源、状态依赖或退出依赖，否则候选和部署语义不一致。

运行增加 `context_baskets.market/category`，每个实际使用的篮子必须显式指定2–12只ETF。不得从单一研究产品推导“市场宽度”。成员×时间使用固定float64矩阵；类型包含 `panel` 与 `condition_panel`，不同成员身份的矩阵不能按位置相加或比较。聚合为时间序列后可以做相对动量。

每个成员仅在任务内读取一次，篮子与研究产品锁定同一行情快照/价格口径。按目标日期对齐，成员上市前/缺行情留NaN，任何成员未知则当日广度/均值未知；不删日期、不变分母、不虚构成分股历史。结果保存成员顺序、源指纹、矩阵校验和与固定篮子偏差提示。

篮子中间步骤可逐成员预览，最终比例可直接预览。默认参数的特征图与训练后最终入场是不同输出；最终入场单独展示，选中参数列于冻结审计，不将默认值图冒称所选参数图。

## 算子颗粒度审查

| 能力 | 唯一职责/可替换中间量 | 复用与不再拆分的理由 |
| --- | --- | --- |
| 普通数学与滚动 | 数值特征 | 复用统一typed注册表和NJIT计划，没有新MFI/CMF黑盒或重复滚动代码 |
| panel_formula / panel_lag | 把同一个系列能力映射到每个成员 | 仅布局/调度适配，调用同一公式计划与同轴内核；行视图可共享 |
| panel_compare | 对成员输出三值条件 | 复用共享condition_compare；类型和阈值等号不变 |
| cross_mean / breadth | 当日固定成员归约/条件比例 | 复用共享轴均值、三值转数值；与趋势计算分开 |
| Alpha-Beta | 水平、斜率、创新的联合递推 | 复用既有耦合内核；缺失重置，一遍观察后终止；外围比较独立 |
| priority_quota | 主信号优先、月内有限补位 | 当月计数与跨月冷却共享因果状态，不能拆成事后月总数；全日期扫描终止，候选不等于成交 |
| encode_states | 显式条件组合编码 | 0/1位编码、任一未知为-1；只承载状态身份，不把数字大小当距离 |
| score_trades | 已成熟交易按状态估计保守效用 | 成熟/分组边界和Welford统计融合避免每组复制；统计结果可独立查看，与选择分开 |
| choose_actions | 有限效用中选过门槛的最高者 | 独立于特征、拟合与应用；同分按配置顺序，不足证据空仓 |
| route_actions | 冻结选择映射为当日入场 | 不训练、不读取收益；状态未知仍未知，现金为明确0 |
| simulate | 推进持仓/现金/交易状态 | 复用既有固定签名T+1内核，规则与评价均在外部 |

所有完整家族只作为组合模板，插入后是真实连接，不注册整套“Axxxx黑盒算子”。上述训练阶段分别可审计，前端低码表单持久化同一类型契约。当前未新增把训练过程序列化为通用指标公式的语言；训练是独立研究阶段，不伪装为普通因果特征。

## 冻结与因果边界

- 训练交易必须满足 `signal >= study_start`、`exit < split - embargo_bars`，且已实际平仓；未完成头寸不强制当标签。
- 冻结日为样本外首日前一个交易日收盘。可用该日信号在首个样本外开盘交易，不能用首个样本外日收益拟合。
- 目标 `mean_return - confidence * sample_std / sqrt(n) - risk_penalty * stop_rate`；交易数不足、无有限效用或未严格超过门槛时空仓。无全局强行回退、不看未来无信号回退。
- global一张表；state按至多3条三值条件编码；month/quarter按信号日期自然月/季度匹配训练历史。整个样本外冻结，不宣称滚动重训。
- 在同训练集选108候选仍有选择偏差，置信惩罚不是独立内层验证或多重检验修正；最终效果必须看预留样本外。
- 图表训练期保持现金，明确“未回填拟合收益”；不显示事后选择的训练绩效冒充已部署策略。原有年度/月度、样本外分段指标继续复用。

## 内存与性能预算

- 输入float64/int64只读，成员×时间采用C布局，每成员行为零拷贝视图；非连续共享输入由固定A布局内核接收，普通公式仍要求读取边界C布局，不在节点内复制。
- 必要复制：读取日期解码/对齐，篮子面板输出分配，三值转数值工作缓冲，信号矩阵和持久化。对齐用原研究序列内核；其只接收可写C数组时，适配复制仅在I/O边界一次发生，并复用结果。
- 最多12研究产品＋两组各12篮子成员，任务内缓存唯一产品，单工作线程、4任务槽；不跨进程序列化大矩阵。
- 图工作区64MiB，训练候选峰值含基础通道、篮子、108条信号及最大单候选工作区，上限128MiB；训练估算工作量不超过20亿，显式拒绝而不是自动缩样本。
- 参数组合的公式计划全部在显式准备/启动预热完成；生产运行禁止重新prepare/compile。每次候选完成释放中间值，只保留信号和小型评分表。

## 验收记录

### 自动化与界面验收

2026-09-10 最终后端回归：**310 passed**，48.16秒，只有既有 Starlette/httpx 弃用告警。命令在项目根目录运行：

```sh
NUMBA_CACHE_DIR=/private/tmp/firp-timing-numba-cache PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_timing_graph.py backend/tests/test_timing_service.py backend/tests/test_timing_routes.py backend/tests/test_timing_data.py backend/tests/test_timing_repository.py backend/tests/test_timing_research_numeric.py backend/tests/test_timing_training_runtime.py backend/tests/test_timing_panel.py backend/tests/test_timing_learning.py backend/tests/test_timing_etf_templates.py backend/tests/test_regime_granularity.py -q -p no:cacheprovider
```

覆盖14个真实展开模板、108候选、因果前缀、冻结策略不受样本外价格扰动影响、成熟交易/隔离边界、证据不足空仓、三值未知、固定篮子身份、逐成员预览、共享数组、旧定义兼容及NJIT实际调用。不同篮子的矩阵不能按行号混算，分别归约后可以组合。

前端择时单测 **20/20**；完整择时浏览器测试 **12/12**；补强“所选参数”断言后的320/768/1440三屏复测 **3/3**。桌面与手机截图已人工查看，无页面横向溢出；候选表只在局部横滑，训练高级设置按需展开。截图使用明确标注的离线交互夹具，不是研究绩效。

生产构建通过。全量前端单测 **797/798**；独立失败为 `frontend/src/i18n/runtime.test.tsx` 的 `/settings/risk-models` 导航文案登记，不属于择时模块，本次未改动。报告位于 `/private/tmp/timing-etf-frontend-final-vitest.json`。不得将此结果写成全仓测试全部通过。

TypeScript检查仍有 `EtlWorkflowEditor.test.tsx:18/20` 两个既有错误（hook返回值、测试夹具published字段），择时文件无类型报错。前端命令在 `frontend/` 运行：

```sh
npm run test -- --run src/pages/TimingResearch.test.tsx --maxWorkers=2 --minWorkers=2
npm run test:e2e -- timing-research.spec.ts --output=/private/tmp/timing-etf-training-browser
npm run test:e2e -- timing-research.spec.ts --grep 'ETF 改编训练' --output=/private/tmp/timing-etf-training-browser-final
npm run test -- --run --maxWorkers=2 --minWorkers=2 --reporter=json --outputFile=/private/tmp/timing-etf-frontend-final-vitest.json
npm run build
npx tsc --noEmit --pretty false
```

路由结构验证（`validate_ai_routing.py --skip-reproducibility`）通过；27个本次改动文件的显式覆盖检查通过，无未覆盖文件。完整路由检查仍因择时与其他模块的新文件尚未纳入Git而失败；未为制造通过而暂存其他任务文件。`git diff --check`通过。

### 真实快照技术验收

报告：`/private/tmp/timing-real-etf.rIxG0f/report.json`；临时运行快照/数组位于同目录 `repository/`。未写生产研究记录、未下载、未修改行情。

技术验收脚本在同目录 `acceptance.py`。项目根目录执行 `env NUMBA_CACHE_DIR=/private/tmp/timing-numba-cache PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend:. /Users/chenjunming/Desktop/myenv_312/bin/python /private/tmp/timing-real-etf.rIxG0f/acceptance.py`；此临时产物不是永久测试依赖，可重复测试以仓库内pytest为准。

- 活跃行情快照 `tushare_snapshot_20260910_merrill_macro01`；研究对象510300.SH，2020-01-01至2026-09-03，样本外起点2024-01-01。
- 市场参考篮子为510300.SH/510500.SH；类别测试篮子为510300.SH/159915.SZ。后者仅是技术输入，不是经验证的类别分类或推荐。
- 14个模板显式选择不复权 `raw`；A2552额外检验 `hfq`，**15/15通过、0预算失败**。全部产生7年/81个月结果，样本外648个交易日。含训练模板冻结日为2023-12-29，拟合截至2023-12-08。
- 三组raw输入与一组hfq输入各读取一次；输入只读，运行前后数组SHA一致。预热全部17模板4.02秒；主流程24.72秒（不含模块导入），单次已准备的产品计算0.013–0.161秒。该耗时不是完整API启动时间或并发压测结果。
- **经济结果并未证明有效**：12个学习模板在本窗口默认设置下均选现金；A2552的hfq也选现金。A160S-106候选成熟训练交易为0，A2074每组最多3笔低于最低5笔，其余候选最佳保守效用未超过门槛。没有为了增加交易而放宽阈值。
- A2552 hfq 的108候选中最多35笔成熟训练交易，最佳效用−0.00139985低于门槛0，故选现金；不是读取或计算失败。
- 固定A2067/A2296均全期22笔、样本外9笔已平仓交易；raw样本外策略收益约−23.98%，持有基准约+31.95%。不能沿用原股票实验的高胜率，更不能据技术验收推荐使用。

不复权价格不代表含分红总回报；固定篮子存在选择/幸存者偏差；当前快照不保证历史修订版本。这里只证明当前数据和算法执行链可用，不是独立ETF有效性、正式PIT认证或收益承诺。

### 本地服务加载

`bash start_services.sh restart` 已成功，完整应用和worker预热271秒；后端PID80510，前端启动PID87566。仅重启本项目8000/5173服务，未终止独立ETL执行器，数据存储无待迁移计划。

实际GET后端8000及前端代理5173的 `/api/timing-research/catalog`，两者均为17模板（3既有＋14个`etf-v1`），步骤上限128、训练候选108、篮子成员12。`/api/health` 返回 `ok=true`、`numba_warmup.complete=true`。因此当前本地服务已加载新实现；不代表远端部署或Git提交，本次没有提交/推送。
