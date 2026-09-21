# 因子研究

本页是特征、因子收益、风格暴露与收益贡献的主要入口。与[指标中心](indicators/README.md)共用数学能力；发布结果进入产品池仍需人工审核，不直接成为 SAA／TAA 约束或交易指令。
## 产品定位与边界

在「设置 → 因子研究中心」维护全系统共享的因子定义、研究方案、检验结果和发布版本。股票、ETF、场外基金使用同一中心，以产品类型、底层资产、用途和数据能力区分工作流。

中心解决三个问题：哪些特征能帮助产品筛选；收益由哪些风险或风格解释；结论如何带着证据进入产品池、资产配置和投后复核。特征得分、风险暴露、因子收益是不同对象，禁止混用。发布表示研究结果可引用，不表示因子已证明有效，更不表示可以直接交易。

本版交付可运行的特征研究闭环、收益风格归因、不可变发布及产品池接入。现有配置与投后页面引用同一发布证据，保留各页面原有决策职责。股票框架共用契约；当前快照只有股票基础信息，缺少完整行情、复权及财务 PIT 数据，禁止伪造股票研究结果。

## 项目链路与复用决策

已按 AGENTS → repo_map → task_routes → pitfalls 阅读，并使用 CodeGraph sync/explore 确认：
- FastAPI app → services 路由 → 领域 service/repository → 固定签名 NJIT。
- 指标体系：custom_indicators 的定义、版本锁及真实序列 provider；typed_types 支持 time×asset 轴。因子中心补充跨产品面板和研究语义，结构化算子不执行任意 Python。
- 数据：market_data 活跃快照解析器；series_provider 批量读取 adjusted_nav 和公告日，保留数据来源和可用性。
- 产品池：ProductPoolService 的 EvaluationPlanGateway 提供稳定扩展边界。因子发布适配为只读评价来源，入池后仍为待审核，沿用原有生命周期和版本发布。
- 下游：产品池版本 → SAA / TAA / 类内产品配置 → 目标组合 → 实际组合与核算 → 投后与反馈。配置算法和交易核算不因发布因子而隐式改变。

相关代码：backend/app.py、backend/custom_indicators/series_provider.py、backend/product_pools/service.py、frontend/src/app/processRegistry.ts、frontend/src/layouts/StageLayout.tsx。

## 功能与验收编号

| 编号 | 本版功能 | 可验收行为 |
| --- | --- | --- |
| FR1 | 因子目录与构建 | 内置模板可复制，参数、方向、适用类型可编辑；保存产生版本，历史引用不变 |
| FR2 | 研究口径 | 研究对象、比较基准、模型、数据集分开；日期、窗口、费用、样本外日期明确 |
| FR3 | 检验与组合 | 原始值/标准化值、IC/RankIC、分组收益、相关性、覆盖率、换手与扣费模拟，样本内外分离 |
| FR4 | 归因 | 本地指数代理 RBSA；显式上传的同市场同币种 FF3 因子收益集；展示暴露和残差，不将其标为选股得分 |
| FR5 | 不可变运行 | 锁定方案版本、因子版本、原始输入快照、数据指纹、计算版本、日期口径 |
| FR6 | 发布与监控 | 从成功运行发布研究版本；可停用；覆盖、失效日期及最新运行相对发布的变化可检查 |
| FR7 | 产品池 | 选定发布与产品池执行入池；保存发布/运行标识、分数和排名，保留人工审核 |
| FR8 | 全流程引用 | 产品研究及投前、投后页面可查看发布证据，显式绑定研究上下文；持仓权重可用于加权因子画像 |
| FR9 | 数据能力限制 | 缺失 NAV/公告日、市场币种不匹配、未预热、空截面、无有效标签均返回可操作错误 |
| FR10 | 真实因子套件 | 使用当前本地数据创建三因子研究方案，运行并保存结果与研究发布；报告真实样本外表现 |

## 信息架构与交互

中心包含「因子库」「研究工作台」「收益归因」「发布与应用」四个页签。

因子库：按用途/资产筛选；显示定义、参数、方向、版本与数据依赖。模板包括中期动量、低波动、回撤控制、短期反转。结构化编辑器实时展示公式说明，保存新定义或新修订。输入校验在前后端执行。

研究工作台：选择/新建版本化研究方案；分块设置对象与样本、基准与来源、因子权重、检验及模拟规则。数据覆盖检查后运行，期间阻止重复操作并显示状态。结果包括区间与数据证据、样本内外指标、净值对照、分组收益、相关性、逐期 IC、最新产品得分、排除原因。支持重新载入历史运行。

收益归因：选择 RBSA 或 FF3，前者选择本地指数代理，后者选择已上传、来源完整的因子收益数据集。上传模板采用显式小数收益，拒绝不匹配市场/币种。列名固定并校验，不猜测百分数单位。返回样本内拟合、固定系数样本外解释度和残差；归因使用已实现收益，不能直接用为历史预测信号。

发布与应用：选择成功运行，填写名称及有效期，发布不可变研究版；从发布详情选择已有产品池和 Top N 导入待审核名单。可登记研究上下文绑定、停用发布、查看引用及监测结果。各投研页面提供紧凑的因子证据入口。

空状态不展示假图或虚构结果；错误提供原因；缺失数值显示「—」而非 0。按钮键盘可达，表格局部滚动，320/768/1440 宽度不得产生页面横向溢出。

## 概念与数据契约

- FactorDefinition：id/revision/name/operator/window/skip/direction/product_kinds/description；模板只读，复制可改。
- FactorStudy：id/revision/name/product_kind/asset_class/market/currency/targets/universe_source/start_date/end_date/oos_date/benchmark/factor_refs/normalization/horizon/quantiles/top_n/cost_bps。
- BenchmarkRef：kind（ETF 净值或指数）/code/label/return_basis；用于业绩比较。
- FactorModel：characteristic_composite、rbsa、ff3；与 BenchmarkRef 分离。
- FactorDataset：id/name/source_url/market/currency/frequency/units/construction/rows/checksum；FF3 行含 date/MKT_RF/SMB/HML/RF，均为小数日收益。声明数据修订局限。
- FactorRun：id/created_at/study_snapshot/factor_snapshots/data_lineage/input_checksum/engine_version/execution/periods/statistics/latest_scores/warnings。
- FactorRelease：id/run_id/name/effective_from/effective_to/usage=research_only；停用记录独立保存，不改历史计算。
- FactorBinding：release_id/context_type/context_id/note/created_at；上下文明确属于研究方案或实际组合，不依靠演示页面上的静态名称。
- 原始数据矩阵采用 float64 C-contiguous，日期/可用日采用 int64；缺失 NaN，输出 JSON 为 null。
- 定义和方案保存 current + history，乐观 revision 冲突 409。运行按 ID 原子持久化，输入与结果可重演；JSON 使用现有 AtomicJsonStore 和跨进程文件锁。
- 数据血缘记录活跃快照目录、文件 size/mtime、输入哈希、字段、研究可用日；读前读后指纹变化则拒绝运行。

## 数值和时间口径

### 特征与横截面

主输入为复权净值，不把未复权收盘涨跌误作总收益。每个研究日只使用此前已公告的 NAV：公告日只有日期时，保守延后一交易日进入特征。不以未来公告或修订值冒充完整历史 PIT。滚动窗按 SSE 交易日轴，缺失不压缩、不补零、不向前填充；过期 NAV 不形成新信号。

动量 = NAV[t-skip] / NAV[t-skip-window] - 1。
波动 = 日简单收益的样本标准差 × sqrt(252)，方向取负。
回撤 = 窗口内 NAV / 历史峰值 - 1 的最小值，越接近 0 越好。
反转 = 负的短期动量。
横截面使用平均秩处理并列；百分位秩或截面 winsorized zscore 在当日可观测产品中计算。所有入选因子均有效才生成组合分数，组合权重固定且归一化，不以未来收益自动寻优。

### 标签与样本内外

月末收盘形成研究信号，下一 SSE 交易日净值为模拟入场点。IC 标签采用入场点至 horizon 个交易日后的净值总收益；只有终点及路径完整时有效。IC 是 Pearson，RankIC 为平均秩 Spearman，常数截面返回缺失，最少三个有效产品。

样本内统计仅含标签终点早于 oos_date 的观察，跨界标签剔除；样本外信号日不早于 oos_date。逐期 ICIR 不冒充显著性检验，不对重复试验给出虚假的 p 值。固定权重无需拟合；归因模型只在样本内拟合，再用固定系数评价样本外。

### 模拟与费用

月频 Top N 多头等权，缺失或不足最小截面则持有现金；现金收益为 0 的显式假设。持仓之间按实现收益漂移后再调仓。费用 = 单边费率 × 买卖绝对权重变化之和，包括首次建仓；费用扣除后再推进净值。组合持仓路径缺失时对应区间不生成伪造回报，不因已知未来缺失而重选幸存产品。

研究模拟使用复权净值，不能还原 ETF 盘口、折溢价、基金申赎确认及容量，因此本版发布仅为研究用途。基准使用同区间、同有效日期，不把价格指数和总回报基准静默互换。

### 收益归因

RBSA：用指数日收益解释产品日收益，权重非负且和为 1；对中心化数据最小化残差平方（等价残差方差），截距单独估计。固定签名 NJIT 投影梯度并检查收敛，拒绝无信息样本；输出系数、R²、残差波动、截距。代理高度相关时提示辨识限制，暴露不是披露持仓。

FF3：产品超额收益 = 截距 + βMKT_RF + βSMB + βHML + 残差，RF 来自所选数据集。只接受明确市场/币种/频率/来源/单位；拒绝以美国官方因子自动解释 A 股并标为匹配研究。秩亏/样本不足返回错误。FF 数据集是因子收益，不能作为普通横截面打分特征直接排名。

## 投研全流程落点

| 环节 | 本版结合方式 | 决策边界 |
| --- | --- | --- |
| 产品研究/评价 | 发布版本的产品得分、因子贡献和数据日期 | 评价辅助证据 |
| 产品池 | 因子发布适配 EvaluationPlanGateway，Top N 导入待审核，人工批准后发布池版本 | 不越过准入与禁用规则 |
| SAA/自动分类 | 研究上下文绑定发布，查看风格和因子画像；可使用批准后的池成员 | 不将动量排名直接当战略风险预算 |
| 历史情景/TAA | 查看版本化信号证据、可用日与适用区间 | 原有情景/TAA 模型仍控制配置边界 |
| 类内产品配置 | 在相同大类预算内参考评分候选及发布证据 | 不跨类抢占预算 |
| 目标/实际组合 | 绑定不可变发布；显式持仓权重计算因子画像并报告覆盖 | 不自动下单 |
| 投后/反馈 | 对照发布与最新运行、覆盖、因子漂移、样本外检验；保留失效历史 | 不将残差等同基金经理能力 |

后续扩展到自动调用 TAA 信号节点、目标版本强引用及真实基金申赎执行前，需要各模块契约的额外实施；本版不把通用证据绑定称为完成自动交易或组合优化集成。

## API 与实现布局

前缀 /api/factor-research：
- GET /catalog：因子、模型、数据能力、默认真实数据研究建议。
- GET/POST /factors，PUT /factors/{id}：定义及修订。
- GET/POST /studies，PUT /studies/{id}：方案及修订。
- POST /studies/{id}/runs：指定 revision 运行；GET /runs 和 /runs/{id}。
- POST /datasets，GET /datasets：不可变 FF3 数据集导入/目录。
- POST /attributions：RBSA/FF3 归因，保存运行证据。
- GET/POST /releases，POST /releases/{id}/retire，GET /releases/{id}/monitor。
- GET/POST /bindings；POST /releases/{id}/portfolio-profile。
- 产品池使用既有 attach_evaluation_plan API，以 factor-release- 前缀分派到只读适配器。

新增 backend/factor_research/{contracts,catalog,data,numba_kernels,repository,service}.py 和 services/factor_research_routes.py；前端 FactorResearchCenter、factorResearch service、FactorEvidencePanel。引用与计算分层，避免在已有巨型指标页面堆代码。

## 需求与概念

在原有入口 `/settings/factor-research` 内设置两个一级模块，不建立互不相通的两套系统。

| 模块 | 核心算法 | 主要产物 | 页面流程 |
| --- | --- | --- | --- |
| 产品特征因子 | 滚动特征 → 当期横截面标准化 → 对未来收益的截面检验 | 日期×产品×特征、分数、IC/RankIC、研究发布 | 因子库 → 研究检验 → 发布与应用 |
| 因子收益率 | 历史组合持有收益 / 双排序组合收益 → 因子收益率序列 | 日期×因子收益、来源与构造证据、回归暴露 | 构建工作台 → 收益率数据集 → 收益归因 |

区别不是投前/投后：两种产物都可服务产品尽调、配置和投后。评价方案依旧负责评价与准入；特征研究负责验证预测假设；收益率研究负责生产及检验解释变量。回归系数是暴露，不是因子收益率。Barra 风格横截面回归和 FF3 组合差额不是同一个构造算法；本次不冒称实现了完整 Barra 风险模型。
## 新领域对象与存储

沿用 AtomicJsonStore、VersionStore 和 ArtifactStore；不引入数据库或第二套数据提供者。

- ReturnPlan：id/revision/name/method、source_run_id 或 source_panel_id、factor_key、quantiles、cost_bps、output_factor。方法为 characteristic_spread 或 ff3_2x3。方案编辑乐观锁冲突返回409；运行精确绑定指定 revision。
- FF3Source：市场、币种、真实来源、构造说明、交易日历、年度六月形成截面、每日股票总收益与前一交易日市值、每日 RF。作为不可变原始输入产物保存。上传不等于证明具备完整历史 PIT。
- ReturnDataset：kind=dataset，factor_names、dependent_return（total/excess）、frequency=daily、units=decimal_return、rows、source_method、构造方案快照/父运行/父面板标识、输入哈希、执行审计、诊断和警告。
- 行以 date 为唯一升序键，其余键必须严格等于 factor_names，excess 口径另有 RF；缺失为 null。不填零、不猜测百分数单位。
- ReturnRun 不再复制巨大收益数据：生成的数据集本身就是不可变运行产物，记录 return_plan_id/revision 及全部构造输入；方案列表与数据集列表区分。
- 特征收益构建从原运行冻结的 NPZ 读取 prices/days/decisions，不重读当前市场快照；读取前验证文件、dtype 和输入校验和。

## 算法A：特征分组收益差额

输入为一个已完成的特征运行，选择其中一个标准化方向后的因子或综合分；不以样本外收益选择最优因子。对每个信号日可得截面按平均秩分组，固定末组为高分、首组为低分；并列不按代码随意拆分，任一端为空则明确不可用。

信号日 t → 下一交易日收盘建仓 → 随后逐日获得收益。各腿组内初始等权、持有期间按净值变化漂移，在下一明确调仓日重置。不把重叠 horizon 标签平均收益直接拼成因子收益。

`F_t = R_high,t - R_low,t`，多头名义1、空头名义1。分别保存高低腿收益、毛差额、双边换手、费用以及扣费差额诊断。默认因子数据列为毛差额，借券、融资、基金不可做空等执行限制单列警告；不能称为可实际交易基金组合。累计展示为收益率算术累计而不是可投资净值。

持仓发生缺失时该腿当日无效，不能按未来数据补齐或剔除缺失产品重配；在下一合法形成日可以重新定义下一持有段，但不掩盖中间断点。当前快照及固定样本池的修订/幸存者偏差随父运行传递。输出名不得使用 MKT_RF/SMB/HML/RF 冒充 FF3。

## 算法B：FF3 风格 2×3 双排序

本机股票正式行情、复权和财务 PIT 接入尚不完整，继续关闭“从本地股票目录自动构建”的按钮。本次提供**基于显式上传的规范时点面板**的真实计算路径；没有原始数据时不给假结果。不是下载服务，不改变 Tushare 权限/频控协议。

输入形成截面：formation_date、asset、market_cap（六月形成日权益市值）、december_market_cap（上一年十二月权益市值）、book_equity（上一财年账面权益）、fiscal_year_end、announced_date、reference_member。权益与市值必须正；财年属于上一年度；账面权益公告日严格早于形成日。形成日必须是输入日历的六月最后交易日。每期独立使用参考池计算规模中位数、B/M 的30%及70%断点；B/M=book_equity/december_market_cap。

独立双排序形成 SL/SM/SH/BL/BM/BH 六组，每组按形成日市值加权，持有至下一年形成日，期间按总收益漂移。空组或不足断点参考样本失败关闭，不任意合并组。每日股票收益包含分红；退市损失应由输入总收益明确反映，不能静默删除退市行。

- SMB = (SL+SM+SH)/3 - (BL+BM+BH)/3。
- HML = (SH+BH)/2 - (SL+BL)/2。
- MKT_RF = 全部当日明确市场成员按前一交易日市值加权的总收益 - RF。
- RF 必须显式逐日提供小数收益，禁止默认为0或把年化利率直接塞入日收益。

六组每年的形成日期、断点、人数可审计。不能以未来收益/未来市值确定分组。已持有成员收益缺失导致相应腿与依赖因子 null，不把剩余股票重新归一化。方法标为 `FF3风格2×3（自定义市场与参考池）`；与 Kenneth French 官方美国数据的证券筛选、交易所断点、退市口径、数据版本不保证相同，不宣称官方复制品。市场集合的完整性和日历真实性由来源声明，不能从上传本身证明。

## 收益率检验及归因

数据集诊断：逐列有效天数、均值、样本标准差、正收益比例、因子相关矩阵；展示原始逐日收益与算术累计路径。常数/缺失时相关性为null；不将统计均值/波动之比称为有无风险调整的 Sharpe。不提供未经多重检验校正的显著性结论。

FF3 和通用因子模型复用既有固定签名 NJIT OLS（带截距）；RBSA 保留非负、和为1约束。按日精确对齐，不前向填充。训练仅使用样本内数据，样本外固定系数评价；不足样本、秩亏返回明确状态。

## API

原前缀 `/api/factor-research` 下新增：

| 方法与路径 | 行为 |
| --- | --- |
| GET /return-catalog | 模块/算法能力及数据契约、空上传模板 |
| GET/POST /return-plans | 方案列表/创建 |
| PUT /return-plans/{id} | 按 revision 更新 |
| POST /return-plans/{id}/runs | 固定 revision 运行并产出不可变数据集 |
| GET/POST /return-sources | 原始 FF3 时点面板目录/导入 |
| POST /return-datasets | 通用因子收益数据导入 |
| GET /return-datasets/{id} | 详情、原始行、诊断、血缘（含旧 FF3 兼容） |
| GET /return-datasets/{id}/export | CSV 导出，有限列名、固定日期，避免公式注入 |

GET /datasets 返回两种来源的统一目录；旧 POST /datasets 继续仅校验原 FF3 契约。特征 run、归因 artifact、return dataset 必须检查 kind，禁止跨类型误读。收益率数据集不允许走原产品评分发布/入池接口。

## 计算架构与性能

`FactorResearchService` 保持现有对外门面；组合新的 `FactorReturnService`，后者独立持有收益方案和构造编排。新增 return_contracts.py、return_kernels.py、return_service.py，避免在既有特征 service 内堆完整新模型。

特征滚动统计、分组持有、权重漂移、FF3 断点/组合、收益诊断全部固定签名 Numba NJIT；业务层仅 I/O、校验、数组边界与封装。与现有主进程 readiness/并发限制共同预热，未就绪503、忙429；无请求新签名、无 Python fallback。float64/int64 C 连续数组统一边界。只复制需要冻结或重排的数组，不使用 pandas apply 数值循环。

JSON 输入有明确行数、资产数、交易日数与面板单元上限，避免上传造成无界内存；本版面向受控研究面板，大规模全市场多年分区导入需后续扩展，界面不隐瞒上限。

## 核心计算契约

### 暴露与每日贡献

总收益模型：r[t] = Σ beta[t,k] f[t,k] + alpha[t] + epsilon[t]。
超额收益模型：r[t] = RF[t] + Σ beta[t,k] f[t,k] + alpha[t] + epsilon[t]。

同一次模型估计的 beta 同时用于暴露展示与贡献计算。每个因子贡献 c[t,k] = beta[t,k] f[t,k]。截距、残差分别列示；RF 只在超额收益模型单列，总收益模型不虚构无风险收益数据。RBSA 的指数代理与其模型约束保持不变：系数非负、和为1。FF3/通用回归系数不解释为真实持仓比例。

每日独立输出实际收益、因子贡献、RF（适用时）、截距、残差、贡献合计、对账误差与状态。误差检查只证明代数闭合；模型解释力仍需看样本外 R²、残差与覆盖，不能因为对账正确就称模型有效。

### 固定与滚动模式

- `exposure_mode=fixed`（默认）：沿用样本内一次拟合，同一 beta 解释全部日期。样本内是事后描述，不能作为当时已知暴露；样本外系数固定。
- `exposure_mode=rolling`：从研究起始日起，积累完整 `rolling_window` 个交易日位置后，使用 [t-window,t) 拟合，绝不读取 t 或之后的收益。每隔 `refit_step` 个交易日重新估计，间隔内沿用最近一次计划估计；拟合失败不偷偷保留旧成功模型。`min_observations` 检查完整有效配对数；缺失日期保留，不把缺失压缩成更长实际窗口。
- 默认滚动窗口126、最少60个完整配对、每21交易日重新估计，可配置。底线 max(30, 5×因子数)。基金收益的两个端点公告日期必须早于本次估计生效日，尚未公告的历史净值不能进入该次拟合。
- 记录估计窗口起止、有效样本数、训练 R²、失败原因和生效日。样本外滚动属于 walk-forward，会使用此前已经过去的样本外数据，不声称是冻结训练集。
- 当前因子数据集没有完整历史修订和发布时间档案，因此滚动结果仍为历史研究，不保证实时 PIT 可交易性。

### 多期贡献：期初财富加权串联

不能把每日贡献简单相加当成复利贡献，也不能分别连乘各因子再求和。对指定区间从财富1开始：

```text
V[t-1] = Π(s<t)(1+r[s])
C[k] = Σ[t] V[t-1] c[t,k]
Σ[k] C[k] = Π[t](1+r[t]) - 1
```

这是明确的顺序财富加权分解，不称为 Carino/Brinson，也不要求因子腿可独立投资。截距、残差、RF 同样串联。每个样本内、样本外、月份、全区间分别从1重置；月份贡献不能直接相加当作全区间贡献。

累计曲线始终明确区间。任一实际收益缺失或不合法，完整区间总收益不可用；任一贡献日期不可计算，完整区间因子贡献和对账不可用，禁止跳过缺口假装完整结果。其后独立月份仍可正常计算。滚动预热期不纳入贡献评价，但明确展示剔除日数、实际评价起止。正常的真实零收益保留为0；缺失保持null。

## API 与持久化

继续使用 POST `/api/factor-research/attributions`，新增可选参数 `exposure_mode`、`rolling_window`、`min_observations`、`refit_step`；老请求默认固定模式。既有结果字段保留，新增 `attribution`（版本1）结构：

- `mode`、`linking_method`、`components`（因子/无风险/截距/残差的类型化ID）、`evaluation_start`、`warmup_days`。
- `products[]`：按产品的 daily（含估计证据、暴露与贡献）、summaries（全区间/样本内/样本外/月度）、curves（按所选样本独立串联）。
- 每个 summary 包含起止、应有/有效日数、实际复利收益、贡献、贡献合计、对账误差、模型R²、残差年化波动和完整性状态。

新增 `attribution_kernels.py` 负责纯数值，`attribution.py` 负责编排与序列化；主 service 保留数据和模型校验。原始输入、公告日、暴露路径、逐日贡献均与运行一起冻结为 NPZ，旧运行不修改、不自动补算。

响应按200万结果单元限制；滚动总拟合次数有限额，超过则拒绝并提示减少产品/区间或降低重估频率。所有新增 kernel 显式固定签名，启动预热纳入 readiness 和实际执行证明，禁止请求期编译或 Python 数值回退。

## 前端

入口不变：因子收益率 → 收益归因，内容升级为风格与收益贡献研究。配置区增加固定/滚动选择，滚动参数折叠显示，明确收益口径和日期含义。

结果独立组件 `AttributionResults`，避免继续堆叠工作台代码。选择产品和评价区间后展示：

1. 完整性、实际累计收益、贡献合计、对账误差、模型解释度。
2. 风格暴露曲线；固定模式明确训练期暴露为事后估计。
3. 各因子与RF/截距/残差贡献柱形、累计贡献与实际收益对照曲线。
4. 月度对账表、分页每日明细（估计窗口与暴露可核对）；导出当前产品的逐日贡献 CSV 和运行 JSON。

贡献用“百分点”，暴露用系数，不能把系数标成持仓占比。默认查看样本外。图表局部缩放，表格局部横滚，320/768/1440宽度不得撑破页面。旧结果无 `attribution` 时显示原回归表及“旧运行未保存贡献，请重新运行”，禁止由前端凭摘要反造。

## 版本与研究资格

特征值、因子收益序列、绩效基准和风格代理分别管理；FF3 不是可以直接用来给单只基金打分的特征。训练标签须成熟，purge 跨边界样本，滚动 IC 按标签成熟日可用；不能把未来收益回填为当日特征。

组别／权重训练后冻结，产品缺行情不按剩余持仓重归一化。RF、总收益口径、复权和成本覆盖明确。收益风格暴露不是实际持仓穿透，alpha 不自动等于经理能力。

首套固定 ETF 实验没有稳定选优证据，完整样本、收益、RankIC、业务对象与输入 checksum 见[因子研究纪要](verification/factors.md)。当前股票／跨市场输入能力与真实 FF3 数据可得性不可由接口存在推断。
