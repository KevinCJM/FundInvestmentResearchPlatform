# Repository Guidelines

## 项目结构与模块组织
- `backend/`：FastAPI 服务入口位于 `app.py`，业务逻辑拆分在 `fit.py`、`optimizer.py` 等模块，`run.py` 提供统一 CLI。
- `frontend/`：React + Vite 源码保存在 `src/`，Tailwind 直接写入 JSX，`dist/` 存放构建产物供后端静态托管。
- `data/`：仅保留小型、可复现实验数据和测试夹具，路径通过 `DATA_DIR` 工具函数解析。
- 根目录：公共脚本如 `T01_get_data.py` 与环境配置 `config.py`，文档放在此处，避免散落。

## 构建、测试与开发命令
- `cd backend && pip install -r requirements.txt`：使用推荐虚拟环境 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12` 安装依赖。
- `uvicorn app:app --reload --host 0.0.0.0 --port 8000`：本地热加载 API 服务，自动提供 `/api`。
- `cd frontend && npm install`：安装前端依赖；后续命令默认在同一路径执行。
- `npm run dev` 与 `npm run build`：前者启动代理到后端的开发服务器，后者输出生产包至 `frontend/dist`。
- `python backend/run.py`：打包前端后的一体化演示入口，方便业务验证。

## 代码风格与命名约定
- Python 采用 PEP 8 与四空格缩进；函数、变量使用 `snake_case`，Pydantic 模型使用 `PascalCase`。
- 前端使用函数式组件与 PascalCase 文件名，局部样式与测试与组件同目录保存。
- 避免硬编码路径，优先复用配置和工具模块；必要注释保持简洁并解释设计意图。

## 代码版本与存量代码治理
- Git 是本项目唯一的代码版本管理机制。源码目录只保留一份代表当前版本、实际参与调用、构建和发布的实现；历史版本统一通过 Git commit、tag 或 branch 追溯，不在当前代码中并行保存。
- 禁止仅为留档、对比或回滚而保留旧实现，包括但不限于 `old`、`legacy`、`backup`、`copy` 等副本、注释掉的大段旧代码、永远不会进入当前调用链的分支，以及同一功能的新旧双实现。
- 替换或重构功能时，必须在同一变更中删除被取代的代码，并同步清理失效的导入、导出、路由注册、配置、测试、文档和依赖；不得以“以后可能用到”为理由保留未使用代码。
- 回滚必须使用 `git revert`、切换 commit/tag/branch 等 Git 能力，不得通过恢复旧文件、旧函数或旧路由到当前源码来预埋回滚路径。
- 只有仍被当前系统真实调用、属于明确兼容契约且有测试覆盖的适配代码可以保留；兼容需求结束后必须立即删除。代码评审与验收需确认实际调用链只指向唯一当前实现，并检查无未引用代码、重复入口和失效配置。
- API/数据格式版本、数据库迁移、数据快照与模型产物版本可因外部协议或审计要求保留，但不得借此在源码中维持同一功能的多套历史实现。

## 算子颗粒度与组合算法治理
- 按“最小独立计算语义”划分算子，不按函数、代码行数、公式长短、输出数量或计算耗时划分。算子应能用一句话说明职责，有明确输入、输出及独立测试；不追求把每次加减或每个循环都变成画布节点。
- 区分基础算子、组合模板和耦合内核。可独立替换的特征计算、条件判断、分类、确认、区间统计和结果展示不得封装成新的黑盒算子；完整业务算法应是可展开、可编辑的组合模板。
- 只有内部步骤具有不可分离的递推状态、联合约束迭代或模型拟合语义，且拆成普通单向连接会改变结果时，才保留耦合内核。必须说明保留原因、状态边界、终止条件及可独立拆出的外围步骤；不能仅因“算法复杂”而豁免。
- 审核新增或修改的节点时必须回答：它只解决什么问题？中间结果能否独立使用或替换？现有算子能否等价组合？继续拆分是否破坏数值、状态或时点语义？答案与验收记录写入该需求设计文档，不把具体实现清单写进本规范。
- 多输出属于同一计算语义、共享一次不可分离求解时可以保留；相互独立的业务指标不能仅为“单次计算”而绑定成一个公共算子。画布语义粒度与执行内核粒度分离，执行层可共享中间结果、融合计算，但须保持各输出可追溯、可独立预览。
- 通用数学、统计和序列能力优先复用统一算子注册表及其 NJIT 内核；领域特有的状态、事件和区间能力扩展领域契约。禁止在不同中心复制同一数学实现，禁止把前端公式或描述文字当作计算真相。
- 算子注册信息应声明职责、颗粒度类别、输入输出类型、参数默认值与约束、缺失处理、边界等号规则、因果性及知识可得时点。前端目录、连线、说明、模板和后端校验必须依据同一份契约；无损版本适配不能靠名称猜测。
- 数值、条件、状态、事件和区间边界必须有明确类型，禁止隐式混用。数据缺失、条件为假、未触发新候选、未分类和中性状态必须按契约区分；不能为了连线方便把缺失填成零或中性。
- 模板插入应产生真实、可编辑、可预览的计算步骤与连线；参数有合理默认值，必填输入与未连接原因直接可见。分类目录将基础算子、组合模板和耦合内核明确区分，历史兼容节点不作为新建方案的默认推荐。展开是显式、可撤销的定义变换，必须重接所有被使用的输出与下游引用；不得仅绘制装饰性子节点，或用单一主输出替换多输出契约。
- 拆分必须保持数据轴、dtype、NaN/Inf、窗口预热与重置、首尾边界、状态编码、阈值等号、确认计数和多输出语义。证明等价前不得自动替换已保存定义；改变数学口径属于新算法版本，不得伪装成纯架构重构。
- 时点语义随所有实际依赖传播。需要后续或全样本数据的结果，即使下游只有普通比较，也仍是事后结果；不得通过拆分、滞后或改名绕过实时/发布/回测门禁。
- 历史定义、运行快照、发布版本和下游引用保持不可变。确有存量契约的旧入口只能做有测试覆盖的薄适配，委托唯一当前计算实现；禁止保留新旧两套数值算法作为回滚副本。
- 数值路径继续遵守下述固定签名 NJIT、预热和禁止 Python 回退要求。仅执行所需依赖，避免无用多输出、重复取数、全量数组复制及每节点重复扫描；融合、缓存、并行优化必须有等价测试和可重复性能证据。
- 验收必须同时覆盖：原始与组合结果等价、边界/缺失/时点、历史契约、新模板真实连线、单节点预览、公式与画布往返、实际 NJIT 调用及前后端回归。删除被替代且无真实兼容用途的实现、导入、目录项和测试，不能只隐藏旧代码。

## NJIT 与零拷贝高性能计算约束
- 除纯 I/O、参数校验、任务编排、结果封装和本节明确允许的第三方模型豁免外，所有由本项目实现或承载的数值计算逻辑都必须提供并默认使用 Numba `njit` 编译路径。范围包括但不限于普通数学、矩阵运算、统计、回归、导数、指标计算、绩效与风险归因、因子计算、组合构建与优化、权重求解、组合回测、情景模拟、压力测试、滚动窗口和批量统计。
- Python 函数只负责参数校验、数据读取、类型与数组转换、任务编排和结果封装；大规模逐行、逐期、逐资产、逐情景循环及核心数学运算必须下沉到 `njit` 兼容的纯数值内核。
- 正式计算请求必须进入已经编译完成的 NJIT 内核；禁止在生产请求期间临时编译新签名，禁止因缓存未命中、类型不兼容或编译失败而静默回退到 Python、object mode、pandas 循环或其他低性能实现。
- 服务启动阶段必须预热全部生产计算计划、参数类型组合和 worker 进程；预热不完整时 readiness 应失败关闭，不能先对外服务再由首个请求触发编译。
- 输入边界应将 pandas/Pydantic 对象转换为稳定 dtype 的 NumPy 数组或 Numba typed container；NJIT 内核不得执行文件、网络、日志、Pydantic、pandas 或其他不兼容的 I/O/对象操作。
- 生产计算过程中不得使用 `pandas.DataFrame` 或 `pandas.Series` 承载数值运算或作为节点间的计算容器。NumPy 数组（`numpy.ndarray`，包括一维序列数组）可以使用且应优先使用；“NumPy Series”在本文中指一维 NumPy 数组，NumPy 本身没有独立的 `Series` 类型。Pandas 仅限数据读写、输入结构校验、必要的边界转换和最终结果封装；数值变换、滚动统计、分组聚合及逐行计算必须使用 NumPy 数组与 NJIT 内核，禁止以 Pandas 向量化、`apply`、`iterrows`、`rolling` 或 `groupby` 替代。数据进入计算链路时统一规范化并尽量零拷贝，节点与循环内部不得反复转换回 Pandas 对象，也不得将数值计算包装成“预处理”绕过本要求。
- 计算数据尽量避免 `object` dtype 和混合类型容器，应在输入边界转换为语义明确、稳定的数值、布尔或时间类型；`object` 数组不得进入 NJIT 数值内核。以 `object` 保存的数值应按原数值语义转换，不能误当作类别编码；无效值与缺失值按契约处理，不得静默填零。纯 I/O、编排和展示元数据不属于数值数组限制范围。
- 对取值集合有限的类别、枚举或标识类 `object` 数据，必须建立稳定、可追溯的映射，转换为固定整数 dtype（如 `int32`/`int64`）后计算；映射应在上下游节点、批次及 worker 间一致，并随持久化结果保留解释关系，禁止每次独立编码导致含义漂移。缺失值和未知类别使用明确且不与合法值冲突的编码或有效性掩码，保留原始语义；类别编码不自动代表大小顺序或数值距离。验收须覆盖映射一致性、编码还原、缺失/未知类别和整数范围边界。
- 数据完成必要解码与边界规范化后，计算、滚动窗口、分组与节点间传递必须默认复用同一底层内存，使用 NumPy 零拷贝视图或向 NJIT 内核传递基础数组及索引范围；禁止为每个节点、窗口或分组重复复制原始数据。
- 连续区间优先使用基本切片视图，或通过 `kernel(base_array, start, end, stride)` 按起止索引和索引步长计算内存块；非连续选择优先在 NJIT 内核中按索引数组间接访问，避免高级索引、布尔筛选、`take` 等物化数据副本。所谓“指针算术”优先由受控数组索引实现，必须校验范围、步长与布局，保持数据轴、缺失值和时点边界；不得通过未经验证的裸指针或越界视图绕过数组契约。
- 解码、dtype 转换、排序对齐、第三方接口连续内存要求等确实无法零拷贝的环节，只能在明确边界执行必要的最小范围复制，并复用转换结果；在设计与验收记录中说明原因、位置和数据规模。`copy=False`、`np.asarray` 或 `np.ascontiguousarray` 不等于零拷贝保证，必须核实实际是否共享内存；视图的布局和只读属性须纳入固定签名与启动预热，不能在请求中临时编译或为匹配签名反复复制。
- 共享输入、缓存、快照及其视图按只读契约使用，持有底层内存所有者直至所有消费者结束；禁止原地写入污染其他节点或历史结果。结果数组与必要工作缓冲区可以分配，但应预分配、按生命周期复用；原地计算只允许写入明确独占的缓冲区，且不得覆盖仍被使用的结果或造成并行写入冲突。零拷贝不等于禁止分配计算输出。
- 跨进程复用大数组时，应使用受控共享内存或内存映射并传递描述信息与索引，避免重复序列化整块数据；必须管理所有权、只读访问、同步和释放时机，不能把进程内地址直接当作其他进程可用的指针。
- 零拷贝验收须检查实际调用链，以 `np.shares_memory` 等验证预期视图共享，覆盖非连续布局、只读输入、空区间、索引边界、别名写入与生命周期；记录必要复制、内存分配及峰值内存的可重复测量，并验证数值与时点语义等价。不能仅凭存在切片、`copy=False` 或 `@njit` 就宣称实现了零拷贝。
- NJIT 豁免仅限无法由 Numba 编译、且自身具有优化执行后端的第三方机器学习、深度学习、神经网络等模型。豁免必须显式登记模型、版本、执行后端和原因；普通数学、矩阵运算、统计、回归、导数、优化、滚动计算，以及通用外部求解器或数学库，不得仅因使用第三方实现而获得豁免。
- 豁免模型必须作为与 NJIT 计算相互独立的执行节点或阶段，只能通过固定 dtype、连续内存的 NumPy `ndarray` 与 NJIT 前后处理内核交换数据；禁止从 NJIT 内核调用第三方 Python 对象、Python callback 或黑盒函数，禁止因此进入 object mode 或触发任何 Python fallback。模型调用前后的特征准备、约束计算、批量评估、状态映射和结果校验仍必须使用 NJIT。
- 新增或修改计算功能时，必须同时提供 NJIT 路径测试，验证其与受控参考实现的数值一致性、NaN/Inf、空样本、边界窗口、dtype 和确定性；豁免模型还必须验证执行隔离、固定数组边界和 `python_fallback=0`。性能敏感改动应保留可重复基准，`parallel=True`/`prange` 只能在实测更快且结果一致后启用。
- 代码评审与验收必须检查实际调用链是否进入 NJIT，而不能仅以存在 `@njit` 装饰器或未被调用的加速函数作为完成证据。

## 测试指引
- 后端测试放在 `backend/tests/`，文件命名 `test_*.py`，使用 `pytest` 验证路由状态码与响应结构。
- 前端测试使用 Vitest + React Testing Library，测试文件与组件同级，命名 `*.test.tsx`。
- 所有测试需可重复执行，使用 `data/fixtures/` 提供固定样本，禁止网络调用。

## 回测窗口与样本要求
- 再平衡与权重反推统一使用 `slice_fit_data`：`window_mode='all'` 表示从首条数据到目标日全部样本，`rollingN` 表示取最近 `data_len` 个上交所交易日（不少于 2 条）。
- 当窗口长度超过可用样本时，接口与回测会返回 400/抛错，需放宽窗口或缩短 `data_len`；静态策略将默认保持旧权重。
- `ensure_valid_rebalance_window` 会自动顺延到首个样本充足的调仓日，首日 markers 与 series 保持对齐；此前日期由前端负责提示。
- `SSE` 交易日日历来源 `data/trade_day_df.parquet`，新增数据时需保证 `exchange='SSE'`、`is_open=1`。

## 提交与合并请求规范
- AI 在创建开发分支、准备 commit/push、创建或更新 PR、执行 PR 审核、合并、发布、回滚及维护提交规则前，必须完整阅读并遵守 [branch_submission_rules.md](branch_submission_rules.md)，不能仅凭记忆或本节摘要操作。
- 分支流向、提交范围、AI 审核、测试证据、合并条件及异常处理由该文档统一规定；不得在其他文档中维护相互冲突的提交规则。
- 按用户已经授权的任务范围执行，不重复询问已授权动作；写文档或修改代码本身不表示已获准提交、推送、合并或修改远端仓库设置。

## 安全与配置提醒
- 配置值由环境变量或 `.env` 读取，不要将密钥或令牌写入仓库。
- CORS 当前仅用于开发，部署前需收紧允许的来源并复核日志策略。
- 若发现仓库外部新改动或异常文件，先暂停操作并与团队沟通后再处理。

## Tushare 下载文档同步协议
- 涉及 Tushare 数据下载、刷新、新增接口、扩展数据源或修改下载脚本时，必须先使用 `$tushare-fetcher`，完整读取其 `SKILL.md`，并按当前接口目录和当前账户条件核定频率、时限、单次返回上限、积分门槛与权限边界；不得凭经验假设或沿用过期限制。
- 接口事实优先读取项目内 `docs/tushare_interfaces_ai_optimized.json`；项目未提供时，才使用 `$tushare-fetcher` 自带的接口目录。账户积分必须通过该技能提供的机制查询；积分不等于接口权限，未明确确认独立权限的接口不得直接生成或启用可执行下载逻辑。
- 下载实现必须把核定后的调用上限转化为可执行约束：所有线程、进程和任务共享同一限流配额，控制最小调用间隔和统计窗口内调用数，并预留安全余量；禁止每个并发任务各自限流后叠加突破额度。
- 每次请求必须设置可配置的连接/读取超时，批量任务必须设置空闲超时和最大运行时限；全历史下载须明确按日期、代码或分页分片，并保存可恢复检查点，禁止无限等待、无限循环或一次性盲拉全量数据。
- 重试必须有界且可观测：只对超时、连接中断、服务端暂时错误和限流错误重试，采用指数退避、随机抖动，并确保限流等待不短于接口要求；鉴权、权限、积分不足、参数、字段或数据契约错误不得盲目重试。达到最大次数后必须失败并保留明确错误原因。
- 单次结果触及接口行数上限、分页未完成、分片缺失或返回疑似截断时，必须继续拆分补齐或失败关闭，不得把不完整结果标记为成功。成功空响应必须独立复核；重试、断点续传和重复运行必须幂等，不能制造重复记录或覆盖已验证数据。
- 新建或修改下载脚本必须先按 `$tushare-fetcher` 要求做 smoke test；默认最多发起 1 次真实请求并写入临时目录，校验通过且脚本哈希一致后才能固化。测试不得污染项目正式数据目录。
- Tushare Token 只能从受控配置或环境变量读取，禁止写入代码、文档、命令输出、日志、异常信息或测试夹具。
- `TushareDownload.md` 是 Tushare 数据现状、接口映射、输出格式和下载逻辑的快速分析契约；代码是执行真相，两者必须保持一致。
- 修改 Tushare API、动作、模块或范围、默认依赖、输出文件名、Parquet schema、业务键、时点口径、全量/增量策略、分页限频、检查点、凭据、验收或快照激活逻辑时，必须在同一变更中更新 `TushareDownload.md`。
- 文档必须区分“活跃快照中已真实下载”“代码已支持但尚未下载”和“本地派生”，不得仅凭接口代码存在就声称数据已下载。
- 判断实际下载状态时，应复核 `data/tushare_active.json`、活跃目录中的真实文件及 Parquet metadata；行数和日期范围属于带审计时间的快照，不是永久常量。
- 完成相关变更后运行 `backend/tests/test_tushare_data_script.py` 的文档契约检查，并按数据采集模块的 minimum regression 验证。

<!-- AI-HERMES-ROUTING-PROTOCOL:BEGIN -->
# AI Hermes Routing Protocol

## Purpose

Machine-first routing protocol for downstream agents operating from the current working directory.

## Scope Boundary

- Treat `.` as the writable project boundary unless higher-priority instructions say otherwise.
- External folders may be read for task understanding, comparison, or integration analysis; do not route edits outside the target project.
- Treat external services, DB schema, and invisible callers/callees as `out_of_scope` unless directly observed from readable files.
- Keep routing facts in JSON files under `docs/`; keep `AGENTS.md` protocol-only.

## Required Read Order

1. `AGENTS.md`
2. `docs/repo_map.json`
3. `docs/task_routes.json`
4. `docs/pitfalls.json`
5. `branch_submission_rules.md` before branch creation, commit/push, PR creation or review, merge, release, rollback, or submission-policy maintenance
6. Routed code, tests, and configs

## Routing Ownership

- `docs/task_routes.json` owns task matching, module expansion, and operational-list merge policy.
- `docs/repo_map.json` owns module facts, operational file lists, tests, configs, and regression commands.
- `docs/pitfalls.json` owns hidden contracts, recurring pitfalls, affected modules, and safe checks.
- `AGENTS.md` owns protocol, required read order, scope rules, and tool workflow only.
- Do not duplicate module-level file, test, config, or regression lists in `docs/task_routes.json`.

## Default Operating Sequence

1. Match the task in `docs/task_routes.json`.
2. Load `first_read_modules` from the selected route.
3. Expand into `expand_to_modules` only when route rule codes trigger.
4. Resolve `first_read_files`, `then_check_files`, `related_tests`, `related_configs`, and `minimum_regression` from `docs/repo_map.json` using `docs/task_routes.json` merge policy.
5. Load linked pitfalls from `docs/pitfalls.json`.
6. Verify claims from code, tests, configs, or command output before promoting them to routing memory.

## AI Routing Validation

- Use `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` after editing `AGENTS.md`, `docs/repo_map.json`, `docs/task_routes.json`, `docs/pitfalls.json`, or matching service routing files.
- The validator checks route/module/pitfall references, routed path existence, git-tracked reproducibility for stable references, minimum regression command targets, and `grounding.fact_status` values.

# AI Routing Self-Evolution

- Treat `docs/ai_routing_evolution_policy.json` as governance only; routing facts belong in `docs/task_routes.json`, `docs/repo_map.json`, and `docs/pitfalls.json`.
- Update `AGENTS.md` only when protocol, required read order, scope rules, or tool workflow changes.
- Promote verified hidden contracts and recurring pitfalls to the correct JSON owner.
- Use `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py` after code, test, config, tool, or routing changes to check coverage.
- For routing-only work, run `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --routing-only` with explicit changed paths.
- Re-run `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` after routing file changes.

## Output Discipline

- Keep routing facts in JSON only.
- Keep `AGENTS.md` protocol-only.
- Stop exploration once routing is sufficient for first-pass narrowing.
<!-- AI-HERMES-ROUTING-PROTOCOL:END -->
