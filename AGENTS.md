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

## NJIT 高性能计算约束
- 除纯 I/O、参数校验、任务编排、结果封装和本节明确允许的第三方模型豁免外，所有由本项目实现或承载的数值计算逻辑都必须提供并默认使用 Numba `njit` 编译路径。范围包括但不限于普通数学、矩阵运算、统计、回归、导数、指标计算、绩效与风险归因、因子计算、组合构建与优化、权重求解、组合回测、情景模拟、压力测试、滚动窗口和批量统计。
- Python 函数只负责参数校验、数据读取、类型与数组转换、任务编排和结果封装；大规模逐行、逐期、逐资产、逐情景循环及核心数学运算必须下沉到 `njit` 兼容的纯数值内核。
- 正式计算请求必须进入已经编译完成的 NJIT 内核；禁止在生产请求期间临时编译新签名，禁止因缓存未命中、类型不兼容或编译失败而静默回退到 Python、object mode、pandas 循环或其他低性能实现。
- 服务启动阶段必须预热全部生产计算计划、参数类型组合和 worker 进程；预热不完整时 readiness 应失败关闭，不能先对外服务再由首个请求触发编译。
- 输入边界应将 pandas/Pydantic 对象转换为稳定 dtype 的 NumPy 数组或 Numba typed container；NJIT 内核不得执行文件、网络、日志、Pydantic、pandas 或其他不兼容的 I/O/对象操作。
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
- 提交信息遵循 `feat:`, `fix:`, `docs:` 等前缀，聚焦单一改动并描述影响面。
- PR 需说明目的、关键变化、运行过的命令，并在涉及 UI 时附上截图或视频。
- 关联相关 Issue 或任务编号，确认无敏感信息泄露后再发起合并。

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
5. Routed code, tests, and configs

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
