# 基金投研平台（Fund Investment Research Platform）

基金投研平台是一体化的资产配置研究与回测系统，前端提供 ETF 与场外公募基金研究、统一产品筛选、策略构建与回测可视化，后端基于 FastAPI 提供风险预算求解、资产大类回测、效率前沿探索等服务，并通过统一的数据目录共享净值与产品元数据。

## 目录
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [数据资源与约定](#数据资源与约定)
- [快速开始](#快速开始)
- [开发与运行流程](#开发与运行流程)
- [测试与质量保证](#测试与质量保证)
- [主要 API 概览](#主要-api-概览)
- [常见约定](#常见约定)
- [路线图与计划](#路线图与计划)
- [版权说明](#版权说明)

## 系统架构
- **backend/**：FastAPI 服务入口 `app.py`，对外暴露 `/api`。风险预算、回测、效率前沿、ETF 分析等业务拆分在 `strategy.py`、`backtest_engine.py`、`fit.py`、`optimizer.py` 等模块；`cal_indicators/` 与 `custom_indicators/` 分别承担受限 LaTeX/DAG 计算内核和工作区指标服务；`run.py` 提供一体化启动脚本。
- **frontend/**：React + Vite 单页应用，`src/` 采用函数式组件与 Tailwind 样式；通过 Vite 代理 `/api` 到后端。构建产物输出至 `frontend/dist` 供后端静态托管。
- **data/**：集中存放可复现实验数据，如 `asset_nv.parquet`、`etf_daily_df.parquet`、`trade_day_df.parquet` 等。所有路径通过后端公共工具 `DATA_DIR` 解析，避免硬编码。
- **docs/**：辅助文档，包括测试记录与需求规划（详见 [`docs/requirements.md`](docs/requirements.md)）。
- 根目录保留数据脚本（如 `T01_get_data.py`）与配置入口（`config.py`）；Tushare Token 由主界面配置并保存在本机忽略文件，后端不回传明文。

## 核心功能
- ETF 宇宙分析仪表盘：`/api/etf/analytics` 提供类型、管理人、发行规模、费用率等统计；前端 Dashboard 页面呈现图表、趋势分析与过滤器。
- 产品研究与详情：`/api/instruments/products?kind=etf|fund` 分别筛选 ETF 与场外公募基金；场外基金详情/对比读取真实 `fund_nav_df.parquet`，ETF 详情继续兼容原接口。
- 自定义研究指标：指标中心支持受限 LaTeX 编辑、DAG 校验、真实 ETF/基金预览和版本化保存；同一指标可在产品详情、产品对比及评价方案中复用，计算异常不会静默按零处理。
- 真实评价方案：评价方案锁定指标 revision，以严格完整样本策略执行方向归一化与加权排名；缺少任一指标的产品会明确标记为不可排名。
- 资产大类构建：在 ETF 与场外公募基金统一候选池中选品，支持等权、风险预算、自定义权重模式，前端可调用 `/api/fit-classes`、`/api/rolling-corr` 获取净值、相关矩阵、滚动指标。
- 模块化数据刷新：主页可组合“基础信息 / ETF / 公募基金”模块，并选择增量或全量模式；全量模式默认由环境开关保护。
- 资产配置与回测：保存大类配置（`/api/save-allocation`），基于 `asset_nv.parquet` 计算效率前沿（`/api/efficient-frontier`）、策略权重（`/api/strategy/compute-weights`、`/api/strategy/compute-schedule-weights`）以及组合回测（`/api/strategy/backtest`）。
- 再平衡与窗口管理：`slice_fit_data`、`ensure_valid_rebalance_window` 等工具统一处理 rolling/all 窗口逻辑，窗口样本不足时返回明确错误码。
- 一体化启动：`python backend/run.py` 会检测前端构建产物并自动启动 uvicorn（可配置端口、自动打开浏览器）。

## 技术栈
- **后端**：Python 3.12，FastAPI + Pydantic，Pandas/Numpy/Scipy/Numba 做矩阵计算，Uvicorn 提供服务。见 [`backend/requirements.txt`](backend/requirements.txt)。
- **前端**：React 18 + TypeScript，Vite 构建，Tailwind CSS，ECharts 用于可视化。
- **测试**：后端采用 pytest（见 `backend/tests/`）；前端使用 Vitest + React Testing Library 运行组件与页面单测。
- **部署建议**：生产环境推荐静态托管 `frontend/dist`，由 FastAPI 挂载静态资源并服务 API，可对接 Nginx/Gunicorn。

## 数据资源与约定
- 必备数据：
  - `data/asset_nv.parquet`：资产大类净值宽表，回测与效率前沿依赖。
  - `data/asset_alloc_info.parquet`：保存的配置明细，用于加载资产组合。
  - `data/etf_daily_df.parquet`、`data/etf_daily_candle_df.parquet`：ETF 历史行情，风险预算与详情页使用。
  - `data/etf_info_df.parquet`：ETF 产品宇宙及元数据。
  - `data/fund_nav_df.parquet`、`data/fund_info_df.parquet`：场外公募基金复权净值及产品元数据。
  - `data/fund_company_df.parquet`：公募基金管理人基础信息。
  - `data/trade_day_df.parquet`：上交所交易日历，为回测窗口校验提供依据。
  - `data/custom_indicators.json`、`data/evaluation_plans.json`：运行时创建的工作区共享指标与评价方案；使用原子 JSON 写入并由 Docker data volume 持久化，不提交版本库。
- 数据更新统一通过 Tushare-only 脚本 `T01_get_data.py` 完成；敏感凭证通过 `config.require_tushare_token()` 校验。
  ```bash
  python T01_get_data.py --etf-info --calendar --stock-basic
  python T01_get_data.py --nav --candle --start-date 20100101
  python T01_get_data.py --latest --etf-info --calendar --nav --candle
  python T01_get_data.py --fund-info --fund-nav --start-date 20100101
  python T01_get_data.py --smoke --all
  ```
- 所有数据文件请保持体积可控，适合作为测试夹具分发。

## 快速开始
1. **Python 环境**  
   推荐使用 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12` 对应的虚拟环境。
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Node 环境**  
   ```bash
   cd frontend
   npm install
   ```
3. **单独运行**  
   - 后端开发模式：
     ```bash
     cd backend
     uvicorn app:app --reload --host 0.0.0.0 --port 8000
     ```
     健康检查：http://127.0.0.1:8000/api/health
   - 前端开发模式（默认代理到 5173 → 8000）：
     ```bash
     cd frontend
     npm run dev
     ```
4. **一体化演示**  
   首次执行需构建前端：
   ```bash
   cd frontend
   npm run build
   ```
   然后在项目根：
   ```bash
   python backend/run.py
   ```
   浏览器访问 http://127.0.0.1:8000 即可体验页面与 API。

## 开发与运行流程
- **数据下载配置**：先在主界面“数据管理”中保存 Tushare Token；Token 只写入本机 `data/.tushare_token`，不会从环境变量读取或在页面回显。数据刷新功能开关仍使用环境变量：
  ```
  DATA_REFRESH_ENABLED=true  # 允许主页启动增量刷新；生产环境默认关闭
  TUSHARE_TOKEN_CONFIG_ENABLED=true  # 允许从主界面保存/更新 Token
  DATA_FULL_REFRESH_ENABLED=false  # 全量历史更新需单独显式开启
  ```
- **代码约定**：
  - Python 遵循 PEP 8，函数/变量为 `snake_case`，Pydantic 模型使用 `PascalCase`。
  - React 组件使用函数式写法，文件名采用 PascalCase，局部样式与测试文件与组件同目录。
  - 避免硬编码路径，统一使用后端 `DATA_DIR`、前端配置。
- **调试技巧**：
  - 通过 `/api/strategy/compute-schedule-weights` 获取包含调仓 markers、缓存 key 的完整结果，可直接用于回测。
  - 若仅有前端静态资源，可运行 `uvicorn app:app` 并访问后端自动挂载的 `frontend/dist`。

## 测试与质量保证
- 后端测试：
  ```bash
  cd backend
  pytest
  ```
  当前覆盖 `strategy` API、回测窗口校验、ETF 分析路由等（19 项用例，全量通过记录见 `docs/test_report.md`）。
- 前端：`npm run test` 使用 Vitest + React Testing Library；开发阶段可配合 `npm run test -- --watch` 进入交互模式。
- 构建校验：`npm run build --prefix frontend` 确认 Vite 构建成功。
- 建议在提交前同步执行上述命令，并在 PR 中记录测试结果、附上 UI 截图。

## 主要 API 概览
| 路由 | 方法 | 描述 |
| --- | --- | --- |
| `/api/health` | GET | 健康检查。 |
| `/api/etf/search` | GET | ETF 模糊检索与分页。 |
| `/api/instruments/search?kind=all` | GET | ETF 与公募基金统一模糊检索与分页。 |
| `/api/etf/analytics` / `/api/etf/analytics/list_trend` | GET | ETF 宇宙统计、上市趋势及维度过滤。 |
| `/api/etf/products` | GET | ETF 产品列表（分页、过滤、排序）。 |
| `/api/instruments/products?kind=etf\|fund` | GET | 分类型产品列表（分页、过滤、排序）。 |
| `/api/instruments/products/{id}?kind=fund` | GET | 场外公募基金详情与真实复权净值序列。 |
| `/api/etf/products/{id}` | GET | 单个产品详情与（若缺失则生成的）价格序列。 |
| `/api/data/refresh/status` | GET | Tushare 刷新任务与本地数据新鲜度。 |
| `/api/data/refresh` | POST | 按固定模块枚举启动增量或全量刷新任务。 |
| `/api/custom-indicators/meta` | GET | 受支持变量、算子模板、周期和计算限制。 |
| `/api/custom-indicators` / `/api/custom-indicators/{id}` | GET/POST/PUT/DELETE | 工作区指标目录、创建、版本更新与并发安全删除。 |
| `/api/custom-indicators/validate` / `/api/custom-indicators/evaluate` | POST | 校验 LaTeX/DAG，并基于真实产品净值按需计算。 |
| `/api/evaluation-plans` / `/api/evaluation-plans/{id}` | GET/POST/PUT/DELETE | 工作区评价方案及锁定指标版本。 |
| `/api/evaluation-plans/{id}/run` | POST | 执行严格完整样本排名。 |
| `/api/fit-classes` | POST | 资产大类净值、相关性、绩效指标计算。 |
| `/api/rolling-corr` / `/api/rolling-corr-classes` | POST | ETF/资产大类的滚动相关分析。 |
| `/api/save-allocation` / `/api/list-allocations` / `/api/load-allocation` | POST/GET | 保存、枚举、加载资产配置。 |
| `/api/efficient-frontier` | POST | 指定配置的效率前沿探索（支持约束、随机探索、SLSQP 精炼）。 |
| `/api/strategy/compute-weights` | POST | 基于当前窗口计算单次权重。 |
| `/api/strategy/compute-schedule-weights` | POST | 生成调仓表、缓存权重序列。 |
| `/api/strategy/backtest` | POST | 执行策略回测，返回组合净值与交易记录。 |

> 更详细的字段说明请参考对应模块的 Pydantic 模型或阅读 `docs/requirements.md`。
> Tushare 增量算法、失败语义与部署边界见 `docs/tushare_data_refresh_design.md`。
> 自定义指标的产品定位、周期语义、安全边界和数据契约见 `docs/custom_indicator_design.md`。

## 常见约定
- CORS 目前仅用于本地联调，部署前应收敛允许的来源。
- `ensure_valid_rebalance_window` 会顺延首个样本充足的调仓日；前端需提示用户窗口不足的日期。
- 当数据或配置文件缺失时接口会返回 400/404，请在部署前准备好 `data/` 文件。
- 遇到仓库外部的异常文件或脏数据，应暂停操作并与团队确认。

## 路线图与计划
- ✅ ETF 仪表盘、ETF/公募基金产品研究、自定义指标研究闭环、真实评价方案、统一资产候选池、资产大类构建、效率前沿、策略回测。
- 🚧 调仓权重复用缓存：已在 `/api/strategy/compute-schedule-weights` 接口返回 `cache_key`，待前后端联动减少重复求解（详见 `docs/weight_reuse_task_plan.md`）。
- 🔄 自动构建大类、产品组合构建页面目前为占位组件，后续将接入实际策略与回测。
- 📈 计划扩展前端测试覆盖率、补充更多市场/基金数据源以及指标对比。

## 版权说明
本项目采用 [LICENSE](LICENSE) 中声明的条款。请在提交前确认未泄露敏感信息，数据文件仅供内部研究使用。
