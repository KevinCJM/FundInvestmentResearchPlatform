# 基金投研平台需求说明

本文档梳理当前基金投研平台的业务背景、功能现状与规划需求，便于产品、研发、测试对齐设计意图。

## 1. 背景与目标
- 构建覆盖「ETF 宇宙 → 产品研究 → 资产大类构建 → 策略回测」的一站式投研工具，缩短资产配置策略的验证周期。
- 支撑多角色协作：量化研究员构建策略、产品经理评估发行产品、数据团队维护行情与产品库。
- 保证每个环节可复现、可追踪（数据来源统一、模型参数可回溯、回测流程可复核）。

## 2. 目标用户与场景
| 角色 | 诉求 | 典型操作 |
| --- | --- | --- |
| 量化研究员 | 查看 ETF 宇宙、构建资产大类、验证再平衡策略 | 调用风控指标、生成调仓表、执行回测并导出结果 |
| 产品经理 | 分析热门 ETF、了解费用与发行、筛选潜在产品 | 在仪表盘查看趋势、在产品库筛选过滤、查看单产品详情 |
| 数据团队 | 维护行情与日历、保证指标准确 | 更新 `data/` 下 parquet 数据、监控接口异常 |

## 3. 功能矩阵
| 模块 | 功能描述 | 前端页面 / 接口 | 状态 |
| --- | --- | --- | --- |
| ETF 宇宙仪表盘 | 统计 ETF 数量、类型、费用率、上市趋势 | `Dashboard` 页面 / `/api/etf/analytics`、`/api/etf/analytics/list_trend` | 已上线 |
| ETF 产品检索 | 多维过滤、排序、分页、关键字搜索 | `ProductResearch` 页面 / `/api/etf/products` | 已上线 |
| ETF 产品详情 | 展示元信息、费用、托管方、价格序列（缺失时生成模拟行情） | `ProductDetail` 页面 / `/api/etf/products/{product_id}` | 已上线 |
| ETF 模糊搜索 | 供资产配置时选择 ETF | `ManualConstruction` 页面 / `/api/etf/search` | 已上线 |
| 资产大类拟合 | 计算净值曲线、绩效指标、相关矩阵，支持 class 级别分析 | `ManualConstruction` 页面 / `/api/fit-classes`、`/api/rolling-corr-classes` | 已上线 |
| 风险预算/目标权重求解 | 基于历史净值与风险指标生成权重，支持窗口模式、约束 | `ClassAllocation` 页面 / `/api/strategy/compute-weights` | 已上线 |
| 调仓表与回测 | 生成调仓 schedule、执行回测、返回净值与交易明细 | `ClassAllocation` 页面 / `/api/strategy/compute-schedule-weights`、`/api/strategy/backtest` | 已上线 |
| 效率前沿探索 | 支持随机探索、量化步长、SLSQP 精炼 | `ClassAllocation` 页面 / `/api/efficient-frontier` | 已上线 |
| 资产配置保存/加载 | 以 Parquet 存储配置明细与净值 | `ManualConstruction` & `ClassAllocation` 页面 / `/api/save-allocation`、`/api/list-allocations`、`/api/load-allocation` | 已上线 |
| 自动构建大类 | 根据 ETF 特征自动聚类并生成资产大类 | `Placeholder` 页面 | 规划中 |
| 产品组合构建 | 跨策略组合、情景分析、导出报告 | `Placeholder` 页面 | 规划中 |
| 调仓权重缓存复用 | schedule 与回测共享权重，降低重复求解 | 后端缓存、`precomputed` 字段 | 开发中（参见 `docs/weight_reuse_task_plan.md`） |

## 4. 数据与集成要求
- **数据文件**：
  - ETF 宇宙：`data/etf_info_df.parquet`（包含 `ts_code`、`name`、`fund_type`、`market`、`issue_amount` 等）。
  - ETF 行情：`data/etf_daily_df.parquet`（复权净值）、`data/etf_daily_candle_df.parquet`（K 线数据）。
  - 资产配置：`data/asset_alloc_info.parquet`、`data/asset_nv.parquet`。
  - 交易日历：`data/trade_day_df.parquet`，字段需满足 `exchange='SSE'`、`is_open=1`，供窗口与调仓逻辑使用。
- **数据更新流程**：
  - 优先通过 `T01_get_data.py` 或内部数据管道更新 parquet，避免在代码层面硬编码路径。
  - 更新后需执行后端测试，确保 schema 未破坏（尤其是指标统计用到的列）。
- **外部集成**：
  - 需设置 `TUSHARE_TOKEN`（环境变量或 `.env`）。
  - 未来若对接第三方行情源，需提供兼容的 parquet schema，并在 README 中补充说明。

## 5. 非功能需求
- **性能**：常用 API（`/api/etf/analytics`、`/api/strategy/compute-weights`、`/api/strategy/backtest`）在常规数据量（近 10 年日度数据、几十只资产）下响应时间应控制在 1~2 秒内；回测允许更长，但需反馈进度状态（前端已通过按钮 loading 体现）。
- **可靠性**：所有接口遇到缺失数据需返回带解释的 400/404/500，并确保 parquet 写入支持幂等。
- **可维护性**：模块化拆分（`services/`、`backtest_engine.py` 等），提供单元测试覆盖主要逻辑；关键函数需附简要注释解释设计意图。
- **安全性**：禁止在仓库提交密钥；CORS 在生产环境需收紧；日志中避免输出敏感字段。
- **可观测性**：推荐后续引入结构化日志、慢查询监控与基础指标埋点。

## 6. 依赖与技术约束
- Python 3.12（推荐），依赖详见 `backend/requirements.txt`。
- Node.js >= 18，使用 npm 进行包管理。
- 构建产物位于 `frontend/dist`，由后端静态托管；部署时需确保写权限或提前构建。
- 测试框架：后端 pytest，前端采用 Vitest + React Testing Library。

## 7. 风险与未决事项
1. **调仓权重缓存**：后端已实现 LRU 缓存与 `cache_key`，但前端尚未完全串联，仍可能重复求解，影响复杂策略的响应时间。
2. **数据完整性**：当前 ETF 行情与产品信息依赖本地 parquet，如未及时更新会导致回测窗口不足或产品统计偏差。
3. **前端测试覆盖有限**：虽已引入 Vitest + RTL，但需要补充关键页面和复杂交互的用例以提升回归保障。
4. **自动策略页面**：`auto-classification`、`portfolio-construction` 仍为占位，需定义清晰的功能边界与接口。

## 8. 里程碑规划
| 时间 | 里程碑 | 说明 |
| --- | --- | --- |
| 近期 | 调仓权重缓存串联 | 完成前端调用改造，回测支持 `precomputed`，减少重复求解（参考 `weight_reuse_task_plan.md`）。 |
| 中期 | 自动构建大类 MVP | 基于 ETF 指标自动聚类，生成大类方案并支持评估。 |
| 中期 | 产品组合策略库 | 支持多策略组合的配置、回测与导出，补齐页面占位。 |
| 中期 | 前端测试体系 | 扩充 Vitest + RTL 用例，覆盖关键页面交互与数据流。 |
| 远期 | 报告导出与监控 | 自动生成研究报告、引入接口监控与告警。 |

## 9. 附录
- 构建与测试命令详见项目根 [`README.md`](../README.md)。
- 更细粒度的缓存、回测设计可参考 `backend/backtest_engine.py` 与 `docs/weight_reuse_task_plan.md`。
