# Tushare 统一基金、ETF、指数与宏观数据刷新设计

## 目标与领域边界

项目同时支持两类产品：交易所上市 ETF（`instrument_type=etf`）和场外公募基金（`instrument_type=fund`）。两类产品在采集与存储层物理隔离，在产品检索和资产配置净值读取层统一。

- ETF 保留交易所、跟踪指数、日 K 线等可交易资产语义。
- 场外公募基金使用申赎净值，不生成或伪造盘中行情。
- 产品研究页面按 ETF / 公募基金切换；场外基金详情与对比使用真实复权净值；资产大类构建可在统一候选池选择两类产品。
- 研究结果文件（如 `asset_alloc_info.parquet`）不由数据刷新任务改写。

## 数据模型与 Tushare 映射

| 模块 | 本地文件 | Tushare 接口 | 增量策略 | 业务去重键 |
| --- | --- | --- | --- | --- |
| 基础 | `trade_day_df.parquet` | `trade_cal(exchange='SSE')` | 从 `max(cal_date)+1` 合并 | `exchange, cal_date` |
| 基础 | `stock_basic.parquet` | `stock_basic` | 小表全量替换 | `ts_code` |
| 基础 | `index_info.parquet` | `index_basic` | 按市场循环后全量替换 | `ts_code` |
| 基础 | `fund_company_df.parquet` | `fund_company` | 全量替换 | `org_code`（源数据标识） |
| ETF | `etf_info_df.parquet` | `fund_basic(market='E')` + `etf_basic` | 按状态/交易所循环后重建 | `ts_code` |
| ETF | `etf_daily_df.parquet` | `fund_nav(market='E')` | 按缺失开放交易日循环 | `ts_code, date` |
| ETF | `etf_daily_candle_df.parquet` | `fund_daily` | 按缺失开放交易日循环 | `ts_code, trade_date` |
| ETF | `etf_index.parquet` | `etf_index` | 全量替换 | `ts_code` |
| 公募基金 | `fund_info_df.parquet` | `fund_basic(market='O')` | 按 L/I/D 状态循环后重建 | `ts_code` |
| 公募基金 | `fund_nav_df.parquet` | `fund_nav(market='O')` | 按缺失开放交易日循环 | `ts_code, date` |
| 公募基金 | `fund_manager_df.parquet` | `fund_manager` | 分页重建 | `ts_code, name, begin_date` |
| 公募基金 | `fund_scale_df.parquet` | 从 `fund_nav` 派生 | 净值完成后本地重建 | `ts_code, observation_date` |
| 公募基金 | `fund_portfolio_df.parquet` | `fund_portfolio` | 按公告日检查点增量 | `available_at, ts_code, end_date, symbol` |
| 公募基金 | `fund_dividend_df.parquet` | `fund_div` | 按公告日检查点增量 | `available_at, ts_code, ex_date, pay_date` |
| 公募基金 | `fund_adj_factor_df.parquet` | `fund_adj` | 按基金分片/按交易日增量 | `ts_code, date` |
| 公募基金 | `fund_benchmark_df.parquet` | `mkt_idx_bmk` | 小表全量替换 | `ts_code` |

`etf_basic` 是 ETF universe 的权威清单，避免把场内 LOF/封闭式基金误判为 ETF。`fund_basic(market='O')` 是场外公募基金 universe。两类信息文件使用相同的核心字段集合，但由 `instrument_type` 明确区分。

`fund_portfolio` 只代表季报公开的股票持仓，不是包含债券、现金和衍生品的完整资产配置。`mkt_idx_bmk` 是官方市场基准目录，不自动等同于每只基金合同中的业绩比较基准；产品级研究仍保留 `fund_basic.benchmark` 原文及后续受控映射。基金资产规模直接复用 `fund_nav` 的 `net_asset/total_netasset`，不把主要覆盖交易所基金的 `fund_share` 冒充完整场外基金规模。

## 宏观数据模块与时点契约

| 范围 | Tushare 接口 | 本地文件 | 主要用途 |
| --- | --- | --- | --- |
| `cycle` | `cn_gdp`、`cn_cpi`、`cn_ppi`、`cn_pmi` | `macro_cn_gdp_df.parquet`、`macro_cn_cpi_df.parquet`、`macro_cn_ppi_df.parquet`、`macro_cn_pmi_df.parquet` | 增长、通胀与景气状态 |
| `money_credit` | `cn_m`、`sf_month` | `macro_cn_money_df.parquet`、`macro_cn_social_financing_df.parquet` | 流动性与信用扩张 |
| `rates` | `shibor`、`shibor_lpr`、`repo_daily` | `macro_shibor_df.parquet`、`macro_lpr_df.parquet`、`macro_repo_daily_df.parquet` | 资金价格与贷款定价环境 |
| `release_calendar` | `cn_schedule` | `macro_cn_schedule_df.parquet` | 发布日校验与 PIT 映射 |

宏观下载不等于数据可直接进入历史回测。每张表统一保留：

- `observation_date`：经济观测期末或行情日期；
- `available_at`：当时最早可得日期；
- `revision`、`vintage`：同一观测期后续修订版本；
- `ingested_at`、`source_api`：本系统取得时间和来源。

GDP、CPI、PPI、PMI、货币供应和社融的历史接口没有给出逐期真实发布日期，因此 `available_at` 保持为空，`availability_status=release_date_unknown`。这些数据可用于事后研究，但正式历史回测必须等待发布日映射或使用可证明的首次发布版本。Shibor、LPR 和回购行情暂按日期级可得，使用 `availability_status=date_only` 提醒调用方缺少日内时间。

月度/季度宏观小表每次重取当前历史，只在值发生变化时追加新 revision；日频利率按安全日期窗口分段。这样既保留修订轨迹，也避免每天重复保存完全相同的历史副本。

## 指数数据模块

指数采集独立为 `index` 模块，并按用途拆成八个范围。选择任一范围都会自动带上 `catalog`；前端默认选择用于情景模拟的核心范围 `catalog + domestic + industry + global`。

| 范围 | Tushare 接口 | 本地文件 |
| --- | --- | --- |
| `catalog` | `index_basic`、`etf_index`、`index_classify`、`ths_index`、`dc_index`、`tdx_index` | 原始目录、`index_catalog_df.parquet` |
| `domestic` | `index_daily` | `index_daily_df.parquet` |
| `industry` | `sw_daily`、`ci_daily` | `index_sw_daily_df.parquet`、`index_ci_daily_df.parquet` |
| `concept` | `ths_daily`、`dc_daily`、`tdx_daily` | 三个来源独立行情文件 |
| `global` | `index_global` | `index_global_daily_df.parquet` |
| `futures` | `fut_index_daily` | `index_futures_daily_df.parquet` |
| `valuation` | `index_dailybasic` | `index_daily_basic_df.parquet` |
| `constituents` | 六类成分接口、`index_weight` | `index_members_df.parquet`、`index_weights_df.parquet` |

统一目录键为 `source_api, ts_code`，行情键为 `source_api, ts_code, trade_date`。`index_coverage_snapshot.parquet` 逐 Parquet row group 生成，保存首末行情日、行数、陈旧天数、境内交易日覆盖率和源文件指纹；指数查询接口只读取目录与覆盖快照，不扫描完整行情。

全量指数按代码并发、单代码日期段顺序执行，每个代码/日期段都有检查点；只有主线程写 Parquet。达到已知接口行数上限时递归二分日期区间。`dc_index` 与 `tdx_index` 本质上是按日快照接口，目录构建固定从截止日向前寻找最近有数据日期，禁止无日期请求聚合全历史后命中 5,000/1,000 行上限。增量严格按本地上交所交易日历重复拉取最近 5 个交易日并覆盖同键旧值。`fut_index_daily` 运行时要求指数代码，代码目录使用 Tushare 文档列明的完整南华指数清单，不以无代码请求猜测。

## 前端任务矩阵

`POST /api/data/refresh` 只接受固定枚举，不接受脚本路径或任意命令参数：

```json
{
  "modules": ["base", "etf", "fund", "index", "macro"],
  "mode": "incremental",
  "module_scopes": {
    "fund": ["info", "nav", "manager", "scale", "benchmark"],
    "macro": ["cycle", "money_credit", "rates", "release_calendar"]
  }
}
```

| 模块 | 内容 |
| --- | --- |
| `base` | 交易日历、股票基础、指数基础、公募基金公司 |
| `etf` | ETF 产品信息、复权净值、日 K 线、ETF 指数 |
| `fund` | 产品信息、净值、经理、规模、持仓披露、分红、复权因子、基准库 |
| `index` | 指数目录及所选指数行情、估值或成分范围 |
| `macro` | 增长通胀景气、货币信用、利率回购、发布日历 |

| 模式 | 语义 | 安全边界 |
| --- | --- | --- |
| `incremental` | 小维表刷新；时序数据从本地最大日期逐交易日补齐 | 新增持仓/分红/复权因子无基线时只建立最近安全窗口；完整历史需显式全量 |
| `full` | 从 `TUSHARE_FULL_START_DATE` 重建所选模块 | 默认关闭；需 `DATA_FULL_REFRESH_ENABLED=true`；公募基金按代码逐只循环，可能运行数小时 |

ETF/基金增量任务会自动带上交易日历依赖，即使用户未选择基础模块，也不会使用过期日历判断“已是最新”。

## 限频、循环与失败语义

所有线程共享一个 `RateLimiter`，并同时执行：

1. 每分钟调用总量上限；
2. 任意两次请求的最小时间间隔；
3. 普通异常的指数退避；
4. 限流异常的最小等待时间；
5. 随机抖动，避免多个请求同时重试。

全量净值/K 线与指数行情按产品/指数代码提交到有界线程池；增量净值/K 线按交易日提交到同一类线程池。指数目录的市场/参数组合、指数成分和最新权重的逐代码请求也共享该线程池策略。单个代码内部的分页或日期分段仍严格顺序执行，全部任务完成后再由主线程做 Parquet 原子归并，因此提高网络等待阶段的并行度不会引入分页乱序或并发写文件。当前项目按已确认的 10,000 积分配置，默认 16 个 worker、450 次/分钟、请求间隔 0.13 秒，为官方常规接口 500 次/分钟上限保留余量；积分较低或接口权限特殊时可通过环境配置下调。

已知有限接口采用下列策略：

- `fund_basic` 单次最多 15,000 行：先按市场和 L/I/D 状态分区，再用 `offset/limit` 分页循环；检测分页不推进并设置最大页数，禁止无限循环或保存疑似截断结果。
- `etf_basic` 单次最多 5,000 行：按 SH/SZ 与 L/P/D 组合循环。
- `fund_daily` 单次最多 5,000 行：历史全量按 `TUSHARE_HISTORY_CHUNK_DAYS` 切日期段循环。
- 公募基金全量净值：按基金代码循环，并把单只基金历史按日期段切分；增量净值：按缺失日期循环，并使用 `offset/limit` 翻页直到空页，规避账户实测的 10,500 行隐含截断。
- `fund_manager` 按 `offset/limit` 分页；分页不前进或达到安全页数上限时失败关闭。
- `fund_portfolio`、`fund_div` 按公告日保存可续跑检查点；单日响应达到行数上限时降级为“公告日 + 单基金”补抓，不保存疑似截断结果。全量分片用 Arrow 流式归并，避免一次读入完整持仓历史。
- `fund_adj` 全量按基金代码和最多 1,200 个自然日切片；增量重复抓取最近交易日并以新值覆盖相同键。
- GDP/CPI/PPI/PMI/货币/社融为版本化小表；Shibor/LPR/回购行情按接口上限切日期段，任何分段触顶即失败。

数据集只在本批次完整成功后落盘。Parquet 先写同目录临时文件，再用 `os.replace` 原子替换；重复批次按业务键去重，保持幂等。增量归并时，未命中本次更新窗口的产品直接以 Arrow 表复制，只有发生重叠更新的产品才进入 pandas 去重；若归并前后内容完全一致，则删除临时文件并保留原 Parquet 的修改时间。网页增量任务发现净值、行情和交易日历文件均未变化且原分析快照不陈旧时，会直接复用快照，避免无数据日仍扫描完整历史。中间日期返回空数据会终止，最后一个开放日尚未发布时保留给下次更新。

## 运行与状态

CLI、网页刷新和本地快照重建共用跨进程文件锁，状态为 `idle | running | succeeded | failed`。`GET /api/data/refresh/status` 返回任务模块、模式、功能开关、Token 是否已配置，以及各 Parquet 的行数、更新时间和最新业务日期；不会返回 Token 明文。

用户在主界面“数据管理”中保存或清除 Token。后端原子写入 `data/.tushare_token` 并设置 `0600` 权限，抓取脚本只读取该本机凭据文件，不再读取 `TUSHARE_TOKEN` 环境变量。Token 不进入命令行参数、持久任务状态或日志；数据刷新运行期间禁止修改凭据。

增量与全量分别使用独立的“无进度输出超时”。只要下载器持续输出节点、进度或归并状态，计时就会重置，不会因为任务总时长较长而误杀。另有默认关闭的绝对总时限；设置为 `0` 表示不限制总运行时间。旧的 `DATA_REFRESH_TIMEOUT_SECONDS` / `DATA_FULL_REFRESH_TIMEOUT_SECONDS` 仍兼容读取，但语义同样是无输出超时。

```dotenv
DATA_REFRESH_ENABLED=true
TUSHARE_TOKEN_CONFIG_ENABLED=true
DATA_REFRESH_IDLE_TIMEOUT_SECONDS=1800
DATA_REFRESH_MAX_RUNTIME_SECONDS=0
DATA_FULL_REFRESH_ENABLED=false
DATA_FULL_REFRESH_IDLE_TIMEOUT_SECONDS=7200
DATA_FULL_REFRESH_MAX_RUNTIME_SECONDS=0
TUSHARE_MAX_CALLS_PER_MINUTE=450
TUSHARE_MIN_CALL_INTERVAL_SECONDS=0.13
TUSHARE_MAX_WORKERS=16
TUSHARE_MAX_FUND_BASIC_PAGES=20
TUSHARE_MAX_FUND_NAV_PAGES=20
TUSHARE_FUND_NAV_PAGE_SIZE=10000
TUSHARE_MAX_FUND_MANAGER_PAGES=20
TUSHARE_MAX_RETRIES=5
TUSHARE_RETRY_BACKOFF_SECONDS=2.0
TUSHARE_RATE_LIMIT_WAIT_SECONDS=15.0
TUSHARE_MAX_LATEST_DAYS=120
TUSHARE_FULL_START_DATE=20100101
TUSHARE_HISTORY_CHUNK_DAYS=3650
```

生产环境默认关闭网页写操作，全量更新另有第二道开关。启用前应确保站点和 `/api/data/*` 已由网关鉴权。当前锁只适用于现有单 worker 部署；多 worker/多实例时应迁移到 Redis、数据库或任务队列。

## 命令行示例

```bash
# 全模块增量
python T01_get_data.py --latest --calendar --stock-basic --index-info --fund-company \
  --etf-info --nav --candle --etf-index --fund-info --fund-nav

# 首次初始化场外公募基金（长任务）
python T01_get_data.py --fund-info --fund-nav --start-date 20100101

# 小样本只写临时目录，不污染项目 data/
python T01_get_data.py --smoke --fund-info --fund-nav --limit 2

# 指数情景模拟核心范围
python T01_get_data.py --index-catalog --index-domestic --index-industry --index-global

# 宏观数据完整范围
python T01_get_data.py --macro-cycle --macro-money-credit --macro-rates --macro-release-calendar

# 公募基金扩展数据；持仓、分红和复权因子首次全量可能耗时较长
python T01_get_data.py --fund-info --fund-nav --fund-manager --fund-scale --fund-benchmark
python T01_get_data.py --fund-portfolio --fund-dividend --fund-adjustment --start-date 20100101
```

## 消费层接入顺序

1. 历史情景中心先增加“最新版本”和“首次可用版本”读取模式；正式回测只允许后者。
2. 使用 GDP/PMI、CPI/PPI、M1/社融、Shibor/回购构建可审计宏观特征，不把单个指标直接标签化为投资结论。
3. 产品研究展示经理变更、规模趋势、分红事件和披露持仓集中度；持仓穿透必须同时展示报告期与公告日。
4. 建立 `fund_basic.benchmark` 到标准基准库的受控映射，并保留无法自动识别和人工确认状态。

最终验收要求：任何正式研究结果都能追溯到数据文件、Tushare 接口、观测期、可用日或未知状态、修订版本和本地取得时间。
