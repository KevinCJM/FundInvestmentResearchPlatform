# 内置时序指标通用滚动迁移：自审核与验收

## 1. 交付范围与最终状态

已按 `builtin_series_rolling_migration_design.md` 实施5个内置时序指标的 revision 3。代码目录默认返回 v3；显式 v1/v2 引用仍可读取、预热和计算。没有提交/推送 Git，没有重启正在运行的后端服务，没有修改其他功能的源码来消除全仓检查错误。

本需求后端、前端指标中心专项及迁移浏览器验收通过。但**不能宣称当前整个工作区已经可构建发布**：最终生产构建及 TypeScript 检查被风险模型模块的语法错误阻塞，见第7节。浏览器2项通过是在该错误阻塞最终构建之前取得的验收结果；未将其当作最终整个工作区的浏览器全量通过证明。

## 2. 已验证的迁移结果

| 内置指标 | 当前版本 | 执行与共享语义 |
|---|---:|---|
| 20 日收盘价均线 | 3 | `rolling_apply(mean(market_close),20)` |
| 20 日布林带 | 3 | 均值、总体标准差分别使用一个共享滚动作用域；三个通道复用这两个结果 |
| 10 日成交量均线 | 3 | `rolling_apply(mean(volume),10)` |
| KDJ（9,3,3） | 3 | 高低价条件极值使用两个通用作用域，最少有效观察数1；K/D递推保持在作用域外 |
| 5 日滚动年化夏普比率 | 3 | 完整来源标量子图进入通用作用域；来源转换协议3.0.0 |

所有当前通道可达计算路径均包含 `rolling_apply`，不含 `rolling_window` 或历史 `rolling_mean/std/min/max`。这不仅检查公式文本，也检查生成计划中的实际区间委托调用及冻结签名。

保留所有原 indicator_id、名称、通道ID/顺序/精度和固定参数默认值。内置定义仍只读，`parameter_schema=[]`；作者复制为自定义指标后才可明确开放窗口/最少观察数，并由原参数系统执行默认值和临时覆盖。

## 3. 关键兼容与安全结论

- 迁移前捕获的10份历史定义 SHA-256（排除创建/更新时间）全部保持不变；测试中保留校验值，不以新代码自行生成期望值。
- 已准备全部15份时序定义（5个指标各3个版本）。真实服务分别验证 v1/v3、v2/v3 调用；未指定版本使用v3，指定旧版本不跳转。
- v2/v3 在正常及NaN/Inf夹具中逐通道比较，缺失位置相同；数值容差为 `rtol=1e-8, atol=1e-9`，不声称浮点位级一致或已穷尽所有输入。
- KDJ另用独立参考计算覆盖首1至8期、短序列、连续缺失、平价/零值、状态延续和J越界。分母沿用原实现的 `abs(denominator)<1e-12` 回退50，不擅自更改为严格等于0。
- 默认滚动仍要求完整有限窗口。显式 `min_periods` 才允许部分窗口，且必须是1至window的整数。只按原日期位置取窗口，不补零、不为凑样本向前扩大、不压缩输入。
- 新 `finite_mask` 是独立逐项有效性判断；零为真、NaN/正负Inf为假。KDJ通过它和既有 `min_where/max_where` 表达缺失处理，没有KDJ专用滚动数值分支。
- BOLL两个作用域、KDJ两个作用域及两次递推均各自只生成一次共享调用；完整区间图不会提前按全历史求值再重复标量。
- 正式计算使用已准备NJIT计划；参数覆盖复用签名。测试拦截请求期编译入口，确认 `request_time_compilation=0`、`python_fallback=0`。
- 画布/公式往返、因果性探针和原生Excel均覆盖。Excel默认完整窗口使用COUNT校验；部分窗口使用SUMPRODUCT/ISNUMBER共同有效计数，条件极值复用原生AGGREGATE函数4/5、选项6。标准差复用标量编译器的 `SQRT(DEVSQ/(COUNT-ddof))`，不是私有Excel函数。

## 4. 测试证据

Python执行环境：项目指定Python3.12。所有后端夹具为临时目录中的确定性本地数据，无外部行情请求。

统一命令前缀：

```bash
PYTHONPATH=.:backend:backend/tests OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest
```

以下后端组不重叠，共 **369项通过**：

1. **87 passed**：`backend/tests/test_builtin_series_rolling_migration.py backend/tests/test_custom_indicator_time_series.py backend/tests/test_rolling_interval_service.py -q --tb=short`。最终一次结果87项通过。
2. **181 passed**：`backend/tests/test_rolling_interval_graph.py backend/tests/test_rolling_interval_safety.py backend/tests/test_typed_indicator_catalog.py backend/tests/test_typed_indicator_compiler.py backend/tests/test_causality.py -q --tb=short`。
3. **101 passed**：`backend/tests/test_typed_numba_v22.py::test_current_registry_has_complete_fixed_signature_njit_coverage backend/tests/test_typed_numba_v22.py::test_numba_operator_families_match_numpy_reference backend/tests/test_typed_numba_v22.py::test_warmed_runtime_never_adds_a_request_signature backend/tests/test_typed_numba_v22.py::test_runtime_does_not_call_python_operator_registry backend/tests/test_custom_indicator_excel_export.py::test_excel_registry_covers_every_current_single_product_operator backend/tests/test_custom_indicator_excel_export.py::test_every_current_operator_generates_an_excel_formula -q --tb=short`。

前端 **108 passed**：

```bash
npm run test --prefix frontend -- --run src/pages/IndicatorStudio.test.tsx src/pages/IndicatorStudio.roundtrip.test.tsx src/components/indicator-graph src/components/indicator-parameters/IndicatorParameters.test.tsx src/services/customIndicatorsExecution.test.ts src/services/customIndicatorsRolling.test.ts
```

迁移浏览器专项 **2 passed**（桌面1440px、窄屏320px）：

```bash
INDICATOR_TEST_PYTHON=/Users/chenjunming/Desktop/myenv_312/bin/python3.12 npm run test:e2e --prefix frontend -- indicator-primitives.spec.ts --project=desktop-1440 --project=mobile-320 --grep 内置时序迁移
```

浏览器使用真实指标API和临时行情夹具，验证5个v3目录定义、实际计算、KDJ画布两个滚动/两个有限值判断节点、首期有效值、Excel下载、无页面横向溢出和无浏览器脚本错误。原8行净值日期夹具保持不变，额外OHLCV数据只写临时目录。

`npm run i18n:check --prefix frontend` 通过；`git diff --check` 通过。Starlette/httpx弃用提示和浏览器兼容数据库过期提示未作为功能错误隐藏。

## 5. 性能与内存实测

基准脚本：`backend/scripts/benchmark_builtin_series_migration.py`。

```bash
PYTHONPATH=.:backend OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_NRT_STATS=1 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/scripts/benchmark_builtin_series_migration.py
```

同一确定性夹具4096个观察位置，使用各内置默认窗口，9次稳态中位数，**不含编译、取数、序列化和页面渲染**。输入只读；每个v3计划额外重复40次，检查NRT分配与释放。

| 指标 | v2中位数ms | v3中位数ms | v3/v2 |
|---|---:|---:|---:|
| 布林带 | 0.0331 | 0.3486 | 10.52 |
| 收盘价均线 | 0.0111 | 0.1197 | 10.76 |
| KDJ | 0.0884 | 0.4923 | 5.57 |
| 滚动夏普 | 0.0331 | 0.0572 | 1.73 |
| 成交量均线 | 0.0111 | 0.0478 | 4.30 |

**这次迁移统一了组合能力，不是速度优化。** 通用区间重算比旧专用滑动内核慢约1.7至10.8倍；本夹具绝对用时仍低于0.5ms，不据此承诺大规模或任意窗口都相同。保留窗口内子图共享、只读切片和资源预算；没有为了掩盖代价再增加按内置名称分派的快速路径。

5个v3计划重复40次后的NRT净存活分配增量全部为0，签名数量不增长。既有作用域安全测试继续覆盖实际NJIT切片地址、非连续输入拒绝和异常分配释放。有限值mask、时序输出及单窗口必要计算缓冲可以分配；没有逐窗口复制原始输入，也没有物化T×W窗口矩阵。NRT存活检查不等同于完整进程峰值内存测量，不宣称零分配。

## 6. 自审核与路由治理

本次改动集中于内置定义、统一min_periods能力、有限值原语、相关注册/参数/历史推断/Excel/翻译、测试与设计记录。历史代码入口仍只服务可真实读取和执行的历史协议；没有新增备用算法、无效路由、回滚副本或按指标名字执行数值的分支。

`docs/repo_map.json` 记录5个v3、KDJ边界和实测性能取舍；`docs/pitfalls.json` 记录部分窗口有效性、递推边界、共享节点及Excel契约。证据来自已读实现、历史hash、实际数值/API/浏览器与基准输出，未将新未跟踪文件路径冒充可从Git复现的稳定引用。

路由治理文件的routing-only检查通过，R70解析通过。全量路由validator仍失败，含工作区内其他需求的未跟踪稳定引用。

本次显式覆盖检查还识别出两个新文件尚未进入稳定路由：`backend/tests/test_builtin_series_rolling_migration.py`、`backend/scripts/benchmark_builtin_series_migration.py`。它们是本需求的真实测试和基准文件，不应伪装为可忽略产物。待确定Git提交范围并纳入跟踪时，应将它们登记到对应模块的related_tests/minimum_regression；本次没有为使校验变绿而擅自暂存文件。

## 7. 尚未通过的整站检查与运行边界

最终两次生产构建失败位置一致：

```text
frontend/src/components/risk-models/ResearchWorkbenchContent.tsx:90
Unexpected closing "section" tag ... opening "div"
Unterminated regular expression
```

`tsc --noEmit --project frontend/tsconfig.json` 同样在该文件报告JSX语法错误。该风险模型文件不属于本次迁移范围，未修改或回退。**因此目前只能确认迁移专项通过，不能确认当前整个前端可构建/发布。** 未将早先浏览器成功等同于最后工作区整站状态成功，也没有宣称完整浏览器套件全部通过。

初始全平台 `warm_numba_plans()` 长测试未取得完成结果，后续时序专项改为真实预热全部15个时序定义，避免在每个时序用例中重复无关的标量/组合批量准备。这没有修改生产启动逻辑，但**不构成全平台readiness验收通过的证据**。

本次未重启运行中后端，服务进程是否已经载入v3需在实际重启/重载后另行确认。现有明确锁定v1/v2的运行及快照不会因重启自动改写。
