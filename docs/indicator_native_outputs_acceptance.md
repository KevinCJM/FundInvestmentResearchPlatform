# 指标中心原生多结果与最大回撤：验收记录

验收日期：2026-09-08。范围：当前工作区中的指标中心、多输出计算、最大回撤及相关消费链；不是全项目全部功能验收。未提交、未推送，未重启正式应用服务。保留工作区其他任务的修改。

## 1. 用户可见变化

- 市场归因已从指标中心内置目录和读取/计算入口剔除，当前不提供该旧内置 ID。本次没有在因子研究中心新增市场归因工作流，也没有接入研究参数中心。
- 新建标量指标可以选择“多结果计算 → 最大回撤分析”，配置一次净值输入，自动得到四个结果；通过勾选决定保留哪些结果。
- 修改共享输入会更新全部端口引用，保留结果身份、名称、精度和展示设置。用户不必逐项编写四条计算公式。
- 支持修改结果名称/格式/评分方向、复制、排序、删除。删后重加生成新 ID；最后一项不能删除。
- 画布显示一个分析节点及四个具名输出端口，可连接下游算术。保存、重载、公式/画布往返保留连接身份。
- 原有单结果及多个独立公式的指标组继续支持。旧最大回撤单标量 ID 仍可被已有快照和评价任务引用；不原地改变其返回结构。

## 2. 最大回撤口径

| 输出 | 契约 |
| --- | --- |
| 最大回撤 | `max(1 - NAV / running_peak)`，正百分比 |
| 最大回撤下跌期数 | 最深回撤对应峰值至谷底的观察间隔数 |
| 最大回撤恢复期数 | 谷底至首次恢复原峰值的观察间隔数；尚未恢复为 null |
| 最长水下期数 | 各回撤段峰值至恢复日或窗口末的最大已观察间隔数，包含活动回撤 |

“期数”不是自然日。无回撤时四项为0。重复最深谷底取首次；下降前相同峰值取最近一次。空输入、非正净值、NaN/Inf 不生成伪结果。

有效分析但未恢复，与非法输入造成整个分析失败，使用不同状态。未选中的恢复期数不可用，不影响最大回撤结果的评分资格。

## 3. 一次计算的证据

新增受限 `record` 类型和静态命名端口。`drawdown_analysis(adjusted_nav).max_drawdown` 是对分析结果的投影，不是重新执行一份最大回撤公式。

同一图、同一输入的四个端口共享一个原生算子节点；测试检查生成源码只包含一次分析调用。固定 tuple 由预编译 NJIT 返回，端口读取不重新扫描净值。结果可继续参与其他计算。record 不能作为资产向量或任意 Python 对象使用，禁止访问未知/内部属性。

核心回撤算法单遍扫描、常数辅助空间；数据读取与输入校验属于独立阶段，不将“算法一次扫描”描述成整个系统完全没有其他扫描。

## 4. 最终执行结果

| 检查 | 结果 |
| --- | --- |
| 后端 A：类型、编译、运行、目录、服务、API、标量/多标量、回撤专项 | 244 passed；129.00 秒 |
| 后端 B：Excel、时序、图往返、并行、分页、组合 | 209 passed；125.57 秒 |
| 后端合计 | 453 passed；两批文件不重叠 |
| 前端专项 | 140 passed / 16 files |
| i18n 校验 | valid；500 system、384 business；350 个字面引用 |
| 生产构建 + 真实隔离 API 浏览器测试 | 9 passed；320/768/1440 三种宽度 |
| 本次已跟踪改动的 `git diff --check` | 通过 |
| 全项目 TypeScript | 未通过；本次新增的两项错误已修复，仍有既有测试夹具/类型错误和 ResearchDataLab 的 nullable 问题 |
| AI Hermes 路由覆盖/治理 | 未全部通过，见第7节 |

最终两批后端和前端/i18n/生产浏览器联合验收均检查关键源码 SHA 前后相同，命令输出 `SOURCE_STABLE_DURING_TESTS`。不将不同版本的测试结果拼成一次通过记录。

### 后端 A

使用项目 Python 3.12 虚拟环境，执行：

```sh
python -m pytest backend/tests/test_typed_indicator_types.py backend/tests/test_typed_indicator_compiler.py backend/tests/test_typed_indicator_runtime.py backend/tests/test_typed_indicator_catalog.py backend/tests/test_typed_numba_v22.py backend/tests/test_typed_indicator_product_service.py backend/tests/test_custom_indicator_variables.py backend/tests/test_custom_indicator_service.py backend/tests/test_custom_indicator_routes.py backend/tests/test_scalar_bundle_execution.py backend/tests/test_scalar_indicator_outputs.py backend/tests/test_scalar_outputs_integration.py backend/tests/test_scalar_outputs_workflows.py backend/tests/test_drawdown_multi_output.py backend/tests/test_drawdown_indicator_workflow.py -q
```

### 后端 B

```sh
python -m pytest backend/tests/test_custom_indicator_excel_export.py backend/tests/test_scalar_outputs_excel.py backend/tests/test_custom_indicator_time_series.py backend/tests/test_custom_indicator_time_series_excel.py backend/tests/test_indicator_graph.py backend/tests/test_indicator_formula_roundtrip.py backend/tests/test_indicator_parallel_engine.py backend/tests/test_evaluation_run_results.py backend/tests/test_portfolio_research.py -q
```

### 前端与浏览器

```sh
npm run test --prefix frontend -- --run --silent src/components/indicator-outputs src/components/indicator-graph src/components/metrics src/pages/IndicatorStudio.test.tsx src/pages/IndicatorStudio.roundtrip.test.tsx src/pages/ProductDetail.test.tsx src/pages/ProductCompare.test.tsx src/pages/ProductResearch.test.tsx src/pages/EvaluationPlan.test.tsx
npm run i18n:check --prefix frontend
INDICATOR_TEST_PYTHON=/path/to/project/python3.12 npm run test:e2e --prefix frontend -- --config=playwright.indicator-output.config.ts --workers=1
node frontend/node_modules/typescript/bin/tsc --noEmit -p frontend/tsconfig.json --pretty false
```

三个浏览器工作流：原有多公式组完整操作；从空白创建共享输入的多结果自定义指标并保存重载；内置回撤的四结果、LaTeX、图端口、预览、导出。API 使用生产路由及临时工作区数据，不修改用户正式指标和市场文件。

开发服务器模式曾出现平板测试接口错误及页面重载/等待超时；期间观察到同工作区源码变动。未把每一次错误都归因于已证明的同一原因。最终使用无热更新的生产构建验收，连续两轮9项通过，最后一轮另有源码稳定性检查。

## 5. 性能测量

```sh
python backend/scripts/benchmark_drawdown_outputs.py --size 100000 --repeat 100
```

实际输出：100,000 个净值点，4个输出，一次内核调用；100次测量中位数0.2689375毫秒，输入800,000字节，常数辅助空间，新增签名0。

该测量仅包含已预热回撤内核，不包含I/O、参数校验、编译、接口序列化；没有测量相对旧应用的端到端提速倍数。

## 6. 导出与边界

同一产品窗口的原生多输出共用一张计算表，汇总逐项引用；不生成四份重复回撤扫描。原生 Excel 公式覆盖输入、峰值、谷底、恢复及最长水下状态。未恢复值留空并说明原因。

自动测试检查单一计算表、单元格引用、原生公式、无 `#REF!` 和平台缓存值。没有使用桌面 Excel 进行独立公式重算，不能把工作簿结构验证表述为桌面 Excel 数值回归已通过。

本次原生多输出仅开放具名数值标量端口，不将日期、任意对象或混合矩阵结果冒充标量。首个内置原生多输出算子为最大回撤分析。

## 7. 治理及工作区说明

CodeGraph 最新检查为697个文件、14,206节点、46,593关系，索引已同步。R70能定位指标相关模块；路由中已有 named output ports / drawdown_analysis 关键词及共享固定签名执行约束。

显式10路径覆盖检查中，typed_types和custom_indicators目录内实现有覆盖；7个新文件尚未纳入稳定路由路径，包括原生算子文件、两项后端测试、共享输入前端组件、端口测试和专用Playwright配置。新文件仍未git跟踪，本次没有为绕过治理检查擅自暂存，也没有将真实源代码加入artifact白名单。

全项目路由验证仍报告其他工作区模块的大量未跟踪稳定引用；未清理、回退或暂存其他任务文件。该检查不是通过状态。

实施过程中发现同工作区另有市场归因清理改动，最终按当前源码验收：市场内置ID已删除，而最大回撤旧单值ID保留。若外部历史研究曾引用被移除的市场ID/变量，不能承诺当前仍可回放；本次没有把它自动改写成新模型。

当前没有执行Git提交、推送或正式服务重启。生产浏览器测试只启动和关闭自己的隔离服务。依赖弃用提示、浏览器映射数据过旧及bundle体积提示尚未通过升级依赖处理。
