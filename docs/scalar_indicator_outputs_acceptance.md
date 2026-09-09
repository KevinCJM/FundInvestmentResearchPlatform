# 指标中心多标量输出：自审核与验收记录

审查日期：2026-09-07。范围：当前工作区的自定义标量多结果功能，不代表全项目所有检查已通过。

## 结论

多标量功能已完成代码接入并通过专项验收：后端 427 项、前端 106 项、三个尺寸的真实 API 浏览器流程通过，Vite 生产构建通过。全量前端、全项目 TypeScript 与路由治理仍有下述未通过项，不能称为“全项目全绿”。

未提交、未推送、未暂存修改；没有回退其他开发工作，也没有重启正式服务。浏览器验收使用临时工作区和固定测试数据，未写入正式指标或行情目录。

## 用户可见能力

保留“标量指标”入口，点击“＋添加结果”扩展为多个结果，最多八项。新结果默认仅展示。支持独立公式、名称、格式、精度、方向；复制和排序保持正确身份；最后一项不能删除。

多结果在详情中按组展示，比较中逐项排列，评价方案按具体结果配置权重。公式复用、组合诊断、快照配置、滚动派生与 Excel 导出已接入具体结果；历史版本保持锁定。

## 自审核发现并修复

| 问题 | 修复与验证 |
|---|---|
| 保存后源码变为可逆 LaTeX，旧校验令牌导致预览失败 | 保存后失效旧校验；预览和导出重新校验当前草稿。真实浏览器验证保存、预览、下载闭环 |
| 多结果组把第一项窗口冒充共同窗口 | 逐项保留窗口；只有一致才声明共同窗口，不一致显示 per_output |
| 缓存命中数固定为零 | 汇总真实命中与未命中计数，并测试重复调用 |
| 新服务实例无法读取已预热组合计划 | 按工作区共享当前进程的显式预热计划；令牌仍由当前实例签发，执行不重新编译 |
| 缺少基准或某个行情字段影响无关结果 | 按输入依赖分组，输出独立诊断；缺失不填零 |
| 评价方案可能选择整组或默认把 Beta 当作越高越好 | 必须指定 output_id；neutral 必须明确指定评分方向，比较不自动标最佳/最弱 |
| 快照同指标多结果写入同一列 | 配置和列身份加入 output_id，禁止把标量结果套用 last_finite |
| 滚动派生不区分来源结果 | 冻结 output_id、revision、源定义哈希和窗口，错误来源被拒绝 |
| 多结果导出缺少独立公式和单位 | 复用原有 Excel 公式编译器，逐输出保存直接输入、公式和对照值，失败结果留空并保留原因 |
| 长指标名与长结果名合并后超出内部单指标校验长度 | 内部验证使用结果本名，外部完整显示名不截断；增加边界测试 |
| 原时序指标校验提示被多结果提示覆盖 | 恢复时序专用提示，并通过旧时序页面回归 |
| 新结果标签缺少键盘切换 | 增加方向键、Home、End、焦点移动和 tabpanel 关联，并测试 |

## 最终专项测试

以下两组后端没有重复测试文件，合计 427 项。

### 后端主链路：156 passed

```sh
python -m pytest backend/tests/test_scalar_bundle_execution.py backend/tests/test_scalar_indicator_outputs.py backend/tests/test_scalar_outputs_integration.py backend/tests/test_scalar_outputs_workflows.py backend/tests/test_scalar_outputs_excel.py backend/tests/test_custom_indicator_service.py backend/tests/test_custom_indicator_routes.py backend/tests/test_custom_indicator_time_series.py backend/tests/test_indicator_graph.py backend/tests/test_indicator_formula_roundtrip.py -q
```

### 后端内核与下游：271 passed

```sh
python -m pytest backend/tests/test_typed_indicator_types.py backend/tests/test_typed_indicator_catalog.py backend/tests/test_typed_indicator_compiler.py backend/tests/test_typed_indicator_runtime.py backend/tests/test_typed_numba_v22.py backend/tests/test_custom_indicator_variables.py backend/tests/test_indicator_parallel_engine.py backend/tests/test_evaluation_run_results.py backend/tests/test_portfolio_research.py backend/tests/test_instrument_analytics.py backend/tests/test_custom_indicator_excel_export.py backend/tests/test_custom_indicator_time_series_excel.py -q
```

实际使用 AGENTS.md 推荐的 Python 3.12 环境。两组均只有 TestClient 依赖弃用提示，没有失败。

### 前端专项：106 passed / 13 files

```sh
npm run test --prefix frontend -- --run --silent src/components/indicator-outputs/ScalarOutputEditor.test.tsx src/utils/scalarOutputReferences.test.ts src/components/metrics/ScalarOutputDisplay.test.tsx src/pages/IndicatorStudio.test.tsx src/components/metrics/MetricDisplay.test.tsx src/pages/EvaluationPlan.test.tsx src/pages/HoldingDiagnosis.test.tsx src/services/portfolioResearch.test.ts src/components/indicator-graph/indicatorGraphAdapter.test.ts src/components/indicator-graph/useIndicatorGraphEditor.test.tsx src/pages/ProductResearch.test.tsx src/pages/ProductDetail.test.tsx src/pages/ProductCompare.test.tsx
```

### 浏览器：3 passed

```sh
INDICATOR_TEST_PYTHON=<项目 Python 3.12 解释器> npm run test:e2e --prefix frontend -- scalar-outputs.spec.ts --workers=1
```

mobile-320、tablet-768、desktop-1440 均通过。覆盖：真实 API 新建、添加结果、切换公式、保存版本、画布还原、固定数据预览、实际 Excel 下载、页面无运行异常及页面宽度检查。

测试过程中的截图、下载文件和失败诊断位于 `frontend/test-results/`，为本地可再生成测试产物，不作为正式用户数据。

### 构建及其他检查

- `npm run build --prefix frontend`：通过；存在项目包体积及浏览器映射数据较旧的提示，没有擅自升级依赖。
- `node scripts/check_i18n.mjs`：通过；新增系统文案具备中英条目。旧页面仍有未翻译文本，不宣称本次完成全站翻译。
- `git diff --check`：通过。
- CodeGraph 已同步；检查时为 672 文件、13,974 节点、45,739 条关系。这是当前工作区快照，包含其他开发任务的改动。
- R70 路由上下文检查可正常输出相关模块。

## 全项目检查的未通过项

### 全量 Vitest

`npm run test --prefix frontend -- --run --silent`：577 passed、2 failed，91 个文件中 89 个通过。

未通过项：

1. `ClassAllocation.test.tsx`：历史情景下拉列表中缺少 `regime-run-1|regime-publication-1`。单独重跑仍未通过。
2. `FactorResearchCenter.test.tsx`：全量运行中未找到“复制构建”按钮；随后与上项单独重跑时四项全部通过，表现为运行时序敏感。

这两个页面不属于本轮多标量功能修改范围，未通过放宽断言或修改其他开发任务来制造全绿。

### TypeScript

`node frontend/node_modules/typescript/bin/tsc --noEmit -p frontend/tsconfig.json` 未通过。发现并修复本次新增的 ES API 兼容和测试夹具字段问题；全项目仍存在原有测试夹具类型不完整、readonly/可变数组不一致、测试 afterEach 返回值和 `ResearchDataLab.tsx` 的 `profile.vintage` 可能为空等问题。Vite 构建成功不等于全量类型检查通过。

### AI Hermes 路由治理

已更新已追踪入口对应的多结果行为、引用及令牌失效契约。治理校验仍因工作区原有大量未追踪稳定路径失败，涉及因子研究、情景研究和 ETL 等模块。新增多结果内核、编辑器及新测试的精确路径也尚未全部进入已追踪路由文件清单。

未使用 artifact 白名单掩盖源代码，未擅自暂存其他文件。待确定 Git 提交范围时，应将新实现与测试纳入版本控制并补齐精确路由清单，再执行完整治理校验。

## 明确边界

- Excel 已验证原生公式、缓存对照值、单位、窗口、版本与不可计算状态；未启动独立 Excel 引擎重算，不把 NJIT 缓存值当作独立 Excel 数值验证。
- 复用同依赖组的输入与公共计算；不同物理依赖组仍可能分别扫描同一来源。没有宣称“全组只读一次”或未经测量的加速倍数。
- 实现通用多标量能力，不新增归因专用数据源或把残差时序伪装成标量。
- 未将本次专项通过解释为真实市场数据口径已全部验收，未以静态测试数据冒充正式投资结果。
