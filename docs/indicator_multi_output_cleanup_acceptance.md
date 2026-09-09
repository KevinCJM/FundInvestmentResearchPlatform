# 指标中心多输出与最大回撤：清理及验收记录

日期：2026-09-08。

## 验收结论

本次指标中心功能专项验收通过；源码清理和全项目发布检查尚未全部关闭。不得把本记录理解为全仓库无死代码、全项目测试通过或正式服务已发布。

- 后端两组专项回归：298 + 192 = 490 项通过。
- 前端指标及消费页面专项：151 项通过。
- 手机320、平板768、桌面1440：9 项浏览器流程通过；在最后一次生产代码清理后重跑，仍为9项通过。
- Vite生产构建、i18n引用检查、指定Ruff规则、git diff --check通过。
- 全量前端：597项通过、2项失败；全项目TypeScript检查未通过。
- 路由治理未通过；一个已清空的旧源码文件仍待物理删除。

## 清理范围与实际修改

### 市场归因

移除指标目录注册、兼容目录注册及其固定指数变量；直接请求此前的内置ID返回404，而不是仅由前端隐藏。清除专用取数分支、固定指数转换特例、相关翻译和无用浏览器数据夹具。

删除前检索本地配置未发现固定市场归因ID或专用变量引用；最后针对 data/custom_indicators.json、evaluation_plans.json、portfolio_runs.json、product_pools.json、research_targets.json 的定向检索亦无匹配。此结论仅针对已检查的本地文件，不代表外部客户端或其他工作区。一次全data目录递归审计超时，没有将其计为完成。

原市场归因实现文件内容已经清空；当前DevSpace动作仅提供读取、写入、编辑、命令检查，没有文件删除动作，因此 backend/custom_indicators/market_attribution.py 仍是0字节空文件。它已无导入及执行路径，但文件物理删除尚未完成。未借助受限命令绕过工具契约，也未将空文件称作已删除。

旧市场归因测试已改为当前移除契约回归：普通/兼容目录均不包含该指标，读取与计算入口404，固定变量不存在，正常组合benchmark_returns保留。文档改为当前职责边界说明，不再保存旧算法。

### 旧执行分支与重复转换器

- 删除无人调用的 _run_fused_builtin_groups（261行）及其 _fast_builtin_code、相关导入。
- 删除无人调用的旧滚动来源校验 _verify_rolling_source_contract。
- 删除旧 rolling_scalar_time_series_definition 构造函数及导出。
- rolling_series.py 成为滚动派生唯一实现；rolling_scalar.py 仅保留当前API实际使用的版本/观察数边界元数据，不再包含第二套AST转换、哈希与来源校验算法。
- 删除未使用的快照规范化函数、仅由测试引用的缓存查询函数、未使用导入和局部变量。
- 保留现行多结果编译器、公共算子、时序数据转换和实际消费者；没有按文件名包含legacy或版本号就机械删除。

### 保留的真实契约

builtin-maximum-drawdown-v2@1 仍由默认快照真实引用，保持单值返回与原定义，交给当前typed NJIT引擎执行；新版最大回撤分析使用命名输出。测试覆盖旧单值结果与新max_drawdown输出的一致性。没有原地改写已保存指标、评价方案、数据快照、源模型哈希或API数据格式。

两个有效滚动派生API复用同一个当前转换器，保留各自请求字段及元数据范围。未删除正常的指数行情数据、组合基准变量或因子研究数据源。

## 最大回撤的关键验证

最大回撤分析一次扫描产生最大回撤、下跌期数、恢复期数和最长水下期数。期数为观察间隔，不冒充自然日；未恢复时恢复期数为空，不填零。

验收不只检查四个返回值，还检查：

1. 编译图只有一个分析节点，生成计划只有一个多输出调用位置。
2. 同一产品/窗口同时选取三个输出，实际dispatcher只执行一次；重复相同请求不再次执行，使用结果缓存。
3. 执行前后NJIT签名不增加，python_fallback=0。
4. 随机受控参考、上涨/平坦/下跌、已恢复/未恢复、重复峰谷、NaN/Inf/非正净值及空样本。
5. 未恢复的恢复期数不阻断最大回撤的评分；选定输出ID和历史指标版本保持稳定。
6. Excel引用同一共享计算工作表，未恢复值为空，不出现伪造零或失效引用。
7. 用户只配置一次净值输入，勾选结果即可；新建、保存、载入、公式/画布往返、预览和导出均有浏览器验证。

## 实际执行的测试范围

后端第一组（298项）：

```sh
python -m pytest backend/tests/test_drawdown_multi_output.py backend/tests/test_drawdown_indicator_workflow.py backend/tests/test_scalar_bundle_execution.py backend/tests/test_scalar_indicator_outputs.py backend/tests/test_scalar_outputs_excel.py backend/tests/test_scalar_outputs_workflows.py backend/tests/test_scalar_outputs_integration.py backend/tests/test_builtin_market_attribution.py backend/tests/test_custom_indicator_service.py backend/tests/test_custom_indicator_routes.py backend/tests/test_custom_indicator_variables.py backend/tests/test_custom_indicator_time_series.py backend/tests/test_custom_indicator_excel_export.py backend/tests/test_custom_indicator_time_series_excel.py backend/tests/test_indicator_graph.py backend/tests/test_indicator_formula_roundtrip.py -q
```

后端第二组（192项）：

```sh
python -m pytest backend/tests/test_typed_indicator_types.py backend/tests/test_typed_indicator_catalog.py backend/tests/test_typed_indicator_compiler.py backend/tests/test_typed_indicator_runtime.py backend/tests/test_typed_numeric_backend.py backend/tests/test_typed_numba_v22.py backend/tests/test_typed_indicator_product_service.py backend/tests/test_numba_finance_indicators.py backend/tests/test_indicator_parallel_engine.py backend/tests/test_evaluation_run_results.py backend/tests/test_portfolio_research.py backend/tests/test_instrument_analytics.py -q
```

本地实际使用AGENTS.md指定的Python3.12环境。第二组首次为191通过、1失败，原因是测试缺少自己的显式预热、依赖之前测试填充缓存；补齐测试初始化后重跑全组192通过，未放宽生产运行约束。

前端专项（151项）：

```sh
npm run test --prefix frontend -- --run --silent src/components/indicator-outputs src/components/indicator-graph src/components/metrics src/pages/IndicatorStudio.test.tsx src/pages/IndicatorStudio.roundtrip.test.tsx src/pages/ProductResearch.test.tsx src/pages/ProductDetail.test.tsx src/pages/ProductCompare.test.tsx src/pages/EvaluationPlan.test.tsx src/pages/HoldingDiagnosis.test.tsx src/services/customIndicatorsExecution.test.ts src/services/customIndicatorsRolling.test.ts src/utils/scalarOutputReferences.test.ts src/services/portfolioResearch.test.ts
```

浏览器（9项）：在frontend目录，设置INDICATOR_TEST_PYTHON为项目Python3.12解释器后执行：

```sh
npm run test:e2e -- scalar-outputs.spec.ts --workers=1
```

浏览器使用临时数据和真实生产路由/计算内核，不污染用户业务配置；它不是正式服务上线测试。

其他检查：npm run build --prefix frontend；npm run i18n:check --prefix frontend；Ruff F401/F811/F821/F841指定源码检查；git diff --check；CodeGraph sync。i18n检查无错误，但其旧文案库存并不等于全项目已经完成翻译。

## 未关闭项

### 清理

backend/custom_indicators/market_attribution.py 的空文件待物理删除，内容清除与取消注册已经完成。

### 全项目检查

- 全量Vitest：597通过、2失败。ClassAllocation.test.tsx 未找到历史情景发布选项；FactorResearchCenter.test.tsx 未找到“复制构建”按钮。本次未修改这两个测试或其业务实现，未跳过失败项。
- TypeScript全量检查仍有不完整测试夹具、readonly类型、目标lib不含Array.at、空值等错误；其中ResearchDataLab.tsx也有空值检查错误。生产构建成功不等于类型检查通过。
- Scoped evolve_ai_routing 检查发现 test_drawdown_indicator_workflow.py 尚未在路由清单覆盖；validate_ai_routing发现多个因子/情景/ETL等模块引用未被Git跟踪的路径。R70 route_task解析通过。没有为消除错误将无关文件暂存、提交或加入豁免名单。
- 未运行全仓库后端测试，490项仅代表列明的专项与关联回归；未对所有外部部署/历史客户端作兼容证明。

## 工作区与交付边界

工作区原有大量其他未提交修改，均未执行全量重置、清理或覆盖。没有Git提交、推送、暂存和服务重启。未新增依赖；没有因为本次功能而卸载仍供其他模块使用的包。未把参数中心设计或因子中心市场模型迁移称作本次已实现功能。
