# 独立指标与算子原语化：实施及专项验收

## 结论

当前独立指标正常调用链、原语构建、共享计算、日期与时长、快照和 Excel 联动已完成本轮专项验证。公开标量契约不再提供父指标/子结果或指标组；正常时序多通道与参数契约继续保留。

2026-09-09 提交整理已完成35个退役文件及两份一次性清理工具的物理删除，历史保留于Git提交 `3ef7d07`。以下为此前专项记录，提交整理后的全项目检查见文末。

## 本轮完成

1. 新增 `backend/custom_indicators/snapshot_execution.py`，分开独立标量/日期与时序通道路径。标量批次显式准备锁定版本的共享计划；同一时序指标版本和目标的多个配置通道只运行一次时序计划。
2. `series_service.py` 内部快照模式在完整请求区间内执行固定签名 NJIT 末个有限值归约，随后记录实际 value_date。与普通图表缓存隔离，不从5000点显示截尾中取值，不把预热观察值当成本期结果。
3. 快照字段保留 channel_id、reducer、通道单位/精度/格式与值日期。数值0保留；无有限值明确不可用；最后有效值早于区间末日明确警示。生成中数据代际变化则失败关闭，任务完成/失败均关闭计算资源。
4. 清除滚动来源中的标量子结果复制/匹配分支；显式拒绝退役子结果字段，不默默绑定到整个指标。正常来源版本、定义哈希、转换协议兼容仍保留。
5. 删除前端残留的多标量端口类型，以及编译器已不可达的属性输出节点特殊分支。删除无生产调用者的旧 shadow 比较函数和仅针对它的测试。
6. API 对退役 scalar_outputs/output_id 字段按键存在性拒绝，包括空列表和 null。新建、计算、导出和滚动入口不再默默接受旧请求。
7. 数学回显自审修复：协方差/相关系数公式显示两个输入；嵌套 argmin/argmax 下标正确作用于整个表达式。
8. 将旧浏览器测试文件内仍有用的时序可变参数测试迁入当前独立指标验收文件，删除旧文件时不会丢失真实契约覆盖。
9. 重写性能基准，比较当前共享 DAG 与当前独立单指标计划，不依赖已退役内核。
10. 补齐指标相关前端测试夹具的实际类型契约：输入字段元数据、锁定展示信息、滚动来源协议及转换信息；修复测试返回值和可空DOM收窄，不放宽生产类型。相关前端专项再次147项通过。
11. 同步当前设计、市场归因移除说明和路由事实。未重置其他工作区改动，未修改业务数据，未提交/推送，未重启正式服务。
12. 补齐新原语的因果审计候选与基线：日期、回撤区间和回归拟合中间状态均可构造有效探针输入；退役 drawdown_analysis 基线项已移除。`backend/tests/test_causality.py` 最新 56 项全部通过，`last_drawdown_interval` 与 `linear_fit` 明确标记为 window_consuming，字段投影/value_at/days_between 为 causal。

## 后端专项：508 项通过

以下为互不重复的四组当前测试文件，取本轮最后一轮结果；不是后端全量测试计数。

### A. 原语、计算图、类型、公式及清理保护：259 passed

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_operator_primitives.py \
  backend/tests/test_independent_drawdown.py \
  backend/tests/test_independent_indicator_workflow.py \
  backend/tests/test_typed_indicator_catalog.py \
  backend/tests/test_typed_indicator_compiler.py \
  backend/tests/test_typed_indicator_runtime.py \
  backend/tests/test_typed_indicator_types.py \
  backend/tests/test_typed_numeric_backend.py \
  backend/tests/test_typed_numba_v22.py \
  backend/tests/test_indicator_graph.py \
  backend/tests/test_indicator_formula_roundtrip.py \
  backend/tests/test_builtin_market_attribution.py -q --tb=short
```

覆盖最后并列谷底、同一区间日期/时长、无事件/未恢复、索引边界、受限 AST、独立根共享、分支裁剪、缓存不执行、错误隔离、数学记号、被移除市场归因不可调用，以及临时目录中的清理安全性。

### B. 服务、API、取数与评价：62 passed

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_custom_indicator_service.py \
  backend/tests/test_custom_indicator_routes.py \
  backend/tests/test_custom_indicator_variables.py \
  backend/tests/test_indicator_parallel_engine.py \
  backend/tests/test_evaluation_run_results.py -q --tb=short
```

### C. 时序、可变参数及混合快照：63 passed

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_custom_indicator_time_series.py \
  backend/tests/test_custom_indicator_time_series_excel.py \
  backend/tests/test_series_runtime_parameters.py \
  backend/tests/test_independent_snapshot_execution.py -q --tb=short
```

新增反例：5020个观察值仅最早10个有有限结果，普通图表截尾后不可用，快照仍正确选择完整请求区间的末个有限值；切换1W区间不得泄漏之前的结果。混合标量、日期和两个KDJ通道分别写入正确字段，同一时序实例只执行一次。

### D. 产品快照、ETL快照及Excel：124 passed

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_instrument_analytics.py \
  backend/tests/test_etl_snapshot.py \
  backend/tests/test_custom_indicator_excel_export.py -q --tb=short
```

共同警告：FastAPI/Starlette 测试依赖弃用提示；不影响以上通过结果，未为此扩展依赖升级。

## 前端专项：147 项 / 16 个文件通过

使用 Vitest 执行 IndicatorStudio、公式往返、产品详情/对比/研究、评价方案、持仓诊断、计算请求、指标显示/偏好、单一结果设置、时序参数、画布适配/常量/展示/状态管理测试。没有把待删除的旧多标量测试计入通过项，也没有更改测试全局排除规则来掩盖它们。

## 浏览器：9 项通过

```bash
cd frontend
INDICATOR_TEST_PYTHON=/Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  node node_modules/@playwright/test/cli.js test \
  --config playwright.indicator-output.config.ts --workers 1
```

手机320、平板768、桌面1440各验证三条流程：
- 独立日期指标的LaTeX、单根画布往返、保存、真实API预览、正确日期及Excel下载。
- 两个独立拟合指标的共享计划只有一个linear_fit调用点；不返回业务多结果；向导正确还原拟合中间状态。
- 时序窗口默认值5→7、保存、画布参数节点、临时覆盖3、复用计划、导出、恢复默认7且不修改已保存版本。

使用生产 Vite 构建和临时数据的真实指标路由/NJIT后端；非本功能API隔离，不是正式数据环境全链路上线验收。无浏览器pageerror，无横向页面溢出。

## 性能证据

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  backend/scripts/benchmark_drawdown_outputs.py --products 100 --observations 1000 --repeats 5
```

固定随机种子的100产品×1000观察值、4个独立回撤指标：共享执行最小耗时0.000555875秒，逐指标执行0.001823667秒，后者约为前者3.28倍。编译时间不计入计时，比较双方均为当前NJIT实现；数值和状态一致，运行期间无新增签名。共享图中drawdown_series/last_drawdown_interval各一个调用点，独立执行图共有4个回撤调用点。仅代表该本地合成数据基准，不宣称所有生产场景均有相同倍数。

## 其他检查

- 生产构建通过（浏览器配置先执行build）；包体积及浏览器映射数据提示保留，未做无关打包升级。
- i18n字面引用检查通过，system514/business395、348个引用、errors为空；本轮删除了已退役多结果/旧回撤算子的无引用词条。旧页面未翻译文本的库存不等于已全部国际化。
- 修改的Python文件语法编译及 `git diff --check` 通过。
- 对当前保留的后端、前端、E2E、脚本做导入扫描，退役模块的当前导入引用为0。
- R70路由解析成功；原语、共享DAG、独立版本锁定及快照通道的过期路由文字已更新。

## 2026-09-09 提交整理

先将全部源码改动保存为Git提交 `3ef7d07`，随后执行既有清理清单。35个退役文件均通过内容哈希、路径和真实存量契约检查；活动Python源码对这些退役模块的导入为0。删除成功后，一次性清理脚本和清单同时删除，共移除37个文件。没有修改业务数据，没有增加测试排除规则。旧实现、清单和清理保护测试均通过Git历史追溯。

项目运行文件和本地导出报告加入忽略规则；源码、测试、配置、文档以及文档引用的配图纳入提交。补齐路由R70引用但未声明的算子因果基线检查规则。

提交整理后检查：Python3.12语法、完整TypeScript检查、Vite生产构建、i18n引用检查和路由结构检查通过；后端专项392项与因果检查56项通过，全后端2195项测试可收集（不等于全量执行通过）。前端全量662项通过、2项失败，仍为整理前已有的ClassAllocation历史情景选项和FactorResearchCenter模板复制按钮测试；未跳过失败项，也未为提交改变这两个业务流程。
