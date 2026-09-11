# 通用区间计算图滚动执行：自审核与验收

验收日期：2026-09-10。范围为本需求的指标引擎、指标中心、参数、画布、Excel 与因果审计。代码保留当前工作区，未提交、未推送；不处理工作区其他并行需求。

## 1. 最终能力

新派生协议 3.0.0 使用 `rolling_apply(完整区间标量子图, window)`。作用域实现版本为 `interval-rolling-1.3`。它不是把各个归约替换成专用 rolling_xxx，也不在全历史上先计算 body。

已有基础算子组成的新指标，经完整依赖、输出类型和区间语义检查合格后，可直接生成滚动时序，不需修改滚动执行器。已验证均值、标准差、方差、极值、累计收益、分位数、条件分位统计、CVaR、最大回撤、Sharpe、Calmar、相关系数、拟合投影和自定义组合。

数值标量是必要但不充分条件。纯取首末值/长度、时序输出、未声明区间语义、依赖窗口外递归状态、嵌套滚动等不自动开放。新的未注册数学算法仍需开发正常算子及契约，不能执行任意 Python 回调。

用户流程：选择已保存的区间指标 → 生成滚动时序指标 → 搜索/确认来源、设置整数窗口 → 生成 → 可在“计算参数”开放窗口并设置默认值 → 保存 → 预览或其他调用页面临时覆盖/恢复默认。画布保留真实 body 节点与滚动作用域；修改不改写原来源指标。

## 2. 代码与算法审核结论

- `rolling_scope.py` 负责能力分析、依赖闭包、静态 body 生成及通用窗口循环。未按业务指标名称分派。
- 外层 emitter 不提前执行仅属于 body 的节点；同一节点存在外部消费者时仍保留外部计算。窗口内公共子表达式只计算一次。
- 收益窗口使用 W 个收益和需要时的 W+1 净值点；纯价格窗口使用 W 点。样本数、自然日数、区间无风险收益按每个窗口重新绑定。最大回撤峰值从窗口起点重置。
- 全部数值执行走预先准备的固定签名 NJIT。保存/校验/预热承担编译；实际运行不创建新签名。窗口变更复用计划，参数哈希和取数历史范围同步改变。
- 原始输入切片不复制。已使用源码级 `np.shares_memory` 探针和实际编译的地址探针验证，窗口起点地址逐个等于基础数组地址加正确偏移；只检查地址，不做裸指针读写。
- 只读输入不变，非连续输入直接进入底层固定签名时拒绝，不偷偷扩展签名或复制；正式数据边界负责已有规范化。
- 输出和有效性前缀一次分配；必要中间结果、排序工作区仍会分配，不能称零分配。分位数移除了一次重复复制。
- 窗口1..5000；按点数×窗口×节点数做工作量估算，按输出、有效性前缀和窗口中间量做内存估算。此门禁不是硬实时保证，复杂指标仍须看成本。
- Scope 版本纳入父计划 ID。新旧结果缓存不会因作用域实现版本遗漏而混用。
- 保留旧派生协议1/2及已存在内置时序 revision；新建派生采用3。历史公式未原地改写。

## 3. 自审核实际发现并修复的问题

1. **Excel 流式行回写失效**：窗口上下文先写占位值、后改公式，在 constant-memory 工作簿中旧行已经刷新，导致样本数/天数/区间无风险收益仍是占位值。现改为在变量行第一次输出时直接写入真实公式，并增加打开工作簿检查公式行的回归。
2. **异常路径 NRT 引用泄漏**：仅验证内层 helper 不足。实际父 NJIT dispatcher 调用子作用域、子作用域校验抛异常时会留下引用。作用域在 Numba IR 层内联，校验先于分配，body 异常在所属函数中正常返回缺失值；通过实际父 dispatcher 重复测试。
3. **动态长度潜在越界**：两个 `T-n` 分支可能拥有不同实际长度。加入多序列节点实际长度检查，防止调用未默认开启 bounds checking 的配对内核时越界。
4. **计划身份遗漏 scope 版本**：已修复并验证版本变化改变计划 ID。
5. **前端类型与测试契约**：修复 window 形状徽标映射、测试选择器类型、旧测试对有限归约改写的过时断言。旧兼容数值仍单独测试，不通过删除历史验证来达成通过。

## 4. 测试结果

后端分两批回归加最后新增地址探针，去重后 **533 项通过**。最终收集命令确认当前这18个测试文件共533项；新增 safety 文件最终8项再次全通过。

第一组：

```
python -m pytest tests/test_rolling_interval_graph.py tests/test_rolling_interval_service.py tests/test_rolling_interval_safety.py tests/test_typed_indicator_catalog.py tests/test_typed_indicator_compiler.py tests/test_typed_numba_v22.py tests/test_causality.py tests/test_indicator_graph.py tests/test_indicator_formula_roundtrip.py -q
```

第二组：

```
python -m pytest tests/test_custom_indicator_time_series.py tests/test_custom_indicator_time_series_excel.py tests/test_custom_indicator_service.py tests/test_custom_indicator_routes.py tests/test_custom_indicator_variables.py tests/test_custom_indicator_excel_export.py tests/test_series_runtime_parameters.py tests/test_typed_indicator_runtime.py tests/test_typed_indicator_types.py -q
```

上述命令在 backend 下使用项目指定 Python3.12，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`。市场数据来自本地测试夹具，无行情网络请求。

前端：**108 项通过**（9个测试文件），涵盖 IndicatorStudio、公式往返、画布、参数编辑与运行请求。

```
npm run test -- --run src/pages/IndicatorStudio.test.tsx src/pages/IndicatorStudio.roundtrip.test.tsx src/components/indicator-graph src/components/indicator-parameters/IndicatorParameters.test.tsx src/services/customIndicatorsExecution.test.ts src/services/customIndicatorsRolling.test.ts
```

浏览器：**8 项通过**，桌面1440px和窄屏320px，各4项。使用隔离的真实 FastAPI+NJIT 后端；产品搜索为固定夹具，其他无关 API 不参与。覆盖新自定义回撤从搜索、生成、参数默认值、保存、真实图节点、临时覆盖、Excel下载到恢复默认，也覆盖旧均线参数、独立日期指标和共享拟合回归。无页面异常、无文档横向溢出。截图保存于 Playwright test-results，可能被后续测试覆盖。

```
INDICATOR_TEST_PYTHON=<项目Python3.12> npm run test:e2e -- indicator-primitives.spec.ts --project=desktop-1440 --project=mobile-320
```

构建、i18n、git diff --check 通过。Excel 检查包含 ZIP 完整性、原生公式、输入只写一次、完整区间图、真实上下文公式、有限缓存值；没有调用桌面 Excel 应用强制重算，不将 XML/缓存检查描述成桌面 Excel 验收。

## 5. 可重复性能和内存测量

```
NUMBA_NRT_STATS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python scripts/benchmark_rolling_interval.py --size 4096 --window 63 --repeats 7
```

本机合成固定数据4096点、63观察窗口、7次稳态测量；排除编译时间、不含HTTP和数据读取。中位数：

| 计算 | 毫秒 |
| --- | ---: |
| 均值 | 0.710 |
| 标准差 | 1.199 |
| 最大回撤 | 1.055 |
| Sharpe | 1.539 |
| Calmar | 1.646 |
| CVaR | 5.154 |
| 自定义组合 | 2.372 |

每项输出32768字节，有效性前缀32776字节，另有单窗口必要中间量。测得整个进程峰值RSS354.41MiB，**包括Numba编译与已加载库，不是单个滚动计算的额外内存**。

7项稳态计算的NRT净存活分配增量均0。成功、零分母、数组除零、数组log定义域错误、无效净值路径、非法日期轴这6类场景分别重复40次，实际父dispatcher的NRT净存活分配增量均0。该结论限于这些已测路径，不宣称穷尽所有潜在内存问题。

复杂度仍是各窗口原子图成本之和。扫描类通常O(TW)，排序类更高；没有声称任意图经NJIT自动变成O(T)，也没有未经测速开启prange。

## 6. 扩展检查中的范围外问题

- 包含 ProductDetail 的扩展前端测试共126项，125通过、1失败：产品详情标签键盘切换断言（并行页面改动）。本次未修改该业务页面来掩盖失败。
- 全项目 TypeScript检查仍有3个诊断：EtlWorkflowEditor.test.tsx 两项，ProductDetail.tsx 的数组 `.at`/目标lib一项。本需求直接涉及的IndicatorStudio类型诊断已修复。
- AI Hermes已有入口与路由文件的限定覆盖检查通过；repo_map purpose及P13已同步完整区间作用域与风险点。全仓validator仍被其他需求的未跟踪稳定引用阻塞。本需求新文件也仍未Git跟踪；提交时需逐项纳入并补充稳定路径索引，不得将其伪装成历史备份或用宽泛忽略规则掩盖。
- 构建存在既有大chunk及浏览器兼容数据过期警告，未为此改动依赖或构建拆包。

## 7. 交付状态

本需求的功能实现、专项审核、数值/历史/因果回归、参数与画布流程、真实后端浏览器验收和可重复性能/内存检查已完成。全仓其他模块并非全部通过。未提交、未推送Git，未重启或修改正式用户数据服务。
