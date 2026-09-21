# C++ AOT 契约验收记录

日期：2026-09-20。工作区：同一 `CppCalAST` 下的平台与 CalMetricsEngine；基线分别为 `9d35563` 和 `f952f37`，本次改动未提交。

## 2026-09-20 验证

| 验证 | 结果 |
| --- | --- |
| 最终 Release wheel 构建、临时目录安装 | 通过，calmetrics-engine 0.3.0，CPython 3.12 / macOS arm64 |
| 引擎完整 Python 回归 | 4062 通过 |
| 原生 CTest | 5/5 通过 |
| 平台 typed DSL、runtime、service、routes、参数与因果性回归 | 344 通过 |
| 最终平台 C++ 接入和执行门禁 | 33 通过；含真实 NJIT 批次的值/状态对照 |
| 前端完整 Vitest | 1333 通过 |
| TypeScript、Vite build、design:check | 通过 |
| AI 路由结构/覆盖 | 在包含新增文件的临时候选索引中通过；真实 Git 暂存区保持不变 |
| Chrome 浏览器 AOT 门禁 | 6 通过：320/768/1440，各含接受和拒绝；覆盖凭据展开、页面溢出、文字对比度 |

平台 344 项回归与最终 33 项存在重叠，不能相加为独立测试数。浏览器是合成响应契约测试，数学等价由真实原生包与后端测试证明。测试使用 `CUSTOM_INDICATOR_DATA_DIR` 临时目录，不向软链接指向的正式 Tushare 数据目录写入。

## 关键边界

- 除零、定义域、算子内部短样本错误只污染依赖根；严格入口仍抛出原异常。根状态在进程传输和 DAG 分支归并后保持列顺序。非法几何与内存/worker 故障不被吞掉。
- N 条 NAV 对应区间内部 N−1 条收益率；短窗口、跨区间边界、历史 DSL/LaTeX、独立参数与利率槽均有测试。开放参数值改变不修改原定义或图指纹。
- `run_snapshot()` 与下一次输出不共享内存，保留只读结果与状态，在 scheduler 关闭后仍有效；`run_audit()` 借用行为保持且明确声明。
- C++ 原生审计通过后端与前端门禁；缺字段、矛盾后端、Python 回退/回调、请求期编译、非法 CPU 配额均拒绝。

## 可重复命令

使用 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。先在引擎目录执行 `python -m build --wheel`，将生成的 wheel 安装至隔离环境，再运行：

```sh
# CalMetricsEngine
python -m pytest tests -q
ctest --test-dir <native-build-dir> --output-on-failure
python tools/check_phase2_performance.py --better-root ../FundInvestmentResearchPlatform --cpu-budget 4 --output-dir <temporary-output-dir>

# FundInvestmentResearchPlatform，已安装上述 wheel
CUSTOM_INDICATOR_DATA_DIR=<temporary-data-dir> PYTHONPATH=backend python -m pytest backend/tests/test_cpp_aot_contracts.py backend/tests/test_compute_policy.py -q

# frontend
npx tsc --noEmit
npx vitest run
npm run build
npm run design:check
npx playwright test e2e/cpp-aot-execution.spec.ts --workers=2
```

原生共享内存与浏览器启动需要正常的本机进程权限。性能门禁是现有严格模式的 500/1000 产品批次与 63/252 观察微批次；不据此宣称所有平台业务、隔离模式或全服务启动已完成性能迁移。

## 范围结论

三类接入契约已实现并验证；平台提供显式单产品标量接入 API。现有生产服务仍沿用原 NJIT 路由和预热，本次没有部署、替换全部指标服务，也未宣称已经解决整台后端的启动耗时。V1 DSL、组合矩阵和时序指标服务迁移仍需单独接入与业务验收。

## 性能记录

最终性能门禁四种负载全部通过，4 个 CPU token、每项 5 次成对重复。500/1000 产品各 12 区间、16 指标的 Native/NJIT 中位耗时比为 0.636/0.603；63/252 观察微批次的 prepared 比值为 0.527/0.668，普通 scheduler 比值为 0.856/0.829。该记录覆盖成功计算路径；随后仅收紧失败状态码映射，数值成功路径未改变。

本机日志：`/private/tmp/calmetrics-contract-full-tests.log`、`/private/tmp/calmetrics-contract-ctest.log`、`/private/tmp/firp-aot-backend-regression.log`、`/private/tmp/firp-aot-contracts.log`、`/private/tmp/firp-aot-vitest-full.log`、`/private/tmp/firp-aot-browser.log`。成对性能 JSON 位于 `/private/tmp/calmetrics-contract-final-performance`。临时文件不会随 Git 提交；上面的命令用于重新生成证据。

## Dev 集成复验（2026-09-21）

提交候选从远端 Dev `dd3ba3046a213e4bf151881aea9b6c02239913ba` 建立，提交前同步至 `d8f530c9924c8ef097da9b2e4ae989035be2c53d`。同步仅涉及文档检查器、文档规范与坑点；前后端代码与上述测试候选逐字一致。既有文档重组保留，AOT 规则归入数值规范，契约与证据登记到文档目录。原工作区的 41 个修改文件哈希不变；测试在隔离 worktree 和临时数据目录执行。

- 原生环境：已构建的 calmetrics-engine 0.3.0，CPython 3.12 / macOS arm64；未在服务启动或请求期间构建机器码。
- C++ AOT 集成与执行门禁：33 项通过，包括真实 NJIT 批次的值和状态对照。
- 指标兼容回归：491 项通过（typed DSL、运行时、服务/路由、参数、滚动与因果性），与上述 33 项使用不同测试文件，合计 524 项。
- 前端：163 个文件、1343 项 Vitest 通过；TypeScript、Vite build、设计棘轮与语言检查通过。
- 保留旧门禁：产品比较和等权接口缺少任一 NJIT 后端声明的 4 个反例在修复前失败；修复后相关 65 项及完整前端回归通过。未安装原生包时平台适配模块导入、NJIT 门禁正常；显式 AOT 构造失败关闭。
- Chrome：6 项通过，覆盖 320/768/1440 宽度的接受、拒绝、凭据展开、溢出与文字对比度。
- 文档检查器：47 项通过（含最新 Dev 检查器修复）；66 份文档、349 处本地链接、暂存候选检查、Hermes 结构与本次文件覆盖检查通过。

历史引擎全量回归和性能记录仍仅适用于上文日期与版本；本次提交没有切换平台默认服务后端，也没有部署。
