# 情景识别剩余能力：后端验收

> 最终集成补充：父任务已完成前瞻服务/预热/路由/进度接口/共享TAA消费接线；最终新增及受影响专项96项、TAA相关66项通过。本文分工范围内“前瞻尚未接入”不再代表最终工作区状态；使用边界和准确命令见 [集成审核记录](regime-completion-final-audit-2026-09-15.md)。

日期：2026-09-15；分支 `ISSUE2609/BetterSaaTaa`。本文件仅描述后端证据。
前端合同见 `regime-completion-api-2026-09-15.md`。不提交、不推送、不接触生产数据；
测试均为新建临时目录、合成数据。现有 dirty work 保留。

## 已实现

- 历史质量 preview / confirm / report / catalog：保存定义精确版本，重执行核对血缘，
  原子不可变报告；预览不创建运行、参考数组、发布或计划清单。
- 覆盖/未知首尾/逐状态区间长度、转换、有效价格段收益。价格语义不能证实或有
  缺失时显示 null/reason；不会把宏观值、上传的任意数字或编码直接当价格。
- 共用有界参数/窗口/适用随机种子/截尾变体。复用已有扰动数学和唯一执行器；
  每个不同图取得正确计划，新建临时计划执行后释放。实时拟合变体沿基准冻结折
  重放，测试输出不会被全样本拟合替换；固定状态轴，不做事后最佳置换。
- 可选 stability/bootstrap policy 默认可执行。真实模型后验才提供选中概率、前两名、
  margin 和 entropy；确定性规则证据为 null，原 raw_type 编码契约保留。
- 固定校准器的成对 moving-block bootstrap，仅独立留出或最终测试。
  五个具名统计量：accuracy、accepted coverage/error、Brier、配对 Brier 改善。
  完整时间轴保留缺失位置，所有指标共享抽样索引；块/完整状态周期/有效重复不足
  不签发区间。不是今天真实状态的置信区间。
- 四个新整数版本 1 的可编辑模板：HMM 风险、GMM 波动、窗口均值变化、趋势共识。
  复用已有特征、模型、分类和组合节点；历史用途默认事后。窗口变化不是 BOCPD/PELT。
  原 PS、CSI300、宏观模板身份和定义不改写。
- 部署资格门禁保留。没有伪造前瞻注册、参考 vintage 或选择历史，报告仍不能使
  新协议 TAA 获得部署资格。前瞻部署资格明确独立、尚未证实。

## 执行预算与内存边界

最多每输入/输出 20000 个观察、64 必需节点、4 拟合模型、100 轮/模型、8 前向折、
8 评价对象、12 变体（默认 6）。数值工作量按数据规模、特征、状态数、迭代和折数
预检，合计上限 100000000；时间预算 120 秒在执行间检查，**不会中途强杀已运行的
NJIT 内核**。bootstrap 最多 500 次、块长 2..250，固定种子 1729。

新增数值逻辑进入启动预热和 PID 所属检查，共 15 个单签名可靠性内核；readonly
任意 stride 固定签名，禁止请求新增签名和 Python 回退。输入解码/状态编码/报告
封装仍分配数组，不能声称端到端零拷贝。比较分配 O(n) 边界缓冲；bootstrap 复用
原数组，分配 (n+1)×10 浮点前缀统计和有界重复缓冲，不物化 n×replicate 数据。

2 万观察、3 状态、500 次 bootstrap，组合质量/边界/概率证据/区间内核重复 5 次：
最终复测中位数 0.009170 秒，Python 跟踪峰值 1,631,734 bytes；进程 max RSS 404,373,504 bytes
（包括导入/Numba 环境，不是本次增量）。四个只读非连续输入均共享底层内存；
15 内核各一个签名，输入未改变。仅内核性能，**不含取数、完整图拟合或 JSON**。
日志 `/tmp/regime-completion-benchmark-final.json`；此前测量保留在 `/tmp/regime-completion-benchmark.json`。

## 实际测试记录

所有命令统一前置（每次使用新目录）：
```sh
regime_test_root=$(mktemp -d /tmp/regime-completion.XXXXXX)
export CUSTOM_INDICATOR_DATA_DIR="$regime_test_root"
export HISTORICAL_REGIME_DATA_DIR="$regime_test_root"
export PYTHONPATH=.:backend
```
Python 固定为 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。

| 阶段 | pytest 范围 | 实际结果 |
| --- | --- | --- |
| M1 首次 | `test_regime_completion_quality.py` | 6 passed / 19.28 秒 |
| M2 首次 | stability + reliability_service | 14 passed，2 failed；小整数扰动取整导致缺少窗口变体，另一个是测试将 UTC 发布日与本地 today 比较 |
| M2 修复复跑 | 同上 | 16 passed / 31.86 秒 |
| M3 首次 | `test_regime_completion_bootstrap.py` | 5 passed / 19.32 秒 |
| 扩展集成首次 | `test_regime_reliability_*.py test_regime_completion_*.py` | 62 passed，5 failed；新增测试错误理解公式解析返回值与定义仓储结构，已修正测试调用 |
| 扩展集成复跑 | 同上 | 67 passed / 40.58 秒 |
| 广泛历史/TAA回归 | 见下方命令 | 782 passed，1 failed / 494.45 秒；失败来自并发新增的 prospective 模块测试，见下节 |
| 逐输出预览与 graph 扩展 | reliability/completion + v2/v2_p1 | 141 passed / 59.81 秒 |
| 临时计划回收、共享输入、最终图谱/TAA复跑 | 见最终定向命令 | **206 passed / 87.91 秒** |

以上集合有交叠，不能相加当唯一用例数。日志：`/tmp/regime-completion-m1.log`、
`/tmp/regime-completion-m2.log`、`/tmp/regime-completion-m2-rerun.log`、
`/tmp/regime-completion-m3.log`、`/tmp/regime-completion-focused.log`、
`/tmp/regime-completion-focused-rerun.log`。

最新回归命令：
```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_historical_regime*.py backend/tests/test_regime*.py \
  backend/tests/test_tactical_allocation*.py backend/tests/test_tactical_walk_forward.py \
  backend/tests/test_portfolio_regime_backtest.py backend/tests/test_research_series*.py \
  backend/tests/test_merrill_clock.py backend/tests/test_manual_historical_events.py -q
```
输出 `/tmp/regime-completion-regression.log`。

基准命令：
```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/tests/benchmark_regime_completion.py
```

## 尚未支持 / 验收不代表

前瞻部署证据、不可变预注册及历史 reference vintage 不在本轮已完成范围；
Student-t HMM、BOCPD、PELT 和自动跨频率映射未新增。统计门槛通过也不是投资有效性。
仅后端测试不能证明浏览器交互或前端集成验收。路由登记按本任务约束不修改，
新文件覆盖待父任务统一处理，不通过暂存/提交消除未跟踪事实。


## 最终定向命令与集成边界

```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest   backend/tests/test_regime_reliability_*.py backend/tests/test_regime_completion_*.py   backend/tests/test_historical_regime_v2.py backend/tests/test_historical_regime_v2_p1.py   backend/tests/test_historical_regime_taa.py backend/tests/test_tactical_allocation_service.py   backend/tests/test_tactical_allocation_bridge.py -q -o faulthandler_timeout=120
```
结果 **206 passed**，日志 `/tmp/regime-completion-owned-final.log`。包含完整窗口/种子变体、
同轴和缺失 oracle、配对 bootstrap 分位数、预算/readonly/stride、PID 预热、所有新模板
公式往返及每个节点每个输出的真实预览、共享临时计划重叠使用与显式 prepare 保留、
预览无业务文件、版本/数据篡改、旧报告与 TAA 门禁。

补充测试调试记录：新增“不同参数必须有不同结构 hash”断言失败后，核对真实契约发现
`graph_hash` 只标识结构，参数改变可以合法复用同一个计划。已改为检查每次执行的
preparation hash 匹配，且实际不同定义确实执行；变体返回额外 `definition_hash` 和差异。
未把结构 hash 改成新语义，也未为了测试强制重复编译。该补充测试已包含最终 206 项。

并发父任务新增了 `reliability/prospective.py`、`prospective_kernels.py` 和
`test_regime_prospective.py`。它们不是本执行者实现的部署能力：

- 广泛回归 782 passed / 1 failed：latent fixture 校准尚未 fitted。
- 后续集成复跑 152 passed / 10 failed：其中 1 项为上述已修复的结构 hash 测试断言，
  另 9 项为当时的 prospective capture 在源文件增长后触发 `SOURCE_FILE_CHECKSUM_MISMATCH`。
  调用为 `_candidate` → `load_definition` → `_formal_source_gate`。
- 保留完整性门禁，没有删除 checksum 验证；父任务独立整合冻结注册与前向输入。
  不能把本轮研究诊断完成等同于前瞻部署已经通过。
- 父任务继续修订后，最后独立复跑 `test_regime_prospective.py`：**21 passed / 49.68 秒**，
  日志 `/tmp/regime-completion-prospective-check.log`。这是合成时钟/数据的技术测试；
  不代表任何当前模型已获得真实前瞻投资证据，也不把最初广泛回归改写为全绿。

API 文件已复读并同步真实字段：结构/定义 hash 区别、整数版本、概率证据单位 nats、
小整数窗口步长、有效样本/完整块/周期、null/reason、临时计划生命周期。
Python py_compile 和 git diff --check 通过。覆盖审计记录
`/tmp/regime-completion-routing.json`：源文件被现有模块目录覆盖，3 个新测试文件尚未
显式登记；按用户约束未编辑路由 JSON/AGENTS，交给父任务统一维护。
