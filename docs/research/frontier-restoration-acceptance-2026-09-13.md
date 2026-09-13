# 可配置空间与有效前沿恢复验收

本轮先编写 [详细设计](frontier-restoration-design-2026-09-13.md)，再改造前后端。历史对照见根目录 `saa-frontier-history-comparison-2026-09-13.md`。原有审核记录未改写；本记录只描述本轮新增改造与本次实际重跑结果。

## 1. 功能对照与交付

| 能力 | 本轮之前的问题 | 本轮结果 |
| --- | --- | --- |
| 多轮约束随机游走 | 从最近 `buckets × 4` 个候选选父点，没有按收益分桶择优 | 首轮保留全部随机可行点；后续轮按收益分桶选最低风险点，并冻结下一轮父候选集合 |
| 单项／联合上下限 | API 会忽略未知资产、覆盖重复组、静默截断部分边界 | 明确校验；约束在采样、精炼、连续网格与离散采用中执行，支持重叠组 |
| 0.1%／0.2%／0.5% 精度 | 取整后再修复、归一化会破坏精度 | 整数单位求解同时满足总量、单项与联合边界；无解与搜索预算耗尽分开报告 |
| 收益、风险与相关参数 | 核心配置折叠在生成模块外；收益 EWM 窗口没有参与计算 | 同一模块直接配置；保留五个收益选项、七个风险选项及普通／对数类型，收益 EWM 窗口统一进入全部计算阶段 |
| 整条前沿 | 已恢复目标网格，但量化时可能采用连续权重 | 保留 20／200 个收益目标与独立单点迭代预算；连续参考坐标不改写，采用时使用另行修复且重算指标的指定精度组合 |
| 三个代表点 | 前几轮修复的不可退化与候选一致性需保持 | 保留独立局部精炼；最终 Pareto 与代表点从同一个可采用集合重选 |
| 散点查看 | 当前全部候选与历史筛选保留点口径不同 | 同时保留全部可行候选与首轮／各轮分桶保留点两种视图，新增轮次统计与可保存随机种子 |
| 前端流程 | 参数和结果分散 | “可配置空间与有效前沿”统一包含方案、区间、指标、约束、探索、精度、网格、生成、图表、逐点状态和权重采用；后续策略与回测保留 |

第 0 轮直接随机采样，不使用扰动步长；前端明确禁用该无效输入。后续轮数、样本点、步长、分桶仍可编辑、增删。`samples` 是尝试预算；实际可行点、失败点和保留点分别列示。变更任一计算输入会清除旧结果，随机种子与其余参数随研究草稿保存。

## 2. 数值链路与边界

`ClassAllocation → /api/efficient-frontier → calculate_efficient_frontier_exploration → explore_portfolios_kernel → repair_weights_kernel → return_bucket_indices_kernel`。

量化时 `repair_weights_kernel` 委托唯一 `integer_weights_kernel`。它先作有界平衡取整，再以剩余总量／组上下界剪枝执行有界整数搜索，最多访问 50,000 个节点。不声称最近点或整数全局最优。没有在整数结果上继续归一化。

成功连续网格点通过 `project_grid_weights_kernel` 产生 `adoption`，保留独立的权重、实际收益／风险、可行状态与 `target_met`。连续解重复数与指定精度组合重复数分开统计。离散修复可能达不到原收益目标，界面如实显示，采用组合不会冒充原连续最优解。

版本为 `3.0.0`，固定签名内核 `25/25`。独立选种与整数可行化接受只读／非连续数组视图，关闭请求期新增编译；整数工作栈、候选数组和结果封装需要独占内存。测试验证共享输入不被改写，不宣称全流程零分配。

收益 EWM 窗口：末尾 N 个观察期使用既有递推规则；0 为全样本，N 超过样本数时使用全部可用样本。风险窗口独立。其他收益公式及无风险利率口径未在本轮改写。

## 3. 本次验证

| 验证 | 结果与范围 |
| --- | --- |
| 后端相关回归 | **152 passed**，含新增分桶／整数约束／EWM 窗口、Optimizer、20／200 点网格、窗口与再平衡、回测、策略 API、分析 API、研究输入、SAA 与 TAA 数据测试 |
| 前端全量 | **946 passed / 126 files**；最终窄屏图例调整后，受影响页面与网格组件 **23 passed** |
| 完整真实浏览器 | **8 passed**，桌面与手机，包含真实隔离 API 的前沿、网格、投资目标、SAA 与 TAA 流程；未操作正式研究数据 |
| 响应式与文字 | 320／768／1440 宽度，等待图表完成 resize 后无整页横向溢出；代表性浅底说明文字实测对比度 ≥ 4.5；表格在各自容器内横向滚动 |
| 数值证据 | 小规模整数空间穷举对照、重叠组、无解／预算区分、非有限值、精度与幂等性、父候选冻结、指标切换改变后续探索、同种子复现、全部候选与前沿／代表点一致性 |
| 既有网格 | 20／200 点解析解及独立求解器对照、失败断点、局部风险状态与只读非连续输入继续通过 |
| 静态与构建 | TypeScript、生产构建、设计检查、语言检查通过；构建保留已有大包体积提示，测试保留既有依赖弃用及 React act 提示 |

后端重跑命令：

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_frontier_sampling.py backend/tests/test_optimizer_strategy_numba.py backend/tests/test_frontier_grid.py backend/tests/test_window_slice.py backend/tests/test_rebalance_window.py backend/tests/test_backtest_output.py backend/tests/test_strategy_api.py backend/tests/test_analytics_routes.py backend/tests/test_research_input_checks.py backend/tests/test_strategic_allocation.py backend/tests/test_tactical_allocation_data.py -q
```

前端：在 `frontend/` 运行 `npm run test -- --run`、`npx tsc --noEmit`、`npm run build`、`npm run design:check`、`npm run i18n:check`、`npx playwright test --config playwright.strategic.config.ts`。

浏览器证据位于 `.pytest_cache/top-down-allocation/browser/`，包含各宽度完整配置模块截图与 20／200 点真实画布。最初 320px 断言在 ECharts 尚未完成 resize 时失败；改为等待最终布局后通过，未屏蔽尺寸检查。人工查看原始截图后另优化移动端指标排版、表格最小宽度、滚动图例与坐标留白。

## 4. 性能与内存抽测

252 个收益观察期、3 类资产，默认六轮共 20,000 个尝试；单项 `[5%,80%]`，组 `(0,1)` 合计 `[30%,85%]`，固定种子 137 生成输入，探索种子 42。进程内预热后每种情况重复 3 次，记录函数返回全部候选的中位耗时：

| 实现 | 不量化 | 0.1% | 0.5% |
| --- | ---: | ---: | ---: |
| 本轮开始前的本地代码快照 | 0.0607 s | 0.0610 s | 0.1310 s |
| 恢复后 | 0.0656 s | 0.0917 s | 0.0915 s |

每次返回 20,000 个候选。旧量化路径不满足严格精度，不能将它与新路径视为完全等价性能比较；样本次数较少且环境存在其他编译任务，不据此宣称加速比例。包含两套代码启动编译的整个基准进程峰值 RSS 为 370,098,176 字节，不是单次请求峰值或零拷贝证明。

以下脚本可重跑当前实现（启动编译在计时之前）：

```python
import time
import numpy as np
import pandas as pd
import optimizer
optimizer.warm_optimizer_numba_kernels()
f = pd.DataFrame(np.random.default_rng(137).normal([.0002,.0005,.0009], [.006,.012,.02], (252,3)))
rounds = [dict(samples=n, step=s, buckets=b) for n,s,b in [(1000,.5,10),(2000,.4,10),(3000,.3,20),(4000,.2,30),(5000,.15,40),(5000,.1,50)]]
for precision in [None, .001, .005]:
    times = []
    for _ in range(3):
        start = time.perf_counter()
        r = optimizer.calculate_efficient_frontier_exploration(f, {'metric':'annual_mean'}, {'metric':'annual_vol'}, rounds=rounds, single_limits=[(.05,.8)]*3, group_limits={(0,1):(.3,.85)}, quantize_step=precision)
        times.append(time.perf_counter()-start)
    print(precision, np.median(times), r['accepted_candidates'])
```

运行该脚本需设置 `PYTHONPATH=.:backend` 并使用上述 Python 环境。

## 5. 路由与本地运行

CodeGraph 已同步；模块与测试入口登记在 `docs/repo_map.json`，新增行为约束登记在 `docs/pitfalls.json` 的 P21。本轮 13 个变更路径覆盖检查通过，无未覆盖项；`validate_ai_routing.py --skip-reproducibility` 结构校验通过。完整校验目前仅因 9 个引用文件尚未纳入 Git 而未通过（含此前已有的未追踪网格文件），没有放宽追踪规则或把新源码标记为豁免产物。纳入 Git 时应重跑完整校验。`git diff --check` 通过。

本地前后端已重启，启动耗时 263 秒，Numba 与全部 worker 预热、页面及 API 代理健康检查通过。最终手机排版调整后另重跑前沿相关真实浏览器 **4 passed**。后端 `8000` 与前端代理 `5173` 均返回 `ok=true`、预热完成、workers ready；运行中的 Optimizer 为 `3.0.0 / 25/25 / python_fallback=0`。可访问 `http://127.0.0.1:5173/pre-investment/saa/allocation-lab` 查看统一模块。本轮未 commit／push，未修改旧审核证据或研究历史快照。
