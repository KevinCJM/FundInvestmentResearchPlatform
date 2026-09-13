# SAA／TAA 第五轮复核：逐目标有效前沿

日期：2026-09-13
范围：当前未提交工作区；HEAD `2329845b345665a9bacc95e36b814868a8e288ea`。本轮复核 [整改设计](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/docs/research/frontier-grid-round4-design-2026-09-12.md) 和相对上一轮新增的网格求解、API、前端及测试，不是远端 PR 或全仓后端审核。

## 结论

**整条前沿的逐目标求解已实际恢复，20／200 的参数语义正确。新增 1 个 P2：奇异协方差下端点求解不能正常停止，导致本来可解的整条前沿无法生成。整体暂不通过。**

此前不恶化、共同 Pareto 集合和最终代表点重选的修复保持有效；原审核侧 4 条复现通过。新增问题不再是“三个代表点代替整条曲线”，而是新 QP 求解器的退化问题处理。

本次只新增本报告和仓库外审核证据，未修改业务代码、开发方测试、整改文档或四轮原始报告；未 commit／push。

## P2-1：重复资产使最低风险端点耗尽预算，整条网格没有开始

### 位置与影响

- [feasible_qp_kernel：正则化方向矩阵](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/qp_numba.py:65)
- [feasible_qp_kernel：以方向长度作为进入 KKT 成功检查的前提](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/qp_numba.py:101)
- [solve_frontier_grid_kernel：两个端点成功后才进入目标循环](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:938)

当两个资产具有相同收益序列时，协方差矩阵是半正定而非正定。这是一个合法且可解的均值—方差问题：可以在两个相同资产之间重新分配权重，风险和收益均不改变。

当前方向方程加了很小的对角正则，但成功检查必须先满足 `max(abs(direction)) <= 1e-9`。在平坦方向上，数值方向未能满足这一条件，即使返回的驻点残差已接近机器精度，也会持续迭代。最低风险端点最终返回 `max_iterations`，随后所有目标都成为 `range_unresolved`，曲线为空。

这不是正常的“不同目标得到重复解”：系统连目标求解都没有开始。单纯增加预算不能解决本次复现。

进一步检查同一解析模型的 KKT 方程：1000 次后权重为约 `[0.35020831, 0.2, 0.44979169]`，原目标驻点残差仅 `2.78e-17`，下一步却仍沿 A／C 之间的平坦方向移动约 `1.39e-7`，大于 `1e-9` 步长门槛。B 的权重及实际经济组合已经正确，算法仍不停止。[KKT 数值证据](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/singular-kkt-detail.json)

### 确定性解析复现

使用固定随机种子 84 构造 80 期收益，正交化后得到：

| 资产 | 年化平均收益 | 年化波动率 | 关系 |
| --- | ---: | ---: | --- |
| A | 3% | 10% | 与 B 不相关 |
| B | 9% | 20% | 与 A 不相关 |
| C | 3% | 10% | 收益序列逐项等于 A |

只有资金约束和每项 `0 <= w <= 1`，没有复杂分组、取整或额外限制。

解析解：最低风险时 `wA + wC = 0.8`、`wB = 0.2`；年收益 **4.2%**、年波动 **8.94427191%**。前沿应与只有 A、B 时完全相同，最高收益端点为 100% B。

独立运行结果：

| 输入 | 目标数 | 单点预算 | 成功目标 | 最低风险端点 |
| --- | ---: | ---: | ---: | --- |
| A、B | 20 | 1000 | 20 | converged |
| A、B、C=A | 20 | 1000 | **0** | **max_iterations** |
| A、B | 200 | 1000 | 200 | converged |
| A、B、C=A | 200 | 1000 | **0** | **max_iterations** |

三资产复现返回：

```text
minimum_risk.return = 0.04200000000000001
minimum_risk.optimality_residual = 5.551115123125783e-17
minimum_risk.iterations = 1000
minimum_risk.status = max_iterations

requested_points = 20
attempted_points = 0
solver_calls = 0
successful_points = 0
unattempted_points = 20
```

另以种子 0–19 构造普通两资产随机收益，再复制第一列成为第三资产；分别使用 300／1000 预算，**40／40 次均未生成网格曲线**。这说明问题不限于正交化解析夹具。

### 要求修复与验收

1. 让主动集 QP 正确处理半正定 Hessian 和平坦方向。以可靠的原问题原始／对偶可行性、驻点和互补条件判断收敛，不能仅依赖病态方向方程产生的步长。
2. 可采用稳定的约束子空间处理等数值方法；不得静默改变经济风险矩阵，也不能直接删掉重复资产并丢失它们各自的单项或分组约束。
3. 将下面两条审核复现纳入回归；20／200 均应成功并与两资产解析前沿一致。补充近共线、重复资产的独立约束、奇异协方差及不可行约束检查，保持诚实的失败状态。

证据：

- [独立回归测试：2 条失败](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/test_singular_frontier.py)
- [pytest 原始失败日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/singular-regression.log)
- [完整响应与两资产参考点](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/singular-detail.json)
- [40 组重复资产探针](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/degenerate_probe.py)、[结果](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/degenerate_probe.json)

复现命令，在项目根目录执行：

```bash
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest -p no:cacheprovider '/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/test_singular_frontier.py' -q
```

## 已确认恢复的功能

| 核验项 | 结论与代码证据 |
| --- | --- |
| 目标数量独立配置 | 默认 20、范围 2–200；单点预算默认 300、范围 1–1000。草稿、请求、后端字段独立。[前端控件](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/components/frontier-grid/FrontierGrid.tsx:4)、[API 模型](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/services/analytics_routes.py:195) |
| 确实逐目标优化 | 先求最低风险与最高收益端点，再等分收益目标，逐个最小化风险；不是增加随机散点。[网格内核](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:915) |
| 约束与口径 | 各目标共享资金、单资产、分组约束，并施加收益下限。连续网格与散点取整需显式确认。[模型构造](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:770)、[输入校验](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1417) |
| 计数和失败留痕 | 请求点数、尝试数、实际求解调用数、成功、失败、未开始及重复解分别返回。[响应封装](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1565) |
| 曲线坐标真实 | 实际风险／收益按目标顺序连线，失败或未保留在最终 Pareto 集合的位置断开；不使用插值补造优化结果。[图表](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1163) |
| 共同候选与代表点 | 成功网格解加入最终候选，统一重建 Pareto 前沿、最大夏普、最小风险、最大收益；来源可为 grid。[候选重建](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1534) |
| 权重采用 | 明细仅允许成功且有候选索引的点触发采用；共用现有采用逻辑，保留连续权重精度。[明细](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/components/frontier-grid/FrontierGrid.tsx:95)、[采用处理](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:717) |
| NJIT | 本次测试进程报告版本 2.4.0、22/22、warmed=true、object_mode=0、python_fallback=0；固定签名和只读视图专项通过。不是全部生产 worker 的认证。 |

因此，第四轮指出的**功能缺口已经补齐**。后续验收应处理上面的新数值缺陷，不再把网格密度改回局部迭代次数。

## 算法判断与实际边界

采用收益目标下的最小风险模型，符合此次平滑整条前沿的意图。均值—方差模型中，固定收益最小化风险可以得到相应风险收益前沿。[CVX Group 官方说明](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)

实现名称也已如实区分：标准差／年化波动／EWM 波动走主动集 QP；其余风险走有限差分 BFGS-SQP，并未冒称 SciPy SLSQP。前沿目标数与单次优化迭代预算应独立，后者在 SciPy 官方接口中由 maxiter 表示。[SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

非二次风险的稳定性仍有限。本轮 12 组普通三资产收益、每组 20 个目标的独立检查中：

| 风险口径 | 完全没有成功目标的组数 | 成功目标／请求目标 |
| --- | ---: | ---: |
| 年化波动 | 0／12 | 240／240 |
| EWM 波动 | 0／12 | 240／240 |
| VaR | 4／12 | 144／240 |
| ES | 5／12 | 110／240 |
| 下行波动 | 4／12 | 135／240 |
| 最大回撤 | 5／12 | 127／240 |

这些是固定合成样本的工程观察，不是市场成功率估计。当前接口和页面保留失败状态，故本轮**不把这些诚实失败另列为缺陷**；但不能把“支持该风险”理解为稳定生成完整平滑曲线，更不能将有限差分小步长等同于全局最优证明。证据：[探针](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/probe.py)、[结果](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/probe.json)。

## 本轮实际验证

| 范围 | 独立复核结果 |
| --- | --- |
| 18 个相关后端文件 | **392 passed**，19 条既有弃用／日期解析警告 |
| 前端全量 | **124 文件，918 passed** |
| 桌面 1440／手机 390 浏览器 | **6 passed**，包含真实 API 的 20／200 网格、曲线变化及预算耗尽状态 |
| 前四轮原审核侧缺陷复现 | **4 passed**，测试文件未改 |
| 本轮新增奇异协方差解析复现 | **2 failed**，即本报告 P2 |
| TypeScript、生产构建 | 通过；构建仍有既有大 chunk 提示 |
| Design check、git diff --check | 通过，无新增静态设计回归 |
| Hermes 结构校验 | 通过；不代表新增未跟踪文件已完成长期路由登记 |

浏览器用隔离 API、离线合成 Parquet，未写正式投资数据。现有 6 条浏览器用例未覆盖“从逐目标明细采用后再完成策略回测”；采用处理另有源码和组件测试证据，不扩大浏览器验收范围。手机截图中曲线可见，但左轴文字仍有裁切，不能据 6 条用例通过宣称所有图表排版都已完备。

原审核侧测试首次单独启动时，因没有加载项目测试目录的路径设置，Numba 缓存引用 `cal_indicators` 导致收集失败；补齐 `PYTHONPATH=.:backend` 后原测试 4 项通过。保留[首次日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/original-regressions.log)和[正确运行日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/original-regressions-with-project-path.log)，未修改测试或以收集失败代替数值结论。

详细日志：[后端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/backend.log)、[前端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/frontend.log)、[浏览器](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/browser.log)、[构建](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/build.log)、[设计检查](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/design.log)、[路由结构](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/routing.log)。

浏览器图像：[桌面 200 点](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/browser/strategic-allocation-real--6c835-ntier-and-preserve-failures-desktop-1440/target-grid-200.png)、[手机 200 点](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/browser/strategic-allocation-real--6c835-ntier-and-preserve-failures-mobile-390/target-grid-200.png)。

## 证据与工作区边界

按 AGENTS → repo_map → task_routes → pitfalls 缩小到 R30／R60，使用 CodeGraph 定位链路，并补读其未返回的新 QP 内核。对本轮开始时记录的 **67 个已有改动／未跟踪文件**做哈希复核，全部未变化；tracked diff 与开始时一致。[工作区校验](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round5-recheck-evidence/source-verification.json)

未重跑全仓后端、真实市场数据和生产部署；未新增投资有效性结论。开发方已披露的新文件长期路由登记仍需在纳入 Git 时完成，本轮没有将结构检查通过当作完整覆盖。
