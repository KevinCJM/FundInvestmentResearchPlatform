# SAA / TAA 整改复核记录（2026-09-12）

**结论：原 7 项整改中，6 项可以关闭；第 6 项“真实精炼”已接入实际 NJIT 计算，但仍有 2 个已复现的 P2 问题，不能按“全部修复”验收。有效前沿按钮、离散点图和默认展开已在真实浏览器中确认恢复。**

本次按[逐项处理记录](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/docs/research/saa-taa-review-response-2026-09-12.md)复核，并对照[上一轮审核](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/saa-taa-review-2026-09-12.md)。这是本地工作区复核，结论为 **BLOCKED：精炼部分仍需修复**；没有修改业务源码，也没有执行提交、推送或合并。

**仍需修复的问题**

**R1 · P2：量化后的再次修复改变了比较起点，可能把更差的权重标成“未恶化”。**

位置：[optimizer.py:629](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:629)，结果替换发生在[optimizer.py:1104](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1104)。

进入精炼的原候选已经通过随机探索的约束修复。新内核再次调用 `repair_weights_kernel`，随后才记录 `before_score`。开启权重量化且存在单项/组约束时，再次量化、归一化和组边界修复并不保证保持原权重。后续搜索仅相对于这个被改过的起点“不恶化”，不能证明相对于原候选不恶化。

实际复现：3 类资产、80 期合成收益、100 个随机候选、0.5% 量化、单项边界与组权重区间 `[43.1%, 76.3%]`，精炼上限 20 次。相同数据、参数、随机种子，仅切换精炼开关：

| 比较项 | 数值 |
| --- | ---: |
| 原候选真实夏普值 | 0.5503851491 |
| 精炼后真实夏普值 | 0.5498503477 |
| 精炼输出声称的 `before_score` | 0.5498002283 |
| 输出状态 | `converged`、`non_worsening=true` |

夏普值由独立 NumPy 参考公式重新计算，确实下降，不是展示舍入问题。原候选收益 6.325744%，精炼后为 6.316710%。界面会采用这个较差的新权重，处理记录中的“不恶化证据”因此不成立。

修复要求：在任何再次量化或修复前，记录原始可行候选的目标值，并保留原权重作为候选；只有新权重满足约束且相对原值不恶化时才替换。若必须改变起点，分别披露原始值、修复后值和最终值。补充“量化 + 非整齐边界 + 组约束”的回归，使用精炼关闭时的结果和独立公式作比较，不能只断言函数返回的布尔值。

证据：[确定性输入与运行脚本](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/refine_probe.py)、[原始结果](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/refine_probe.json)、[独立回归检查](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/test_review_regressions.py:24)。

**R2 · P2：精炼更新了特殊候选，却没有更新有效前沿，曲线保留已被淘汰的点。**

位置：[optimizer.py:1062](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1062)、[optimizer.py:1113](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1113)。

`scatter` 和 `frontier` 在精炼之前构造；精炼只替换 `max_sharpe`、`min_variance`、`max_return`，最后仍返回原前沿。这使同一份结果中的表格/标记与前沿曲线不一致。这里要求的是对**本次已计算出的可行点**保持非支配关系，不是要求局部算法证明全局最优。

实际复现：3 类资产、160 期合成收益、100 个候选，不开启量化。精炼前后 `scatter`、`frontier` 完全相同；返回的 14 个前沿点中，有 1 个已被精炼后的夏普候选同时在风险、收益两方面超过：

| 本次返回的点 | 年化风险 | 年化收益 |
| --- | ---: | ---: |
| 仍留在“有效前沿”的旧点 | 10.898052% | 33.091006% |
| 同次返回的精炼夏普候选 | 10.483893% | 33.450141% |

以上仅为复现用合成数据。后一方案风险更低、收益更高，前一方案不应继续作为当前候选集合的有效前沿。

修复要求：将通过验收的精炼点纳入本次候选集合，复用唯一的前沿筛选实现重新构造曲线及计数；可分别记录随机采样数量和精炼点数量，避免混淆。增加“任何返回的可行候选都不能支配返回前沿点”的回归。当前只精炼 3 个代表候选，仍不等同于恢复原来的整条前沿求解能力；界面和文档应清楚说明这一范围。

证据：[复现脚本](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/frontier_dominance_probe.py)、[原始结果](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/frontier_dominance_probe.json)、[独立回归检查](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/test_review_regressions.py:36)。

**原审核逐项复核**

| 原审核项 | 本次结论 | 当前代码与验证 |
| --- | --- | --- |
| 1. 前瞻主动风险未计算 | 可关闭 | `expected_active_risk_kernel` 实际计算权重差的协方差风险；同时核对本次上限与政策上限。预览与产品应用复用政策门禁，界面区分历史 TE 和前瞻 TE。公式参考、超限阻断和既有产品桥接回归通过。 |
| 2. 政策日与行情日混淆 | 可关闭 | `initialRequest` 分离 `as_of` 与 `end_date`，默认主动风险上限也继承政策。真实浏览器显示政策日/决策日 2026-09-12、行情截至 2026-09-11，政策门禁通过。 |
| 3. 单折不可行导致整个 walk-forward 失败 | 可关闭 | 固定假设允许返回不可行选择用于诊断，随后将该折标记 `blocked` 并继续；不偷换强度。多折、成熟期剔除、只读视图和固定假设阻断回归通过。 |
| 4. 全部资产共同缺日仍按 252 年化 | 可关闭 | 历史风险参考按 SSE 开放日核对完整日期轴，并拒绝非 252 年化。全资产共同缺一个交易日的回归已覆盖。当前仅认证 SSE 日频契约，其他市场/周月频仍需独立契约。 |
| 5. 按钮与图表不可发现 | 可关闭入口问题 | 政策页新增直接可见的历史实验链接；历史页有“生成可配置空间与有效前沿”按钮。桌面与手机端实际修改采样数、量化、精炼配置，真实 API 返回非空点集，ECharts 默认显示；图表计算一致性问题见 R2。 |
| 6. SLSQP 名称与实现不一致 | 部分完成，不能关闭 | 旧假 SLSQP 与附加随机轮次已删除；旧字段显式拒绝，新的受约束局部搜索确实进入固定签名 NJIT。但 R1、R2 尚未解决。 |
| 7. 历史实验加载/切换跳错页 | 可关闭 | 历史页使用独立 `allocationLabPath`；策略政策仍使用政策路由。实际路由组件测试覆盖切换，浏览器覆盖空入口选择后留在 `allocation-lab`。 |

补充问题：海外债券/商品优先于地域的分类修改及回归通过；追加新行情可通过冻结历史前缀校验，历史修订仍被拒绝。旧基线缺少前缀元数据时保留严格行为，没有伪造兼容证据。

**流程与算法边界判断**

保留“投资目标 → 前瞻 CMA → SAA 政策 → TAA → 产品配置”的分层是合理的。本轮对前瞻主动风险、决策日期、日频证明和不可变政策的数据生命周期做了实际修复。历史前沿与前瞻 CMA 已在界面区分，能够减少将历史表现直接解释成长期预期的混淆。

同意处理记录对扩展范围的判断：跨币种/对冲、负债与现金流、BL/HRP/CVaR、更丰富信号和真实执行，需要新的数据、模型与业务契约，不应为了关闭本轮缺陷而表面补字段。该判断不改变原报告对这些能力尚未实现的说明。当前局部搜索是替代算法，必须先满足自己声明的比较与约束保证；采用 NJIT 不代替数值正确性验收。

**本次实际验证**

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 后端 17 个相关测试文件 | **357 passed**，161.00 秒，19 个警告 | [backend.log](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/backend.log) |
| 前端全量 Vitest | **123 个文件、911 passed**，27.55 秒 | [frontend.log](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/frontend.log) |
| Vite 生产构建 | **通过**，6.41 秒；仍有大包提示 | [build.log](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/build.log) |
| 隔离 API + 真实浏览器 | **4 passed**，49.5 秒；1440 与 390 宽度 | [browser-retry.log](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/browser-retry.log) |
| 本次新增的正确性复现检查 | **2 failed**，与 R1、R2 分别对应 | [检查代码](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/test_review_regressions.py)、[失败输出](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/review-regressions.log) |
| 工作区 diff 空白检查、Hermes 结构校验 | 通过 | [routing-validation.log](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/routing-validation.log) |

浏览器证据：[桌面前沿](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/browser/strategic-allocation-real--096f9-le-and-renders-real-ECharts-desktop-1440/historical-frontier-real-echarts.png)、[手机前沿](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/browser/strategic-allocation-real--096f9-le-and-renders-real-ECharts-mobile-390/historical-frontier-real-echarts.png)、[TAA 日期与政策门禁](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/browser/strategic-allocation-real--b7493-n-and-TAA-on-desktop-mobile-desktop-1440/taa-validation.png)。前沿图与 TAA 桌面截图另经人工视觉核对。浏览器首次因沙箱禁止绑定本地端口退出，获准启动隔离服务后完整复跑通过；没有复用生产服务。

后端范围：`test_strategic_allocation`、`test_tactical_walk_forward`、`test_strategy_research_dates`、`test_tactical_allocation_numeric/data/service/bridge`、`test_historical_regime_taa`、`test_portfolio_research`、`test_optimizer_strategy_numba`、`test_analytics_routes`、`test_auto_asset_class`、`test_research_input_checks`、`test_window_slice`、`test_rebalance_window`、`test_backtest_output`、`test_strategy_api`。未将处理文档中的全后端测试结果当成本次运行结果。

新增复现检查的复跑命令（现版应得到上述两条断言失败；修复后应通过）：

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest -p no:cacheprovider /Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/test_review_regressions.py -q
```

**版本、范围与未验证项**

- 分支：`codex/frontend-design-quality`；HEAD：`2329845b345665a9bacc95e36b814868a8e288ea`。审核对象是该 HEAD 加当前未提交修改，包含未跟踪的新 SAA 模块。
- 开始时记录 57 个修改/新增文件的 hash，结束时全部一致，HEAD 未变化：[基线](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/baseline.json)、[结束核验](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-recheck-evidence/final-source-verification.json)。本次只新增复核报告与仓库外诊断证据。
- 此次不是远端 PR 审核或合并门禁认证，没有核验最新远端 base、检查发布身份及远端保护状态。Hermes 结构通过不等于新模块覆盖完整；新增模块登记仍是处理文档已披露的待提交事项。
- 使用离线夹具与合成数据。未认证正式 PIT、完整真实产品池应用资格、真实投资业绩或交易执行。浏览器主链截止 TAA 预览，产品应用风险门禁的证据来自相关后端回归。
- 新增复现首次运行仅出现 pytest 缓存目录无写权限警告，两条数值断言均已执行并失败；这不是测试被跳过。

**整改完成条件：修复 R1、R2，补充独立数值与前沿一致性回归，并据此更新逐项处理记录。其余已通过的整改无需重新设计。**
