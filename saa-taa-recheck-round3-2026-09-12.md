# SAA / TAA 第三轮复核记录（2026-09-12）

## 结论

**上一轮 2 个 P2 均可关闭；补查仍发现 1 个 P2，暂不建议将整项精炼整改标为全部通过。**

原始两条失败复现未经修改，在当前代码上均通过。新增问题是：精炼后的前沿已经正确更新，但三个特殊候选没有从最终候选集合重新选取，可能将并非集合内最优的点标为“最大夏普”或“最大收益”。

本轮仅审核并保存报告，没有修改业务代码，也没有 commit / push。原始 [首次审核](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/saa-taa-review-2026-09-12.md) 和 [第二轮复核](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/saa-taa-recheck-2026-09-12.md) 均保留。

## 1. 尚需处理：P2 — 特殊候选未从最终集合重新选取

**位置：** [optimizer.py:1171](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1171)，以及 [最终返回:1182](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1182)。

当前每个精炼分支只更新自己的 `special_points[key]`。三个精炼点加入 `combined_*` 后，只重新计算 Pareto 前沿，没有重新比较整组候选的最大夏普、最小风险、最大收益。因此，一个分支产生的新点可能优于另一个分支对外返回的代表点。

接口原样返回这些特殊点；图例仍明确显示“最大夏普率”“最大收益”，见 [analytics_routes.py:323](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/services/analytics_routes.py:323) 和 [ClassAllocation.tsx:1144](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1144)。用户也会通过候选表采用对应权重。

### 确定性复现 A：默认随机种子下，“最大夏普”低于已返回的候选

参数：3 个资产、120 期收益、100 个采样点、年化收益 / 年化波动率、无风险利率 0、不量化、精炼 1 次。1 次位于当前接口允许的 1–200 范围内。**不传 optimizer seed，使用接口调用所用的默认值 42。**

```python
rng = np.random.default_rng(17)
values = rng.normal(0, .01, (120, 3)) + rng.uniform(-.001, .002, 3)
result = calculate_efficient_frontier_exploration(
    pd.DataFrame(values, columns=["a", "b", "c"]),
    {"metric": "annual", "days": 252},
    {"metric": "annual_vol", "days": 252},
    rounds=[{"samples": 100, "step": 1., "buckets": 40}],
    use_local_refine=True,
    refine_iterations=1,
)
```

| 返回点 | 权重 | 独立重算 Sharpe |
|---|---|---:|
| `max_sharpe` | [0.332695692, 0, 0.667304308] | 0.924119167832816 |
| 同次返回的 `max_return`，也在 scatter 中 | [0.037761592, 0, 0.962238408] | **0.962150364509974** |

用原始收益矩阵和返回权重独立计算 `mean(portfolio) * 252 / (std(portfolio, ddof=1) * sqrt(252))`，确认差异不是展示舍入造成的。

这项要求是“从已经返回的有限候选集合选出最优代表点”，不涉及连续空间的全局最优保证；增加局部迭代次数也不能代替最终集合的统一选择。

### 确定性复现 B：“最大收益”低于另一返回候选

同样的收益生成公式改为数据随机种子 1，optimizer seed 19，精炼 1 次：

- 返回 `max_return` 的年化收益：**-7.2292377589%**。
- 同次返回集合中已有候选的年化收益：**-6.7907924516%**。
- 差异约 **43.84 bp**。

两个场景已固化为审核侧独立测试；当前结果为 **2 failed**，均是预期行为断言失败，不是环境或导入错误：

- [测试代码](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/test_representative_consistency.py)
- [失败日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/representative-regression.log)
- [默认 seed 数值证据](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/default_seed_probe.json)

### 修复要求

1. 所有被接受的精炼点加入集合后，从同一个最终集合重新选择三个特殊候选，再与 scatter / frontier 一起返回。
2. 选择过程使用唯一的固定签名 NJIT 实现，沿用当前无风险利率、零风险、有限值和并列值规则。
3. `refinement.items` 继续记录各精炼分支相对其原始候选的改善；区分“分支精炼结果”和“最终集合选出的代表点”，避免审计字段对应错权重。
4. 回归覆盖：精炼关闭、1 次、常规迭代、量化与 group constraints；断言三个特殊点分别达到整个返回集合的对应极值，同时保留原始两条缺陷复现及全候选支配检查。

## 2. 上一轮两项关闭证据

| 原问题 | 本轮核验 | 判定 |
|---|---|---|
| R1：二次 repair 改变比较基准，真实 Sharpe 下降却标记不恶化 | 原候选直接作为基准；仅严格改善才应用。原始 seed 1461 / optimizer seed 19 复现：前后 Sharpe 均为 **0.5503851490687908**，before / after score 相同，`non_worsening=true`、`applied=false` | **关闭** |
| R2：精炼后 frontier 未同步 | 随机探索与最终集合复用 `pareto_frontier_indices_kernel`。原始复现返回采样 100、精炼新增 3、合计 103、前沿 17；对全部 103 个返回候选检查，被支配的前沿点数为 **0** | **关闭** |

[原始两条复现本轮日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/original-regressions.log) · [精确数值与计数](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/closure-metrics.json)。

候选数量已在 API 和页面区分为采样、新增精炼、总候选、前沿点数；生成入口、参数传递和真实 ECharts canvas 均通过浏览器测试。

## 3. 本轮实际验证

| 检查 | 结果 |
|---|---|
| 原始审核侧两条复现，未改测试 | **2 passed**，8.26 秒 |
| 与第二轮相同范围的后端 17 个测试文件 | **359 passed**，139.22 秒 |
| 前端全量 | **123 文件 / 911 passed**，21.81 秒 |
| 隔离真实 API + Chrome，1440 桌面 / 390 手机 | **4 passed**，46.7 秒 |
| TypeScript `npx tsc --noEmit` | 通过 |
| Production build | 通过，9.21 秒 |
| Design check | 无新增回归 |
| AI routing validation | 通过 |
| `git diff --check` | 通过 |
| 新 Pareto 内核与独立有限集合参考比较 | **52 组通过**，含空集、单点、相同风险、重复点、负收益 |
| 新增特殊候选一致性复现 | **2 failed**，对应上文同一个 P2 |

日志：[后端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/backend.log) · [前端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/frontend.log) · [浏览器](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/browser.log) · [构建](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/build.log) · [TypeScript](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/typescript.log) · [Design](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/design.log) · [Routing](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/routing-validation.log)。

通过的回归与新增失败复现应同时保留，不能用现有全绿测试替代这个缺失断言。后端有 19 条依赖弃用 / 日期推断警告；构建仍有大 chunk 警告，Design 仍有既有超目标项，本轮未将这些列为新缺陷。

运行核验显示 optimizer 版本 **2.2.0**、固定签名内核 **13/13**、`nopython=true`、`object_mode=0`、`python_fallback=0`；预热后的测试调用没有新增编译签名。证据：[独立探针代码](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/independent_probe.py) 与 [运行审计](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/independent_probe.json)。这些是本轮测试进程证据，不等同于所有生产 worker 的验收。

## 4. 范围与代码链路

- 基于 AGENTS、项目路由文档、上轮报告、整改记录及当前 CodeGraph 调用链复核。
- 当前分支：`codex/frontend-design-quality`；HEAD：`2329845b345665a9bacc95e36b814868a8e288ea`，包含尚未提交的工作区改动。
- 本轮相较上轮主要变化在 optimizer、analytics API、ClassAllocation、对应测试和整改 / 设计 / 验收文档，共 9 个文件；继续对 SAA / TAA 原同范围执行后端回归。
- 关键链路：`ClassAllocation → POST /api/efficient-frontier → calculate_efficient_frontier_exploration → explore_portfolios_kernel → refine_special_candidates_kernel → pareto_frontier_indices_kernel → API / 图表 / 候选采用`。
- 浏览器使用临时夹具数据，覆盖生成图像及 Goal → CMA → Policy → TAA；不代表真实投资数据、生产部署或投资有效性验收。本轮没有重做前次行业研究。
- [工作区校验](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-final-recheck-evidence/source-verification.json)：复核开始时记录的 58 个文件哈希未改变，HEAD 与已有 tracked diff 未改变；原始两份报告未被覆盖。
