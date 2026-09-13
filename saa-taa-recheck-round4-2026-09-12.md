# SAA / TAA 第四轮复核与前沿功能澄清（2026-09-12）

## 结论

**第三轮 P2 已修复，可以关闭；但原始“可配置网格点数、逐点优化形成密集有效前沿”的功能仍未恢复。**

本轮将原始功能恢复列为 **F1 / P2，待整改**。此前对局部精炼的正确性验证，不代表它与原来的整条前沿求解功能等价。前几轮审核漏查了“精炼点数”的参数语义和整条曲线能力，本报告补正这一验收范围。

已保留[第三轮原始报告](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/saa-taa-recheck-round3-2026-09-12.md)及前两份报告；本轮只新增复核记录和仓库外测试证据，没有修改业务代码、commit 或 push。

## 1. 用户确认的原始需求

沿收益轴或风险轴设置目标网格，例如 **20 个或 200 个点**，逐点执行受约束优化，再用这些优化结果绘制更密集的有效前沿。

- 收益网格：对每个目标收益，求满足要求的最小风险组合。
- 风险网格：对每个目标风险上限，求满足要求的最大收益组合。
- 本次验收接受这两种方式服务于同一个前沿加密目标，不把等分轴的选择作为缺陷。
- **网格点数控制整条曲线的取点密度；最大迭代次数控制单个优化问题的求解预算。两个参数必须分开。**

SLSQP 是受约束数值求解器，曲线上的网格点由外层逐点调用产生；网格点数与求解器的 `maxiter` 是不同概念。[SciPy 官方说明](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

## 2. F1 / P2：20–200 个“精炼点”被替换成三个代表点的迭代次数

### 当前事实

| 能力 | 当前实现 |
|---|---|
| 可配置随机采样点数及散点图 | 保留 |
| 三个代表点的受约束局部精炼 | 已实现，数值缺陷已通过复核 |
| 从最终集合重建 Pareto 前沿并重选代表点 | 已实现 |
| 按指定数量生成收益 / 风险目标网格 | **未实现** |
| 对网格上的每个目标逐点求最优组合 | **未实现** |
| SLSQP 实际调用 | **没有，旧请求字段还会被显式拒绝** |
| 当前填写 200 的含义 | 最多执行 200 次局部搜索迭代，仍只处理 3 个初始代表点 |

当前[页面参数](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1071)已改名为“局部精炼最大迭代次数”；[请求构造](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:761)将原变量 `refineCount` 发送为 `refine.iterations`。[API](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/services/analytics_routes.py:282)拒绝 `use_slsqp`，改为读取局部精炼开关及迭代次数。

[当前计算入口](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1116)的流程是：

```text
随机采样
→ 对最大夏普 / 最小风险 / 最大收益三个初始候选做局部搜索
→ 最多加入三个严格改善的结果
→ 在最终有限集合中筛选 Pareto 前沿、重选代表点
```

流程没有收益 / 风险目标网格，也没有按目标点数逐点优化。[前沿图层](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1138)目前仍使用 scatter 展示。因此，提高随机样本数或局部迭代次数，均不能作为“生成指定数量的优化前沿点”的验收依据。

### 历史代码证据：用户记忆正确

核查提交 `7fdcc49afeabd338d79c73cf0383b4de62f39b92`，即删除实际 SLSQP 路径的 `88e4e75143163759de7084f5a0f935a0f6fad5b0` 的第一父提交：

1. 当时前端 `refineCount` 默认值为 **20**，输入名称为“精炼数量”，发送 `refine.count`。输入没有阻止填写 200。
2. 当时实际接入页面开关的 `calculate_efficient_frontier_exploration` 使用：
   ```python
   grid = np.linspace(rmin, rmax, int(max(2, refine_count)))
   for rt in grid:
       # 在 risk(w) <= rt 下，最大化收益
       res = minimize(neg_ret, w0, method='SLSQP', ...)
   ```
   因此填写 20 / 200，确实控制 **20 / 200 个目标网格点及对应的求解尝试**，不是每个点的迭代次数。该路径当时单独固定了 `maxiter=300`。
3. 同一历史文件还存在 `calculate_efficient_frontier`，使用 **100 个等分目标收益率**，逐个通过 SLSQP 在收益等式约束下最小化波动率；但该历史 API 入口调用的是第 2 项的风险网格版本。

两种历史实现都印证“网格加密整条前沿”的目的。点求解失败时，旧代码可能不返回该点，因此指定数量并不保证等量成功点。历史存在这些函数，也不等于所有旧实现当时都在页面调用链内。

可逐行复核：[历史前后端代码摘录](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/historical-frontier-evidence.txt)。历史记录显示，实际 SLSQP 数值路径在前次报告定位的 PR #8 祖先提交 `88e4e75` 中删除；近期整改进一步把点数参数改成局部迭代次数，原能力尚未补回。

### 必须补回的功能与验收

1. **恢复明确的“前沿目标点数”参数**，支持用户以 20、200 等数量控制网格密度；单点最大迭代次数单独配置或采用明确默认值。请求及草稿字段不得将两种语义混用。
2. 在当前资产、单资产约束和 group constraints 下确定可行的前沿区间；按收益轴或风险轴生成目标网格，逐点求解。随机散点、前沿优化点和三个代表点分别保留其用途。
3. 每个返回前沿点应有实际权重、实际收益 / 风险、对应目标值和求解状态；失败、不可行或重复目标应如实计数，曲线坐标来自真实求解结果。
4. 前端绘制按目标顺序组织的前沿曲线，保留散点背景、代表点和权重采用功能。画面的平滑设置属于展示层，求解精度与约束满足须单独验证。
5. 按项目现行 NJIT / 固定签名 / 预热要求实现数值路径。若继续使用 SLSQP 名称，实际算法必须相符；若换用其他求解器，也必须保留并证明上述逐目标求解能力，不能以三个代表点局部搜索替代。
6. 用 **20 与 200 个网格目标**验收请求、实际求解次数、成功 / 失败返回及图像；在可解析小样本或受控参考下验证目标约束、最优风险 / 收益、单资产和 group constraints，并覆盖不可行目标、边界及可重复性。
7. 当前整改与设计文档将范围限定为“三个代表点、不做连续前沿求解”，须根据本次用户澄清同步修正，不能继续将它作为原功能完整恢复的结论。

以均值—方差模型为例，固定收益后最小化风险是构造风险收益前沿的标准方式；在该模型及对应约束下可以采用目标网格逐点求解。[Stanford / CVX Group 组合优化说明](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)

密集取点的验收应看逐点求解和覆盖情况。启用严格离散权重时，可行组合可能不连续，不能承诺任意目标都恰好有解或曲线处处平滑；须明确图上连续权重前沿与离散候选各自使用的约束域。

## 3. 第三轮 P2 关闭证据

[representative_indices_kernel](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:429)已成为探索阶段及精炼最终阶段共用的固定签名代表点选择实现。当前流程在[合并候选后](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1207)统一重建前沿、重选三个代表点。

核对结果：

- 同一有限候选集合统一使用无风险利率、有限值、零风险排除及并列首个候选规则。
- `refinement.items` 保留各局部精炼分支的改善信息。
- [final_representatives](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:1222)单独记录最终代表点的候选索引和 sampled / refined 来源；独立检查确认索引、权重、坐标与 scatter 一致。
- 第三轮复现 A：`max_sharpe = 0.962150364509973`，最终索引 102，来源 refined。
- 第三轮复现 B：`max_return = -0.06790792451618546`，约 **-6.790792%**，最终索引 100，来源 refined。

第二轮两条、第三轮两条原始审核测试均未修改，合并运行结果为 **4 passed**。第二轮的比较基准与前沿同步修复保持通过。

本轮没有发现该代表点重选修复的新数值缺陷；F1 是用户原始功能尚未恢复的问题，不能用本节关闭结果抵销。

## 4. 本轮实际验证

| 检查 | 结果 |
|---|---|
| 第二、三轮原始审核侧复现 | **4 passed**，13.67 秒 |
| 同范围后端 17 个文件 | **361 passed**，175.81 秒 |
| 前端全量 | **123 文件 / 911 passed**，32.71 秒 |
| 隔离真实 API + Chrome，桌面 1440 / 手机 390 | **4 passed**，56.8 秒 |
| TypeScript | 通过 |
| Production build | 通过，14.00 秒 |
| Design check | 无新增回归 |
| AI routing validation、git diff --check | 通过 |
| 独立代表点边界参考检查 | **47 组通过**，包括空集、NaN/Inf、零风险、阈值、并列值、有效 count 前缀 |
| 非法轴 / count | **3 组正确拒绝** |
| 独立重算与来源一致性 | **18 组通过**，覆盖关闭精炼、1 / 20 次、量化与 group constraints、非零无风险利率 |

Optimizer **2.3.0**，固定签名内核 **14/14**，`nopython=true`、`object_mode=0`、`python_fallback=0`；独立探针调用前后没有新增签名、没有改写选择器输入。

证据：[原始复现日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/original-regressions.log) · [后端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/backend.log) · [前端](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/frontend.log) · [浏览器](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/browser.log) · [独立探针](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/independent_verification.py) · [数值及来源审计](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/independent_verification.json) · [构建](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/build.log) · [TypeScript](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/typescript.log) · [Design](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/design.log) · [Routing](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/routing-validation.log)。

后端仍有 19 条既有依赖弃用 / 日期推断警告；构建有既有大 chunk 警告。通过的测试验证当前已实现的算法，**尚未覆盖 F1 要求的网格求解功能，因为当前代码没有该功能**。

## 5. 审核边界

- 当前分支 `codex/frontend-design-quality`，HEAD `2329845b345665a9bacc95e36b814868a8e288ea`，审核对象包含尚未提交的改动。
- 相比第三轮，任务内变化为 optimizer、对应测试及三份整改 / 设计 / 验收文档，共 5 个文件；另有 AGENTS 和前端设计规范的既有改动，本轮读取当前规则并保留。
- 本轮依据 AGENTS、路由文档、CodeGraph、当前源码、Git 历史、测试及用户澄清完成。浏览器使用临时数据夹具；本轮结果不是生产部署、全部 worker 或投资有效性认证。
- [工作区校验](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-round4-recheck-evidence/source-verification.json)确认本轮开始记录的 61 个文件哈希、HEAD、已有 tracked diff 均未改变。
- 本报告只新增第四轮记录，不覆盖前三轮历史证据。**当前结论：第三轮数值 P2 关闭；原功能恢复 F1 / P2 待整改。**
