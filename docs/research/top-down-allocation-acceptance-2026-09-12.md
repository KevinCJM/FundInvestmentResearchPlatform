# 自上而下配置研究：实现、自审核与验收

日期：2026-09-12。配套文件：`top-down-allocation-research-2026-09-12.md`、`top-down-allocation-design-2026-09-12.md`。

工作区基线：执行期间原有前端设计/指标改动被提交为 `2329845`。本轮保留该提交，以当前工作区运行回归；没有提交、推送、切换分支或操作正式投资组合。

第四轮补充：原“精炼数量”被误解为三个点的迭代次数，前三轮数值修复不等于原始功能完整恢复。现已恢复整条前沿的逐收益目标求解，独立设计与证据见 `frontier-grid-round4-design-2026-09-12.md`。

## 1. 实际交付

| 环节 | 已实现 | 主要代码 |
| --- | --- | --- |
| 投资目标 | 真实目标版本：币种、期限、最低预期收益、波动上限、流动性边界、战术主动风险预算及复核日期。后端计算真实消费这些约束。 | `backend/strategic_allocation/contracts.py`、`service.py`；`InvestmentObjectivesWorkspace.tsx` |
| 大类经济解释 | 沿用已有大类和产品映射，要求确认经济用途、流动性与代理理由。不按产品名字猜测宏观敏感度，也不把统计聚类自动视为经济分类。 | `AssumptionEditor.tsx`、CMA 资产契约 |
| 资本市场假设 CMA | 明确的同币种、同期限、年化算术总收益假设；波动、相关性、均值不确定半宽；来源与确认记录；对称/半正定校验；不可变保存。 | `backend/strategic_allocation/{contracts,kernels,service,routes}.py` |
| 历史风险参考 | 读取真实共同净值，复用样本协方差内核，采用明确指定的对角收缩系数。当前要求 SSE 连续开放日日频并固定 252 年化；所有资产共同缺失开放日同样阻断。只回填波动和相关性，不把历史均值自动写成未来收益。 | `historical_risk_kernel`、`risk_reference` |
| 长期政策 SAA | 比较同一可行候选集合中的较低风险、名义效用、区间稳健效用、较高预期收益。确认后复算校验并直接保存为原 TAA 可消费的政策基线，冻结目标、CMA、产品映射、约束和理由。 | `policy_candidates_kernel`、`preview_policy`、`publish_policy` |
| 战术 TAA | 新政策的主动风险预算不能擅自扩大；当前目标按冻结 CMA 同时检查总波动和相对 SAA 的前瞻主动风险 `sqrt((w_TAA-w_SAA)'Σ(w_TAA-w_SAA))`。导出及直接产品组合入口共用门禁。增加滚动/扩展训练的分段验证；固定假设某折超限只阻断该折，不改选且继续后续折。 | `policy_gate.py`、`tactical_allocation/walk_forward.py`、`service.py`、`portfolio_bridge.py` |
| 历史 SAA 研究区间 | 点权重求解、调仓计划与回测统一使用可选 `end_date`；缓存区分结束日，响应列出实际区间。前端改变结束日后清除旧计算权重并拒绝迟到结果。 | `backend/services/strategy_routes.py`、`ClassAllocation.tsx` |
| 历史前沿目标网格 | 20／200 个收益目标逐点最小风险求解，目标点数与单点迭代预算分开；真实坐标曲线、逐点权重/状态/采用、失败断线及重复计数；网格解加入共同集合重算 Pareto 和代表点。连续权重与取整采样需要显式区分。 | `backend/qp_numba.py`、`optimizer.py`、`FrontierGrid.tsx`、`ClassAllocation.tsx` |
| 操作体验 | 研究范围 → 长期假设 → 政策比较 → 确认与交接；历史实验与前瞻政策分工明确。主 SAA 页面直接提供历史有效前沿入口；历史页有“生成可配置空间与有效前沿”按钮、默认展开真实散点/近似前沿，并展示样本与候选数量；方案切换不离开历史实验。 | `StrategicAllocationWorkspace.tsx`、`ClassAllocation.tsx`、新组件、路由和导航 |

入口：`/pre-investment/objectives`、`/pre-investment/saa/policy`。原 `/pre-investment/saa/allocation-lab` 保留为历史有效前沿/风险预算实验，并非另一套前瞻政策实现。

## 2. 调研如何落实到设计

调研使用 CFA 与 J.P. Morgan、BlackRock、Vanguard、Bridgewater、State Street、AQR 的公开第一方资料，具体日期与链接见调研文件。

落实的是机构公开方法中的共同原则：先目标和风险预算，再前瞻假设，再权重；经济分类与产品代理分开解释；考虑均值估计的不确定性；TAA 相对 SAA 进行受限偏离；留出验证不得反选训练赢家。没有声称取得或复刻机构专有的 VCMM、VAAM、MRI、All Weather 完整模型或参数。

## 3. 最终执行结果

| 验证 | 实际结果 |
| --- | --- |
| 后端回归 | 第四轮实际运行相关18文件 **392 passed**，19条既有警告；含网格专项31项。第二、三轮审核侧四条原始复现未修改，**4 passed**。第一轮历史全仓结果3365项不代表本轮重新全仓验收。 |
| 全前端 `npm run test --prefix frontend -- --run` | **124 个测试文件、918 项测试全部通过** |
| 真实浏览器 + 隔离 API | **6 passed**：桌面1440与手机390；每种视口分别覆盖Goal→CMA→Policy→TAA、原历史前沿/局部精炼、新增20/200目标真实求解及曲线更新/失败状态。 |
| TypeScript | `tsc --noEmit -p frontend/tsconfig.json` 通过 |
| 生产构建 | `npm run build --prefix frontend` 通过 |
| Python 全量语法静态检查 | 第四轮使用项目Python3.12，对Git列出的全部 **515个Python源码** 解析通过 |
| 设计规范 | `npm run design:check --prefix frontend` 无新增回归 |
| 差异空白检查 | `git diff --check` 通过 |
| CodeGraph | 第四轮已同步，**965文件、19719节点、67197条边**，索引最新 |

前几轮新增的战略、日期和TAA测试继续保留。本轮新增 `test_frontier_grid.py`，前端新增 `FrontierGrid.test.tsx` 并扩展 `ClassAllocation.test.tsx`；上述计数已包含相应测试，不重复叠加。

浏览器使用 `frontend/playwright.strategic.config.ts` 与 `backend/tests/strategic_allocation_app.py`：临时目录中的真实 Parquet、真实服务路由、真实 NJIT 内核和不可变成果库；不使用模拟计算结果代替新功能。只对无关服务返回明确的不可用响应。主链完成目标创建、历史风险读取、CMA 校验与保存、政策求解与采纳、前瞻主动风险门禁、TAA 分段验证；历史实验链实际修改随机探索、权重量化和局部精炼参数，验证请求真实传入后端、候选数量与实际样本可见、真实 ECharts canvas 默认显示，并验证切换方案不跳出 `allocation-lab`。新增网格流程在20与200之间切换，断言真实API的点数、求解次数、成功数、权重约束及候选计数，并比较实际canvas图像发生变化；单点预算1会触发真实端点未完成，不返回假曲线。两种视口均检查整体横向溢出和页面 JavaScript 异常。该流程未认证完整生产 PIT 设置、正式产品池应用资格或真实交易。

## 4. 审核重点及实际发现

- **数值一致性**：样本协方差/收缩、组合均值/方差、稳健效用和风险贡献与受控 NumPy 参考勾稽；无效矩阵不自动修复；零惩罚下稳健与名义候选一致；更大的不确定性按声明的目标改变选优。
- **时间一致性**：每段训练尾部尚未成熟收益只作尾部剔除；内部未知/未可得收益阻断该段，不压缩中间日期。改变本段留出收益不改变本段训练选择。历史 SAA 截止日之后价格变化不改变截至日内结果。
- **版本一致性**：保存由服务端重算并核对 hash；输入和风险参考变化会阻断旧预览采纳。政策基线冻结建立时的净值历史前缀：后续只追加新观察允许继续研究，冻结前历史若被修订则失败关闭；已保存决策继续读取自身冻结数组。客户端不能提交自造绩效、篡改政策预算或绕过直接产品入口门禁。
- **前端一致性**：新目标/新假设不预填收益；编辑和迟到响应不串版本；确认才持久化。选择名义或稳健候选不是外部审批。
- **首次全前端回归**发现两处配套遗漏：旧 SAA 返回链接测试、导航语言目录缺少新入口；均已修正后全量重跑通过。
- **浏览器走查**发现百分比重绘显示 `7.000000000000001`；新增统一显示边界转换，保留底层假设数值，浏览器仍断言显示为 `7`，未放宽测试掩盖问题。首轮下拉框定位改为可访问角色选择器后继续完整走查。
- **静态检查**首轮误用了系统 Python 3.9，对已有 Python 3.12 语法报错；改为项目 Python 3.12 后全部通过，未修改该既有文件。

## 5. NJIT 与内存实证

新前瞻计算及本轮新增前瞻主动风险、局部精炼、Pareto 前沿与代表点筛选内核均为固定签名 NJIT，并在启动阶段预热；服务缺少预热时失败关闭。历史前沿的旧假 SLSQP 已删除，改为 `bounded_pairwise_pattern_search_njit`。精炼直接以探索阶段返回的原始可行候选计算 `before_score`，不会为了量化再次改写比较起点；只有相对原候选严格改善的结果才加入最终候选集合。最终集合统一经过 `pareto_frontier_indices_kernel` 重建非支配前沿，再由 `representative_indices_kernel` 按同一无风险利率、零风险和并列规则重选最大夏普、最小风险、最大收益三个代表点。`refinement.items` 保留分支改善审计，`refinement.final_representatives` 单独记录最终代表点来源，避免把分支精炼结果误当作最终代表。三个点局部精炼现为可选补充；整条前沿由独立目标网格运行N个约束优化，实际算法为主动集QP/可行BFGS-SQP，不冒用SLSQP名称。均值—方差以解析解和独立测试求解器验证精度，非二次风险只报告局部数值收敛。成功网格解加入共同集合，前沿和最终代表点一起更新，来源包含grid。Optimizer kernel version为 **2.4.0**，固定签名内核 **22/22**，`python_fallback=0`。TAA 多段验证仍调用原 `evaluate_candidates → _taa_path_kernel`，没有新增平行回测引擎。

定向测试覆盖只读、非连续数组与实际 `np.shares_memory`；分段训练/验证使用原历史数组的切片，未将成熟间隔拼接成新数组。CMA 协方差成果通过只读 mmap 读取。Parquet 解码、轴对齐、dtype 规范化和小型约束边界仍有必要分配；权重候选、输出与独占工作缓冲允许分配，不宣称整个请求零分配。

前期CMA基准（不是本轮网格求解耗时）：12 资产、5000 个随机候选，另加等权及顶点共 5013 个候选；纯候选 NJIT 内核五次运行中位数 **约 2.13 ms**，不含取数、序列化、导入或预热，不能当作 API 总耗时或跨机器保证。2000×12 历史风险输入为共享底层内存的非连续只读视图。运行前后每个新内核都保持一个签名；重复计算后 NRT 存活分配差为 **0**。进程峰值 RSS（含导入和预热）为 228278272 字节，不等于单请求新增内存。

## 6. 尚未覆盖的业务与治理边界

1. 大类仍来自现有产品篮子；本轮已修正“美国国债 QDII 被地域规则覆盖”的窄分类错误，经济资产类型优先于地域，但这仍不等于完成独立经济资产分类库、币种/对冲、久期/信用或穿透风险模型。自动聚类/PCA 算法沿用现有实现。
2. 本轮是同币种、long-only、无杠杆、正波动率资产假设研究。独立现金账本、跨币种对冲、负债/私募现金流、衍生品实施、概率目标达成率和完整 Black–Litterman 尚未实现；不以示例结果冒充。
3. 稳健 SAA 是透明的有限候选比较，不保证全局最优。历史风险参考的对角收缩系数是显式研究参数，不是自动估计的 Ledoit–Wolf。
4. TAA 分段验证选择的是预声明规则的偏离强度，不是完整嵌套的宏观模型重训练。各段独立从 SAA 起步，不拼成连续可交易净值，也不改变主研究选择。反复查看留出后改规则仍会损害独立性。
5. 研究版本可复现不等于完整历史 PIT、当时已部署、外部审批或真实盈利记录。数据可得时间仍按日期校验；ETF 开盘、基金申赎与结算等细节不在本轮认证范围。
6. 现有Hermes结构校验通过；新CMA等前期文件的登记边界保持不变。第四轮定向覆盖检查另识别出新QP文件、目标网格后端测试、前端组件/测试及浏览器测试尚未登记；结构通过不等于路径覆盖完整。当前新增文件未纳入 Git 跟踪，按该项目的可复现路径门禁，新增模块的长期路由登记保留至源文件纳入提交范围时同步处理；未修改忽略规则。CodeGraph 已覆盖本轮新代码。
7. 构建仍提示较大 JavaScript bundle；后端保留既有 Pydantic/Starlette 等弃用或日期解析警告，设计检查保留既有预算内项目。没有借本需求修改无关依赖或提高检查预算。

## 7. 本地证据与复跑

第四轮实际日志位于 `.pytest_cache/saa-taa-round4-implementation/`（backend/targeted/frontend/browser/build/typescript/design/original-review-regressions等）；前期证据仍留在 `.pytest_cache/saa-taa-review/`。浏览器截图按独立配置写入 `.pytest_cache/top-down-allocation/browser/`。这些均为本地诊断产物，不进入业务数据或正式研究成果库。

后端复跑须在导入服务前将默认研究工作区指向临时目录；本轮隔离 `CUSTOM_INDICATOR_DATA_DIR`、`HISTORICAL_REGIME_DATA_DIR`、`TACTICAL_ALLOCATION_DATA_DIR`、`STRATEGIC_ALLOCATION_DATA_DIR`、`TIMING_RESEARCH_DATA_DIR`。数值执行使用项目 Python 3.12，`PYTHONPATH` 同时包含项目根和 `backend`。

```bash
PYTHONPATH="$PWD:$PWD/backend" <项目Python3.12> -m pytest backend/tests/test_strategic_allocation.py backend/tests/test_strategy_research_dates.py backend/tests/test_tactical_walk_forward.py -q
npm run test --prefix frontend -- --run
npm run build --prefix frontend
npm run design:check --prefix frontend
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts
```

浏览器配置提供独立测试 API 与前端端口，不复用已有服务；测试结束清理临时行情与研究目录。没有启动或重启生产服务、改写正式行情、切换活跃快照、下载行情或执行真实投资操作。
