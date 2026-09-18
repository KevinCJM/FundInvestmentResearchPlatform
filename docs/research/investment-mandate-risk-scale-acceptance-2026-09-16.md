# 投资目标与边界：实施、自审核与验收记录

任务日期：2026-09-16。项目：BetterSaaTaa。工作分支：`ISSUE2609/BetterSaaTaa`，开始时HEAD为`fa42a5e`。本轮未创建提交、未暂存文件、未推送或修改远端。

配套设计：`investment-mandate-risk-scale-implementation-design-2026-09-16.md`。依据原统一设计第5、13、19章，实施范围限于投资目标、必要的风险标尺接入、资金诊断及SAA/TAA消费。本文以真实命令和浏览器记录为证据，不把设计或模拟示例当作已验证的投资收益。

## 1. 验收结论

**前后端核心改造及受影响功能验收通过；仓库级路由治理检查仍有已记录的待处理项，不能描述为全部仓库检查均通过。**

|检查|结果|证据|
|---|---|---|
|改动前基线|125项通过|`.tmp_mandate_20260916/baseline-tests.log`|
|受影响后端完整回归|301项通过，4条既有依赖/性能警告|`backend-final.log`|
|全量前端Vitest|153个文件，1186项通过|`frontend-final.log`|
|TypeScript|通过|`types-final.log`|
|前端生产构建|通过；保留既有较大分包提示|`build-final.log`|
|前端设计检查|无回归；不是全仓历史问题归零|`design-final.log`|
|i18n检查|通过，无错误|`i18n-final.log`|
|输入类型生成及风险中心类型生成校验|两组`--check`通过|实际命令输出|
|真实浏览器|6项通过：320/768/1440三种宽度|`browser-final.log`|
|固定签名/共享内存/数值对照|通过|后端测试、`benchmark.json`|
|Git whitespace检查|通过|`git diff --check`|
|任务路由R87解析|通过|`routing-r87.log`|
|仓库路由验证|未通过：17个既有未跟踪引用；本轮没有新增该类验证错误|`routing-validation-initial.log`、`routing-validation-final.log`|
|本轮实现文件路由覆盖|尚有未跟踪的新文件待正式纳入稳定清单|`routing-evolution.json`|

上表未写前缀的日志均位于`.tmp_mandate_20260916/`。后端301项是受影响模块回归，不是对整个仓库所有后端测试的宣称。前端1186项是本工作区全量Vitest结果。

## 2. 用户可见变化

默认工作流由四步改为：**资金任务 → 成功标准 → 风险等级与预算 → 量化诊断 → 核对与确认**。

现金预算与成功标准分离：绝对收益、期间支付与期末余额、相对基准三种目标均可使用现金预算。切换目标不会删除本金或现金流；只清除不适用的成功条件。预算金额、名义/实值、费用和来源可以核对。

风险入口保留三种用途：按明确授权等级选择、在授权内请求资金建议、有依据的高级数值授权。等级模式冻结风险标尺ID/hash，由后端解析cap；C3表示“不超过C3”，不是“必须达到C3”。没有可用标尺可保存待补充研究，不虚构C3或15%上限。

政策参数有来源、日期和显式确认。新建目标不隐藏填入成功概率、现金保障窗口或机构最高风险等级。风险厌恶、路径数和随机种子保留在高级设置，不把它们当作机构风险测评。

诊断分别显示全局参考与实际CMA。系统先搜索候选，再冻结候选并做独立验证，显示最低已验证参考等级、正式授权和本次拟确认上限。参考结果不能取代实际SAA核验。

当前搜索未找到达标候选时，若仍有合法候选，提供一个固定候选下的追加本金测算；金额按同一规则回灌验证。该结果明确标为调整诊断，不自动修改原目标、不提高风险授权，也不假装推荐已通过。

原有不可变历史、复制为新研究、PIT失效、迟到响应保护、机构背景、资产与分组限制、旧数值授权及原三种成功标准保留。

## 3. 后端结构与调用链

主链：`InvestmentObjectivesWorkspace → /mandates/preview → 统一预算/授权解析 → 参考或实际CMA诊断 → /mandates/confirm → 不可变目标 → 实际SAA预览/采纳 → TAA政策门禁`。

|职责|实现|
|---|---|
|资金/政策/授权请求契约|`contracts.py`、`mandate_contracts.py`|
|新旧资金输入唯一适配、标尺授权解析|`mandate_inputs.py`|
|月度资金与资本补足唯一数值实现|`goal_kernels.py`、`planning.py`|
|参考前沿编排、独立验证、诊断性调整|`mandate_diagnosis.py`|
|候选约束检查、批量资金搜索、确定性选取|`mandate_kernels.py`|
|保存、实际CMA/SAA消费、独立采纳验证|`service.py`|
|冻结现金用途下限的下游复核|`policy_gate.py`|

以上文件位于`backend/strategic_allocation/`。Settings与目标共用同一个风险标尺服务；`risk_scale_service.py`只增加读取已核验冻结上下文的接口，没有复制前沿或分类算法。

旧`funding_paths_kernel`保留被真实调用、有测试的输入适配路径，委托唯一月度资金递推；不是保留一套旧数值实现作备份。没有生产Python回退，也没有用前端公式代替后端计算。

新目标保存`effective_cash_reserve_weight`。可交易流动资产下限与现金用途下限分别执行；TAA门禁缺少冻结现金证据时失败关闭，不按零通过。参考与目标的有效期共同限制后续政策使用。

实际SAA采纳冻结所选候选后使用独立验证流，失败不保存政策。它仍只证明声明模型下的研究结果；TAA短期风险门禁不被描述成完整长期资金续算。

## 4. 里程碑审核与修复

M0：读取项目规范、代码图谱、路由和真实源码。先记录已有203个变更文件hash，再执行125项基线测试。

M1：契约与消费改造。142项阶段回归通过；确认三类成功标准共享预算、旧请求不改语义、预算与储备不重复扣减、资金保障落入现金角色约束。

M2：参考诊断与数值内核。完成约束后101点前沿、共享搜索样本、独立验证、无解措辞、金额回灌与固定签名测试。补测非法轴、NaN、只读非连续输入、资金支付失败永久保留、旧适配与新共享内核逐值一致。

M3：五步工作台。发现新增验证种子会被通用草稿形状检查视为旧草稿不兼容；在目标边界做有限适配，保留原输入，没有放宽全局草稿校验。修复新文案插值、表单可访问标签和标尺分页迟到响应；读取失败与空目录不再混显示。

M4：跨模块验收。发现TAA共同门禁仍只读取原机构现金字段，已改为新版冻结有效现金下限，并补缺失证据及真实下游测试。补上搜索未达标时的资本调整诊断，禁止把该诊断升级为授权。浏览器测试的服务端/UTC日期差异通过显式对齐夹具时钟修正，没有放松生产可得日期校验。

最终审核确认：没有复制第二套支付/概率算法；原来所有有关历史、实际CMA、授权收紧及缺失诊断阻断的测试保留；新增代码的生产数值路径实际进入已预热NJIT内核。

## 5. 浏览器验收证据

命令：

```bash
npm run test:e2e --prefix frontend -- --config=playwright.mandate.config.ts
```

使用真实Chrome、真实FastAPI路由、临时目录、离线合成数据、正式数值路径。测试不写项目正式研究成果，不访问外部数据网络。

中文链路覆盖：填写资金 → 成功标准 → 标尺和政策 → 101候选搜索与独立验证 → 明确勾选 → 服务端确认 → 只读 → 进入实际SAA并重新计算 → 返回复制 → 高目标未达标 → 查看追加资本且不产生自动授权。

每个关键页面检查整页无水平溢出、标题焦点、主要说明文字对比度至少4.5:1，并保存截图。英文检查覆盖新建入口、空输入不放行、键盘输入与Tab焦点，不宣称所有旧诊断详情均已完成英文化。已执行自动浏览器布局和对比度检查；没有把截图存在冒称逐像素人工视觉审核。

18张截图和基准摘要已保存在：

`docs/research/screenshots/mandate-boundaries-20260916/`

各宽度目录：`desktop-1440`、`tablet-768`、`mobile-320`。每组包含资金、风险授权、独立诊断、冻结确认、未达标资本诊断及英文入口。完整运行附件位于`.pytest_cache/mandate-boundaries/browser/`。

## 6. 性能与内存

基准命令：

```bash
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  backend/tests/benchmark_mandate_reference.py \
  --output .tmp_mandate_20260916/benchmark.json
```

本机macOS arm64，101候选 × 120月 × 2000路径，预热后的批量搜索内核三次约为0.193、0.196、0.193秒。该值不包含完整HTTP、前沿求解与页面加载，不是端到端响应时间，也不是收益表现。

共享输入1,923,384字节，候选摘要输出2,424字节；`np.shares_memory`确认抽样视图复用所有者且只读。每次NRT计数为6次原生分配，**不是零分配**。没有候选×路径×月的大结果立方体。整体进程峰值RSS为496,730,112字节，包含依赖导入与启动预热，不能解读为单内核峰值内存。

固定签名无增长、`python_fallback=0`、`request_time_compilation=0`；三次输出确定性一致。没有测得或宣称相对Python的倍数加速。

## 7. 可重复的主要命令

后端：

```bash
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_mandate_boundary_contracts.py \
  backend/tests/test_mandate_reference_diagnosis.py \
  backend/tests/test_investment_mandate.py \
  backend/tests/test_mandate_diagnostic_integrity.py \
  backend/tests/test_strategic_allocation.py \
  backend/tests/test_bettersaataa_m1.py \
  backend/tests/test_bettersaataa_scope_journey.py \
  backend/tests/test_cma_models.py \
  backend/tests/test_cma_model_integration.py \
  backend/tests/test_strategic_risk_budget.py \
  backend/tests/test_bettersaataa_final_audit.py \
  backend/tests/test_risk_scale_service.py \
  backend/tests/test_risk_scale_integration_audit.py \
  backend/tests/test_tactical_allocation_bridge.py -q
```

前端和契约：

```bash
npm run test --prefix frontend -- --run
(cd frontend && npx tsc --noEmit)
npm run build --prefix frontend
npm run design:check --prefix frontend
npm run i18n:check --prefix frontend
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 scripts/export_mandate_contracts.py --check
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 scripts/export_risk_scale_contracts.py --check
git diff --check
```

## 8. 路由治理待处理项与范围保护

本轮仅在三个正确的JSON所有者内更新已核实的模块语义、R87匹配词和P22注意事项；没有改AGENTS协议，没有改前端规范，也没有新增一套互相冲突的规则。

`validate_ai_routing.py`在本轮路由语义更新前后均报告相同17个问题，全部是先前情景研究的未跟踪文件引用；错误集合无新增。R87上下文解析正常。这些问题没有用伪造白名单、删除原功能或暂存无关文件绕过。

针对本轮41项实现/测试/设计文件执行`evolve_ai_routing.py --changed-file ... --json`，12项新文件尚未登记为稳定路径，其中包括新增测试、浏览器夹具、类型生成脚本及设计文档。按现有治理要求，稳定路径须已由Git跟踪；本轮没有获得提交/暂存授权，因此将清单保留在`routing-evolution.json`，待将来授权纳入Git时一并登记。不能把该覆盖检查说成已通过；此项不改变上述功能测试结果。

开始时已有203个改动文件。基线hash核对确认，本轮只在5个原本已变更的共享文件上继续改动：战略路由共享实例、参考契约去重、风险服务冻结上下文、增加目标i18n文案，以及仅更新`repo_map.json`中的战略模块语义。其他原情景、风险中心实现和原统一设计文档未被本轮改写。`task-files.json`记录本轮具体范围，避免将整棵dirty worktree误算成本任务。

## 9. 明确保留的边界

本次政策是Mandate内的不可变确认快照，没有新建完整OrganizationPolicy管理/审批平台。请求类型来自Pydantic生成；新响应仍沿用战略模块类型加严格运行时校验，没有谎称全部响应已自动生成。

保留整数年、等长模型月；不输出无法回灌的小数年期限。只支持有限候选研究，不能把最低已测试等级理解为严格全局最低风险。所有概率都依赖声明分布和现金流假设。

未扩展多CMA联合优化、完整ALM、随机负债、全自动多变量目标修复、全自动TAA预算或真实交易。旧历史/高级研究能力仍可用；当前验证不是模型有效性认证、外部审批或投资收益保证。
