# 投资目标与边界：实现、自审核与验收

日期：2026-09-13。配套文档：[优化方案与产品评审](investment-mandate-optimization-2026-09-13.md)、[调研与前后端详细设计](investment-mandate-research-design-2026-09-13.md)。

> 本文第 1—7 节保留修复前的历史验收；本轮 1.2.0 修复和最终提交验证以 [修复复核报告](../../investment-mandate-fix-recheck-2026-09-13.md) 为准。旧版本用时与测试数量不作为本轮证据。

## 1. 验收结论与范围

本次需求已形成“资金与成功标准 → 风险与限制 → 量化诊断 → 核对与确认 → SAA消费”的可运行研究流程。本次续做在工作区已有未提交的目标研究实现上完成独立核对、两处一致性修复和最终回归，不把此前已有代码全部算成本次新增。

**业务回归通过；仓库发布准备尚未全部通过。** 全后端3492项、全前端948项以及8项真实浏览器测试通过。AI Hermes路由校验当前退出码1：并行前沿任务的路由引用了尚未Git跟踪的源文件、测试和设计文档。本任务没有更改这些引用、放宽门禁或擅自暂存文件；投资目标新增文件的长期路由覆盖也须在正式提交范围确定后补齐。不能将代码测试通过解释成可直接发布。

未执行commit、push、切换分支、生产服务重启、行情下载、正式组合修改或交易。原始SAA/TAA审核报告保留。历史前沿页及其路由文档存在并行改动，本轮未改写该页。

## 2. 实际实现

| 环节 | 当前能力 | 实际代码 |
| --- | --- | --- |
| 成功标准 | 绝对预期收益、期末金额与期间支付、相对真实基准三种目标；按目标类型展示和验证输入 | `contracts.py`、`MandateFields.tsx`、`model.ts` |
| 资金事实 | 总资金、组合外储备、可投资本金、名义/实际购买力、月末投入及支付、费用、必要支付窗口 | `goal_kernels.py`、`planning.py` |
| 确定性算账 | 所需年度有效回报、压力净支出对应流动性预算、支付缓冲和缺口；储备不重复扣减 | `funding_summary_kernel`、`required_return_kernel` |
| 前瞻诊断 | 引用真实已保存CMA，复用SAA四类代表组合；计算成功概率、Wilson采样区间、终值分位、未支付金额及回撤预警 | `service._candidate_calculation`、`funding_paths_kernel` |
| 调整代价 | 给定组合和共同路径集下的所需本金、追加本金、增加本金10%/降低终值10%的敏感性；点估计与采纳区间下界两种资本口径分开 | `capital_gate_kernel`、`MandateResults.tsx` |
| 目标约束 | 资产/组授权绑定大类方案，下游只能收紧；流动性由资金计划与手工下限取更严；基准TE与TAA相对SAA的TE分开 | `service._constraints`、`policy_gate.py` |
| 版本与历史 | 预览不保存；确认重算并核对hash；旧版本只读、显式复制；历史研究可保存，当前应用独立检查过期 | `routes.py`、`service.py`、`InvestmentObjectivesWorkspace.tsx` |
| 人类操作 | 四步页面，只显示当前所需字段；无CMA先完成资金算账，不编造概率；技术参数和历史列表收起，图表同时有可读年度表格 | `InvestmentObjectivesWorkspace.tsx`、`components/investment-mandate/` |

后端表格中的相对路径位于`backend/strategic_allocation/`；前端页面及组件位于`frontend/src/`。

缺少CMA时允许保存“仅输入版本”，不是可行性认证。已诊断但未达标也可保留研究记录；SAA采纳必须重新诊断被选候选并实际满足目标门槛，不能凭“已保存”绕过。

## 3. 自审核发现与修复

### 3.1 资本调整与采纳口径一致

原点估计分位数本金不能直接声称满足Wilson下界采纳条件。当前另计算能使同一路径集的成功计数满足区间下界门槛的最少本金；并列样本、样本量不足和零资本情况单独处理。独立测试将所需本金回灌同一随机路径，验证实际门槛；原点估计指标仍保持其既有含义。

### 3.2 诊断缺失不能变成通过

前后端拒绝缺失`goal_check`、非有限概率、倒置区间、错误门槛或与区间不一致的通过标志。直接SAA采纳入口同样检查；不以`.get(..., True)`将未知结果当成成功。授权资产和组必须绑定大类方案，禁止仅因资产同名迁移授权。

### 3.3 冻结资金模拟种子不能被SAA搜索设置替换

本次续做先复现失败：确认目标使用种子917，SAA搜索使用19，原`funding_model.seed`错误变为19。现分开候选搜索种子与目标保存的模拟种子/路径数。无已确认模拟设置的仅输入版本仍沿用本次政策请求设置，不制造历史证据。

后端独立测试验证搜索种子19变53而资金模拟保持917。真实浏览器采用目标模拟种子917、SAA搜索种子19，最终政策产物同时保留各自实际值。

### 3.4 历史快照与未保存预览的时钟行为分开

本次续做先复现失败：知识截止日变化会清空已保存只读版本的诊断，使确认步骤没有内容。现保留历史版本保存时的结果，并提示不代表新截止日下可用；未保存预览仍取消并失效。复制历史版本后必须遵守当前研究时钟，不能覆盖原版本。

对应前端测试验证不额外请求、不清空冻结诊断、不允许覆盖保存，并验证未保存预览仍不能继续确认。两处修复均保留失败复现思路，没有删除业务断言换取通过。

## 4. 最后一次实际执行结果

| 验证 | 实际结果与证据 |
| --- | --- |
| 全后端 | **3492 passed，0 failures / 0 errors / 0 skipped**，29条警告，869.38秒；`continuation-backend-full.xml`及`.log` |
| 投资目标后端专项 | **60 passed**，1条Starlette弃用警告；包括资金数学、概率、资本回灌、时钟、授权和直接采纳负向测试 |
| 全前端 | **126个文件、948 passed**，32.67秒；`final-frontend-full.log`。数量是当前整个工作区的实际结果，不是本需求新增测试数量 |
| 投资目标/SAA客户端与页面专项 | **3个文件、37 passed**；`continuation-targeted.log` |
| 真实浏览器 | **8 passed**，约1.2分钟；`continuation-browser.log`。1440桌面、390手机，金额诊断另覆盖320px无整体横向溢出 |
| TypeScript | `tsc --noEmit --project frontend/tsconfig.json`通过；`final-typescript.log` |
| 生产构建 | 通过，6.79秒；`final-build.log`，保留既有大bundle警告 |
| 设计检查 | 无新增回归；`final-design.log`，未提高检查预算 |
| Python静态检查 | **504个backend Python文件AST解析通过**；不是声称全仓所有脚本都执行过 |
| 语言引用检查 | `node scripts/check_i18n.mjs`返回valid=true；既有未翻译文本目录仍存在，不等于全功能多语言验收 |
| 差异空白检查 | `git diff --check`通过 |
| CodeGraph | 已同步，状态up to date；最后读取978文件、19964节点、68217边，属于检查时点的索引计数 |
| AI Hermes治理 | **未通过**；`final-routing-validation.log`记录未Git跟踪的前沿相关引用；目标新增文件覆盖见`continuation-routing-coverage.json` |

上述日志除数值探针另注明外均位于`.pytest_cache/investment-mandate-2026-09-13/`。全后端使用临时研究目录隔离后运行，最终XML计数与完成状态均已读取；没有用此前3458项日志代替本次结果。

浏览器验证的是临时Parquet、真实API和NJIT执行，不使用伪造概率替代金额诊断。覆盖现金流、CMA读取、诊断、确认、SAA实际采纳及独立模拟种子；同时回归无CMA路径、TAA交接和已有前沿。页面通过自动化状态、表格、canvas、异常和宽度断言。截图已保存，但本次读取截图工具超时，**不将其表述为已完成人工逐像素视觉审查**。

## 5. 数值、执行和内存证据

资金模型版本`mandate-funding-monthly-lognormal/1.1.0`。资金递推、所需收益根求解、Wilson区间、资本门槛等均使用固定签名NJIT，复用既有随机数与分位数内核；进程启动预热，禁止临时新增签名或Python回退。

复跑`.pytest_cache/investment-mandate/numeric_evidence.py`：120个月、2000条路径、7次内核调用；单候选中位数**4.39ms**。随机数和现金流输入均验证实际共享内存、只读及跨步布局；签名增长0，NRT存活分配差0、meminfo差0。进程峰值RSS为424181760字节，包含导入、预热及探针数组，不是单请求额外内存。结果见同目录`numeric-evidence.json`。

上述时间不含读取CMA、序列化和完整API开销，不作为跨机器性能保证。解码、排序/类型对齐、输出及独占工作缓冲仍有必要分配，不宣称整个请求零分配。

## 6. 建模与业务边界

- 本轮是一个期末财富目标与多笔必要支付的资金研究，不是完整多目标优先级优化、随机负债/寿命/税务模型。
- 月度独立对数正态组合代理只匹配CMA年度简单收益均值和波动；不模拟厚尾、状态转换、逐资产再平衡或实时交易。概率完全依赖声明的模型与CMA。
- Wilson区间只表示有限模拟样本误差，不包含模型错误、CMA估计错误或现实收益保证。保守条件是敏感性，不给它编造发生概率。
- 比较现有SAA的四类代表组合，不保证找到全局最高目标成功概率；代表组合未达标不证明所有组合都无解。
- 所需固定复合回报与CMA算术预期收益不同；不得提高CMA来使目标看似可行。
- 流动性预算依赖人工liquid标签及声明窗口内现金流，不认证各基金的赎回、结算、冲击成本或压力变现。支付缓冲也不是可承受市场亏损额度。
- 追加本金是同一组合和相同随机路径下的调整测算，重新选择目标或本金后仍需重新计算约束及SAA，系统不会自动加本金、融资或放松限制。
- TAA只继承其明确的短期政策检查，不把战略长期成功概率冒称为任意战术路径的保证。历史研究可保存不等于当时实际已部署、正式PIT认证或当前可执行。

## 7. 复跑与交付位置

```sh
PYTHONPATH="$PWD:$PWD/backend" <项目Python3.12> -m pytest backend/tests/test_investment_mandate.py backend/tests/test_mandate_diagnostic_integrity.py -q
# 全后端使用隔离研究目录的本地诊断runner，不指向正式成果库。
<项目Python3.12> .pytest_cache/investment-mandate-2026-09-13/run_backend_continuation.py
npm run test --prefix frontend -- --run
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts --workers=1
```

页面入口仍为`/pre-investment/objectives`，无需新增一套目标编辑路由。优化方案、详细设计和本验收记录均保存在`docs/research/`。浏览器截图位于`.pytest_cache/top-down-allocation/browser/`；这些是本地诊断产物，不是正式研究成果或投资业绩证据。
