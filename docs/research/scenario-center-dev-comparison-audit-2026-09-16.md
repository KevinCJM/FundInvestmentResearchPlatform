# 情景算法中心：对比远端 Dev 的变更总结与审核

审查日期：2026-09-16。

## 结论

当前改造主体是情景算法中心及市场状态研究，以及其必要的 TAA、PIT、启动预热和测试联动。没有新增独立 LTCMA 研究中心，也没有在本次差异中重写 SAA 优化引擎。

**审核结论：暂不通过。** 已有回归通过，但额外边界检查复现了 5 类问题：旧情景预览兼容、实际模板的前瞻验证准入、部分资格下的概率消费、月频持续期计算、压力路径终点判断。当前应视为“主要研究流程已经建立，尚需修复与收口”，不能继续沿用“全部优化已完成”的结论。

本轮仅执行远端同步、代码/契约审核、隔离测试和报告整理，没有修改业务实现，没有提交、推送或创建 PR。

## 1. 远端同步与比较范围

执行：

```bash
git fetch --no-tags origin refs/heads/Dev:refs/remotes/origin/Dev
git merge --ff-only origin/Dev
```

结果：

- 当前开发分支：`ISSUE2609/BetterSaaTaa`。
- 本地 HEAD 与 fetch 后的 `origin/Dev` 都是 `a4c30843870c7b457a273756557fe98c9a02bf29`。
- 提交差距为 `0 / 0`；合并返回 `Already up to date.`，没有产生新的合并提交。
- 审核的业务差异来自工作区已修改文件和未跟踪文件，而不是 `origin/Dev...HEAD` 的提交差异。只看三点提交 diff 会遗漏本轮全部工作。
- 本次复核在合并前后比较了 142 个变动文件的 SHA-256（含审核报告自身），结果 `WORKTREE_PRESERVED=True`，未使用 stash、reset 或 clean。

审核起点的范围，排除本轮审核产物：

| 分类 | 文件数 |
| --- | ---: |
| 后端实现与研究脚本 | 32 |
| 前端实现、类型及夹具 | 25 |
| 测试、浏览器用例及测试配置 | 41 |
| 研究/设计/验收文档及图片 | 23 |
| 小计，不含临时文件 | **121** |
| 临时任务记录、日志、PID 文件 | 20 |

本次复核的 Git 状态口径：32 个已跟踪文件修改，110 个未跟踪文件，其中包括本审核报告和 20 个临时文件。上表不含报告自身及临时文件。已跟踪文件相对 Dev 为 +973 / -216 行；此行数不包含新增文件，不能用来代表全部改造规模。

## 2. 已经改了什么

### 2.1 市场状态研究工作流

情景中心采用“市场状态研究 / 全球历史事件库 / 情景模拟与压测”三个入口。市场状态内部串联：

```text
定义历史参考 → 建立实时识别 → 验证识别能力
```

历史定义与实时定义分别保留草稿；实时构建和验证共用同一模型。兼容原 historical/realtime 深链接，避免把一侧 definition/revision 带入另一侧。已增加键盘切换和响应式检查。

主要文件：`frontend/src/pages/ScenarioCenters.tsx`、`MarketStateResearchCenter.tsx`、`HistoricalRegimeWorkbench.tsx`、`regime-workbench/regimeStudy.ts`。

需要准确描述：第一步的“下一步”目前是导航，第二步仍通过历史参考下拉框显式选择发布版本；不是自动把第一步刚发布的 reference 无缝带入第二步。界面整合已经实现，自动交接仍可改善。

### 2.2 历史参考、实时模型与验证报告分离

新增 `Study` 契约，区分 `historical_reference` 与 `realtime_recognition`。实时模型绑定准确的 reference run/publication/hash，可显式配置状态映射。校准、资格、模型和数据版本通过哈希校验，编辑后取消旧资格绑定。

历史质量诊断包括区间数、完整区间、覆盖率、参数/窗口/种子敏感性，以及 Horizon 持续期分布。实时诊断包括分类、区间重合、转折延迟、概率校准、Brier、时序相关性区间和逐状态证据。

状态结果区分 verified / partially_verified / insufficient_evidence / failed。它们是相对所选参考和门槛的研究结论，不是市场真值或投资盈利保证。

### 2.3 前瞻记录与数据续接

新增候选登记、服务端捕获判断、后续成熟参考评估、资格采用及撤销校验。日志保留签名/哈希链，禁止回填与覆盖原始预测；支持受限的单一日频指数快照续接。

**这一设施确实存在，但并未与当前全部推荐模型打通，详见 F02。**

### 2.4 实时识别与 LTCMA 解耦

实时报告提供 `recognition-evidence`。旧 `cma-evidence` 入口明确拒绝，返回 `REALTIME_CMA_EVIDENCE_REMOVED`。两个 TAA 消费入口统一调用校准与资格解析，而非直接信任原始 confidence。

历史参考继续提供后续 LTCMA 研究所需的区间与样本证据。**不存在已经完成的独立 LTCMA 模块和完整 Historical Reference → LTCMA 消费接口。** 当前 `quality.py` 的 conditional_estimation 主要是完整区间数量门槛，不是长期收益/协方差估计质量认证。

### 2.5 算法和算子

保留沪深300主趋势事后参考及 SMA9 实时模板；增加波动率 HMM 事后参考、固定阈值实时波动状态，以及回撤周期事后/实时模板。

新增回撤周期耦合内核及其领域注册。常量节点允许参与研究参数扰动，显式 `parameter_role=structural` 的常数不扰动。新增计算接入固定签名 NJIT 与启动预热。

GMM 波动分组、窗口变化、多窗口共识、旧风格轮动及两个日频峰谷模板，从新建目录隐藏。旧模板仍可按精确 ID 读取，已保存工作区副本也没有被删除；不能将其描述为已彻底清除全部旧算法。

Style Cycle 尚未作为新的正式可信配对写入系统。研究记录与原始模板的隐藏，不等于已完成可用的风格识别新模型。

### 2.6 情景模拟的 Horizon 提示

增加终点变化值、零变化尾部长度、传导模型滞后阶数，以及 `economic_horizon_status=not_empirically_established`。这是路径统计，不是完整经济持续期验证。

该提示有两种已复现误判，详见 F05。历史类比、恢复模型、累计滞后尾部检查、流动性期限和 LTCMA 入口准入尚不能据此宣称完成。

## 3. 必须处理的问题

### F01 — P1：旧的已发布情景预览会导致页面崩溃

位置：`frontend/src/pages/PublishedScenarioCenter.tsx:114`；类型契约 `frontend/src/services/riskModels.ts`。后端 `PublishedScenarioService.resolve_release` / preview 读取仍返回原有不可变内容。

新增界面直接读取 `preview.horizon_evidence.terminal_status`，并把 `horizon_evidence` 声明为必填。Dev 版本保存的 preview 没有该字段，且旧不可变成果不应被就地重写。

**浏览器复现成功：** 使用不含 Horizon 字段的旧版情景夹具，打开情景库并点击“查看路径”，抛出：

```text
Cannot read properties of undefined (reading 'terminal_status')
```

建议：采用显式版本化/可选字段读取，旧成果显示“此版本未提供期限证据”，不得默认已验证或重写历史。增加旧发布版本读取的浏览器回归。

### F02 — P1：实际推荐实时模板无法走通前瞻登记/捕获

位置：`backend/historical_regimes/reliability/prospective.py:267–289`、`:338–350`。

存在三个独立限制：候选只支持日频；执行白名单排除标准 latent/threshold 以外的 `model.*`；任何非空 `evaluation_targets` 都被拒绝。

对实际模板定义执行入口校验，得到：

| 模板 | 已复现的执行入口结果 |
| --- | --- |
| `csi300-maintrend-sma9-realtime-v3` | `PROSPECTIVE_UNSUPPORTED_GRAPH` |
| `csi300-volatility-hmm-reference-recognition-v1` | `PROSPECTIVE_UNSUPPORTED_AXIS` |
| `csi300-drawdown-cycle-realtime-v1` | `PROSPECTIVE_UNSUPPORTED_GRAPH` |

另外，SMA9 和回撤周期的月频输出还会在候选检查阶段被 `PROSPECTIVE_BINDING` 阻止。表中是执行入口的独立复现，不代表完整登记请求的错误优先顺序。

因此不能说这些模型现在只是“等待积累未来数据”就能自动获得资格：当前实现先阻断了它们进入该流程。

建议：在不放宽 PIT 要求的前提下，支持实际使用的可证明因果算子和闭合月频观测；纯展示对象不应成为数值执行的额外依赖。对每个推荐模板新增“登记→捕获→成熟参考→采用→TAA消费”的端到端测试。暂未支持的模型应在前端提前解释，不能让用户登记后才遇到技术白名单错误。

### F03 — P1：部分资格只约束选中标签，未获准状态仍参与 TAA 概率混合

位置：`backend/historical_regimes/reliability/consumer.py:125–143`；`backend/historical_regimes/taa.py:931–976`；`backend/tactical_allocation/service.py`。

当前检查只判断 `chosen` 是否在 `qualified_states`，之后却返回所有状态的完整校准概率。TAA 使用完整概率向量乘各状态偏离。

**数值复现成功：** 只允许 Bull，当前标签也为 Bull，校准结果为 Bull 80% / Bear 20%。消费函数返回完整概率且 `reason=None`。若权益偏离假设分别是 Bull +10%、Bear −50%，实际概率混合得到 −2%，未获准的 Bear 仍直接影响决策。这是合成边界例子，不是实际交易记录。

建议：必须明确“资格只认证分类标签”还是“也授权状态条件决策”。按当前 `do_not_authorize_unverified_states` 契约，不能静默使用未授权状态偏离。保守实现可以在需要不完整状态概率时整体回退；或保留原始概率用于展示，将未授权状态的偏离贡献置零、对应概率留在 SAA 基准，不把它重分给获准状态。两个 TAA 入口必须使用同一消费契约和测试。

### F04 — P2：月频完整状态被算成 1 个日历天

位置：`backend/historical_regimes/reliability/diagnostic_kernels.py:94–101`。

日历持续期计算采用：

```python
dates[end - 1] - dates[start] + 1
```

它测量的是首末观测点覆盖的日期跨度，不是月频状态区间本身。合成日期 2020-01-31、2020-02-29、2020-03-31，状态为 A/B/A 时，完整 B 区间的统计为“1 个观测，1 个日历天”；按到下一状态边界的口径应是 31 天。

建议：先统一“状态从观测时点起生效”或“标签描述观测所属期间”的边界语义，再基于完整期间/已知切换边界计算持续天数。不能直接把稀疏采样点跨度命名为经济持续期。补日/周/月、不规则日期、单观测区间、未知隔断、首尾删失测试。

此前真实报告中 Recovery 的日历持续中位数等可能受此口径影响；应生成新质量报告，不覆盖旧报告。

### F05 — P2：零末期变化被过度解释为“回到无冲击基线”

位置：`backend/scenario_stress/published.py:114–139`；`backend/scenario_stress/numba_kernels.py:45–89`；界面同 F01。

复现一：路径收益 `[-10%, 0%]` 被标成 `returned_to_baseline`，但累计财富仍为 0.9。末期不再变化并不等于价格水平已恢复。

复现二：两期输入 `[1, 0]`，模型只在滞后两期响应。当前两期输出为 `[0, 0]`，系统标记已回基线；将输入延长一个零期后，输出变为 `[0, 0, 1]`。冲击尚未传导出来，终点检查却已报“回到基线”。

当前虽然记录 `maximum_transmission_lags`，并没有用于确认尾部覆盖；多段传导还需要考虑累计响应跨度。

建议：分别展示“末期增量是否为零”“剩余模型响应是否消失”“累计水平是否恢复”。按模型链的有效滞后推导/延展零输入尾部，再检查剩余响应。不要把这些结论合成一个含义过强的 baseline 状态。

## 4. 其他未收口点

- 历史质量报告的 `conditional_estimation_status` 只按完整区间数判断；没有检查长期期限适配、各资产研究代理、收益口径、协方差可估性或估计误差。界面的“LTCMA条件估计可用”应限定为样本计数提示。
- 没有完成独立 Historical Reference → LTCMA Evidence 消费接口和 LTCMA 消费门禁。实时接口移除是已实现的，历史端全面联动是待实现的。
- 历史→实时的一键参考交接仍可改善；现有实现是导航加显式下拉选择。
- 当前算法研究的历史一致率不能当作未见数据的市场预测能力。报告已记录 reference label availability 与模型选择史缺失等限制，应保留这些限制，不把 `verified` 当生产授权。
- 临时目录 `.tmp_regime_completion_20260915/` 有 20 个未忽略文件，含任务、原始日志和 PID，不能混入提交。未检查其私密内容，不应把这些原始日志登记成稳定路由事实。
- 路由结构 validator 通过，但 evolve 未通过：本次复核有 74 个未覆盖路径（含本审核报告），其中 20 个是上述临时文件。正式新实现、测试与文档需在提交准备阶段按规则完成跟踪和路由登记；“validator通过”不能替代“覆盖检查通过”。

## 5. 本轮执行的验证

| 检查 | 本轮实际结果 |
| --- | --- |
| 情景、状态、参考、校准、前瞻、数据续接、TAA与序列相关后端回归，49 个文件 | **916 passed** |
| 前端全量 Vitest，146 个文件 | **1119 passed** |
| 市场状态三步流程、旧工作台、发布情景/产品应用浏览器回归，320/768/1440 | **19 passed，2 个原有桌面限定用例跳过** |
| 旧发布情景缺字段的缺陷复现 | **成功复现页面崩溃；不是产品验收通过** |
| 月频持续期、延迟冲击、零变化非恢复、部分资格、模板准入的数值/契约探针 | **成功复现上述问题** |
| TypeScript / design / i18n / build | **通过** |
| `git diff --check` | **通过** |
| AI routing validator | **通过** |
| AI routing evolve | **未通过：uncovered_files** |

后端测试使用隔离目录，浏览器采用离线 API 夹具，没有重新生成正式市场数据或覆盖已保存研究成果。本轮未重新训练全部真实市场模型，不将代码回归结果表述为新的真实市场统计结论。

非阻断警告包括现有 React act、Starlette/Pydantic 弃用、浏览器兼容数据过旧及前端大包警告；不以隐藏警告替代问题修复。

本地证据目录：

```text
frontend/test-results/dev-audit-recheck-20260916/
  backend-result.json / backend.xml
  frontend-checks.json / frontend-final.log
  typescript.log / design-final.log / i18n-final.log / build-final.log
  reproduce.py / reproductions.json / reproductions.log
  audit-repro.browser.ts / playwright.audit.config.ts
  browser-reproductions.json
  browser-regression.log
  evolve.json
```

该目录为隔离审核产物，不作为正式业务实现或提交内容。本次在当前业务源代码未变的基础上重跑了 49 个后端测试文件（916 项）和全部前端单测（146 文件、1119 项），并重新复现上述数值与浏览器缺陷。

测试隔离说明：首次前端复核误收集了审计目录里的 Playwright `.spec.ts`，导致 1119 个业务测试通过但测试进程退出非零；审计用例已改为 `.browser.ts`，由独立 Playwright 配置显式收集，随后完整 Vitest 重跑退出 0。普通 Playwright 默认输出清理会移除 `test-results` 下的临时文件，因此后续回归显式使用独立子目录 `--output=test-results/dev-audit-recheck-20260916/browser`；本报告列出的是重新保存后的证据位置，不依赖被清理的早期临时日志。业务测试配置、实现代码和正式研究数据没有为此改动。

## 6. 建议的收口顺序

先修 F01 的旧版读取兼容、F03 的 TAA 资格消费边界，再打通 F02 的推荐模型前瞻链路；随后统一 F04/F05 的期限与终点语义。补齐针对这些缺陷的失败测试并回归后，再更新之前的验收文档和路由覆盖。

这不需要重建情景中心架构，重点是修正当前接口、资格和统计解释的不一致。上述问题修复前，不建议宣称本批变更已全面验收通过，也不建议据此授权生产 TAA。

## 7. 最新 Dev 与当前工作区再次复核（2026-09-16）

本次重新执行 `git fetch origin` 和 `git merge --no-edit origin/Dev`，仍返回 `Already up to date.`。当前分支仍为 `ISSUE2609/BetterSaaTaa`，HEAD 与 origin/Dev 均为 `a4c30843870c7b457a273756557fe98c9a02bf29`，提交差距为 0 / 0；没有 stash、reset、clean、commit、push 或 PR 操作。

复核起点有 142 个变动文件：32 个已跟踪修改，110 个未跟踪文件；其中包括上一份本审核文档和 20 个临时文件。审计测试前记录各文件 SHA-256，测试结束、补充本节之前逐一核对，142 个既有文件均未改变。除本节文档更新外，本次只产生隔离审核产物，没有修复或修改业务代码，也没有重新训练正式市场模型。

### 7.1 重新执行的结果

| 检查 | 本次结果 |
| --- | --- |
| 同一组 49 个相关后端测试文件，分四组运行 | **916 passed；0 failures / errors / skipped** |
| 原样执行前端全量命令 | **退出码 1：146 个业务测试文件通过，旧审计 Playwright 文件被 Vitest 误收集并报框架冲突** |
| 排除 `**/test-results/**` 后重跑前端全部业务测试 | **146 files / 1119 passed；退出码 0** |
| 市场状态三步流程、旧工作台、情景发布及产品/组合应用浏览器回归 | **19 passed，2 个原有桌面限定用例跳过** |
| 单独执行旧情景缺 Horizon 字段的浏览器缺陷探针 | **再次复现 `Cannot read properties of undefined (reading 'terminal_status')`** |
| 月频持续期、延迟冲击、零变化非恢复、部分资格概率、推荐模板捕获准入 | **再次复现 F02–F05 所述问题** |
| TypeScript / design / i18n / build / git diff --check | **全部通过** |
| AI routing validator | **通过** |
| AI routing evolve | **退出码 1，74 个 uncovered_files** |

缺陷探针的断言是“证明缺陷存在”，其退出码 0 不表示产品验收通过。原样前端全量命令的退出失败也不能省略；排除目录后的通过属于显式隔离后的业务测试结果。

本次前端误收集来源是既有的 `frontend/test-results/dev-merge-audit-20260916/audit-repro.spec.ts`。这是审计产物隔离问题，不是新的业务实现；后续应固定测试收集范围或将诊断文件使用不参与 Vitest 收集的命名，不能只依赖 Git ignore。这里未删除或改写旧审计文件。

### 7.2 结论与证据位置

**本次复核继续维持“暂不通过”。** F01–F03 是优先修复项；F04–F05 必须在继续宣称 Horizon 已完成、或据此做长期研究准入前修正。Historical Reference 的完整区间数量提示不等于 LTCMA 估计有效性认证；历史参考自动交接、独立 LTCMA 消费接口和完整经济期限验证也不应列为已完成。

本次曾将输入文件哈希、四组后端命令/日志/JUnit XML、前端原始与隔离后结果、浏览器回归与缺陷复现、数值探针和静态/路由检查写入 `frontend/test-results/dev-audit-recheck-20260916/`。

**最终留档限制：** 在四组后端完成汇总、142 个既有文件哈希核对通过并补充本节之后，再次核对时发现该目录和此前的 `dev-merge-audit-20260916/` 都已不存在，而 `frontend/test-results/` 仍存在。清理来源尚未确认，本次没有执行删除这两个目录的命令。测试完成、缺陷复现与当时哈希核对的工具输出已经返回，但不能据此声称本地原始日志/JUnit/截图仍完整留存。最终的再次哈希核对因清单文件缺失未能完成。报告保留上述已观察到的事实，不重新伪造原始日志，也不将此次留档异常解释为业务测试失败或已被修复。

## 8. 本轮续审结果（2026-09-16，当前交付以本节为准）

### 同步与变更保护

重新执行 `git fetch origin --prune` 和 `git merge --no-edit origin/Dev`，结果仍是 `Already up to date.`。分支未切换，HEAD 与 origin/Dev 均为 `a4c30843870c7b457a273756557fe98c9a02bf29`，没有新建提交、推送或 PR，也未使用 stash/reset/clean。同步前后的 142 个既有变动文件逐一校验，合并操作没有改变其内容。

继续审核期间，末次哈希核对发现 `docs/research/pre-investment-dual-path-cma-saa-design-2026-09-16.md` 被其他操作修改；本轮未对该文件执行编辑，也没有覆盖它。其他既有变动文件与审核起点一致（本节写入前核对）。发现并发文档改动后停止进一步变更，仅终止了命令及输出目录可明确对应本轮审核的 Playwright 测试进程及其子进程；未停止用户原有开发服务。

### 本次实际验证，不与此前结果混记

| 检查 | 当前续审结果 |
| --- | --- |
| 后端四个完整回归组，共 47 个测试文件 | **896 passed；0 failures / errors / skipped**，已保存四份 JUnit XML |
| `test_historical_regime_indicator_source.py`、`test_research_series_routes.py` | 两个测试文件的运行超过本轮时限；拆分重试也未完整结束，**不计作通过** |
| 前端全量 Vitest | 日志最终显示 **146 files / 1119 passed**；但外层命令先在 125 秒超时，捕获退出码为 124，因此不能表述为本轮全量命令成功退出 |
| TypeScript、design、i18n、production build | **命令均退出 0**；保留现有大包及浏览器数据过旧提示 |
| 浏览器常规回归 | 出现页面交互超时及 Playwright 连接错误，外层命令亦超时；随后停止本轮遗留测试进程。**未获得有效的完整浏览器通过结论** |
| 旧预览浏览器缺陷探针 | 本次在点击“查看路径”之前超时，**没有再次完成崩溃复现**；F01 的代码缺陷及前轮复现记录仍保留 |
| F02–F05 合成边界探针 | **再次复现**推荐模板捕获准入被阻断、未授权状态概率参与消费、月频单观测记为1天、零增量被当恢复、滞后冲击被截断 |

F02 的本次探针只隔离执行入口校验，不冒充完整注册/捕获端到端测试。F03 的示例使用合成概率与偏离，不是实际交易记录。数值探针成功执行表示缺陷被复现，不表示对应业务验收通过。

本次复核进一步确认：所谓历史 Horizon Profile 是经验持续期统计；完整区间计数的 `conditional_estimation` 提示不等于 LTCMA 期限准入或收益分布估计有效性认证。独立 LTCMA Center、Historical Reference → LTCMA 正式消费接口和完整经济期限验证仍未完成。

### 留档和结论

为避免 Playwright 清理默认 `test-results` 影响证据，当前续审产物写在 Git 本地元数据目录，不进入源码和正式市场数据：

```text
.git/agent-audits/scenario-dev-20260916-final/
  worktree-before.json
  backend-targets.json / backend-groups.json
  backend-3.xml / backend-4.xml / backend-5.xml / backend-6.xml
  backend-*.log / backend-*.json
  frontend.log / frontend-checks.json
  typescript.log / design.log / i18n.log / build.log
  reproduce.py / reproductions.json
  browser-defect.log / browser-regression.log
  browser-defect/ / browser-regression/
  process-cleanup.json / summary.json
```

**审核结论仍为暂不通过。** 当前确认的五类代码/契约问题尚未修复，且本轮部分验证未完整完成。没有把此前的 916 项后端或 19 项浏览器通过记录冒充本轮重跑结果。本轮仅同步、审核、隔离复现和追加报告，未修复业务实现或重训正式模型。优先处理 F01/F03，再打通 F02，最后统一 F04/F05 的期限口径；修复后重新取得完整回归和浏览器验收证据。
