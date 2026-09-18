# 风险标尺与投资目标提交审核

日期：2026-09-18。审核起点：当时 `origin/Dev` 与当前开发分支均为 `42f068fb655f45496d52501df3a971150270fa72`。

## 范围

- 风险等级配置中心：参考数据冻结、历史矩参数、受约束前沿、C1–C5、草稿、发布、默认、比较与退休。
- 投资目标与约束：列表与两步编辑、三种目标、现金预算、标尺引用、参考诊断、不可变替代与退出列表。
- 必要依赖：共享 QP、成果幂等存储、启动预热、正式 SAA/TAA 的目标约束消费、路由、翻译和测试。
- LTCMA 中心的源码、入口、模型与 SAA 选择器迁移全部排除。上位设计文档仅作为两模块共用的需求依据，不表示实现其中其余规划。

提交候选在仓库内隔离 worktree 组装；共享文件按内容拆分，测试不使用原工作区的 LTCMA 实现或正式数据。

## 审核发现与处理

1. SAA 独立验证种子由参考验证种子异或派生。原输入只检查搜索种子与参考验证种子不同；`seed=11, validation_seed=11 ^ 0x9E3779B9` 可使 SAA 重用搜索样本。新增输入拒绝和冻结旧目标的采纳期复核，并测试失败时不保存政策。
2. 目标摘要的待填写值使用 `text-slate-400`，违反浅底对比度规范；改用已有 `slate-600`。
3. 设计、路由说明与浏览器用例仍写三步/四步流程；同步为实际两步。旧目标资金及机构契约改由真实离线 API 建立，继续验证后续 SAA/TAA 和实施映射界面；新目标完整交互由专项浏览器用例覆盖。
4. 生成的 Mandate 类型漏了 `stated_benchmark`；按后端契约重新生成。
5. 补齐新增模块的 AI Hermes 入口、测试、生成器与回归命令，保留历史验收与本次复验的时间边界。

## 验证

- 后端范围回归：495 passed（Risk Scale、Mandate、SAA、历史 CMA、TAA walk-forward、前沿与共享成果存储消费者）。
- 种子修复后的 Mandate/参考诊断复验：52 passed，含两个新增边界用例。
- 前端范围回归：21 files / 161 tests passed；TypeScript、构建、design:check、i18n 通过。
- Risk Scale 浏览器：1 passed，覆盖五种分档、人工微调、草稿、发布、默认、只读及中英响应式。
- 目标浏览器：9 passed，覆盖桌面/平板/手机、中英、增改删、双前沿、键盘、真实文字对比度。下游战略配置：10 passed；无产品战略范围及实施映射：1 passed。
- Mandate / Risk Scale 生成契约检查、AI Hermes validate/evolve、暂存差异空白检查通过。

提交前 Dev 已前进到 `b63f1cf`（产品走势图修复）；五个改动文件中仅 repo_map 同名，修改的是不同模块，业务实现不与本提交交叉。PR 自审核仍按最新基线核对合并结果。

审核结论：本次范围内发现的问题已修复，相关验证通过，允许进入 PR 自审核。

测试使用临时目录和合成夹具，未发布真实业务成果。已有 Numba 布局性能提示、Starlette 弃用提示、浏览器兼容数据过期提示及构建大块警告不等于失败。源码和测试证据不构成金融模型有效性、历史 PIT 或实盘部署认证。

## 复现入口

```sh
PYTHONPATH=.:backend python3 -m pytest backend/tests/test_risk_scale_frontier.py backend/tests/test_risk_scale_independent_audit.py backend/tests/test_risk_scale_reference.py backend/tests/test_risk_scale_segmentation.py backend/tests/test_risk_scale_service.py backend/tests/test_risk_scale_storage.py backend/tests/test_mandate_boundary_contracts.py backend/tests/test_mandate_reference_diagnosis.py backend/tests/test_mandate_diagnostic_integrity.py backend/tests/test_investment_mandate.py backend/tests/test_strategic_allocation.py backend/tests/test_bettersaataa_m1.py backend/tests/test_bettersaataa_scope_journey.py backend/tests/test_bettersaataa_final_audit.py backend/tests/test_strategic_risk_budget.py backend/tests/test_cma_models.py backend/tests/test_cma_model_integration.py backend/tests/test_tactical_walk_forward.py backend/tests/test_frontier_grid.py backend/tests/test_published_risk_models.py -q
npm run test --prefix frontend -- --run InvestmentObjectives RiskScale riskScales flowMonths RiskScaleInputs RiskScaleResults ReferenceEditor StrategicAllocationWorkspace StrategicAllocationModels strategicAllocation CmaModelEditor PolicyCandidatesM2 StrategicScopeWorkspace StageLayout ResearchUI ClassAllocation FrontierGrid --maxWorkers=2 --minWorkers=2
npm run test:e2e --prefix frontend -- --config=playwright.risk-scales.config.ts
npm run test:e2e --prefix frontend -- --config=playwright.mandate.config.ts
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts
npm run test:e2e --prefix frontend -- --config=playwright.m1.config.ts
PYTHONPATH=.:backend python3 scripts/export_mandate_contracts.py --check
PYTHONPATH=.:backend python3 scripts/export_risk_scale_contracts.py --check
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```
