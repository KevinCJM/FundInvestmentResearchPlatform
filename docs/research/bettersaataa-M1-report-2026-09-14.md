# M1 里程碑证据（2026-09-14）

设计第 2、3 节已实现生产后端、前端和隔离测试。当前分支为 `ISSUE2609/BetterSaaTaa`；没有提交、切换分支、修改正式数据、秘密、AGENTS、四份共享文档或路由 JSON。浏览器验收尚未完成，完整 TypeScript 检查仍有并行 CMA 集成错误，不能据此宣称整体验收通过。

## 实际修改

- `backend/strategic_allocation/contracts.py` 与新 `institution_contracts.py`：可选机构上下文、现金用途下限、同日同币种经济快照、五项人工核验记录。省略上下文不增加原请求的门槛；原目标诊断的 goal readiness、资金计划与模拟流程保留。
- 新 `institution.py`、`institution_kernels.py`：总资产和扣除已确认负债后的净资产诊断；缺失保持 null，未出资承诺单列，不并入资产或负债。两个独立固定只读签名 NJIT 内核，由 `service.warm()` 预热并校验 PID；禁止新签名和 Python fallback。
- `service.py`、`policy_gate.py`：现金仅计入显式 `role=liquidity` 且 `liquidity=liquid` 的资产；候选求解加入独立现金分组下限，TAA/应用共享门禁再次核对。普通可交易风险资产不满足该下限。人工核验未完成、未生效或过期时允许保存研究、拒绝当前应用；没有自动税务/监管引擎或独立审批声称。
- 新 `universe_contracts.py`、`universes.py`、`sources.py`：独立不可变战略范围和显式实施映射；保留稳定资产 ID、经济角色、流动性、币种、来源、代理理由、缺口。确认重算 preview_hash，读取校验版本类型；纯预览不写库。
- `routes.py` 新增 `/api/strategic-allocation/universes/{preview,confirm}`、`/universes/{id}` 和 `/implementation-maps/{preview,confirm}`、`/implementation-maps/{id}`；目录增加 `strategic_universes`、`implementation_maps`。沿用已注册 router 和现有 startup，不增加重复入口。
- CMA 接受 `strategic_universe_id`、可选 `implementation_mapping_id`，此路径 `alloc_name=null`；原 `alloc_name` 路径仍调用原实际产品源逻辑。无产品时支持手工前瞻 CMA、候选计算及不可变政策保存，不生成虚拟净值、不删除缺口、不重分配原预算。M1 此路径只接受 manual 风险来源。
- `backend/tactical_allocation/data.py`：完整映射才能进入真实 TAA 读取/应用；核对冻结范围/映射 ID、内容哈希、产品域、成员、资产轴和源配置。真实代理通过显式 ID 在唯一 I/O pivot 前换轴，不从展示名猜测身份；保存后的范围、映射、CMA、政策不回写。
- `portfolio_bridge.py`：直接产品交接也调用 M3 `validate_clock_application(preview)`（仅新 decision_policy 请求），封闭绕过 TAA service 的入口。保留原省略策略请求行为。
- `InvestmentObjectivesWorkspace.tsx`、`components/investment-mandate/*`：可选机构输入、经济诊断、诚实人工核验状态、独立战略授权；新组件为 `InstitutionalFields`、`InstitutionalResults`，旧目标流程保留。
- `ProductPoolSelection.tsx` 与新 `components/strategic-scope/*`：原产品池/手动/自动构建路径仍在；增加 `?scope=strategic` 的独立范围预览、确认、只读历史、复制编辑与真实代理映射。输入变化/切换/卸载使旧异步响应失效；修复原产品范围确认的迟到响应覆盖问题。
- `StrategicAllocationWorkspace.tsx`、`services/strategicAllocation.ts`、新 `services/{strategicScope,institutionalContext}.ts`：独立范围选择、无产品手工前瞻输入、可空真实方案类型、实际应用阻断原因与不完整映射的 TAA 禁用。保留原历史风险参考和假设编辑器。
- `app/allocationJourney.ts`：增加目标/战略范围/映射引用；产品域变化保留独立战略定义，改变依赖时清除下游活动引用。草稿仅存定义及已保存 ID，不存预览计算证据。
- 新 `backend/tests/test_bettersaataa_m1.py`、隔离浏览器夹具 `bettersaataa_m1_app.py`；新范围/机构/journey 测试、产品池迟到响应回归、`frontend/e2e/bettersaataa-m1.spec.ts` 和 `frontend/playwright.m1.config.ts`。

## 必要设计细化

1. 机构上下文存在时，五项核验主题必须完整保留；`researcher_checked` 和 `not_applicable` 都需要理由、证据、核验日和有效期，避免删除主题或空写“不适用”绕过门禁。全部记录仍仅是研究员记录，不是独立审批。
2. 一个真实代理大类在同一映射内不能重复分配给不同战略资产，避免相同产品承担重复战略预算。部分映射可保存，但仍拒绝 TAA 和应用。
3. 补齐映射后创建引用新映射的新 CMA/政策；不把映射补写进旧无产品政策。普通流动性下限与现金用途下限分别执行。
4. 真实产品读取在 I/O 边界进行必要的过滤/对齐/pivot 分配；新战略换轴在同一次 pivot 前完成，不复制一套完整历史数值矩阵。未声称整条既有数据链零分配。

## 测试证据

后端使用指定解释器、`PYTHONPATH=.:backend`，测试数据全部来自临时目录：

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_bettersaataa_m1.py backend/tests/test_bettersaataa_scope_journey.py backend/tests/test_strategic_allocation.py backend/tests/test_investment_mandate.py backend/tests/test_mandate_diagnostic_integrity.py backend/tests/test_tactical_allocation_data.py backend/tests/test_tactical_allocation_bridge.py -q
npm run test --prefix frontend -- --run InvestmentObjectivesWorkspace.test.tsx StrategicAllocationWorkspace.test.tsx ProductPoolSelection.test.tsx allocationJourney.test.tsx allocationJourney.m1.test.tsx strategicAllocation.test.ts StrategicScopeWorkspace.test.tsx InstitutionalFields.test.tsx
```

- 后端 **149 passed，39.68s**；前端 **70 passed，3.46s**。包含原目标/资金门禁、旧请求、纯预览、不可变读取、哈希冲突、缺失/NaN/Inf/bool、现金边界、人工证据过期、错误域/映射/代理、真实 TAA 服务接入、显式换轴、导航依赖和迟到响应。既有后端 HTTP/dateutil 与 React act 警告仍在。
- `npm run build --prefix frontend` 通过，保留既有大包警告；`npm run design:check --prefix frontend` 无回归；`node scripts/check_i18n.mjs` 无错误。静态检查不代替浏览器验收。
- `git diff --check` 通过。`skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` 通过；对 M1 文件运行 `evolve_ai_routing.py --changed-file ... --json` 提示新文件覆盖未齐，由协调者维护路由 JSON。
- 固定签名验收：`np.shares_memory=true`、只读 stride=16 输入、前后输入完全一致、每内核单一签名、冷 PID 拒绝。独立临时测量预热后 10000 次双内核约 0.0601s；2 元素结果 16B，tracemalloc 峰值 1398B（仅这次小内核调用循环，不是全服务峰值），fallback/object mode/request compilation 均为 0。可复跑正确性入口为 `test_readonly_strided_numeric_reference_and_readiness`。

## 未完成验收与集成要求

- **浏览器受环境阻断**：`npm run test:e2e --prefix frontend -- --config=playwright.m1.config.ts` 在 API 预热后因 `[Errno 1] ... bind ... ('127.0.0.1', 8129): operation not permitted` 退出，未执行浏览器用例。协调者须在可监听环境运行；脚本已覆盖 320/768/1440、真实目标→无产品 CMA/SAA→显式映射、只读/复制、键盘和文字对比度。没有合格截图或视觉通过结论。
- **完整 TypeScript 尚未通过**：`node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json` 最近错误涉及并行新增 `EffectiveCmaResult.tsx` 引用尚未完成端到端集成的 `CmaPreview.effective_assumptions/model_result` 字段及连带 implicit-any、`RiskBudgetEditor.test.tsx` 的联合字典类型、`StrategicAllocationModels.test.tsx` 的 cleanup 返回类型。没有为通过检查而伪造模型输出；协调者完成 M2 实际模型服务/有效假设/前端集成后重跑。
- M2 必须在实际 CMA 计算/冻结/政策消费中接入其模型，而不是仅渲染模型组件；独立范围仍需保留上述稳定资产轴和手工无产品能力。保留 `preview_mandate` 的 goal readiness，并合并 `service.warm()` 中机构审计。
- M3 共享门禁已接入 `portfolio_bridge.py`。M4 继续验证完整产品写入交接和全部 worker startup/readiness；当前已验证真实映射进入 TAA 服务，但不是外部交易执行认证。
- 协调者将新 scope/institution 服务、组件、测试、浏览器夹具及命令加入对应模块路由，并同步以上必要设计细化和正式验收状态。工作日志位于 `/tmp/m1-{backend,vitest,build,tsc,design}-final.log`、`/tmp/m1-browser.log`、`/tmp/m1-routing-coverage-final.json`。
