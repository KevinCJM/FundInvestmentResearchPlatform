# M3 里程碑证据（2026-09-14）

M3 后端与前端已实现，当前分支保持 `ISSUE2609/BetterSaaTaa`。后端完整相关回归 147 项通过；浏览器验收尚未完成（沙箱不能监听端口）。未提交、切换分支、修改正式数据、AGENTS、四份共享文档或路由 JSON；其他工作者改动未覆盖。

## 实际修改

- `backend/tactical_allocation/contracts.py`：可选 `decision_policy`，日/周/月/季决策及执行机会、滞后、最小持有期、权重阈值、独立费用口径；有来源/方法/完整资产轴的日期化标准信号及组合权重；实际持仓日期。
- 新 `clocks.py`：因果周期掩码、有效信号期数、信号到期、阈值、当前执行资格与共享 `validate_clock_application(preview)`。缺实际持仓/日期、等待滞后、持有期不足、未达阈值均有原因；旧持仓不能用于今天的交接。
- 新 `signals.py`：多窗口动量及 value/carry/macro/risk_sentiment 外部标准值融合。必需分量尚不可得/过期时整体失效；中性零不同于缺失；无缺失重归一化，无 NAV 伪价值/Carry。
- `historical_regimes/taa.py`：唯一 `_taa_recursive_kernel` 推进 SAA/TAA；非交易期持仓漂移；实际交易产生换手与费用；到期目标失效。原 `_taa_path_kernel` 是固定签名薄适配，直接返回原宽度连续路径，无第二递推引擎、无全路径兼容复制。
- `numeric.py`、`service.py`、`walk_forward.py`：训练、留出、多折复用时钟/信号/递推和独立初始化；选强度只用成熟训练标签。实际交易前及期末持仓的资产/分组/偏离越界进入候选可行性，保留越界期数；固定假设可保留不可应用诊断。全部新内核纳入固定签名、禁止再编译和 PID 预热审计。预览无持久化，保存重算并冻结新输入。
- `frontend/src/pages/TacticalAllocationWorkspace.tsx`、`frontend/src/services/tacticalAllocation.ts`、新 `components/tactical-allocation/TaaPolicySignals.tsx`、`TaaResults.tsx`、`TaaScenarioExperiments.tsx`：新研究显式时钟、渐进设置、多信号编辑/校验、原日频模式、执行状态/原因、实际漂移超界、过期持仓交接禁止。补齐当前真实政策 `assumptions` 类型用于 M1/M2 接口集成。
- 新 `backend/tests/test_tactical_allocation_clocks_signals.py`（22 项）、`backend/tests/benchmark_taa_clocks.py`、`TaaPolicySignals.test.tsx`；更新原页面测试和 `frontend/e2e/tactical-allocation.spec.ts`。旧交接夹具显式选择原 daily-target，以继续测试原合同；新增浏览器用例独立测试新模式。

## 明确的设计细化

1. 周/月/季使用周期首个已知共同观察点；历史区间首点视为首次评估机会。不是交易所执行日历。新模式滞后至少 1 个共同观察期；最小持有期也使用共同观察期。
2. `cost_basis` 独立显式选择：默认 `half_turnover` 保留原半数绝对权重变化费用分母；可选择 `gross_traded_weight` 按买入加卖出权重计费。改变时钟不会自动改变费率含义。审计记录费用口径，路径同时输出单边换手与双边成交权重。
3. 决策先按滞后成熟，再可执行；新决策不会不断重置旧目标的成熟时间。当前拟议权重取最近成熟决策，另报最新决策日期、是否仍等待及本次执行机会。
4. 实际持仓预算按交易前和期末检查，目标可行不代表全路径可行；不为避免事后越界而提前使用未来信息交易。原省略策略请求保持原结果。
5. 外部标准值严格在 [-1,1]，方法/来源由研究输入明确提供。动量用已知完整窗口收益中心化、按最大绝对偏差标准化；合成后中心化并限制幅度，随后使用既有边界缩放。没有自动获取或生成外部研究模型。
6. 现有情景工具继续是“当前目标 + 指定冲击/历史收益、每期恢复目标”，显式标为 `current_target_fixed_stress`；使用所选费用口径，不冒充新时钟的策略路径重演。

## 测试与审核

后端命令前缀统一为：

```sh
CUSTOM_INDICATOR_DATA_DIR=/private/tmp/bettersaataa-m3-tests PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12
```

|命令（后端接在上述前缀之后）|结果|
|---|---|
|`-m pytest backend/tests/test_tactical_allocation_clocks_signals.py backend/tests/test_tactical_allocation_numeric.py backend/tests/test_tactical_allocation_service.py backend/tests/test_tactical_walk_forward.py backend/tests/test_historical_regime_taa.py backend/tests/test_tactical_allocation_bridge.py backend/tests/test_tactical_allocation_data.py -q`|147 passed；原 HTTP、数据、版本、桥接与回测兼容测试包含在内。|
|`-m pytest backend/tests/test_tactical_allocation_clocks_signals.py -q`|最终只读数组细化后 22 passed。|
|`-m pytest backend/tests/test_tactical_allocation_clocks_signals.py::test_new_service_preview_frozen_inputs_and_clock_export_gate -q`|1 passed；共享桥接接入后直接调用同样拒绝缺失实际持仓，预览文件字节/mtime 不变。|
|`npm run test --prefix frontend -- --run TacticalAllocationWorkspace.test.tsx TaaPolicySignals.test.tsx --maxWorkers=2 --minWorkers=2`|30 passed。|
|`node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json`|最后一次通过。此前其他工作者开发中曾出现 SAA/scope 类型错误，未修改其文件。|
|`npm run build --prefix frontend`|通过；保留既有大包及浏览器数据库版本警告。|
|`npm run design:check --prefix frontend`；`node scripts/check_i18n.mjs`|通过；无设计规则回归，语言检查不等于全部文案已翻译。|
|`git diff --check -- <M3-owned paths>`|通过。|
|`npm run test:e2e --prefix frontend -- --config=playwright.taa.config.ts --workers=1`|未执行用例：启动 Vite 被拒绝，`listen EPERM 127.0.0.1:4199`。不声明视觉、对比度或浏览器流程已通过。|
|AI Hermes `evolve_ai_routing.py --changed-file ... --json`，只传 M3 文件|返回 uncovered_files：新增后端测试与 benchmark 尚未录入路由；未越权修改 JSON。|

兼容额外对照：从当前 Git HEAD `aa8ca7a8a707ed1cc10f4ef624f1ceb5894ebada` 只读提取原 `_taa_path_kernel`，在临时解释器中作受控参考；种子 3/19/91 × 费用 0/10/100 bps，100 期×4 资产，四组输出数组全部逐元素一致。参考实现未保存在生产代码。

新增覆盖：周/月/季与前缀稳定性；滞后、最小持有期、阈值等号；独立持仓财富参考与费用对账；未来数据扰动；外部可得/过期/中性/缺轴/NaN/bool；训练未知知识日期；独立留出不选优；多折；预览文件字节和 mtime 不变；冻结输入/旧版本；冷 worker 拒绝；只读/非连续内存共享及重复调用无污染；实际持仓超界阻止应用。

内存/时间命令：`backend/tests/benchmark_taa_clocks.py`。本机 2000×8、readonly、strides=(256,16)，`shares_memory=true`；输出 688000 B，tracemalloc 峰值 1396861 B；5 次约 0.19–0.30 ms，单一签名、fallback=0。进程峰值 RSS 392101888 B 包含导入/预热，不能当作单请求增量。必要分配仅输入解析/时钟元数据、信号及约束结果、持仓缓冲和输出；未宣称整个系统零分配。

## 集成与剩余验收

- 已观察当前 `backend/app.py:134` 调用 `tactical_service.warm()` 并检查 complete；本次未编辑 app。协调者仍须验证实际全部 worker/readiness。
- **桥接接入要求**：`portfolio_bridge.validate_decision_application` 必须调用 `from backend.tactical_allocation.clocks import validate_clock_application; validate_clock_application(preview)`，以覆盖 `/product-allocation` 以外的直接组合写入。报告完成时已观察 M1 在共享桥接加入此调用；该文件不属于本次修改。M3 service 也保留入口检查。M4 需跑完整产品写入集成。
- 协调者将新测试/benchmark 加入 `tactical_allocation_workbench` 的 `related_tests`/`minimum_regression`，同步设计细化与验收记录，再运行路由 evolve/validate。
- 协调者在允许监听端口的环境运行现有及新增 E2E，覆盖 320/768/1440、展开/禁用/保存/晚返回/键盘及实际文字对比度。当前仍不能称 M3 浏览器验收完成。
- 外部真实交易、交易所/基金执行时刻认证和投资有效性不在本次证明范围。
