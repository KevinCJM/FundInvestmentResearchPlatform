# LTCMA 中心第一版：实施与验收记录

- 核验日期：2026-09-18。
- 范围：本地 `ISSUE2609/BetterSaaTaa` 工作区的 LTCMA 中心及其 SAA 交接；不代表生产部署、投资模型有效性或整个投前体系已完成。
- 入口：`/pre-investment/ltcma`。
- 对应设计：`ltcma-center-implementation-design-2026-09-18.md`；总体设计：`pre-investment-dual-path-cma-saa-design-2026-09-16.md`。

## 1. 已交付的功能

| 层次 | 实际交付 |
|---|---|
| 独立中心 | 已确认版本与草稿列表、搜索／方法筛选、分页、新建、复制研究、只读详情、明确停止新引用 |
| 研究工作台 | “研究输入 → 结果与确认”；选择产品大类或独立战略范围；草稿可不完整，正式计算前严格校验；输入变化使预览及确认失效 |
| 方法 | 直接假设、历史统计、Black–Litterman、Bayesian NIW、人工情景、历史状态；后两项属于同一情景／状态方法族 |
| 结果 | 同一资产轴上的收益、波动、相关矩阵、不确定性解释；样本与来源、限制；历史占用率与应用概率分列，零权重未估计状态保留 |
| 后端 | CmaResearchService 为唯一研究编排；共享取数／证据与固定签名数值内核；复用 ArtifactRepository、现有 BL／情景混合及冻结读取；旧 API 为薄委托 |
| 生命周期 | 只读预览、明确确认、预览指纹核对、V2 幂等发布、不可变结果；草稿乐观版本检查；停止新引用不删除历史结果 |
| SAA | 只选择已确认 LTCMA，不重复编辑模型；V2 CMA 不绑定 Mandate／实施映射；映射在 SAA 创建政策时单独选择；实际 TAA／产品门禁保持 |
| 清理 | 删除已无调用者的 AssumptionEditor、EffectiveCmaResult、ModelAssumptionFields、ForwardAssumptions；迁移三组旧浏览器回归，不保留隐藏的旧编辑流程 |

主要实现位置：`backend/strategic_allocation/cma_service.py`、`cma_evidence.py`、`cma_statistical_models.py`、`cma_statistical_kernels.py`、`cma_application.py`，以及 `frontend/src/pages/Ltcma*.tsx`、`frontend/src/components/ltcma/`、`components/strategic-allocation/LtcmaSelection.tsx`。

## 2. 关键审核与修复

1. **战略代理切回产品大类的残留输入。** 原 `applyScope` 保留了统计模型的旧 `proxy_inputs`，与产品大类来源冲突。补了 Historical／NIW／Regime 三个负向复现，先失败再修复；现切换时清除代理，保留显式方法，不替换数据来源。
2. **过期日历造成窗口静默截尾。** 原逻辑在日历和净值同时截短时可能接受更短样本。新增请求结束日的日历覆盖检查，并保留明确闭市日期；同时测试真实覆盖的闭市结束日不被误拒。
3. **状态结果可解释性。** 来源冻结状态名称，前端直接展示历史占用率、应用概率、观察数和估计状态；缺失不填零，未估计不展示成功；手机宽表有横向滚动提示，重复警告仅在展示层去重。
4. **旧浏览器消费者。** 把战略范围、资金目标和 BL／情景研究用例从旧 SAA 编辑按钮迁至独立 LTCMA，再验证真实发布、SAA 风险预算和 TAA 冻结参数传递；没有删除原断言所覆盖的业务能力。

核查了保存版本不重算、过期／退役阻断新用途、输入轴与时点、NIW 先验与新增证据、Student-t 分位数、条件矩混合及年化、零波动与缺失处理、实际 NJIT 调用、只读和非连续数组。

## 3. 实际测试证据

| 检查 | 结果 | 日志 |
|---|---|---|
| 后端相关完整回归 | 27 文件，639 项通过 | `.tmp_ltcma_20260918/continuation-backend-full.log` |
| 同工作区并行目标修改后的兼容补测 | 161 项通过，与上一组有重叠，不相加 | `continuation-final-compatibility.log` |
| 前端全量单测 | 157 文件，1233 项通过 | `continuation-frontend-final.log` |
| LTCMA 真实浏览器流程 | 15 项，320／768／1440 三种尺寸 | `continuation-browser-final.log` |
| 原战略／历史前沿／资金目标浏览器回归 | 10 项通过，含真实 20／200 点前沿、SAA→TAA | `continuation-strategic-browser.log` |
| 战略范围与 BL／情景完整交接 | 15 项通过，三个尺寸 | `continuation-better-saataa-browser.log` |
| 独立战略与实施映射流程 | 1 项通过，内部检查三个尺寸 | `continuation-m1-browser.log` |
| TypeScript、生产构建、前端设计与语言检查 | 通过 | `continuation-build.log`、`continuation-design.log`、`continuation-i18n.log` |
| Pydantic→TypeScript 合约 | `export_ltcma_contracts.py --check` 通过 | 命令输出 |
| 现有 AI 路由结构验证 | validate 与 R87 context 通过 | `continuation-routing-validation-final.log`、`continuation-route-context.json` |

浏览器合计 41 个用例通过。使用隔离的临时数据和本机 loopback API，不操作真实业务数据。覆盖创建、编辑、预览、确认、刷新后只读、草稿恢复／删除、停止新引用、错误／空态、风险门禁、文字对比度和页面无横向溢出；已查看桌面列表和手机状态结果截图。

首次续测中曾有一次 Chrome `ERR_NETWORK_CHANGED` 导致多个静态模块加载失败，已保留 trace 并完整重跑通过；未放宽断言或用自动重试掩盖。迁移旧测试时出现的标签定位不匹配也已修正为当前可访问控件名称。手机宽表提示和警告去重完成后，再次执行 LTCMA 全部 15 项浏览器测试、生产构建及静态检查，均通过。

仍有非失败警告：既有 Numba 非连续矩阵性能提示、Starlette 测试客户端弃用提示、部分 React 测试 act 提示、浏览器兼容数据库过期，以及现有大前端 bundle 警告。没有为消除这些警告升级依赖或扩大重构范围。

### 可复现命令

使用项目认可的 Python 环境，先设置 `PYTHONPATH=.:backend`：

```sh
python -m pytest backend/tests/test_ltcma_lifecycle.py backend/tests/test_ltcma_statistics.py backend/tests/test_ltcma_evidence.py backend/tests/test_cma_models.py backend/tests/test_cma_model_integration.py backend/tests/test_strategic_allocation.py -q
python scripts/export_ltcma_contracts.py --check
python backend/tests/benchmark_ltcma.py
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run test:e2e --prefix frontend -- --config=playwright.ltcma.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.strategic.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.better-saataa.config.ts --workers=1
npm run test:e2e --prefix frontend -- --config=playwright.m1.config.ts --workers=1
python skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```

639 项的扩大回归包括 `backend/tests/` 中 `test_ltcma_*`、`test_cma_*`、`test_strategic_*`、`test_risk_scale_*`、`test_mandate_*`、`test_investment_mandate*`、`test_bettersaataa_*`、`test_tactical_allocation_*`、`test_tactical_walk_forward*`、`test_frontier_grid*`，共 27 文件。

## 4. 性能与内存观察

固定合成样本 10000×30，输入为只读非连续共享视图；预热后各执行五次：

| 内核 | 中位执行时间 | Python／NumPy 可追踪分配峰值 |
|---|---:|---:|
| Historical | 55.798 ms | 15838 bytes |
| NIW | 55.183 ms | 30800 bytes |
| 状态条件统计 | 5.731 ms | 23128 bytes |

`np.shares_memory` 为真，输入内容未改变，请求期间签名数量增加为 0，`python_fallback=0`。这是本机一次可复现观察，不是跨硬件承诺或相对旧算法的加速倍数。计时不包含预热；tracemalloc 不是完整 native heap 测量。进程全生命周期 RSS 高水位约 355 MB，包含启动和输入准备，不能当作单内核内存占用。脚本为 `backend/tests/benchmark_ltcma.py`。

## 5. 当前边界与提交前事项

- 统计证据当前支持 CNY、SSE 日频和 252 年化。跨币种／FX、月频、多期状态转移、所有先验强度模板与衰减策略不属于本版已实现范围。
- BL 的参考权重、风险价格与利率仍需明确来源或人工输入；没有假装已接入全市场市值与自动机构预测。
- 数学上确定的零波动现金在支持它的路径显式处理；NIW 正定先验要求不因现金存在而以假噪声绕过。
- 历史净值或指数来源的实际收益口径和 PIT 局限持续披露；算法通过不证明未来预测有效，更不保证投资结果。
- 多 CMA 参数融合／联合约束、无映射 class-level TAA 和最终产品实施新契约仍是后续工作。下一步优先模式 A，不直接开发模式 B QCQP。
- 本轮识别到同工作区投资目标页面、其测试与共享 contracts 的并行修改，已保留且不覆盖；兼容补测通过不等于代替该并行任务进行需求验收。
- 未执行 git add、commit、push、merge 或生产部署。新 LTCMA 文件目前尚未 Git 跟踪。按 AI Hermes 规则，未将这些未跟踪源码／测试路径伪装成可由 Git 复现的稳定路由引用；已更新现有模块事实、关键词和隐藏契约。提交前需将授权的新文件纳入 Git 后，补齐精确的路由文件／测试清单并重新执行 coverage。当前 `evolve_ai_routing` 的未覆盖新文件提示已明确保留，不称其全部通过。

## 6. 2026-09-20 分支审核修复

本节为 `e5d72fb` 之后的四项修复；上文的测试数量及提交状态保留为 2026-09-18 的历史记录。

| 问题 | 修复与回归边界 |
|---|---|
| 大研究包保存成功但无法读取 | 共享 ArtifactRepository 在提升目录、登记索引前检查最终 UTF-8 字节数；写入与读取共用 8,000,000 字节上限。覆盖含中文对账的真实候选、恰好上限、超出 1 字节、幂等重试、临时文件清理和既有列表／历史可读 |
| NIW 先验在计算后停止引用仍可发布 | 发布最终检查与保存共用生命周期锁；覆盖计算后退役、写入时跨文件描述符锁排他，以及发布完成后先验退役仍可幂等读取原后验 |
| 切换后验续更携带隐藏先验强度 | 切换清除均值／协方差先验观察数和复用确认；续更请求不携带旧强度，切回重设时显示空输入并要求补全 |
| SAA 深链接覆盖手动选中 CMA | 初始引用只消费一次；覆盖异步读取第二版本、清空，以及实际政策预览提交所选版本 |

本轮测试：相关后端 **508 passed**；全前端 **161 文件、1299 passed**；新增组件测试调整为先通过 `completeCma` 再序列化，并单独复验通过。TypeScript、生产构建、设计检查、语言检查均通过。数值内核未修改。

扩大审核补测：QP／前沿、资金目标、TAA walk-forward、双路径交接及风险预算 **122 passed**，与上述 508 项的文件不重叠，合计 **630 项后端测试**通过。

Chrome 离线浏览器 **18 passed**，1440×1000、768×1000、320×900。覆盖真实 NIW 新证据续更并保存、CMA 深链接后异步切换及所选版本的政策计算；各检查状态无页面横向溢出，对比度检查通过，已查看桌面与手机的相关截图。路由 validate／evolve 和 LTCMA 生成契约核对通过。CodeGraph 已同步；未把测试证据视为真实投资模型或生产部署认证。

复现命令：

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_implementation*.py backend/tests/test_funding_continuation.py backend/tests/test_ltcma*.py backend/tests/test_cma*.py backend/tests/test_multi_cma*.py backend/tests/test_strategic_allocation.py backend/tests/test_mandate_diagnostic_integrity.py backend/tests/test_tactical_allocation_service.py backend/tests/test_tactical_allocation_bridge.py backend/tests/test_published_risk_models.py -q
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_optimizer_strategy_numba.py backend/tests/test_frontier_grid.py backend/tests/test_tactical_walk_forward.py backend/tests/test_investment_mandate.py backend/tests/test_bettersaataa_m1.py backend/tests/test_bettersaataa_scope_journey.py backend/tests/test_strategic_risk_budget.py backend/tests/test_bettersaataa_final_audit.py -q
node frontend/node_modules/vitest/vitest.mjs run --root frontend
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run test:e2e --prefix frontend -- --config=playwright.ltcma.config.ts --output=/private/tmp/bst-fix-browser-final
```

容量门禁防止新写入形成不可读成果，不迁移或重写已有超限文件。浏览器及 API 测试使用本地隔离夹具，不访问正式账户或市场数据。
