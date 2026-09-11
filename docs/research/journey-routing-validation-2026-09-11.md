# 配置研究链路的 AI Hermes 路由验收

日期：2026-09-11。范围：本次产品池 → 大类 → SAA → TAA → 产品配置优化中的共享上下文、标签页隔离、输入质量检查、TAA 预检、指标按需展开与对应测试。没有全量扫描或重写其他任务的未提交代码。

## 本次登记

只修改两个事实文件：

- `docs/repo_map.json`：`frontend_allocation_workflows` 补共享草稿/导航、标签页隔离、精简指标展示入口及测试；`backend_analytics_routes` 补固定签名输入检查；`tactical_allocation_workbench` 补训练可得时间预检、尺度异常门禁与保存情景实验的当前行为。
- `docs/task_routes.json`：只在 R30、R60、R86 添加对应定位关键词；没有复制模块文件列表或改动其他路线。

`AGENTS.md`、`docs/pitfalls.json`、治理策略和其他模块/路线保持检查开始时的内容。修改前后按 JSON 条目比较，确认只有上述 3 个模块及 3 条路线发生本次修改；SAA PID/PIT 同步修复及其他已有 dirty 工作未覆盖。

事实依据为当前调用代码、实际测试与命令结果。CodeGraph 自动同步关闭且未索引本次新文件，因此只用于首次导航；最终事实以当前文件为准。未将真实行情质量或正式 PIT 认证标为通过。

## 验证结果

- 对 `/tmp/journey-routing-scope.json` 中明确选定的 **39 个源码/测试路径**运行 `evolve_ai_routing.py --changed-file ... --json`：39 个覆盖，0 个遗漏，退出码 0。范围为原 32 个路径，加 4 个 TAA 组件、数值测试、前端夹具及 TAA E2E。最初遗漏的两个路径为 `backend/research_input_checks.py` 及其测试，现已登记。
- `evolve_ai_routing.py --routing-only --changed-file docs/repo_map.json --changed-file docs/task_routes.json --json`：通过。
- `route_task.py --route-id R30/R60/R86 --mode context --format json`：三条路线均可解析。
- `validate_ai_routing.py --skip-reproducibility`：通过，仅证明结构、路径、引用和回归命令可解析；不替代标准检查。
- **标准 `validate_ai_routing.py` 尚未通过**：检查前有 93 个未跟踪的稳定引用，最终为 99 个（加入指标折叠测试前为 98 个）。剩余问题全部是 Git 可复现性，没有路径不存在、结构或模块引用错误。
- `git diff --check`：本次路由变更通过。

新增的 6 个待纳入 Git 引用：

1. `backend/research_input_checks.py`
2. `backend/tests/test_research_input_checks.py`
3. `frontend/src/app/allocationJourney.ts`
4. `frontend/src/app/allocationJourney.test.tsx`
5. `frontend/src/layouts/StageLayout.test.tsx`
6. `frontend/src/components/HorizontalMetricComparison.test.tsx`

这些文件属于本次实现，不是假设性文件；目前尚未被 Git 跟踪，因而仅完成工作区路由登记，不能宣称 Git 可复现验收完成。本次没有增加 artifact 允许列表，也没有擅自暂存、提交或处理另外 93 个既有引用。正式提交范围确定后需再跑标准验证。

## 实际回归

```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_research_input_checks.py \
  backend/tests/test_tactical_allocation_service.py -q
```

25 个测试通过。覆盖实际 NJIT 固定签名、只读/非连续视图、尺度阈值边界、不可得训练标签、预检不自动改搜索、质量异常不能以固定比较绕过，以及具名情景保存恢复。

```sh
/bin/sh -c 'cd frontend && npm run test -- --run src/components/HorizontalMetricComparison.test.tsx src/app/allocationJourney.test.tsx src/layouts/StageLayout.test.tsx src/pages/ManualConstruction.test.tsx src/pages/ProductPoolSelection.test.tsx src/pages/ProductPools.test.tsx src/pages/ClassAllocation.test.tsx src/pages/PortfolioConstruction.test.tsx'
```

8 个测试文件、50 个测试通过。覆盖核心指标按需展开、跨标签页不串导航、范围元数据清理、按明确 universe 恢复及新快照身份隔离。SAA 交接保留范围研究日的最终加强断言另定向复跑 11 项。前端命令使用显式 `cd frontend`，使路由验证器能按真实工作目录解析测试路径。TypeScript 检查通过。

最终结构与覆盖证据：`/tmp/journey-routing-final-validate.txt`、`/tmp/journey-routing-final-structure.txt`、`/tmp/journey-routing-final-coverage.txt`、`/tmp/journey-routing-final-routing-only.txt`。SAA 范围日期断言证据：`/tmp/journey-saa-range-metadata-tests.log`。

临时证据：`/tmp/journey-routing-before-validate.txt`、`/tmp/journey-routing-after-validate.txt`、`/tmp/journey-routing-after-structure.txt`、`/tmp/journey-routing-after-coverage.json`、`/tmp/journey-routing-routing-only.txt`、`/tmp/journey-routing-tests.log`、`/tmp/journey-routing-frontend-tests.log`。这些是本机验收输出，不作为可复现源码引用。

## 最终趋势口径复核

最后一次窄范围维护只修改 `tactical_allocation_workbench` 的说明及定位符号：`available-window-relative-momentum/2.0.0` 按决策时已可得的最近完整连续窗口生成信号；未知或已过期窗口不生成信号，预检显示训练/验证有效信号期数，无有效训练信号时禁止强度选优。依据当前 `service.py`、`numeric.py`、数值/服务测试和 `TaaResearchContext.tsx`，没有修改业务代码。

最终 39/39 路径覆盖、routing-only、R86 解析及结构验证通过；标准验证仍只有上述 99 个未跟踪引用限制。证据：`/tmp/journey-routing-momentum-coverage.txt`、`/tmp/journey-routing-momentum-routing-only.txt`、`/tmp/journey-routing-momentum-R86.txt`、`/tmp/journey-routing-momentum-structure.txt`、`/tmp/journey-routing-momentum-validate.txt`。

主任务报告最终全前端 118 个文件、849 项测试及 2 项 E2E 通过；本路由任务没有重复执行该完整测试集。服务重启和实际页面复验由主任务完成，不能用路由验证替代。
