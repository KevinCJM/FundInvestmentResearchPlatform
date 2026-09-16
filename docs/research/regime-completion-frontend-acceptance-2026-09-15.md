# 情景剩余能力：前端实施与验收

> 最终集成补充：父任务已完成18项新情景浏览器验收、4项前瞻浏览器验收、2项真实前后端联动和22项旧流程浏览器回归；最终1103项前端单测通过。早期启动受阻记录仍保留，最终结论以 [集成审核记录](regime-completion-final-audit-2026-09-15.md) 为准。

日期：2026-09-15。范围：`frontend/src`、专用 `frontend/e2e` 与本文。保留起始已授权脏文件；未提交、推送、重置、暂存或发布真实参考，未修改后端、主设计、AGENTS、路由与共享 dist。

## 实施交接

已完整读取 AGENTS、前端设计准则、R76 / frontend_regime_research 路由、剩余能力设计与上一轮最终审计；CodeGraph 用于代码定位。开始时后端交接尚未出现，接入 API 前读取了 `regime-completion-api-2026-09-15.md`，交付前再次复核，核对当前 contracts、routes、quality、diagnostics、bootstrap、report 实现。

- **历史质量**：独立 `RegimeQualityPanel`，按需检查已保存精确修订；未保存草稿不能拿旧修订检查。摘要显示覆盖、未分类、首尾、区间与转折，支持/持续时间/收益、参数扰动/窗口/种子/截尾与边界移动按需展开。收益缺失说明价格语义边界。质量报告预览、确认保存、目录与重载使用后端专用 API，不复制画布、不自动发布参考。
- **历史模板**：现有 API 目录与完整可编辑图继续复用，未写第二份 HMM/GMM/窗口/组合模板清单。增加服务端新增任意模板 ID 的目录回归。主流程保留“历史状态定义”“历史参考”；原有实时深链接、事件库与模拟中心保留。
- **实时证据**：按报告实际状态显示相关时序区间和参数稳定性；老报告继续显示未执行/不可用。默认“检查稳定性”与高级块重采样设置接入实际请求；块长度是观测数，完整时间块与状态周期分开。历史指标区间不解释为今日状态概率。所选状态概率、前两名间距、熵只展示服务器证据。确定性编码不解释成 100% 可信。
- **不可变与过期**：质量和可靠性请求使用 AbortController 与渲染身份检查；草稿、模型、参考、策略、截至日、PIT 上下文、离开任务或取消均使迟到结果失效。报告载入核对精确修订/参考，质量额外核对截至日。服务器规范请求、预览 hash、存储 hash 原样保留；确认 HTTP 只发送 request / preview_hash，绝不上传客户端报告。重载报告明确使用保存时参数与数据，上方设置用于下次检查。
- **界面**：复用 ui.tsx 原语、accent/slate、12px 与 40px 触控下限、tabular numbers；摘要先于表格/血缘，无新增依赖。错误原因转为中文；不适用、失败、预算不足与缺证据不标成成功。未取消任何部署门禁。

主要新文件：`services/regimeDiagnostics.ts`、`RegimeDiagnosticControls.tsx`、`RegimeDiagnosticsView.tsx`、`RegimeQualityPanel.tsx` 及配套测试夹具。仅在现有 workbench / regimeGraph / reliability 组件接线。`ScenarioCenters.test.tsx` 的一处同步断言改为等待事件库实际渲染，未修改事件功能。

## 阶段自审与测试

以下数量有交叠，不相加为唯一测试总数。

| 阶段 | 命令 / 结果 |
| --- | --- |
| M1 | `npm run test --prefix frontend -- --run RegimeReliabilityPanel.test.tsx RegimeReliabilityReportView.test.tsx RegimeDefinitionLibrary.test.tsx --maxWorkers=2 --minWorkers=2`：3 文件 / 13 passed；独立 tsc 通过 |
| M2 初次 | `npm run test --prefix frontend -- --run RegimeQualityPanel.test.tsx RegimeReliabilityPanel.test.tsx RegimeReliabilityReportView.test.tsx regimeReliability.test.ts RegimeDefinitionLibrary.test.tsx --maxWorkers=2 --minWorkers=2`：5 文件 / 25 passed；tsc 发现测试夹具 family 字面量、ByRoleOptions 的 exact 字段、保存报告 union 收窄问题，全部修正 |
| M2 扩展 | 相同命令增加 `HistoricalRegimeWorkbench.test.tsx ScenarioCenters.test.tsx` 后 56 passed / 1 事件页异步断言失败；改为 findByRole 后 `ScenarioCenters.test.tsx RegimeReliabilityReportView.test.tsx` 2 文件 / 12 passed |
| 全量第一次 | `npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2`：142 文件 / 1059 passed，77.90 秒 |
| 最终类型 | `node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json`：退出码 0 |
| 最终构建 | `npm run build --prefix frontend -- --outDir /tmp/bettersaataa-regime-completion-build`：退出码 0，8.67 秒；不覆盖共享 dist |
| 最终设计 | `npm run design:check --prefix frontend`：退出码 0，无预算回归 |
| 最终语言 | `node scripts/check_i18n.mjs`：退出码 0，valid=true |
| 格式 | `git diff --check -- frontend/src frontend/e2e docs/research/regime-completion-frontend-acceptance-2026-09-15.md`：退出码 0 |

最终全量复跑：`npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2`，**142 文件 / 1061 passed，84.19 秒，退出码 0**。新增测试覆盖：质量预览/确认/重载、源/模型/策略/PIT/截至日/离开工作区后迟到响应、旧新不可变报告、缺完整块与真实区间、种子不适用/变体失败、模型间距/熵、目录动态模板、默认控件及重采样有效次数约束。

日志：`/tmp/regime-completion-m1-unit.log`、`/tmp/regime-completion-m1-tsc.log`、`/tmp/regime-completion-m2-unit.log`、`/tmp/regime-completion-m2-tsc.log`、`/tmp/regime-completion-m2-recheck.log`、`/tmp/regime-completion-all-unit.log`、`/tmp/regime-completion-final-unit.log`、`/tmp/regime-completion-final-tsc.log`、`/tmp/regime-completion-final-build.log`、`/tmp/regime-completion-final-design.log`、`/tmp/regime-completion-final-i18n.log`。保留 React act、旧依赖与 bundle 大小告警，未降低阈值掩盖。

## 浏览器交接：尚未通过

已实际执行：

```sh
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-completion-build npm run test:e2e --prefix frontend -- --config e2e/regime-offline.config.ts --workers=2
```

18 个用例均在 Chrome 启动阶段失败，页面测试耗时 0 ms。错误为 `browserType.launch: Target page, context or browser has been closed`；Chrome 退出 `SIGABRT`，清理进程 `kill EPERM`。日志 `/tmp/regime-completion-browser.log`。这不是页面断言通过，也没有本轮截图验收证据；不重复尝试绕过环境限制。

最终 `REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-completion-build npm run test:e2e --prefix frontend -- --config e2e/regime-offline.config.ts --list` 成功发现 18 项测试（仅验证收集，不代表页面执行），日志 `/tmp/regime-completion-browser-list.log`。父任务可在允许启动 Chrome 的环境使用上述同一命令，使用最终临时构建复跑。配置无监听服务，页面资源从临时构建路由读取，API 全部固定离线夹具。6 场景 × 320 / 768 / 1440 共 18 项，含真实计算图节点、质量区间表、稳定性表、可靠性 canvas、保存重载、迟到响应、横向溢出与渲染文字对比度。截图/trace 输出 `/tmp/bettersaataa-regime-browser-results`。新增 `regime-full-stack.spec.ts` 属其他任务文件，本任务未修改或运行。

## 边界

本任务验证前端契约与交互，不认证后端数值、真实数据 PIT 或投资有效性。后端数值及集成审查由对应任务负责。质量报告不会自动发布真实参考，诊断报告不会获得部署资格。现有路由文件与新增文件登记留给父任务处理，本任务没有越权修改。
