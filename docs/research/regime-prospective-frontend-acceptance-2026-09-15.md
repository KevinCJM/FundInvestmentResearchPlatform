# 前瞻验证前端接入验收

> 最终集成补充：父任务已修正浏览器就绪等待及真实保存入口的测试契约，4项前瞻浏览器用例全部通过，覆盖320/768/1440px和未通过拒绝路径。最终1103项前端单测通过。可使用仓库内 `frontend/e2e/regime-prospective.config.ts` 复跑；下文启动受阻仅是早期记录，最终见 [集成审核记录](regime-completion-final-audit-2026-09-15.md)。

日期：2026-09-15；工作区：`ISSUE2609/BetterSaaTaa`。

## 交付范围

- 已保存、拟合完成且匹配当前定义修订/精确参考的可靠性报告下，显示“前瞻验证”。未保存、未拟合、脏模型、无效定义或非活跃工作区不能登记。
- 显式登记只发送 `{calibration_id}`，采用后端默认冻结策略；界面说明 252 个参考轴位置、至少 120 个有效配对、95% 覆盖及逐类/完整块门槛。没有虚构高级字段。
- 显式“记录当前判断”发送 `{}`，由服务端决定最新合格 UTC 昨日观测；前端没有日期、概率、置信度或 eligible 输入。界面解释登记后无数据、过期、未知状态、已发布标签及下一步。
- 复用历史参考目录，仅提供同一定义 ID/修订、登记之后发布的新精确参考。检验发送 `{reference:{run_id,publication_id,content_hash}}`；来源、成熟时间和资格仍由服务端验证。
- 目录项作为冻结协议读取，实时状态取 `progress.latest_assessment`，不使用冻结协议的初始 pending 状态冒充当前结果。展示捕获数、最近观测、冻结登记时间、证据/原因、到期时间；重新导航载入报告和刷新进度只调用读取接口，不自动捕获或检验。
- qualified 后显式“采用已验证校准”，先读取资格凭据、核对协议/校准/模型绑定/到期，再通过可靠性面板向工作台同时回写 `study.calibration_id` 与 `study.qualification_id`。用户需保存模型新修订。旧报告 `deployment_eligible` 保持原值；真实兼容旧制品的原采用入口保留。
- 图、公式、参数、状态、参考和校准方法变更清除两种绑定，并使当前报告失效。旧定义序列化不新增空的资格键。
- 新 API 复用 `regimeGraph.ts` 的 `regimeRequest`，无第二套 HTTP 错误处理。新面板使用安全中文原因映射，未知错误不输出原始堆栈、路径或令牌。
- 定义、参考、报告、校准、PIT 和活跃状态共同决定请求会话。会话切换同步取消旧请求并丢弃迟到响应；写操作使用同步 ref 锁防止重复点击。资格读取中的旧采用操作也不能修改新模型。

## 本次文件清单

新增：

- `frontend/src/services/regimeProspective.ts`
- `frontend/src/services/regimeProspective.test.ts`
- `frontend/src/pages/regime-workbench/RegimeProspectivePanel.tsx`
- `frontend/src/pages/regime-workbench/RegimeProspectivePanel.test.tsx`
- `frontend/src/pages/regime-workbench/regimeProspectiveFixtures.ts`（仅离线测试引用）
- `frontend/e2e/regime-prospective.spec.ts`
- 本验收文档。

在已有未提交工作基础上增量修改：

- `frontend/src/services/regimeGraph.ts`：请求函数别名、可选资格字段。
- `frontend/src/pages/HistoricalRegimeWorkbench.tsx`：采用回写、编辑失效。
- `frontend/src/pages/regime-workbench/RegimeReferenceBinding.tsx`：修改映射清除两种绑定。
- `frontend/src/pages/regime-workbench/RegimeReliabilityPanel.tsx`：嵌入新面板、透传采用/校准方法失效回调。
- `frontend/src/pages/regime-workbench/RegimeReliabilityPanel.test.tsx`：校准方法切换回归。
- `frontend/src/pages/regime-workbench/regimeStudy.ts`、`regimeStudy.test.ts`：采用/失效辅助函数及旧序列化回归。

未修改后端、共享 dist、父任务的 `regime-full-stack.spec.ts` 或 `playwright.regime-integration.config.ts`。未提交、推送、重置或暂存。

## 已完成验证

| 验证 | 最终结果 | 日志 |
| --- | --- | --- |
| 定向 Vitest | 5 个文件、76 项全部通过 | `/tmp/regime-forward-scoped.log` |
| 全量 Vitest | 144 个文件、1103 项全部通过；82.13 秒 | `/tmp/regime-forward-full-tests.log` |
| TypeScript（定向后及全量阶段） | 两次最终执行 exit 0，无诊断 | `/tmp/regime-forward-tsc.log`、`/tmp/regime-forward-final-tsc.log` |
| design:check | 无回归 | `/tmp/regime-forward-design.log` |
| i18n 检查 | valid=true，errors=[] | `/tmp/regime-forward-i18n.log` |
| Vite 构建 | exit 0；7.66 秒，产物仅在 `/tmp/bettersaataa-regime-forward-build` | `/tmp/regime-forward-build.log` |

最终命令：

```sh
npm run test --prefix frontend -- --run regimeProspective.test.ts RegimeProspectivePanel.test.tsx regimeStudy.test.ts RegimeReliabilityPanel.test.tsx HistoricalRegimeWorkbench.test.tsx --maxWorkers=2 --minWorkers=2
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run build --prefix frontend -- --outDir /tmp/bettersaataa-regime-forward-build
```

单测覆盖精确写入载荷、拒绝客户端时间/概率/资格字段、未保存/未拟合门禁、登记→捕获→qualified 和双 ID 回调、rejected/pending 禁止采用、目录/进度重载、参考过滤、全部异步入口在 PIT 切换后的取消及迟到结果丢弃、模型/报告/校准/参考身份切换、空值、中文错误脱敏、到期资格和旧序列化。夹具仅用于测试，不代表真实市场资格。

全量测试有既有 React `act(...)` 提示；Vite 提示部分 chunk 大于 500 kB，均未导致验证失败。首次定向执行发现测试断言使用了当前 Vitest 不支持的 matcher，改为兼容断言；上表记录修正后的最终结果。

## 浏览器验收：启动受阻，须父任务补跑

已新增 4 个全离线用例：320 / 768 / 1440 px 的登记、捕获、qualified、重载、显式采用并保存双 ID，以及 rejected 禁止采用。用例包含页面真实渲染的 `auditTextContrast`、全页横向溢出检查及截图。所有页面/API 请求由 Playwright route 拦截；不启动监听端口，不连接真实业务后端。

本环境执行：

```sh
node frontend/node_modules/@playwright/test/cli.js test --config=/tmp/bettersaataa-regime-forward.playwright.config.mjs --workers=1
```

4 个用例均在 Chrome 启动阶段失败（0–1 ms）：`browserType.launch: Target page, context or browser has been closed`，Chrome 进程 `SIGABRT`。页面步骤尚未执行，**不宣称三宽度视觉、对比度或浏览器流程已通过**。日志 `/tmp/regime-forward-browser.log`，失败 trace 在 `/tmp/bettersaataa-regime-forward-browser/`。未尝试突破沙箱限制。

临时配置（已写入上述 `/tmp` 文件，供父任务在可启动 Chrome 的环境复用）：

```js
import { defineConfig } from '/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa/frontend/node_modules/@playwright/test/index.mjs'
export default defineConfig({
  testDir: '/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa/frontend/e2e',
  testMatch: 'regime-prospective.spec.ts',
  reporter: [['list']],
  outputDir: '/tmp/bettersaataa-regime-forward-browser',
  use: { channel: 'chrome', headless: true, trace: 'retain-on-failure' },
})
```

默认读取 `/tmp/bettersaataa-regime-forward-build`；也可用 `REGIME_OFFLINE_BUILD` 指定隔离产物目录。该配置无 `webServer`，不修改父任务集成配置。父任务真实栈负责质量和可靠性集成，本前瞻 UI 用例仅离线模拟。

## 路由覆盖交接

已运行 `evolve_ai_routing.py` 的显式文件检查，检查 5 个文件、覆盖 3 个；新增 `frontend/src/services/regimeProspective.ts` 与 `frontend/e2e/regime-prospective.spec.ts` 未登记（exit reason: `uncovered_files`）。日志 `/tmp/regime-forward-routing.log`。路由 JSON 不在本次授权编辑范围，交由父任务统一补充。

本交付只证明上述前端与离线契约检查。没有真实前瞻采集、真实模型资格、统计显著性或真实市场准确率结论；服务端接线和 TAA 消费资格由父任务负责。
