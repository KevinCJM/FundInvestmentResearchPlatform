# 历史参考与识别验证：前端验收

日期：2026-09-14。分支：ISSUE2609/BetterSaaTaa。

**最终更新：运行环境阻断已解除。本轮重新检查并修正 TypeScript 测试清理回调、固定工作区用途徽标和旧事件测试入口；最终前端 141 文件 / 1043 项单测、类型、构建、设计、i18n 均通过，四组浏览器测试 34 项通过、2 项按既有桌面限定规则跳过。**

本文件下文保留阶段记录，不把当时的环境失败改写成成功；最新命令、数值和仍未实现的部署级功能以 `regime-reference-confidence-final-audit-2026-09-14.md` 为准。

仅修改 frontend/src、专用 frontend/e2e 测试及本文。后端由另一 worker 维护。起始已有模板/测试、研究临时目录、设计文档与图片改动保留。未提交、推送、reset、stash、下载或修改真实数据。

## 文件与实现

- ScenarioCenters.tsx：历史状态定义 / 实时状态识别 / 情景模拟与压测 / 全球历史事件库四页签，方向键/Home/End 导航，各工作区懒挂载保留草稿。
- HistoricalRegimeWorkbench.tsx：唯一共享编辑器与执行路径；两项新任务固定模式，保留画布、公式、向导、节点预览、来源、实验、旧版本管理。新增用途、分类、来源、状态摘要与参考/验证组件。
- regimeStudy.ts：明确 URL 兼容、可选 study、新草稿用途、精确参考和状态映射。旧保存版本不自动改写；缺失 study 的序列化保持缺失。旧直达 workbench 保留显式双模式兼容。
- RegimeReferenceBinding.tsx：读取 references 目录；显示精确修订、频率、截至日、范围；数据源从该参考的精确定义读取。支持刷新、空/错误态和手动状态映射。参考只作评价目标，不进入特征图。
- RegimeReliabilityPanel.tsx：保存当前模型后，显式 prepare，再 preview；confirm 只发送 canonical request 和 preview_hash。按模型修订和精确参考筛选、重载已保存报告。
- RegimeReliabilityReportView.tsx：样本、排除、参考未知、预测拒识、完整区间、逐类表现、混淆矩阵、IoU、转折和延迟、概率评分/可靠性 ECharts 与表格。显示原始证据种类与校准方法，不把 one-hot=1 当可信度。原始分类覆盖与校准门槛后的覆盖分开呈现。
- RegimeSavePanel.tsx：历史参考复用保存定义 + enableRegimeResearchVersion；无第二参考存储。保存与发布迟到响应不能覆盖新草稿/PIT。
- ResearchContext.tsx：只读口径身份，区分未知与无截止日，覆盖同日期下运行模式和快照变化；不创建新默认值。
- regimeGraph.ts：按 handoff 精确请求类型、报告类型和 API 方法；保留原有错误封装与旧 serializer。
- RegimeDefinitionLibrary / RegimeNodePreviewPanel / RegimeTimelineChart：新任务移除模式切换，保留旧入口契约；原始概率补充未校准语义。
- 新增专用单测、离线报告夹具、regime-reference-confidence.spec.ts 和不监听端口的 regime-offline.config.ts。测试夹具未被产品组件导入。

## F1：任务入口与参考绑定

立即执行：

```sh
npm run test --prefix frontend -- --run ScenarioCenters.test.tsx regimeStudy.test.ts regimeDraftIdentity.test.ts --maxWorkers=2 --minWorkers=2
```

结果：3 文件、11 测试通过。随后参考空态/错误重试/精确绑定组件测试 2 项通过。

自审与修正：

- center=historical&mode=realtime 保持 definition/revision，定位实时任务；切换页签清除跨任务目标参数，草稿独立保留。
- 两项任务共享一个组件实现，没有复制工作台或算法。
- 新任务与节点预览固定 mode；旧直达路由保留原双模式和完整能力。
- 定义/模板按 study/default_mode/时点契约识别，不依据中文名、颜色或顺序猜状态。
- 参考改变清除映射与校准绑定；不同状态集合必须显式映射，原图不变。
- 新任务固定 mode + study 的实际请求另加 2 项工作台测试，已进入最终全量回归。

## F2：验证、报告、保存

立即执行：

```sh
npm run test --prefix frontend -- --run RegimeReliabilityPanel.test.tsx RegimeReferenceBinding.test.tsx --maxWorkers=2 --minWorkers=2
```

结果：2 文件、9 测试通过。初次失败为测试定位器匹配多段说明，修正为具体报告文本后通过。

后续自审：

- 未保存草稿不能验证旧修订；参考不可用、状态对应错误、分段日期错误均明确阻断。
- 模型、参考、映射、策略、PIT 身份变更或取消会中止请求并丢弃迟到响应。保存响应也不能覆盖后续编辑。
- 确认方法从客户端响应中仅提取 request/hash，禁止上传客户端 report。
- 保留全样本拒识分母；未知参考不补为震荡；null 指标显示无法估计。
- 展示后端 selective_classification，避免把原始接受覆盖冒充校准门槛后的覆盖。
- calibrated_confidence 直接展示服务端最终状态值，不在前端取 max(probabilities)。
- 保存的诊断报告可重载；校准采用是显式操作，需后端 deployment_eligible。当前后端不生成该资格，因此按钮禁用并说明原因。
- CI/bootstrap 未实现，明确显示无法估计；时点探测已执行不等于参数敏感性或部署检验通过。
- 已对照 handoff 最终说明，以及 backend reliability/contracts.py、references.py、routes.py、service.py、report.py、execution.py 的实际字段。未运行真实后端或真实数据。

## F3：最终检查

| 检查 | 实际结果 |
| --- | --- |
| 全量 Vitest | 140 文件、1041 测试通过，89.88 秒 |
| 后续仅测试配色修正的报告组件回归 | 1 文件、2 测试通过 |
| TypeScript --noEmit | 通过 |
| Vite production build | 通过，最后构建 6.31 秒 |
| design:check | 无回归，预算未调整 |
| i18n 检查 | valid=true |
| git diff --check | 通过 |
| 路由覆盖审计 | 主实现已覆盖；3 个新增测试/配置路径待登记 |
| 浏览器 320/768/1440 | 未完成，详见下节 |

命令：

```sh
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend -- --outDir /tmp/bettersaataa-regime-frontend-build
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
git diff --check
```

日志保存在 /tmp/regime-final-full-vitest.log、/tmp/regime-final-build.log、/tmp/regime-final-design.log、/tmp/regime-final-i18n.log。Vitest 有 React act 提示；构建保留既有大 bundle 警告，未调整阈值或引入依赖。设计检查曾发现测试夹具新增两种色值，已改用现有分类色，回归通过。

## 浏览器尝试与未验收边界

第一次使用既有 Playwright 配置：

```sh
npm run test:e2e --prefix frontend -- regime-reference-confidence.spec.ts --workers=2
```

Vite 无法监听 127.0.0.1:4173：listen EPERM，测试未开始。

第二次使用不监听端口的离线构建配置，页面、资源及 API 均由 Playwright 拦截提供：

```sh
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-frontend-build \
npm run test:e2e --prefix frontend -- --config=e2e/regime-offline.config.ts --workers=2
```

12/12 用例在 Chrome 启动阶段失败，0 个进入页面断言。日志为 browserType.launch: Target page, context or browser has been closed，进程 SIGABRT，并有 kill EPERM。未尝试提高权限，当前环境不允许申请提权。

已编写但尚未执行到的断言：

- 四任务与旧深链接、独立草稿、键盘页签与焦点。
- 无参考、目录错误、禁用与下一步。
- 不足/仅回顾评分/不可部署、确认保存、刷新后精确重载。
- 参考切换阻止验证旧修订，迟到报告不能出现。
- ECharts 实际绘制、数据表、320/768/1440 无页面横向溢出、既有 auditTextContrast 检查。

`--list` 成功列出 12 项。日志 /tmp/regime-browser.log、/tmp/regime-browser-offline.log；启动失败 trace 在 /tmp/bettersaataa-regime-browser-results。这些是环境失败证据，不是视觉验收结果。没有截图或对比度通过结论。

## 剩余事项

1. 在可启动 Chrome 的环境复跑上述 12 项，并复跑原 historical-regime-workbench.spec.ts；必要时修正真实布局问题。当前不能宣称浏览器验收完成。
2. 按最新后端契约，本版报告仅 retrospective_only / insufficient_evidence；未完成部署选择历史验证、CI/bootstrap，前端未伪造资格或区间。TAA 数值/消费者由后端 worker 负责，本次未修改其前端选择语义。
3. 全量 changed-file 路由覆盖审计提示以下路径未登记：frontend/e2e/regime-offline.config.ts、frontend/e2e/regime-reference-confidence.spec.ts、frontend/src/services/regimeReliability.test.ts。无 missing_required_files。该阶段明确划分了路由 JSON 的维护职责，因此仅记录交接项；这不是用户禁止更新路由。另须遵守新增稳定引用先纳入 Git 的约束；日志 /tmp/regime-final-routing-coverage.json。
