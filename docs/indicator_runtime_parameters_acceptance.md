# 时序指标可变参数：自审核与验收记录

日期：2026-09-08。对应设计：`docs/indicator_runtime_parameters_design.md`。

## 1. 交付结论

前后端已实现系统识别可参数化输入、作者开放/共享参数、默认值与约束设置、页面运行覆盖、恢复默认、实际参数溯源。入口为指标中心及产品详情“时序指标研究”。旧固定指标、标量和多标量的原契约不被自动改为可调。

本记录只声明下列已执行的专项检查通过，不代表全项目测试、全量类型检查或部署验收通过。本轮未提交、推送、重启正式服务或改写正式指标数据。

## 2. 最终验证结果

| 检查 | 结果 |
| --- | --- |
| 后端专项回归 | 198 passed，1 warning |
| 前端专项回归 | 9 个文件、108 tests passed |
| 生产构建 | `npm run build` 通过；浏览器验收也使用重新构建的生产包 |
| 真实后端浏览器验收 | 桌面 1440px、窄屏 320px，2 passed |
| 国际化检查 | `npm run i18n:check`：valid=true |
| 修改格式检查 | `git diff --check` 通过 |
| 路由及语言 JSON 语法 | 通过 |
| 本次新增/修改的功能源码与新组件测试类型诊断 | 0 项 |
| 全量 TypeScript | 未通过，其他文件仍有 33 项诊断 |
| 全量 AI Hermes 路由校验 | 未通过，83 条引用路径未纳入 Git 的诊断 |

### 后端命令

在项目根目录使用 AGENTS 指定的 Python 3.12 环境执行：

```bash
python -m pytest \
  backend/tests/test_series_runtime_parameters.py \
  backend/tests/test_custom_indicator_time_series*.py \
  backend/tests/test_indicator_graph.py \
  backend/tests/test_custom_indicator_excel_export.py \
  backend/tests/test_custom_indicator_routes.py \
  backend/tests/test_typed_indicator_product_service.py \
  -q --disable-warnings --maxfail=3 --tb=short
```

覆盖真实 parquet 测试数据、MA 默认/覆盖参数、嵌套历史窗口、KDJ 三结果、部分参数覆盖、范围推导、版本默认值锁定、缓存命中与隔离、Excel 实际参数、API 严格类型、非法位置/值、画布往返。独立参考包括 pandas rolling 与既有 KDJ 内核。正式执行仍走预热 NJIT；测试确认不同参数复用相同编译计划、`python_fallback=0`、`request_time_compilation=0`，并用禁止编译的替身防止计算请求临时编译。

### 前端命令

在 `frontend/` 执行：

```bash
npx vitest run \
  src/components/indicator-parameters/IndicatorParameters.test.tsx \
  src/components/indicator-graph \
  src/pages/IndicatorStudio.test.tsx \
  src/pages/IndicatorStudio.roundtrip.test.tsx \
  src/pages/ProductDetail.test.tsx \
  --reporter=dot --silent
npm run build
npm run i18n:check
```

验证默认显示、输入未应用不计算、非法/空值不发送、恢复默认不改全局定义、版本切换、过期识别结果丢弃、旧计算响应不覆盖新参数结果、原指标中心和产品详情回归。

### 真实浏览器命令

在 `frontend/` 设置 `INDICATOR_TEST_PYTHON` 为项目 Python 3.12 解释器后执行：

```bash
npx playwright test \
  --config=playwright.indicator-output.config.ts \
  --project=desktop-1440 --project=mobile-320 \
  --grep '可变参数真实' --workers=2
```

使用临时隔离后端和测试数据，未访问正式工作区指标。完整流程为：固定均线 → 识别输入 → 开放窗口 → 默认值改为 7 → 保存新版本 → 画布参数节点恢复 → 默认 7 计算 → 本次改为 3 → 同一 NJIT 计划计算 → 导出参数为 3 的 Excel → 恢复默认 7 → 再读定义确认版本/默认值未被运行覆盖。检查无 pageerror、320px 页面无横向溢出。

截图与工作簿由 Playwright 写入 `frontend/test-results/` 对应案例目录；这些是本地测试产物，可能在下次测试时被覆盖。产品详情的新面板由组件及页面回归验证，本轮真实浏览器全流程针对指标中心，不宣称产品详情也做了相同真实后端端到端验收。

## 3. 自审核发现并已修复

1. 画布原上下文校验把参数代码视为未知数据变量：将声明参数加入独立类型上下文，保留节点身份及指纹。
2. Excel 编译器原来只接受字面量窗口：改为允许明确声明、已验证的本次运行参数；读取实际值，不退回默认值。
3. 旧固定指标的导出请求不应新增空参数字段：仅新参数契约发送覆盖字段。
4. 旧固定版本拒绝运行覆盖时的顶层错误代码被改变：恢复历史 `SERIES_PARAMETERS_FIXED_IN_DEFINITION`，保持已有 API 回归契约。
5. 缓存必须区分有效参数和展示点数；历史前取长度及输出范围不能继续使用默认值。
6. 参数设置未应用、画布未应用或旧请求迟到时，不能保存或展示错误上下文的结果。
7. 多个相同数值常量保持独立；只有显式关联才共享，候选绑定携带公式摘要并重新校验来源。

没有因参数值变化新增数值执行器或 Python 备用计算路径。保留旧数据契约的默认值冻结仅用于已有历史版本兼容，不是并行维护一套旧功能源码。

## 4. 尚未通过的全量检查与范围限制

全量 `npx tsc --noEmit` 仍有 33 项诊断，涉及旧测试夹具、`ResearchDataLab.tsx` 等非本功能文件。本次功能源码、新组件测试及涉及的图编辑源码没有类型诊断。未为消除全量报错扩大修改到无关模块，也未把生产构建通过当成全量类型检查通过。

`validate_ai_routing.py` 仍报告 83 条引用未纳入 Git 的问题，涉及既有因子研究、情景、ETL 等模块；本轮路由修改前后均存在该障碍。`evolve_ai_routing.py` 另外识别到新前端参数组件尚未成为稳定路由引用。已更新既有 R70 关键词和指标模块风险契约；遵守稳定引用必须 Git 可追溯的规则，未擅自暂存整个工作区以绕过检查。后续纳入 Git 时需同时补齐新增源文件、测试与文档的稳定路由清单，再执行完整校验。

第一批能力限于窗口/最少观察数、平滑、滞后/差分、裁剪边界；不是任意数字参数化，也不是逐日变化的窗口或自动寻优。页面接入限于指标中心预览和产品详情，未给标量评价方案等页面新增时序计算。Excel 是导出时有效参数的计算快照，不能通过手改参数单元格自动重建全部历史数据和滚动单元格范围。

工作区原有未提交、未跟踪的其他修改均予以保留；本轮没有 Git commit、push、reset、stash 或批量暂存操作。
