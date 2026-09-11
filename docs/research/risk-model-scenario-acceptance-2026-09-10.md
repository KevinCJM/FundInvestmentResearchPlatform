# 风险模型与情景模拟闭环：验收记录

日期：2026-09-10。对象：当前本地工作区；未提交、未推送，不代表生产部署。
设计依据：`risk-model-scenario-design-2026-09-10.md`。

## 1. 最终持久化规则

本轮根据产品要求统一为：**只有用户明确点击“确认发布”才保存研究成果。**

| 操作 | 是否写磁盘 |
|---|---|
| 风险模型“计算敏感度” | 否，只返回当前页面预览 |
| 宏观传导模型计算 | 否，只返回当前页面预览 |
| 债券价格 / 久期 / 凸度计算 | 否，只返回当前页面预览 |
| 情景路径预览 | 否，只返回当前页面预览 |
| 产品 / 组合“计算情景影响” | 否，只返回当前页面结果 |
| 确认发布风险模型 / 宏观传导模型 | 是，保存冻结运行、数组和发布记录 |
| 确认发布情景 | 是，保存冻结情景路径和发布记录 |
| 明确导入自有时序数据 | 是；这是数据导入，不属于未发布研究结果 |

预览返回确定性的 `preview_hash`。用户确认发布时，后端使用完整定义重新计算/编译并再次计算哈希；只有与刚才确认的预览一致，才允许原子落盘。参数、数据或上游模型发生变化时发布会失败，不会保存旧结果。

旧版本曾经落盘的 impact 结果不主动删除，`GET /impacts/{id}` 仅保留只读兼容；新版本不再创建 impact 文件，也不在正常页面提供“已保存压测结果”入口。

## 2. 当前交付能力

| 入口 | 当前能力 |
|---|---|
| 设置 → 风险模型中心 | ETF/基金多因子 OLS、时间留出验证、可选验证后重估、只读系数；固定确定现金流价格/久期/凸度；预览不保存，确认发布才保存 |
| 情景算法中心 → 情景模拟与压测 | 市场、宏观变量、宏观事件三个入口；两段宏观传导研究；预览不保存，确认发布情景才保存 |
| 产品详情 → 风险与压测 | 只读取已发布敏感度和情景；计算产品影响但不自动保存 |
| 组合构建 / 持仓诊断 | 使用不可变组合运行 ID，由后端还原期末持仓；压测结果只存在当前页面 |
| SAA / TAA | 选择明确产品组合快照；可在同一情景和模型口径下比较两个组合；比较结果不自动保存 |
| 投后情景入口 | 使用真实研究组合快照，不把演示账户当真实持仓；结果不自动保存 |

敏感性研究归风险模型中心；宏观传导研究归情景中心；产品、SAA、TAA、组合和投后页面只消费已发布成果，不在那里重新训练模型。

## 3. 关键审核结论

1. 未确认预览不调用研究成果 `ArtifactRepository.save`；真实数据模型预览和直接市场情景预览均实测文件清单前后不变。
2. 发布请求不信任浏览器传入系数，而是提交完整定义 + 预览哈希；后端重新计算后校验一致性。
3. 负数、小数和空输入保持正确语义，不把 `-10.5` 逐字输入误成正数或零。
4. 风险模型的新鲜度使用真实联合样本 `data_as_of`，不能把请求日期改成今天就刷新模型有效期。
5. 现金流法只接受债券到期收益率平行变动契约，不把 Shibor、政策利率或任意 bp 因子等同于债券收益率。
6. 已发布成果消费阶段只加载必要的系数或情景路径，不加载训练面板，也不重新训练。
7. 精确发布版本失效时不静默替换；草稿变化、数据变化或上游模型变化后旧预览不能继续发布。
8. 组合使用最后一期收益后的期末持仓权重；任何非零持仓缺少暴露都阻断，不删除资产、不重新归一权重。
9. 买入持有与固定权重零成本再平衡是显式不同假设；资产与因子贡献必须与终值损益对账。
10. 确定性情景不生成伪概率、VaR 或 ES。
11. 产品/组合 impact 现在是瞬时结果，返回 `transient=true`，不创建磁盘 artifact；相同输入可得到相同内容 ID，但这不是持久化缓存。
12. 原历史重演、随机模拟、状态条件模拟和反向压力仍作为高级实验保留，不自动把人工 Beta 认证成已训练风险成果。

## 4. 自动测试

### 后端相关回归：148 项通过

```sh
python3 -m pytest \
  backend/tests/test_published_risk_models.py \
  backend/tests/test_scenario_stress.py \
  backend/tests/test_scenario_stress_routes.py \
  backend/tests/test_factor_research_numba.py \
  backend/tests/test_factor_research_service.py \
  backend/tests/test_portfolio_research.py \
  backend/tests/test_data_storage.py \
  -q --tb=short
```

结果：`148 passed`。其中新闭环专项文件 `test_published_risk_models.py` 为 `30 passed`。

专项新增/更新断言包括：

- 风险模型、现金流和情景预览后成果目录仍为空。
- 修改定义后使用旧 `preview_hash` 发布失败，并保持零成果写入。
- 确认发布后才出现冻结 run / preview / release。
- 产品/组合 impact 返回 200 的临时计算结果，磁盘 impact 目录不增加文件。
- 发布并发幂等、只读 mmap、制品校验、路径保护、固定签名 NJIT、`python_fallback=0`。

后端仅有现有 FastAPI TestClient/httpx 弃用警告，不影响测试断言。

### 前端相关回归：97 项通过

```sh
npm run test -- --run \
  src/pages/TacticalAllocationWorkspace.test.tsx \
  src/pages/HoldingDiagnosis.test.tsx \
  src/pages/PortfolioConstruction.test.tsx \
  src/services/riskModels.test.ts \
  src/pages/ScenarioCenters.test.tsx \
  src/components/risk-models/ResearchUI.test.tsx \
  src/pages/ScenarioAlgorithmCenter.test.tsx \
  src/pages/PublishedRiskFlow.test.tsx \
  src/pages/ClassAllocation.test.tsx \
  src/ProcessFramework.test.tsx \
  src/pages/ProductDetail.test.tsx \
  src/App.test.tsx
```

结果：12 个测试文件，`97 passed`。存在部分既有 React `act(...)` 警告，但无失败。

前端明确验证：计算阶段显示“当前预览 · 未保存”；确认发布前不会调用发布 API；确认发布时提交完整定义和 `preview_hash`；产品/组合结果展示为“本次压测结果”，不再提供新 impact 的保存/历史入口。

### 浏览器：3 个尺寸全部通过

```sh
npm run test:e2e -- --config=playwright.risk.config.ts --workers=1
```

结果：`3 passed`，覆盖 320×800、768×1024、1440×1000。

浏览器链路覆盖：风险敏感度预览 → 确认发布 → 情景预览 → 确认发布 → 产品临时压测 → 产品详情读取已发布敏感度。断言确认研究预览阶段没有模型发布请求、情景预览阶段没有情景发布请求，并且不存在旧式 `/models/.../runs` 写入流程。

### 前端生产构建

```sh
npm run build
```

结果：通过。仅保留既有 bundle size、Browserslist 数据版本提示。

## 5. 真实本地数据无副作用验证

使用现有本地数据，不下载、不发布正式成果。

### 风险模型预览

对象：华泰柏瑞沪深300ETF `510300.SH`；市场因子：`cn-equity-csi300`；月频；2021-01-01 至 2026-08-31；验证从 2025-01-01 开始。

实际结果：

- `transient = true`
- `data_as_of = 2026-08-31`
- 发布候选 Beta：`0.9987637771569366`
- 独立验证 R²：`0.9969241817309038`
- `data/risk_models` 文件数：预览前 `0`，预览后 `0`
- 文件清单：完全一致

这说明真实行情回归预览没有落盘。数字仅用于工程验收，不是预测或投资建议。

### 直接市场情景预览

输入：沪深300单期 `-10%`。

- `transient = true`
- 内部路径：`-0.1`
- `scenario_stress/published` 文件数：预览前 `7`，预览后 `7`
- 文件清单：完全一致

目录中的既有文件来自之前开发过程，不在本轮自动删除；当前预览没有新增文件。

## 6. 持久化边界

当前正式保存对象只有：

```text
统一数据根目录/
├─ model_variables/                  # 用户明确执行的数据导入
├─ risk_models/                      # 确认发布后的产品风险运行、冻结数组、发布记录
└─ scenario_stress/
   ├─ transmission_models/           # 确认发布后的宏观传导运行与发布记录
   ├─ published/                     # 确认发布后的情景路径与发布记录
   └─ impacts/                       # 旧版本历史兼容；当前版本不再写入
```

未确认的草稿、模型预览、现金流计算、情景预览和新产品/组合压测不会成为磁盘研究成果。

## 7. 未清零项与范围边界

- 当前宏观传导仍是研究级多变量 OLS / 有限滞后条件统计响应，不是 BVAR/SVAR 或已经识别的经济因果模型。
- 尚未实现完整 Barra 协方差模型、含权债/违约全估值、流动性反馈、真实交易执行或自动调仓。
- 全仓独立 TypeScript 静态检查此前存在其他数据源测试文件的类型问题；未为本需求修改那些无关代码。当前 Vite 生产构建与本需求前端回归均通过。
- 工作区还有大量其他任务的未提交改动。本轮没有清理、回退或提交这些文件。
- 没有删除旧版本本地研究 artifact；删除历史数据属于独立数据治理动作，不在本需求范围。
