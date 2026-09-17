# 情景算法中心：研究版交付与集成审核

日期：2026-09-14。开发分支：`ISSUE2609/BetterSaaTaa`。

## 结论与范围

已实现并验收“历史状态定义 → 冻结历史参考 → 实时识别 → 参考匹配与校准诊断”的核心研究流程。**不是完整的部署级置信度系统，不能据此宣称实时识别已经具备投资预测能力。**

未执行 commit、push、合并、真实数据下载或生产研究成果发布；保留原有沪深300模板和图表改动。

设计见 `regime-reference-confidence-design-2026-09-14.md`，接口见 `regime-reference-confidence-api-2026-09-14.md`。前后端阶段记录分别见同目录 backend-acceptance、frontend-acceptance 文档。本文记录最后一轮协调审核，以本轮实际命令结果补充早期记录。

## 已实现

- 情景算法中心拆为历史状态定义、实时状态识别、情景模拟与压测、全球历史事件库。历史/实时任务固定各自运行模式，共用原有画布、公式、向导、模型和执行器，各任务保留独立草稿。
- 历史定义直接执行并生成区间，通过已有确认保存/发布流程形成精确参考。不额外创建第二套标注编辑器、PIT 系统或参考大数组存储。
- 实时模型可绑定精确参考运行、发布记录、hash 和显式状态映射。没有参考仍可探索，但不能伪造参考匹配准确率。暂不支持跨频率自动映射。
- 后端复用因果执行和时点审计；带拟合模型时生成分折样本外输出。校准与后续评分按时间分段，测试标签不参与校准参数拟合。
- 实现混淆矩阵、逐类 Precision/Recall/F1、Balanced Accuracy、逐日 IoU、区间一对一匹配、转折延迟和误报/漏检。参考未知与预测拒识分别统计；拒识保留在全样本分母中。
- 概率来源区分确定性编码与可核验的模型后验。Temperature 只接收可追溯的后验；规则模型使用同类历史平均。Brier、LogLoss、可靠性分箱和 ECE 不被称作获利概率。
- 报告可以预览、确认保存并按精确模型/参考重载。确认只接收规范请求及服务器预览 hash，重新核验数据；不信任客户端上传的报告或通过标志。
- 两个 TAA 消费路径共用校准检查。新 study 协议缺失、失配、过期或不可部署的校准明确回退；旧无 study 的冻结回放保留兼容契约。未更改 CMA/SAA 主体算法。

## 本轮发现并修复

### 1. 有参考样本不等于有校准样本

原校准门禁只检查参考类别及完整区间数。构造三类参考充分、预测全部拒识或仅输出单一类别的反例，实际复现 `fitted=True`。

已增加 `calibration_support_kernel`，仅在校准时间段统计可用的“参考—预测”配对，并检查参考轴和预测轴的逐类样本。Temperature 还排除非法/缺失概率。总量或类别不足时，`fitted=False`、校准概率为空，并显示 `insufficient_usable_calibration_predictions` 的中文原因。测试覆盖未来样本不影响统计、只读非连续视图、无效概率、空类和越界。

没有把未知预测补成震荡，也没有为了让测试通过而降低样本要求。

### 2. 类型检查不能由单测代替

最终独立 TypeScript 检查发现新 API 测试的 afterEach 返回 VitestUtils，不符合清理回调的 void 契约。已改成显式无返回值代码块。修复后类型检查和对应测试通过。

### 3. 页面分离后的用途标识

历史定义工作区可以复用本来支持实时运行的计算模板，但其徽标不应仍显示为“实时识别”。固定工作区按当前用途显示徽标；实时工作区仍排除仅支持事后的模板。新增两项组件测试。

### 4. 旧事件浏览器回归

旧人工事件测试寻找改名前的“历史情景识别”页签，在其他交互均完成后超时。已把两处入口断言更新为“历史状态定义”，保留事件编辑、重叠区间、预览、切换、结果保留和对比度断言，未删除或跳过失败用例。

## 实际验收结果

各命令有交叠，**不能把以下数量相加当作唯一用例总数**。

| 范围 | 结果 |
| --- | --- |
| 后端主回归：历史情景、TAA、组合引用及初版可靠性 | 417 passed；阶段日志记录 315.19 秒 |
| 后端扩展回归 | 434 passed / 1 启动夹具失败；失败定位为真实存储租约未隔离 |
| 修复启动夹具后的启动 + 可靠性 + TAA 复跑 | 77 passed；284.93 秒，包含此前失败的启动测试 |
| 本轮校准配对门禁补丁后的四个可靠性测试模块 | 30 passed；43.36 秒 |
| 最终前端全量 Vitest | 141 文件，1043 passed；82.73 秒 |
| TypeScript --noEmit | 通过，单独核对退出码 0 |
| Vite build | 通过；6.69 秒，已生成当前 frontend/dist |
| design:check / i18n | 通过；未调整设计预算 |
| 最终四组浏览器测试 | 34 passed / 2 既有桌面专属用例跳过；46.0 秒 |
| 现有 AI 路由结构验证 | 通过；新增文件覆盖登记仍有待办，见下节 |

浏览器覆盖 320、768、1440 宽度，实际运行页签和键盘导航、独立草稿、精确参考、空/错误态、报告保存重载、迟到响应、ECharts、表格、页面横向溢出与文字对比度。使用隔离夹具，不是生产数据上的统计有效性验收。

第一次本轮离线浏览器运行误指向旧 `frontend/dist`，因此仍看到旧界面，12 项失败；改用本次新构建后 12 项通过。最终又重新构建实际 dist，并通过默认 Vite 配置运行四组 36 项测试，其中 34 项通过、2 项按原规则仅在桌面执行。

截图由浏览器测试生成，位于 `frontend/test-results/` 下对应测试目录，文件名为 `historical-definition-workspace.png`、`reference-confidence-report.png`、`manual-events.png`。这些截图包含明确离线夹具，不是当前市场结论。

### 最终前端复现命令

```sh
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run build --prefix frontend
npm run test:e2e --prefix frontend -- regime-reference-confidence.spec.ts historical-regime-workbench.spec.ts regime-manual-events.spec.ts regime-temporal-event-library.spec.ts --workers=2
```

后端复跑先使用临时目录设置 `CUSTOM_INDICATOR_DATA_DIR`、`HISTORICAL_REGIME_DATA_DIR`，再用项目既有 Python 环境执行四个 `test_regime_reliability_*.py` 模块。不可让测试获取生产数据存储租约。

本轮日志：`/tmp/bettersaataa-regime-final-vitest.log`、`/tmp/bettersaataa-regime-final-types.log`、`/tmp/bettersaataa-regime-final-build.log`、`/tmp/bettersaataa-regime-final-design.log`、`/tmp/bettersaataa-regime-final-i18n.log`、`/tmp/bettersaataa-regime-final-browser.log`、`/tmp/regime-final-focused.log`。React act、依赖弃用和既有 bundle 体积告警仍需与失败区分，未通过压低阈值或吞掉退出码隐藏错误。

## 数值与血缘审核

现有和新增数值能力共用编译执行路径。新增校准样本统计纳入启动预热，当前可靠性模块共 10 个受检查的固定签名内核；只读、非连续视图和进程预热检查进入测试。Python 负责 I/O、类型边界和报告封装，不新增 pandas 数值回退。

早期 2 万行基准的 9 内核测量保留在后端阶段记录中，是补丁之前的实测，不冒充本次新增内核后的全链路测量。输入解码、状态编码、概率映射和结果分配仍会产生必要分配；不声称整个功能零拷贝。

旧定义不含 study 时保持旧序列化和 hash；旧成果不被重算覆盖。新的诊断数字只相对指定参考版本，不代表“真实牛市概率”。

## 尚未完成 / 不得宣称已支持

1. **前瞻部署级校准验证及可核验的模型/参考选择历史。** 当前服务明确不签发 deployment_eligible；保存的是回顾性实验报告。相应采用操作不可用，新协议 TAA 不以这些诊断结果放行。不能把“门禁已实现”写成“部署校准已完成”。
2. **相关时间序列的 bootstrap 置信区间。** 返回 unavailable，不填示例区间。
3. **新可靠性报告内的统一参数/窗口/种子稳定性汇总。** 当前报告执行因果探测；原有实验与稳定性工具继续可用，但报告不把未运行项目标成通过。
4. **跨频率参考映射、扩展概率来源及新识别模型。** 当前严格同频同对象；未经核验的组合/映射后验不直接做 Temperature。未新增 Student-t HMM、BOCPD 等模型。
5. **新增未跟踪文件的稳定路由登记。** 现有路由校验通过；覆盖审计列出新增测试/配置和任务文档。按路由技能要求，不把未纳入 Git 的源文件伪装成稳定引用，也未擅自暂存、提交或调整忽略规则来消除告警。提交准备时应连同对应源码、测试与路由登记一起处理。

因此本轮可以验收核心研究版，但**不能宣称原先讨论的全部部署级功能已经完成**。
