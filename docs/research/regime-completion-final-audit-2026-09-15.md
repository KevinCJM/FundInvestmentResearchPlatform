# 情景识别补齐：最终集成审核与验收

日期：2026-09-15。分支：`ISSUE2609/BetterSaaTaa`。

**后续收口已完成单一日频指数的固定快照续接**，最新测试为后端285、前端1108、浏览器49通过/2原有跳过，见 [数据续接最终验收](regime-source-continuation-acceptance-2026-09-15.md)。本文其余测试表保留本阶段的原始执行记录，不视为后续变更后的最终复测。

本文汇总最终工作区的真实执行结果，优先于各分工验收文件中早期的“待父任务接线”“浏览器受阻”状态；原始失败与修复记录保留。本轮未提交、推送、合并、reset 或 stash，未下载或修改生产数据。

## 一、已落地的业务流程

### 历史状态定义

保留唯一计算图、画布、公式、向导和系统级 PIT。保存精确模型修订后，可按需“检查划分质量”，查看分类覆盖、首尾未分类、逐状态区间数/持续时间、可解释的价格区间收益、参数与边界敏感性，确认后保存不可变质量报告并按版本重载。质量预览不等于发布历史参考，未知和未完成尾段不补成震荡。

增加四个可编辑组合模板，旧 PS、沪深300和宏观模板保持不变：

- `historical_hmm_risk_v1`：HMM 历史风险状态。
- `historical_gmm_volatility_v1`：GMM 历史波动分组。
- `historical_window_mean_change_v1`：窗口均值变化状态。
- `historical_trend_ensemble_v1`：历史趋势多窗口共识。

前两项使用波动特征，状态表示相对高/中/低波动，不冒充牛熊真值；窗口变化检测没有被包装成 BOCPD 或 PELT。模板的公式往返、节点及各输出预览有实际测试。

### 实时识别与信心分析

精确绑定历史参考，沿独立时间块验证。参数、窗口、适用随机种子和截尾检查复用同一诊断执行链。固定规则与需要拟合的模型分别执行因果回放/冻结训练边界的前向回放，不使用全样本拟合代替样本外预测。

原始证据、状态稳定性、参考匹配表现及校准概率分别展示。概率读取最终判断状态对应的分量，不简单取最大概率；规则 one-hot 仍不是100%可信。

新增成对移动块 bootstrap：仅在留出/最终测试段估计一致率、接受覆盖/错误率、Brier 和相对类别基准改善的统计区间。保留缺失的时间位置、固定校准器、固定随机种子；完整块、状态周期或有效重复不足时明确拒绝估计。区间条件于给定模型、参考及校准器，**不是今天“真实牛市”的置信区间，也不覆盖全部模型选择误差**。

### 前瞻验证与 TAA

新增显式操作：登记冻结候选 → 记录登记后的当时判断 → 等后续参考成熟 → 检验固定前瞻窗口 → 通过后显式采用校准和资格两个 ID。未完成、未通过、过期或来源不一致不得采用。

协议、捕获和检验记录使用本地签名追加日志，时间由服务端决定。已发布标签不能补录预测；源历史前缀修订、执行器语义改变、状态映射改变和客户端伪造时间/资格均被阻断。状态按独立资格记录读取，旧诊断报告的 deployment_eligible 不被改写。

已完成服务构造、路由、每进程预热、进度读取接口、Study 可选 qualification_id 和共享 TAA 消费器接线。资格只授权实际检验后、未过期且非未来的决策日期；不追認历史投资权限。旧无 study 冻结回放与非 Regime 信号保留原契约。

## 二、必须保留的使用边界

**单一日频指数固定快照续接已在后续收口中实现。** 系统保留原始绑定，通过显式检查/确认和历史前缀校验接入新源；仅数据变化的历史参考修订须单独生成并发布。模型、原参考、报告和旧文件不改写。上传、多源、宏观及跨频续接不在本版支持范围，不能由单指数验收推广为全部模型可用。具体契约和实测见上方最新验收。

历史质量、实时诊断、统计区间、受控前瞻流程及单指数续接均已实现；通用跨频映射、新 Student-t HMM/BOCPD/PELT、自动定时采集没有新增，CMA/SAA 主体也未修改。

本次前瞻通过标准是预注册的一致率、覆盖、逐类区间及完整块改善门槛，不声称统计显著性。实际市场资格需要真实后续数据；合成时钟测试只能证明代码和门禁，不代表任何真实模型已获得投资资格。本地签名保护针对 API/存储意外篡改，不等于外部独立审计或能抵抗掌握本机密钥的管理员。

## 三、父任务审核发现与修复

1. 稳定性变体可能只扰动已被显式参数覆盖的兼容项，造成假稳定；现在跳过 deprecated/shadowed 参数，实际检查峰谷左右窗口、阶段/周期等。画布 source.constant 的阈值纳入模型参数，外部行情数据源不被扰动。
2. 小窗口百分比变化被取整吞掉；新诊断明确至少移动一个观察期，原有实验数学契约不改写。
3. 执行器 code marshal 字节会受运行期引用状态影响，导致未改代码却误判指纹变化；改用规范化不可变代码语义记录，并保留真实代码改变的拒绝测试。
4. 确认固定快照不能增长后，保留原校验和门禁，明确来源模式和 pending 行为，不把兼容未绑定来源的测试推广为全部来源可用。
5. 质量/可靠性报告重载在模型初始校验未完成时可能被后续身份变更清空；选择与载入控件在未就绪期间禁用，浏览器按就绪状态交互。
6. 移动端画布允许已有可访问降级视图；验收检查真实节点内容，不把 React Flow 的 CSS 类当成唯一合法呈现。
7. 前瞻资格采用的浏览器测试修正为实际主流程“保存并用于研究”，核对请求内同时保存 calibration_id/qualification_id；没有为测试新增第二条保存入口。
8. 真实联动夹具只有720个日观测时，最后时间段仅有2个完整状态周期，服务正确返回区间不可估计；扩大隔离合成夹具至900观察验证可估计路径，没有降低生产统计门槛。

## 四、最终测试结果

所有后端写入使用临时目录。前端离线夹具与真实 API 集成明确区分。下表后端命令覆盖有交叠，不能合计为唯一测试数。

| 验证 | 最终结果 | 证据日志 |
| --- | --- | --- |
| 新增/受影响后端质量、稳定性、bootstrap、校准、前瞻、集成 | **96 passed**，66.83秒 | `/tmp/regime-completion-parent-final-backend2.log` |
| TAA、策略服务、产品桥与前瞻消费最终复跑 | **66 passed**，64.07秒 | `/tmp/regime-completion-final-taa.log` |
| 后端分工最终历史图谱/模板/TAA定向回归 | **206 passed**，87.91秒 | `/tmp/regime-completion-owned-final.log` |
| 前端全量最终复跑 | **144文件、1103 passed**，71.85秒 | `/tmp/regime-completion-final-vitest2.log` |
| TypeScript | exit 0 | `/tmp/regime-completion-final-tsc2.log` |
| 设计/i18n/隔离生产构建 | 均通过，未调整预算 | `/tmp/regime-completion-final-design.log`、`/tmp/regime-completion-final-i18n.log`、`/tmp/regime-completion-final-build2.log` |
| 情景/历史质量/稳定性区间浏览器 | **18 passed**，320/768/1440 | `/tmp/regime-completion-final-browser18b.log` |
| 前瞻登记/捕获/检验/重载/采用浏览器 | **4 passed**，含三个宽度与拒绝路径 | `/tmp/regime-completion-final-browser4b.log` |
| 隔离真实前后端质量与可靠性 API 联动 | **2 passed**，23.4秒 | `/tmp/regime-completion-final-realstack.log` |
| 原工作台、人工事件与时点事件库浏览器 | **22 passed、2 skipped**，26.7秒 | `/tmp/regime-completion-final-legacy-browser.log` |

浏览器共 **46项通过，2项原有桌面限定用例在非桌面项目跳过**。覆盖实际 DOM/图表或既有可访问降级、键盘、文字对比度、无页面横向溢出、精确版本保存重载及迟到响应。对比度脚本不等于完整 WCAG 人工认证。

早期广泛后端回归为782通过/1失败，失败发生在尚在修改的前瞻夹具；之后前瞻21项独立复跑和父任务96项集成均通过。保留原记录，不把不同时间的运行改写成一次全绿。

既有提示：React act、Starlette/httpx弃用及Vite大chunk警告；未因此降低门槛或加依赖。

## 五、可复现命令

```sh
# 后端必须先设置两项临时存储环境，不能指向生产数据。
TEST_ROOT=$(mktemp -d /tmp/regime-final.XXXXXX)
export CUSTOM_INDICATOR_DATA_DIR="$TEST_ROOT" HISTORICAL_REGIME_DATA_DIR="$TEST_ROOT"
export PYTHONPATH=.:backend
python3 -m pytest backend/tests/test_regime_completion_*.py backend/tests/test_regime_reliability_*.py backend/tests/test_regime_prospective*.py -q
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run build --prefix frontend -- --outDir /tmp/bettersaataa-regime-completion-final-build
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-completion-final-build npm run test:e2e --prefix frontend -- --config=e2e/regime-offline.config.ts --workers=2
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-completion-final-build npm run test:e2e --prefix frontend -- --config=e2e/regime-prospective.config.ts
REGIME_INTEGRATION_BUILD=/tmp/bettersaataa-regime-completion-final-build npm run test:e2e --prefix frontend -- --config=playwright.regime-integration.config.ts
npm run test:e2e --prefix frontend -- historical-regime-workbench.spec.ts regime-manual-events.spec.ts regime-temporal-event-library.spec.ts --workers=2
```

实际后端解释器使用项目规定的Python3.12；浏览器真实栈可用 INDICATOR_TEST_PYTHON 指定。隔离真实栈 `backend/tests/regime_completion_app.py` 在导入路由前设置临时存储、生成合成源，再真实执行发布、校准和保存；没有替换数值响应。生产构建保存在 `/tmp`，未覆盖共享 dist。

## 六、版本与路由记忆

旧研究引用、定义序列化和运行结果保持不可变；本轮源码仍只有一条主执行链。新增领域接口和统计通过共享存储与固定NJIT入口复用，未将无用旧实现复制成备份。

本轮所有变更尚未暂存、提交或推送。新文件是实际待交付源码/测试，不应为了通过路由检查将其伪装成临时产物。路由覆盖检查和最终 git diff --check 结果在本节追加；新稳定路径需在正式提交范围确定并被Git跟踪后登记，不能通过降低验证器要求消除未跟踪事实。
