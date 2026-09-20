# 多 CMA 模式 A：实现与验收记录

- 日期：2026-09-19。
- 基线：`ISSUE2609/BetterSaaTaa` 本地工作树，包含任务开始前已有、尚未提交的 LTCMA 第一版工作。
- 范围：[统一设计](pre-investment-dual-path-cma-saa-design-2026-09-16.md) 的 P4 模式 A／E2 参数平均，以及必要的冻结消费、TAA 风险诊断与前端交接。
- 不覆盖：模式 B 联合约束、E3 原生混合路径、跨频率／跨币种／不同 vintage 适配、TAA 剩余资金续算、产品实施新求解器、全投研审批执行闭环。
- 未提交、推送或部署；保留原有工作树改动与历史产物。

## 实际执行链

`PolicyRequest → StrategicAllocationService.preview_policy → multi_cma.resolve → 原单模型候选求解 → cross_model_results → preview_hash → create_policy → ArtifactRepository`。

下游通过 `cma_application` 读取冻结派生矩，`policy_gate` 用融合参数执行门禁，再对每个原模型生成诊断。TAA 展示原模型风险与 TE，资金 `goal_check=null`，没有继承 SAA 成功率。

前端使用 `MultiCmaSelection`、`StrategicAllocationWorkspace` 和 `CrossModelResults`。多选、显式研究权重、预览、采纳、历史恢复及 TAA 交接共用现有工作台。模式、成员、权重、范围和知识截止日变动均废弃预览；晚到请求不能恢复旧候选。

## 数值与契约

| 项目 | 当前行为 |
|---|---|
| 参数平均 | `mu = sum(p_m * mu_m)`；`Sigma = sum(p_m * Sigma_m)`，复用已有 NJIT 混合内核的 within 输出 |
| 模型分歧 | between 单独保存；不加入 E2 协方差，不平均各模型独立最优权重 |
| 不确定性 | 新固定签名 NJIT 对声明的半宽加权；明确标注未联合校准，不称新置信区间 |
| 资金模拟 | 复用融合年度矩的对数正态代理；不随机抽取原 CMA 路径 |
| 原模型诊断 | 收益、波动、风险贡献、适用的基准与资金检查；诊断失败可见，模式 A 采纳由融合模型判断 |
| 兼容门禁 | hash、V2 显式收益／费用／FX 口径、同 as_of／币种／期限／资产轴与角色／经济来源；统计代理一致 |
| 生命周期 | 来源须可新引用；确认阶段再次核验退役；冻结完整来源及派生矩，历史结果不重新拟合 |
| 旧契约 | 单 CMA 请求序列化与预览 hash 保持原形状；融合证据残缺或未知模式失败关闭 |
| 计算上限 | M≤20、N≤30；资金路径月 `(M+1) × (4 或 5) × paths × months ≤ 50,000,000`；中央／保守最多两倍 |
| 保存上限 | UTF-8 紧凑 JSON ≤7,000,000 bytes，与仓库编码一致；仓库读上限 8,000,000 bytes。源读取／冻结、预览及最终保存前核验，失败不落盘 |

颗粒度：新增内核只负责“按明确权重聚合各模型声明的半宽”；可独立测试和替换。均值／协方差继续复用统一现有内核，来源、校验、保存属于编排。没有新增耦合优化器，也未将多模型完整流程封装成不可解释的新数值黑盒。

内存边界：JSON 冻结证据在读取边界转为稳定 float64，小型 `M×N` 均值／半宽及 `M×N×N` 协方差需要一次堆叠；冻结 JSON 元数据深复制是不可变审计需要。不存在历史面板的按模型重复拟合或 `M×候选×路径×月份×资产` 大张量。只读／非连续输入在固定签名预热与等价测试中覆盖；这不表示整条链路完全零分配。

## 本次执行证据

| 检查 | 结果与边界 |
|---|---|
| 后端相关模块回归 | **605 passed，4 warnings，134.44 秒**；26 文件，覆盖新融合、LTCMA、原 SAA、Mandate、风险标尺、前沿与 TAA。不是全仓库 pytest |
| 最后源修补后定向回归 | **42 passed，49.81 秒**；`test_multi_cma.py`＋`test_multi_cma_performance.py`，覆盖最后的冻结标记剥离拒绝及增量容量检查 |
| 独立容量复核 | **3 passed**；真实中文超限证据、路径月预算及保存前失败，检查失败后文件状态不变；未参与实现的审查者复核编码与 envelope 余量 |
| 前端全量单测 | **158 文件、1,243 passed，120 秒**；包含 8 项新增融合工作台测试 |
| 前端静态／构建 | `tsc --noEmit`、`npm run build`、`design:check`、`check_i18n.mjs` 通过 |
| 生成契约 | LTCMA、Mandate、RiskScale 三份生成类型与当前 Pydantic 一致 |
| 多 CMA 浏览器验收 | **6 passed**；1440／768／320，选择与非法权重、刷新草稿、预览、采纳、冻结历史、真实 TAA 诊断、不兼容口径拒绝；各截图状态横向溢出与文字对比度检查通过 |
| 原单模型浏览器回归 | **10 passed**；1440／390，真实前沿、资金目标、未知 PIT、SAA 采纳与 TAA |
| 投前 01–04 浏览器回归 | **15 passed**；1440／768／320，双路径、历史恢复、真实 BL／Scenario、风险预算、SAA／TAA 与实施缺口提示 |
| M1 双路径浏览器回归 | **1 passed**；真实目标 → 无产品战略范围 → SAA → 显式实施映射，含响应式与只读状态 |
| 风险标尺浏览器回归 | **1 passed**；历史共同样本、五种分段方法、发布与不可变读取 |
| 目标边界浏览器回归 | **9 passed**；1440／768／320，两步目标、基准授权、双前沿、无 PIT 与英语页面 |
| 路由覆盖 | 显式审核本次 28 个改动文件，**28 covered、0 uncovered**；本地结构／路径检查通过 |
| 路由可复现检查 | 完整 `validate_ai_routing.py` 仍因新增源码／文档／测试尚未纳入 Git 跟踪而失败；没有为通过此项擅自 stage／commit，也没有放宽规则。后续授权提交时须重新检查 |
| CodeGraph | 已全量重建并在实现后 sync；当前 status 为 up to date，1,227 文件项、24,742 节点、83,644 关系 |

后端首次回归中的 6 项失败来自离线测试日历只含工作日，研究日为周六时超出覆盖范围。仅修复测试 fixture，补齐截至当天的闭市日记录，净值样本与生产日历门禁未改变；上表 605 项为修复后结果。浏览器共用 fixture 同步修正。

以上六套真实 Chrome 验收共 **42 项通过**，通过隔离的本机服务和合成 fixture 完成，不访问线上账户或修改生产数据。浏览器配置分别为 `playwright.multi-cma.config.ts`、`playwright.strategic.config.ts`、`playwright.better-saataa.config.ts`、`playwright.m1.config.ts`、`playwright.risk-scales.config.ts`、`playwright.mandate.config.ts`。

已有告警包括 Starlette 弃用提示、Numba 非连续布局性能提示、React 测试 `act` 提示及 Vite 大包提示；没有据此宣称全局性能优化或部署验收完成。

已查看的稳定截图：[1440 政策比较](../../.tmp_multi_cma/browser/multi-cma-frozen-multi-CMA-1e6c4-nd-immutable-source-history-desktop-1440/cross-model-results.png)、[768 历史来源](../../.tmp_multi_cma/browser/multi-cma-frozen-multi-CMA-1e6c4-nd-immutable-source-history-tablet-768/immutable-history.png)、[320 选择与权重](../../.tmp_multi_cma/browser/multi-cma-frozen-multi-CMA-1e6c4-nd-immutable-source-history-mobile-320/frozen-selection.png)。它们是本地临时验收产物，不是正式发布资产。

## 可重现基准

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/tests/test_multi_cma_performance.py
```

固定 seed=20260919，M=20、N=30、5 候选，预热后开启 tracemalloc，测量冻结血缘核验、E2 矩计算与原模型诊断，**排除进程启动、完整候选求解和资金模拟**。

独立进程三次为 1.303／1.426／1.333 秒；输入矩 153,760 bytes，traced 峰值 2,485,279 bytes，进程 RSS 历史峰值增长 5,062,656 bytes（440,532,992 → 445,595,648）。RSS 是进程历史高水位，包含 Numba 启动基线，不能当作该功能全部常驻内存。

最终定向测试后的同进程复测为 1.130／1.114／1.125 秒，traced 峰值 2,484,352 bytes。两次均核对参考数值、`python_fallback=0`、新增请求签名为 0；不设置机器相关的绝对秒数断言。

## 后续开发边界

下一算法阶段是模式 B 的联合二次约束与独立验证；P5 另需大类研究和产品应用解耦、剩余资金续算及实施预算／暴露模型。模式 A 的来源诊断不替代这些能力。原 2026-09-16 文档数学样例是历史证据，本次没有重新执行，也未把它计入上述生产实现验收。
