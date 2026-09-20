# 投前研究统一设计：投资目标、双代理大类、五类 CMA 与单／多 CMA 的 SAA

- 项目：BetterSaaTaa
- 整体审核与修订日：2026-09-19
- 文档性质：业务与算法详细设计，并同步记录当前已落地能力；**设计完成不等于其余规划能力已经开发完成**。
- 代码核对基线：本地 `ISSUE2609/BetterSaaTaa` 当前 working tree；风险等级配置中心与 Mandate 2.0／投资目标优化已经进入真实前后端调用链，单／多 CMA 后续能力仍以当前代码事实为准。
- 本次范围：核对当前工作树的投研流程与投前模块，修正已过时的实现描述，并按第 21 节顺序完成 P4 模式 A 与模式 B 的线性收益基础版。保留已有未提交的 LTCMA 工作；代码、测试、浏览器验收分别记录，不将设计要求或过往验收当作本次执行证据。
- 文件名沿用历史名称，避免断开现有引用；本文不再表示“两套独立的 SAA 流程”。

> **最终主线：投资目标 → 构建大类及代理 → 形成一套或多套 CMA → 单／多 CMA 的 SAA → 大类层 TAA → 产品映射与配置 → 验证与定稿。**
>
> Settings 中的 Universal 风险标尺是可复用的前置研究成果，不是每次投前研究都要重新执行的一步。CMA 是独立模块；目标设计中，多 CMA 的“参数融合”和“兼容配置”是两个并列的 SAA 模式。
>
> **当前实施里程碑（2026-09-19）：风险等级配置中心、投资目标／约束、Product-first 与 Strategic-first 两条 LTCMA 上游路径及独立 LTCMA 中心第一版均已有真实链路；两条路径已经汇合到同一冻结 LTCMA → SAA 主流程。现已实现模式 A 的 E2 参数平均及模式 B 的逐模型联合约束、线性收益 minimax regret／maximin、冻结保存与 TAA 门禁，单 CMA 保持兼容。下一阶段重点转向 class-level TAA 与产品应用解耦、产品实施契约和统一验证；模式 B 五等级比较、凹效用扩展、E3 原生混合分布路径作为后续增强。具体执行证据见第 21.4–21.6 节。**

## 阅读顺序

| 章节 | 主要问题 |
|---|---|
| 0–3 | 这次修正了什么；最终流程、对象、日期与收益口径如何统一 |
| 4–6 | Universal C1–C5、投资目标、两种大类代理如何衔接 |
| 7–12 | 五类 CMA 的输入来源、公式、自动化与限制 |
| 13–16 | 资金模拟、单 CMA、参数融合与跨 CMA 兼容配置怎样计算 |
| 17–19 | TAA、产品实施、版本、前端与接口如何闭环 |
| 20–22 | 算法验收、实施依赖、本次审核证据与外部参考 |

---

# 0. 本轮审核结论与修订原则

## 0.1 原文不是所有环节都已闭环

原文积累了多轮讨论，方向基本一致，但存在重复、目标架构与代码现状混写、相同术语口径不同，以及“列出模型名称，却没有定义跨模块算法”的问题。重点修正如下。

| 编号 | 原文的问题 | 本文的统一处理 |
|---|---|---|
| A01 | 多 CMA 只有比较、共识、regret 等名称，缺少完整求解契约；参数加权模式被弱化 | 明确保留参数融合和共同配置两种模式；给出各自公式、求解器、检查矩阵及发布条件 |
| A02 | 资金所需年有效收益直接与 CMA 年化算术收益比较 | 分开保存；资金可行性用同口径分布适配与现金流递推检验，不用混口径收益不等式代替 |
| A03 | 历史状态均值、协方差先年化，再使用年化情景混合 | 历史占用型状态先在观测频率混合，再转换；年度替代情景仍按年度混合 |
| A04 | 贝叶斯风险和均值估计不确定性的年化关系不完整 | 资产风险在独立增量近似下乘 a；年化均值估计协方差乘 a²；两者分开 |
| A05 | 暗示现金流能唯一推出风险承受能力或最大可接受风险 | 现金流给出目标所需风险与政策条件下的资金可行性；机构授权仍须治理规则与确认 |
| A06 | 把风险等级所在区间与“最高允许等级”混用 | C3 授权默认表示不超过 C3 上限，不强迫组合至少承担 C3 下限风险 |
| A07 | “C3 可行、C5 不可行”的例子未区分区间和上限 | 高风险区间内组合可能不满足资金目标；放宽风险上限后的可行集合不能反而缩小 |
| A08 | 前沿斜率分段被描述为客观发现风险偏好，缺少退化处理 | 定义为项目级校准规则；采用有界角度、弧长加权连续分段，公开参数、稳定性与人工覆盖 |
| A09 | 有效前沿右端与全空间最大方差、无杠杆字段含义混淆 | 右端为最高收益面上的最小方差点；无杠杆为非负权重、合计 1、总敞口 1 |
| A10 | 有限候选没找到方案就称“无解”；模拟选优与验证未分开 | 区分已验证可行、数值未收敛、有限搜索未找到、可证明不可行；搜索与确认使用独立模拟样本 |
| A11 | 参考组合的 CVaR、回撤似乎自动成为同档全部组合的保证 | 仅是参考画像；对最终组合重新计算，只有明确实现并复核的约束才能称硬护栏 |
| A12 | SAA 多 CMA 后，TAA／产品门禁仍只认识一个协方差 | 冻结整个风险评估契约；兼容模式的后续目标必须逐 CMA 复核 |
| A13 | 大类代理和产品映射只有功能描述，缺少权重时点、预算方程 | 定义代理再平衡与收益归属；区分类别归属矩阵和统计 Beta，明确产品预算与跟踪误差优化 |
| A14 | 投资目标依赖 CMA、CMA 又似乎依赖投资目标，首次使用循环 | 允许独立构建 Universal 资产与参考 CMA；投资目标先引用全局参考，实际 CMA 完成后补充验证 |
| A15 | 部分“业内做法”外推过强，引用与主张没有一一对应 | 区分公开方法支持、项目设计选择和未实现能力；不宣称机构普遍采用同一套五分类或 C1–C5 分段 |
| A16 | 非资金目标的现金流、TAA 后的剩余现金流没有独立交接定义 | 将资金预算与成功标准分开；新增资金续算上下文，不继承旧 SAA 的成功率 |
| A17 | 只说复用 QP，未核对等式数量与无解证明能力 | 当前内核只有首行等式、其余为不等式；产品多预算等式、Phase-I 和对偶证据须显式补齐 |
| A18 | 参数平均后的资金模拟与风险护栏可能换了语义 | 参数平均与分布混合分别指定路径模型；显示指标、候选过滤与已执行硬约束分开 |

## 0.2 三种“完成”状态必须区分

1. **设计闭环**：每个环节有输入、算法、输出、异常、下游消费者与测试标准。
2. **代码实现**：接口和实际执行链已经按该设计完成。
3. **模型有效**：经过足够数据、样本外检验和业务治理后，具备明确适用范围。

本文整体仍以设计闭环为主。当前 RiskScaleVersion、Mandate 2.0、独立 LTCMA 及模式 A 参数平均已有代码实现；各自的验收范围以第 2 节、第 21 节及对应验收记录为准，不将局部完成扩展成整套投前研究均已上线。

## 0.3 保留的业务选择

- 手动风险等级与现金流推导建议并存，不把手动选择一概解释为“拍脑袋”。
- 历史统计允许成为正式 CMA 基础方法，但标注历史统计基线的预测限制。
- 多 CMA 加权是合法、明确的研究方式，不强制所有研究都使用最坏情形优化。
- Bayesian 模型内部不默认对每次后验抽样求解 SAA；模型间稳健性在 SAA 层处理。
- 原有历史配置实验、风险预算、手动情景及人工覆盖能力保留；有授权的实现改造再做替换与去重。
- 当前阶段只做资产管理：long-only + cash、无杠杆、无做空；ALM、衍生品对冲与负债随机过程不属于本阶段。

---

# 1. 最终业务流程与先后依赖

## 1.1 全局设置与单次研究分开

```text
项目级设置／预先准备
  Universal 大类与 Proxy
            + 参考 CMA + 标准约束
            ↓
  Universal 有效前沿 → C1–C5 风险标尺 → RiskScaleVersion
            + 组织资金政策 → OrganizationPolicyVersion

单次投前研究
  01 投资目标与边界：MandateVersion
            ↓
  02 大类代理构建
      ├─ 产品代理：冻结产品池 → 单产品／产品组合
      └─ 指数代理：指数库 → 单指数／复合指数
            ↓
      AssetClassVersion + ProxyBindingVersion + ReturnPanelVersion
            ↓
  03 CMA 独立研究中心：形成一套或多套 CmaVersion
            ↓
  04 SAA：同一 Mandate、同一资产轴、一个或多个 CMA
      ├─ 单 CMA：直接求解
      ├─ 多 CMA／模式 A：参数融合 → 一套有效参数 → 求解
      └─ 多 CMA／模式 B：各模型候选 + 联合约束 → 一套兼容权重
            ↓
      SaaStudyVersion → 确认 SaaPolicyVersion
            ↓
  05 大类层 TAA：保留 SAA 中枢与风险授权
            ↓
  06 产品实施
      ├─ 产品代理：在原授权产品池范围内配置
      └─ 指数代理：选择已发布产品池或全市场候选 → 暴露匹配
            ↓
      ImplementationMappingVersion → ProductAllocationVersion
            ↓
  07 产品择时引用、组合合成、风险与资金复核、统一验证
            ↓
      ResearchPackageVersion／外部审批记录
```

两种代理方式不是两套 CMA、SAA 或 TAA 数学模型。第二次分叉发生在产品候选范围与实施关系的确定上。

## 1.2 支持中途进入，但不是无条件跳过依赖

用户可直接进入 CMA、SAA 或 TAA，选择已经保存且符合当前用途的上游版本。比如直接进入 SAA，应选择 Mandate、资产版本及一个或多个 CMA，不必重做产品池和大类。

版本选择器应显示：资产轴、币种、研究日、预测期限、口径、发布状态、失效原因；默认过滤不兼容项，允许查看原因，不静默替换成“最新”。

“存在文件／已保存”不等于“可用于当前采纳”。草稿、研究保存、正式确认、当前应用资格分别管理。

## 1.3 首次使用如何启动

Universal 大类、Proxy 和参考 CMA 可独立创建，**不要求先存在客户 Mandate**。这是项目研究基础设施。

当前 **Mandate 2.0 标准前端流程要求先选择研究日有效的 RiskScaleVersion**；没有可用标尺时，投资目标页引导用户进入风险等级配置中心，并阻断新版本确认，不能编造 C1–C5 或自动填入风险上限。后端仍保留 `reference_pending`／`explicit_numeric` 等兼容语义供既有契约和非标准调用使用，但不作为新前端主流程。

Global 诊断是参考，不替代本次实际资产范围验证。实际 CMA 比全局参考乐观或悲观时，应显示差异并重新检验，而不是修改原目标来迎合模型。

---

# 2. 当前代码事实与统一领域对象

## 2.1 已核对的实现事实

下列“已有”仅指本轮读取的本地代码；没有把文档中规划的能力算成实现。

| 领域 | 当前事实 | 主要代码依据 |
|---|---|---|
| 投资目标 | **Mandate 2.0 已落地**：三类成功标准、独立 `cash_budget`、RiskScale 风险授权、资金建议／参考诊断、实时确定性资金回显和不可变确认均进入真实调用链；主页面先列出现有版本，支持新增、修改为替代版本及删除／退役；编辑页已收敛为“目标与约束 → 结果确认”两步 | `frontend/src/pages/InvestmentObjectivesCenter.tsx`；`InvestmentObjectivesWorkspace.tsx`；`backend/strategic_allocation/contracts.py`；`mandate_contracts.py`；`mandate_diagnosis.py`；`service.py` |
| 资金算法 | 已有月度现金流、二分所需收益、对数正态组合代理、Wilson 下界、资本补足重放；填写阶段另有只跑资金算术、不触发 CMA／模拟的 `mandate_funding()` 回显 | `backend/strategic_allocation/goal_kernels.py`；`service.py` |
| 风险等级配置中心 | **已落地**：独立 RiskScale 工作台、草稿、预览、确认、不可变版本、默认版本、启用／退役、比较、参考输入和研究日资格检查已经形成完整前后端链路，并被 Mandate 2.0 直接消费 | `backend/strategic_allocation/risk_scale_service.py`；`risk_scale_routes.py`；`risk_scale_store.py`；`reference_inputs.py`；`frontend/src/pages/RiskScaleCenter.tsx`；`RiskScaleWorkspace.tsx`；`RiskScaleVersionView.tsx`；`services/riskScales.ts` |
| 大类代理 | 产品构建、自动分类、类内权重、大类净值已有；完整统一指数 Proxy 层仍需完善 | `frontend/src/pages/ManualConstruction.tsx`、`AutoAssetClassification.tsx`；`backend/fit.py:323–403` |
| LTCMA 双路径上游 | **两条路径第一版均已打通**：Product-first 从产品池／产品大类进入 `allocation:*`；Strategic-first 从独立 `StrategicUniverse` 进入 `universe:*`，统计方法可为战略资产配置研究 Proxy，不要求先完成实施产品映射。两条路径最终均进入同一 LTCMA 契约 | `frontend/src/pages/ProductPoolSelection.tsx`；`components/strategic-scope/StrategicScopeWorkspace.tsx`；`components/ltcma/model.ts::applyScope()`；`LtcmaInputFields.tsx` |
| CMA | **独立 LTCMA 中心第一版已实现**：Manual、BL、人工 Scenario、Historical、NIW、历史状态六个明确入口（五类方法族）；列表、可编辑草稿、预览、幂等确认、冻结读取、复制与停止新引用已有。统计方法当前限定 CNY／SSE／日频 252；SAA 只选择已保存版本，不再内嵌编辑器 | `backend/strategic_allocation/cma_service.py`；`cma_evidence.py`；`cma_statistical_models.py`；`cma_statistical_kernels.py`；`cma_application.py`；`frontend/src/pages/LtcmaCenter.tsx`；`LtcmaWorkspace.tsx`；`LtcmaVersionView.tsx` |
| 历史风险参考 | 显式区间、SSE 日频、252 年化、至少 20 个共同收益观测、固定强度对角收缩；返回历史均值但不自动当长期收益 | `backend/strategic_allocation/service.py` 的 `risk_reference()`；`kernels.py` |
| CMA 消费 | SAA 读取冻结有效参数，不重新拟合；Historical／NIW 等统计结果提供 `mean_uncertainty` 时使用模型结果，其他方法保留请求半宽；BL 后验均值协方差不自动转成该半宽 | `backend/strategic_allocation/cma_application.py`；`cma_statistical_models.py` |
| SAA | **三种模式已接通**：single 保留 `cma_id`；parameter_average 保留四类有限候选与可选风险预算；compatible_all_models 不使用模型权重，先求独立收益锚点，再直接求解共同约束下的 minimax regret／maximin，并逐模型执行采纳门禁 | `backend/strategic_allocation/contracts.py`；`multi_cma.py`；`compatibility.py`／`compatibility_solver.py`／`compatibility_kernels.py`；`service.py`；`frontend/src/pages/StrategicAllocationWorkspace.tsx` |
| 前沿 | 历史收益输入的确定性目标网格与 QP/SQP 内核已有；网格点数上限 200 | `backend/optimizer.py`；`backend/qp_numba.py` |
| 状态联动 | 历史状态 CMA 已从核验过的事后运行快照取得同一状态日期轴，并对共同多资产收益估计条件矩、历史占用率和应用概率；先混合再年化。未实现 Markov 多期预测、自动新情景参数补造或全部期限适配 | `backend/strategic_allocation/cma_evidence.py`；`cma_statistical_models.py`；`frontend/src/components/ltcma/LtcmaRegimeResults.tsx` |
| TAA | 读取冻结单 CMA 或模式 A 政策；模式 A 按融合风险执行门禁，并展示原模型矩风险及跟踪误差诊断，`goal_check=null`，不继承 SAA 资金成功率；战略来源仍受实施映射限制，class-level TAA 与产品应用门禁尚未按目标设计拆开 | `backend/strategic_allocation/cma_application.py`；`policy_gate.py`；`backend/tactical_allocation/portfolio_bridge.py` |
| Settings | 风险等级配置中心已标记 `available`；研究参数中心仍为 `prototype` | `frontend/src/app/processRegistry.ts` |

特别纠正此前讨论中的两个判断：当前目标模块**已经有量化诊断**；当前通用前沿内核也**不等于已经支持多协方差二次约束的联合稳健求解器**。

### 2026-09-20 审核修复补充

- 研究成果统一按最终 JSON 的 UTF-8 字节数执行 8,000,000 字节容量门禁，包含元数据、校验值和数组描述；超限时不提升临时目录、不注册索引，既有研究包仍可读取。
- NIW 发布计算后，在与停止引用共用的生命周期锁内重新检查先验资格并保存；已完成操作的幂等重放继续读取原不可变版本。
- 前端切换到 NIW 后验续更时清除均值／风险先验强度及数据复用确认；切回重设先验时重新填写，不提交隐藏的旧数值。
- SAA 的 CMA 深链接只用于初始化。之后的手动选择、清空及范围变更优先，不因异步读取清空当前结果而恢复链接中的旧 CMA。

复现用例及本轮验证见 [审核修复记录](ltcma-center-acceptance-2026-09-18.md#6-2026-09-20-分支审核修复)。本次不改变数值模型、历史冻结结果或投资资格边界。

## 2.2 目标对象及唯一职责

| 对象 | 负责保存什么 | 不负责什么 |
|---|---|---|
| OrganizationPolicyVersion | 成功概率门槛、现金保障窗口、治理授权、人工覆盖规则 | 自动猜测机构损失偏好 |
| AssetClassVersion | 稳定大类 ID、经济定义、资产轴、角色、准入定义 | SAA 权重或长期预测 |
| ProxyBindingVersion | 产品／指数成分、权重规则、币种、收益口径、再平衡规则 | 最终实施产品权重 |
| ReturnPanelVersion | 同轴收益、有效性、数据与可得时点、快照哈希 | 把最新修订冒充历史可得 |
| CmaVersion | 一套可解释的有效矩、可选分布／不确定性、方法与来源 | 用户风险授权 |
| RiskScaleVersion | Universal 校准输入、前沿、C1–C5、参考画像 | 机构问卷结论或法定适当性评级 |
| MandateVersion | 资金事实、成功标准、选定授权、数值硬约束及参考诊断 | 最终组合或唯一绑定某个实际 CMA |
| SaaStudyVersion | 所选 CMA 集、融合模式、求解、交叉验证及候选 | 未经确认替换正式政策 |
| SaaPolicyVersion | 一套最终大类权重、Mandate 与完整风险评估契约 | 只留一个 C3 标签让下游猜参数 |
| TaaDecisionVersion | 相对 SAA 偏离、信号、时点、回测与资格 | 改写长期 CMA |
| ImplementationUniverseVersion | 已筛选的实施候选、来源产品池或市场目录、时点准入证据 | 因为来自“全市场”就自动允许投资 |
| MappingPolicyVersion | 匹配规则、统计验证窗口与有来源的阈值 | 用历史 Alpha 自动定义资产类别 |
| ImplementationMappingVersion | 实施范围、类别归属、暴露证据、产品限制 | 把 Beta 当份额预算 |
| ProductAllocationVersion | 产品权重、预算守恒、成本及实际风险复核 | 继承参考产品的风险保证 |
| ValidationReportVersion | 证据范围、通过／失败／未验证、来源与算法版本 | 把局部成功称为全局最优 |

新对象是目标设计。实现时复用现有不可变 artifact 存储，不另建一套平行数值或历史版本实现。

## 2.3 全投研流程与当前交接边界（2026-09-19 核对）

真实流程应按成果及引用关系理解，不能只按导航页面是否存在判断完成度。

| 阶段 | 当前真实能力与交接 | 尚未闭环的部分 |
|---|---|---|
| 产品研究 | 市场概览、产品详情／比较、指标评价、择时研究与产品池；已生效 `ProductPoolVersion` 可生成冻结的 `InvestableUniverseSnapshot` | 持仓穿透仍为原型；研究发布不等于投资准入自动通过 |
| 投前研究 | RiskScale → Mandate；产品池／战略范围 → 大类代理 → 冻结 LTCMA → SAA baseline → TAA decision → 产品组合研究 | 组合合成、统一验证和审批入口仍有原型；本章后续设计不因此全部可用 |
| 投中执行 | 组合、账户与交易分配已有交互演示及数值校验 | 流程注册仍标记 `prototype`；没有据此证明真实订单、托管回报或执行闭环 |
| 基金会计 | Booking、账户分配及双主体报表有专用演示页面 | 仍为原型，不代表真实入账、对账及法定账簿 |
| 投后管理 | `/post-investment/research-diagnosis` 消费真实研究对象与运行快照 | 实际组合绩效、归因、监控及报告仍为原型 |
| 反馈迭代 | 已定义导航与业务框架 | 当前为原型，尚无自动反馈并更新研究版本的业务闭环 |

投前各子功能的输入输出如下：

1. **投资目标**：选择冻结风险标尺，填写三类成功标准与独立现金预算，经预览／确认形成不可变 Mandate 2.0；风险授权与参考资金诊断分别保留。
2. **范围与大类**：**两条 LTCMA 上游路径第一版均已完成。** Product-first 先冻结产品池与可投资域，再手动／自动分类及类内构建，并以 `allocation:*` 进入 LTCMA；Strategic-first 先创建独立 `StrategicUniverse`，可配置研究 Proxy，并以 `universe:*` 进入 LTCMA，不要求预先完成实施产品映射。两条路径共用同一 LTCMA／SAA 主链，而非两套模型。
3. **LTCMA**：对明确资产范围进行人工、历史、NIW、BL 或状态／情景研究；草稿、预览、冻结发布、复制与停止新引用各有独立语义。SAA 引用已发布版本，不在页面重复编辑模型。
4. **SAA**：已确认 Mandate、冻结 CMA 与权重／分组约束生成四种代表候选，可另加风险预算候选；确认时复核预览 hash，冻结来源、参数、约束及最终权重。资金目标另做固定候选的独立样本验证；采纳仍是研究员确认，`independent_approval=False`。
5. **TAA**：读取精确的服务端 SAA baseline，结合信号、观察／决策／执行时钟、训练验证与费用形成决策；继续检查授权、复核日期及实施映射，不将 SAA 成功率当作战术路径的资金保证。
6. **产品配置与诊断**：现有桥接要求同一可投资域、冻结产品归属、手动产品权重与逐类预算守恒，生成 ResearchTarget／PortfolioRun。它是静态目标的历史回放，不等于动态 TAA 回放。择时 release／binding 是独立研究引用，不自动改写权重。
7. **合成、统一验证与审批**：目前导航连接已有单项工具或原型，尚无完整 ResearchPackageVersion／外部审批闭环。

代码依据：`frontend/src/App.tsx`、`app/processRegistry.ts`、`app/prototypeRegistry.ts`、`pages/ProductPoolSelection.tsx`、`pages/PortfolioConstruction.tsx`、`backend/strategic_allocation/service.py`、`policy_gate.py` 和 `backend/tactical_allocation/portfolio_bridge.py`。`allocationJourney` 保存浏览器选择与草稿引用，不能作为冻结模型、数值或 PIT 证据；TAA 返回 SAA 时读取原 baseline。

本次先执行 `codegraph index` 全量重建：1,218 个索引文件项、24,574 个节点、83,013 条关系。4 个 ENOENT 均为当前工作树已删除的旧 CMA 编辑组件，错误记录位于 `.codegraph/errors.log`；不是四个仍在调用链的实现。实现后执行 `codegraph sync`，状态显示最新索引为 1,227 文件项、24,742 节点、83,644 关系。CodeGraph 用于定位与调用关系，最终完成度仍由当前源码和执行验收确定；临时审计目录的同名符号不作为生产调用证据。数字仅为 2026-09-19 本次快照。

---

# 3. 统一数理口径：所有模块必须遵守

## 3.1 资产轴与维度

用 N 表示大类数量，T 表示观测数，M 表示 CMA 模型数，S 表示情景数，a 表示每年观测期数。

\[
R\in\mathbb R^{T\times N},\quad \mu\in\mathbb R^N,\quad \Sigma\in\mathbb R^{N\times N}.
\]

M 套 CMA 在内部可暂存为 `means[M,N]`、`covariances[M,N,N]`；传统单模型优化器仍接受一个向量与一个二维矩阵。三维数组只是多模型批量容器，不是额外的一阶／二阶统计量。

均值是一阶矩；方差和协方差都是二阶中心矩；偏度涉及三阶矩。不能把协方差称为三阶矩。

资产 ID、经济定义和 Proxy 版本共同确定暴露含义。同名但不同大类定义不得自动对齐。顺序变化应显式排列并记录，缺资产不能补零。

## 3.2 四种收益概念不得混用

| 名称 | 定义／用途 |
|---|---|
| 单观测期简单收益 | `r_t = NAV_t / NAV_(t-1) - 1`，历史统计的原始单位 |
| 年化算术参数 | `mu_a = a * mean(r_t)`，`Sigma_a = a * Cov(r_t)`；协方差换算依赖独立增量近似 |
| 一年复合后的简单收益随机变量 | `R_1Y = product(1+r_t)-1`；其期望和方差不一般等于上一行 |
| 资金所需年有效收益／CAGR | 常数复利增长率，满足现金流与期末目标；不是市场预期均值 |

保留当前 `return_basis = annual_arithmetic_total_return` 的历史字段。新版本同时增加：

```text
observation_frequency
periods_per_year
moment_period
moment_semantics: annualized_periodic_arithmetic | one_year_simple
forecast_horizon_years
annualization_method
fee_basis
currency
fx_hedging_basis
covariance_role: asset_return | predictive
included_uncertainty_components: [] | [mean_estimation] | [model_state_variation] | [...]
```

五类方法在新建研究中默认采用明确的年化算术参数出口；若输入是实际一年简单收益矩，则标识为 `one_year_simple`。跨语义融合必须先通过明确、已验证的适配，不能只因都有“年化”二字而直接相加。

历史 artifact 的既有口径保持原状；无法确定其细分语义时允许只读，新的跨方法融合须重新确认来源／创建新版本，不回写旧记录。

## 3.3 协方差与年化

若基础期收益在时间上独立且使用加法累计近似：

\[
\mu_a=a\mu_p,\qquad \Sigma_a=a\Sigma_p.
\tag{U1}
\]

若存在自相关，加法收益的 a 期协方差为：

\[
\operatorname{Cov}\!\left(\sum_{t=1}^{a}r_t\right)
=a\Gamma_0+\sum_{h=1}^{a-1}(a-h)(\Gamma_h+\Gamma_h^\top).
\tag{U2}
\]

因此“日频风险乘 252”是有假设的近似，不是对所有市场状态或产品都成立的恒等式。基础版保留简单年化，但必须记录局限；长期情景路径另由分布适配器处理。

样本协方差默认统一 `ddof=1`；状态经验分布重构允许 `ddof=0`，须显式记录。收缩矩阵不是原始样本无偏协方差。

## 3.4 数据时间与 PIT

分别保存：观测日期、源数据可得时间、模型训练截止、状态确认所需信息截止、研究 as_of、实际计算时间、发布时间。

正式历史时点研究要求其信息集合不超过 as_of。今天重算一个历史 as_of 的研究可以，但必须证明数据、模型选择与状态划分都只用当时信息；不能仅截去曲线后半段。

历史数据缺少可靠 availability 时可做 `retrospective_research`，明确“历史可得性未验证”；不能自动获得正式 PIT 资格。没有时点证据不是零延迟证据。

严格共同收益轴应先从合法相邻 NAV 构造收益，再按共同日期取样；不能删掉中间日期后，把跨数日收益当作一天收益。月／周聚合必须记录端点、有效间隔与日历。

## 3.5 统一风险函数与测量边界

对于第 m 套 CMA：

\[
\mu_m(w)=w^\top\mu_m,\qquad
\sigma_m(w)=\sqrt{w^\top\Sigma_mw}.
\tag{U3}
\]

相对基准 b 的主动风险：

\[
TE_m(w,b)=\sqrt{(w-b)^\top\Sigma_m(w-b)}.
\tag{U4}
\]

投资授权基准的 TE 与 TAA 相对 SAA 的 TE 是两个约束，不能共用名称而覆盖彼此。

同一数值波动上限跨模型比较，须满足币种、收益单位和风险时间尺度一致；结果仍是“在该模型下的估计风险”。风险标尺不是所有模型天然等价的证明。

---

# 4. Settings：Universal 全局风险标尺

## 4.1 Universal 是什么

Universal 是**项目维护的代表性大类投资机会集**，不是声称覆盖全球每一个可投资产品，也不是从全部基金找两只最高／最低风险基金。

选取现金／货币、适当久期利率债、信用债、权益、黄金等具有不同经济暴露的代表性大类；具体名单、Proxy、币种与准入由项目版本定义，不写死收益和风险数值。

单一／复合指数代理优先用于稳定的经济暴露标尺；允许产品代理，但应披露其费用、跟踪误差与生命周期。多只同指数 ETF 不应被当成多个独立风险来源。

正式全局标尺应有经数据与经济定义验证的低风险／流动性锚，不是只把一个大类角色改名为 `rates` 就通过。只有股票与黄金的集合可以研究，但不能无提示地发布为覆盖低至高风险的项目全局标尺。

## 4.2 冻结输入与约束

```text
Universal AssetClassVersion
ProxyBindingVersion / ReturnPanelVersion
Reference CmaVersion
base_currency / risk_measure_basis
StandardConstraintProfile
frontier_algorithm_version
segmentation_algorithm_version
secondary_metric_settings
```

标准可行集：

\[
\mathcal W_U=\{w:\mathbf1^\top w=1,\ w\ge0,\ l\le w\le u,\ g_l\le Bw\le g_u\}.
\tag{R1}
\]

字段使用 `allow_leverage=false`、`allow_short=false`、`gross_exposure=1`，不使用含糊的 `leverage=0`。现金是资产轴中的显式资产，不能靠缺失权重自动补成现金。

Universal 标准约束和具体 Mandate 可以不同；前者定义参考空间，后者定义本次授权。因此必须再做实际机会集验证，不宣称两者完全相同。

## 4.3 前沿端点与求解

左端是全局最小方差组合：

\[
w_{GMV}\in\arg\min_{w\in\mathcal W_U}w^\top\Sigma_Uw.
\tag{R2}
\]

若最小方差解不唯一，在保持最小方差的容差内选择最高预期收益，防止保留被支配端点。在 PSD 二次目标与凸可行域中，两个最小方差解的差位于 Sigma 的零空间；可在已求得 w0 后增加线性等式 `Sigma*w=Sigma*w0`，再最大化收益，并复核原方差值。这样无需为全局标尺的这个并列处理依赖尚未实现的通用 QCQP。数值容差、等式秩和退化矩阵必须验证。

先求最高预期收益 `r_max = max mu_U' w`，再在最高收益面上求最小方差，得到右端点：

\[
w_{right}\in\arg\min w^\top\Sigma_Uw
\quad\text{s.t. }w\in\mathcal W_U,\ \mu_U^\top w\ge r_{max}-\varepsilon_r.
\tag{R3}
\]

右端是有效前沿的最高收益端，不是全部可行权重中的最大方差。无杠杆且允许 100% 持有任一资产时，最大组合波动不超过最大单资产波动；最低组合波动则可能低于所有单资产波动。

在两个收益端点之间建立目标收益网格，逐点解：

\[
\min_{w\in\mathcal W_U}w^\top\Sigma_Uw
\quad\text{s.t. }\mu_U^\top w\ge r_j.
\tag{R4}
\]

基础设置使用 101 个目标点；稳定性复核可用 200 点。当前底层契约上限为 200，不再写成可直接使用 201 点。

输出每个目标的权重、真实收益风险、约束残差、收敛状态与端点状态。失败目标保留缺口，不能插值后假装求解通过。非支配性检查、去重只作用于展示及分段，不删除失败证据。

## 4.4 C1–C5 自动切分：项目规则，不是自然常数

公开资料支持均值方差、风险预算和稳健组合方法，但**不能据此声称存在业内统一的“按前沿形状自动发现五个真实风险偏好等级”的标准**。

这里选择五档，是项目产品规则；选择形状分段，是透明的校准方法。即使算法完全确定，也仍依赖资产、CMA、约束、坐标尺度和算法参数。标尺发布后冻结，不能随当天前沿波动自动改变用户授权。

保留两种设置方式：`automatic_frontier_shape` 与 `manual_volatility_bands`。不输入个人风险厌恶系数来定义等级。

## 4.5 自动算法：弧长加权角度连续动态规划

原设计使用裸斜率平方误差。GMV 附近可能出现接近竖直的切线，裸斜率会发散，并对网格密度敏感。因此采用 `frontier_shape_dp_v2`；当前 `risk_scale_kernels.py` 已默认执行该算法，以下为其设计契约。历史已保存标尺保持原版本，不随实现更新重算。

**步骤 A：标准化。** 对已验证、按风险排序的前沿点：

\[
x_i=\frac{\sigma_i-\sigma_{min}}{\sigma_{right}-\sigma_{min}},\quad
 y_i=\frac{\mu_i-\mu_{GMV}}{\mu_{right}-\mu_{GMV}}.
\tag{R5}
\]

收益或风险跨度低于数值阈值时，不强行生成五档，返回 `DEGENERATE_FRONTIER`；允许管理员选择其他参考输入或进入有依据的人工模式。

**步骤 B：局部角度与弧长。**

\[
\theta_i=\operatorname{atan2}(y_{i+1}-y_i,x_{i+1}-x_i),\quad
 h_i=\sqrt{(x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2}.
\tag{R6}
\]

角度有界，弧长权重避免仅因某段采样更密而获得更大权重。不要求先用 k-means；风险分段必须连续。

**步骤 C：区间代价。** 对边集合 `[l,r)`：

\[
E(l,r)=\sum_{i=l}^{r-1}h_i\theta_i^2-
\frac{(\sum_{i=l}^{r-1}h_i\theta_i)^2}{\sum_{i=l}^{r-1}h_i}.
\tag{R7}
\]

使用三组前缀和实现 O(1) 区间查询。极小负值仅在明确的舍入误差容差内置零；不能掩盖无效输入。

**步骤 D：动态规划。** 设边数为 J：

\[
D(k,j)=\min_{i<j}\{D(k-1,i)+E(i,j)\},\quad D(0,0)=0.
\tag{R8}
\]

其余初值为无穷大；回溯 `D(5,J)` 得到四个边界。候选分段须满足最少边数和最小标准化风险跨度；建议工程起点为每档至少 5 条有效边、风险跨度至少 0.05。参数随算法版本冻结，属于数值分辨率规则，不是投资者偏好。

并列最优在容差内选择字典序最早的四个切点，保证确定性。复杂度 O(5J²)，J 不超过当前有效前沿边数。

**步骤 E：近直线与稀疏处理。** 若全部角度变化小于版本化阈值（初始研究值 0.02 弧度），使用标准化累计弧长的 20/40/60/80% 切分。选择满足跨度条件的最近有效节点；无法得到四个不同边界时返回不足，不制造五个相同区间。

若 DP 无合法解且不是近直线，提示分辨率不足／约束退化，不偷偷换成另一种风险标尺。可以重新计算更密的有效前沿或显式切换人工模式。

**步骤 F：稳定性。** 用 101 点和 200 点复核边界；输出标准化边界最大移动量。超过配置阈值（研究起点 0.02）应显示校准不稳定并要求复核，不能据此宣称金融参数已统计验证。

分段只处理风险收益曲线，不重新优化每个贝叶斯后验样本。代表组合选段内累计弧长的中位有效节点；它是展示锚，不是强制持有组合。

## 4.6 分类区间、风险上限与超界

自动分段产生 `b1 < b2 < b3 < b4 < b5`，其中 b5 为前沿右端风险。正式分类采用：

```text
C1: 0 <= sigma <= b1
C2: b1 < sigma <= b2
C3: b2 < sigma <= b3
C4: b3 < sigma <= b4
C5: b4 < sigma <= b5
sigma > b5: above_scale，不静默归入 C5
```

把 C1 下界延伸至 0 仅用于分类覆盖，不声称 Universal 存在零风险组合。低于参考 GMV 的实际组合可标为 `C1 + below_reference_range`，不能因为它比参考集合更低风险而拒绝。

Mandate 选择 C3 默认含义是 `authorized_max_level=C3`、`sigma_cap=b3`。**不加 `sigma >= b2` 约束**。最终组合可能实际属于 C1／C2，页面同时显示“授权不超过 C3”和“估计风险 C2”。

恰好等于边界时属于较低的一档，和 `sigma <= cap` 保持一致；容差固定并进入测试。

## 4.7 人工阈值与同步风险指标

管理员可直接填写阈值或覆盖自动边界；需非负、严格递增、无间隙，并保存自动值、覆盖值、原因和操作者。超出参考前沿范围的人工区间允许作为政策阈值，但该部分参考收益／代表组合应标记 `not_calibrated`，不能外推假值。

每档保存：边界、参考前沿子集、代表权重、参考收益、参考波动、ES／CVaR、历史回撤、可选压力与模拟回撤。

这些是**该参考组合／参考区间的画像**，不是所有同波动组合的风险上界。默认作为诊断；若组织将其升级为硬约束，必须对每个实际候选重新计算，并要求求解器／最终门禁真实消费相同风险定义。第 13 节规定风险测量公式。

同一个“C3”必须连同 RiskScaleVersion、币种和风险口径解释；不宣称它是法律意义的适当性等级，也不宣称跨项目存在唯一绝对阈值。

---

# 5. 投资目标与边界：事实、治理、推导与确认

## 5.1 功能定位与保留项

保留 `InvestmentObjectivesWorkspace` 和三个成功标准：

| 类型 | 用户提供 | 系统计算 | SAA 主要消费 |
|---|---|---|---|
| `absolute_return` | 明确收益要求及治理来源、期限、币种 | 同口径全局参考与实际可达性 | 年化算术预期收益下限与风险约束 |
| `funding_goal` | 资本、储备、投入、支付、期末最低余额 | 所需有效收益、资金缺口、成功率、可行等级集合 | 现金流成功条件，不把所需复合收益硬塞进算术收益字段 |
| `benchmark_relative` | 正式基准、超额目标与主动风险预算 | 预期超额、TE、最低必要主动风险 | 绝对风险上限与相对风险约束同时满足 |

机构情境优先展示 family office、asset manager、corporate treasury。现有 personal 和资产负债背景信息保留，但不将背景负债字段称为已实现 ALM。

这种“资金用途、成功标准、期限、流动性与授权先于组合选择”的组织方式与 IPS 框架一致；企业现金可以先盘点未来流入流出及流动性需求再作分层。本文据此设计事实与治理输入，但不把机构公开原则外推成唯一的量化风险容忍公式。[R11][R14]

## 5.2 哪些数字由谁确定

| 责任 | 示例 | 产品要求 |
|---|---|---|
| 已知事实 | 当前资本、币种、已签约费用、确定支付 | 用户／业务系统输入，保存来源 |
| 业务计划 | 未来投入、资本开支、期末保有金额、期限 | 标注确定／计划／条件性，不把所有预算称为已知事实 |
| 治理要求 | 成功概率下限、支付失败上限、保护底线、最高授权等级、复核规则 | 来自 OrganizationPolicyVersion 或显式确认；不存在系统可发现的唯一正确数字 |
| 市场假设 | 通胀、未来回报、风险、条件性投入压力比例 | 来源于参考 CMA、政策模板或研究假设，显示版本与不确定性 |
| 模型输出 | 所需收益、流动性金额、模拟成功率、建议等级 | 系统计算，不让用户填写“计算结果” |

原代码中的 80% 成功率、50% 压力投入比例、20% 回撤预警等仅是默认值，不作为机构标准。没有已确认的治理模板时，应提示待确认，不通过换一个隐藏常数解决“拍脑袋”问题。

## 5.3 两种风险等级入口

**手动选择。** 展示冻结 C1–C5 的波动范围与参考画像，用户选择授权上限。系统解析数值上限；高级覆盖记录原因和最终实际授权，不把更高数值仍标为原 C2。

**现金流推导。** 在冻结 Universal 参考条件和资金政策下，检验参考候选是否能完成资金计划，推荐其中风险最低的已验证候选及其等级。返回推荐值、全体已测试结果和选择理由，由用户确认。

必须分别保存：

```text
minimum_tested_feasible_level       模型下已找到的最低参考等级
funding_feasible_candidates         哪些候选通过资金检验
authorized_max_level               机构正式允许到哪一档
selected_max_level                 本次确认采用的上限
realized_model_risk_level           某一组合在特定模型下的风险归类
```

现金流不能唯一推出心理意愿，也不能脱离资本保护政策证明“最高可承受风险”。缺少授权证据时只能给建议，不能替机构完成授权。

## 5.4 现金流展开与所需收益

可投资本金：

\[
W_0=Capital-OutsideReserve>0.
\tag{F1}
\]

组合外储备只扣一次，不参与组合收益和支付；组合内现金仍是组合资产，不能再次从 W0 扣除。

基础版以等长模型月为周期。当前代码只支持整数年目标；短于一年或非整数年计划不能直接宣称已支持，扩展到任意模型月需同步修改契约、最后不完整年度的输出和测试，不能四舍五入成一年。实值计划转换为名义值：

\[
CF_t^{nom}=CF_t^{real}(1+\pi)^{t/12}.
\tag{F2}
\]

若期末目标 K 以研究日购买力输入，也须用 `K_nominal=K_real*(1+pi)^H` 转换；名义值不再次加通胀。

年有效扣费前常数收益 r、年有效费用 f 对应：

\[
g(r)=((1+r)(1-f))^{1/12},\qquad
W_t=W_{t-1}g(r)+C_t-O_t.
\tag{F3}
\]

求最小 r，使各期不发生资金不足且期末满足目标 K。复用现有二分法及 `[-0.99,5.0]` 搜索范围；范围下界已满足、范围内找到、超出上界是不同状态，超界不能返回 500% 当作已求得解。

由于测试路径在失败前资产非负、增长因子随 r 单调，可使用可行性二分。复杂融资、负资本或非常规现金流模型不自动复用此结论。

结果名称必须是 `required_effective_return_gross_of_model_fee`。例如无现金流时：

\[
r_{required}=\frac{(K/W_0)^{1/H}}{1-f}-1.
\tag{F4}
\]

它不是 CMA 的 `mu`，也不是对真实市场未来收益的预测。

## 5.5 流动性与现金保障

压力认可比例 eta 下，前 L 月所需资金：

\[
L_{need}=\max\left(0,\max_{1\le t\le L}\sum_{i=1}^t(O_i-\eta C_i)\right),\quad
\ell_{funding}=L_{need}/W_0.
\tag{F5}
\]

`max(0,...)` 必须保留。保障窗口、投入压力规则是政策假设；不自动把未来不确定投入当成现金。

有效流动性下限：

\[
\ell_{effective}=\max(\ell_{policy},\ell_{funding}).
\tag{F6}
\]

若大于 1，返回资本／支付冲突。禁止裁成 100% 后宣称问题解决。

需要区分“可交易资产”与“可承担现金保障的资产”。股票 ETF 即使标为 liquid，也不能自动满足经营现金保全。沿用现有 cash-role 硬约束，另存 `cash_eligible`／结算或赎回证据；尚无产品映射时只能做大类参考资格，最终产品阶段复核。

## 5.6 全局诊断与实际诊断

全局诊断使用 RiskScaleVersion 绑定的参考 CMA，不依赖本次尚未生成的实际 CMA。实际资产和 CMA 完成后，SAA 再做第二次诊断。

资金目标的最终检验按第 13 节模拟现金流。**不再把 `mu'w >= r_required` 当作资金可行性的充分条件，也不强制建议等级必须高于这种混口径筛选产生的等级。**

对于同口径绝对算术收益目标 r_target，才可以求：

\[
\min w^\top\Sigma w\quad\text{s.t. }\mu^\top w\ge r_{target}
\tag{F7}
\]

作为目标对应的最低风险参考。资金目标的 deterministic required return 仍可展示，但应与具体分布下的财富结果并列，不跨口径直接比较。

相对基准目标只有在基准可明确映射到 Universal 资产轴时才能做全局主动前沿；否则标为 `awaiting_actual_scope`，不按同名资产猜测。实际研究中计算 U4 与 `(w-b)'mu`。

## 5.7 风险区间不是风险上限

若只提高授权波动上限，其他条件不变：

\[
\mathcal F(b_1)\subseteq\mathcal F(b_2)\subseteq\cdots\subseteq\mathcal F(b_5).
\tag{F8}
\]

因此上限为 C3 时已有可行组合，上限为 C5 时同一组合仍可行。可以说“实际风险落在 C5 区间的参考组合支付失败较多”，不能说“允许 C5 反而不存在 C3 的可行组合”。

页面同时提供“各风险区间内候选的表现”和“在某个风险上限内可找到什么”，不混为一张不明含义的表。

## 5.8 保存、授权与诊断状态

Mandate 保存的是事实、治理、已选风险上限和诊断证据；即使暂时无模型可行结果，也可保存 `needs_revision`／`reference_pending`，不得因此禁止继续研究其他实际 CMA。

采纳 SAA 则要求最终权重通过本次真实输入下的硬约束和所要求的资金检验。Universal 参考不足不能冒充真实不可行证明；真实硬约束冲突不能被一个“全局参考通过”覆盖。

上述五项是业务依赖，当前界面已收敛为两步：“目标与约束 → 结果确认”。资金任务、成功标准及风险授权在第一步共同填写，诊断及确认在第二步；不再把旧五步页面顺序当作当前界面。技术参数如随机种子、路径数、误差阈值折叠，计算依据与警告可展开。

现有 `risk_aversion` 保留为效用模型软参数，不定义 C1–C5。若使用同一冻结参考前沿校准，局部平滑内点有 `lambda = 2 dmu/d(sigma²)`；端点、平台段或斜率不稳定时不强行反演，效用候选标为未校准，默认使用明确风险上限下的收益目标。不能对每套 CMA 各自任意校准 lambda 后再声称效用数值可直接比较。

---

## 5.9 资金预算应独立于三种成功标准

旧 `schema_version=1.0` 将 `funding_plan` 与 `objective_kind=funding_goal` 绑定；其读取及兼容校验保持原语义。当前 `schema_version=2.0` 已将 `cash_budget`、`funding_target`、`cash_protection` 分开，三个目标可以共享一份预算，不能继续把旧版限制描述为当前全局限制。

当前 2.0 已解决这一基础契约问题：**企业以绝对收益为考核目标，也可能有每季度支付；以相对基准为目标，也仍需保留运营现金。** 以下字段展示其领域职责；任意月份续算等更完整适配仍按第 17.5 节单独实施：

```text
cash_budget（可选，三个 objective_kind 都可使用）
  capital / outside_reserve / confirmed_balance_as_of
  inflows / necessary_payments / currency / model_calendar
  fees / cash_eligibility_policy / liquidity_window

success_criterion（按 objective_kind 选择）
  absolute_return:  同口径预期收益要求
  funding_goal:    支付完成 + 期末余额要求
  benchmark_relative: 超额收益要求 + 主动风险预算

cash_protection_policy（有预算时按组织政策明确）
  payments_only | payments_and_terminal_floor
  failure_probability_limit / protected_balance_rule / source_version
```

F1–F6 对共同预算计算一次。绝对／相对收益目标如有现金预算，则在收益目标检查之外，另做所要求的支付保障检验；没有期末金额目标时不编造一个。允许只以期间必要支付为资金成功标准，但“无支付、无余额目标”的空条件不得被展示为投资成功率 100%。

旧 `funding_plan` 可在边界映射为一份预算与一份成功条件，旧结果继续按原契约只读。新请求同时携带重复金额来源时拒绝，而不是把旧字段与新字段相加。组织资产负债快照仍只是背景，不因新增现金预算就成为 ALM。

# 6. 大类构建：统一 Proxy Contract

## 6.1 产品代理与指数代理

两种模式均输出 AssetClassVersion、ProxyBindingVersion 和收益面板。

| 项目 | 产品代理 | 指数代理 |
|---|---|---|
| 来源范围 | 已冻结、可追溯的产品池 | 有明确数据权限和口径的指数库 |
| 代理成分 | 单产品或多个 ETF／基金 | 单指数或复合指数 |
| 构建操作 | 手动或自动分类、类内权重 | 手动组合或对指数集合自动分组 |
| 后续 CMA／SAA／TAA | 共用统一资产轴及模型 | 共用统一资产轴及模型 |
| 实施范围 | 原授权产品池及其使用限制 | TAA 后选择指定产品池或全市场候选 |

产品池中的产品数量不决定战略预算。分类是分类，不用 SAA 的最优权重反过来改写类别定义。

## 6.2 手动与自动分类的边界

保留当前 rule、hierarchical、k-medoids、k-means、spectral、GMM，以及 correlation、去噪相关、metrics、PCA、blend 等能力。统计相似性用于辅助分组；合同分类／经济暴露约束作为独立、可选择且可见的规则。

指数模式复用同一特征和分类框架，但需要独立的指数类型、成分范围、总收益口径适配，不通过把指数代码伪装成 ETF 进入现有产品接口。

自动分组结果须明确成分归属、代表代理、类内权重、未分配项目及原因；与上游固定产品池／指数集合版本绑定。分类规则、权重算法改变须新建版本。

## 6.3 代理收益与权重时点

统一公式：

\[
r_{class,t}=\sum_i h_{i,t-1}r_{i,t},\qquad
NAV_{class,t}=NAV_{class,t-1}(1+r_{class,t}).
\tag{P1}
\]

若每期使用固定目标权重，这是“每个观测期再平衡的合成代理”，不是买入后不动的组合。若是买入持有：

\[
h_{i,t}=\frac{h_{i,t-1}(1+r_{i,t})}{1+r_{class,t}}.
\tag{P2}
\]

月度／季度再平衡则在指定时点重设目标，其余时间漂移。绑定中必须保存 rebalancing rule、权重可得日、费用、初始净值和日历；不能把今天挑出的优秀产品回填到历史，宣称当时已知道。

现有 `fit.py` 使用固定类内权重形成加权收益；其容错路径可能略过未映射成分并归一化。新正式 Proxy 发布须对所有请求成分覆盖做独立门禁，不能把既有函数存在当作“无缺失、无隐式范围变化”的证明。原历史调用行为不能在本次文档任务中直接修改。

## 6.4 收益、币种、费用与缺失

优先使用可验证总收益／复权净值。价格指数、全收益指数、基金净值不互相冒充。外币未对冲转换满足：

\[
1+r^{base}_t=(1+r^{local}_t)(1+r^{FX}_t),
\tag{P3}
\]

FX 报价方向、时点、对冲规则必须冻结；无 FX 数据则不能声称自动完成同币种转换。

产品 NAV 已扣管理费时，不在代理和资金模型中重复扣相同费用；实施交易成本、顾问费等另行明确。异常跳点默认阻断或展示，不默认 winsorize 掩盖真实损失；高级清洗须保留原值与规则。

“共同成立以来”指共同可用收益起点，不是让每个资产用不同样本长度后直接拼矩阵。产品缺失不能用其他产品静默替换，复合指数也不得对剩余成分自动再归一化。

---

# 7. CMA 独立中心：五类方法及自动化边界

## 7.1 方法不是五套独立数据管道

统一入口：`AssetClassVersion + ProxyBindingVersion + as_of + horizon + currency`。共用 Evidence Builder 处理收益轴、历史区间、风险矩、来源核验与数据质量。

| 方法 | 目标 method | 自动计算 | 默认留给用户的决策 |
|---|---|---|---|
| 历史统计 | `historical_statistics` | 样本截取、收益、均值、协方差、年化、误差提示 | 1Y／2Y／3Y／5Y／10Y／共同起点／自定义；高级风险估计参数 |
| 手动／直接假设 | `manual`，兼容原 `model=None` | Proxy 风险参考、矩阵与口径校验 | 长期收益假设、来源；高级风险覆盖 |
| 贝叶斯 | `bayesian_niw` | 先验与似然核验、解析后验、风险与均值不确定性 | 先验版本、证据窗口、已版本化先验强度 |
| Black–Litterman | `black_litterman` | 风险、参考权重解析、风险价格、观点矩阵与后验 | 参考组合来源、观点及可信程度；高级覆盖 |
| 状态／情景 | `scenario_mixture` 加明确 source kind | 事后状态桥接、共同轴条件统计、占用率、矩混合 | 状态版本、区间、应用概率与有依据的覆盖 |

这个五分类是项目基础工具箱，不是业内统一的强制分类。CFA 公开方法还包括现金流折现、风险溢价等；这些可作为未来独立生成器或先验来源。[R1]

Monte Carlo／Bootstrap 是分布、估计误差与验证工具，可被上述方法使用；并非一份没有参数来源的“自动第六个预测”。机构方法可以组合统计估计、前瞻判断和模拟，不意味着必须选择五套 CMA 或把它们等权处理。[R2][R3]

## 7.2 通用 CmaVersion

```text
id / definition_version / method / source_kind
asset_class_version_id / ordered_asset_ids / proxy_bindings
as_of / created_at / available_at / valid_until
currency / forecast_horizon_years / horizon_applicability
return_basis / moment_semantics / observation_frequency / periods_per_year
fee_basis / fx_hedging_basis / covariance_role
base_period_mean / base_period_covariance (如适用)
effective_returns[N] / effective_covariance[N,N]
mean_uncertainty_status / posterior_mean_covariance (可选)
mean_uncertainty[N] (可选，附来源与解释)
distribution_spec (可选)
data_lineage / model_definition / overrides
quality / limitations / execution / content_hash
```

缺少统计估计的不确定性不等于没有不确定性。`unknown` 不自动填零；旧兼容接口需要数值半宽时，0 只能表示本次未施加该惩罚，并保留 `uncertainty_not_estimated`。

默认输出资产实现风险，均值估计不确定性另存。采用 posterior predictive 风险在数学上可以合理，但必须明确其已含哪些不确定性，下游不能再按另一字段重复计入。

## 7.3 自动值、人工覆盖与合法性

每个参数保存 `source_value / applied_value / source_ref / override_reason`。人工修改自动风险后，不再沿用原始“完全来自历史”的认证哈希；保存为有覆盖的新 CMA。

验证资产轴、有限值、对称性、PSD、概率合计与数据日期。相关矩阵各元素落在 [-1,1] 不足以保证 PSD；不在用户未确认时偷偷投影修复。

现有收益限制如 [-50%,200%] 与协方差对角限制是既有契约；新历史／状态参数超出范围时不能裁剪后发布。新细分口径须设计对应范围验证，迁移旧接口需显式版本与回归，不能用错误的年度简单收益边界约束所有年化参数。

## 7.4 不同模型不能共享的东西

相同数据准备可以共用，信息含义不能共用：Historical 的均值是输出；Bayesian 的样本是证据；BL 的参考组合是先验来源；Regime 的占用率是历史混合权重。任何一个模型的输出不能仅通过改标题变成另一个方法。

---

# 8. Historical CMA：样本区间与统计估计

## 8.1 窗口解析

提供 `1Y / 2Y / 3Y / 5Y / 10Y / common_since_inception / custom`。

相对窗口以研究 as_of 为锚，结束于该时点实际可得的最后完整观测；显示请求区间和实际区间。近一年是日历回溯，不自动等于“最近 252 条”；高级观察数窗口与日历窗口是两种不同参数。

不得静默缩短用户指定的区间。无法覆盖时提供：调整窗口、修改代理、明确接受缩短并形成新请求。月频最后一个不完整月份不进入完整月度统计。

同一 CMA 的均值和协方差默认取同一共同有效样本。不同模型可以采用不同历史窗口，这正是模型比较的一部分，但模型内部不得无提示地混用样本。

## 8.2 内置估计器

\[
\bar r=\frac1T\sum_{t=1}^T r_t,\qquad
S=\frac1{T-1}\sum_{t=1}^T(r_t-\bar r)(r_t-\bar r)^\top.
\tag{H1}
\]

可选择原始样本协方差或对角目标收缩：

\[
S_\lambda=(1-\lambda)S+\lambda\operatorname{diag}(S),\quad0\le\lambda\le1.
\tag{H2}
\]

再按 U1 转到年化算术参数。当前实现的 lambda 是固定配置值，不是自动估计的 Ledoit–Wolf 系数。未来加入自动收缩时必须独立注册目标矩阵、强度估计公式与测试，不能改名字就声称已实现。[R1]

数学最低要求 T>1；生产样本门槛由版本化数据质量策略给出。当前风险参考最低 20 条仅是技术门槛，不等于长期估计可靠。T 不大于 N 时原始样本矩阵可能秩不足；PSD 允许秩不足不代表估计稳定，需显式选择收缩／先验或停止要求正定的模型。

## 8.3 误差与异常提示

在独立同分布假设下，基础期均值标准误约为 `sqrt(S_ii/T)`；其年化值乘 a，而不是乘 sqrt(a)。存在自相关时该公式仅作 iid 参考，应使用有明确口径的 HAC／区块重采样扩展，或标记未校准。

样本均值与协方差来自同一历史数据不意味着估计难度一样，也不意味着样本均值天然“有偏”。应区分抽样误差、非平稳与结构性偏差，不能用笼统的“不准”禁止历史方法。[R1]

1Y／2Y 可以使用，但显示周期覆盖不足、均值不稳定与最近状态敏感提示；不自动把短期样本称为 LTCMA 的最佳长期中枢。

输出 `forecast_semantics=historical_sample_baseline`。只有用户明确选择本方法时，历史均值才成为该 CMA 的有效预期输入；其他方法不被后台默认替换成它。

---

# 9. Manual CMA：手动收益，默认自动风险

用户选择资产版本、日期、币种、期限后，系统加载 Proxy 风险参考：

\[
\Sigma=D\,Corr\,D,\qquad D=\operatorname{diag}(\sigma_1,\ldots,\sigma_N).
\tag{M1}
\]

默认只填写各大类年化算术预期收益、判断依据与必要治理字段；不要求用户手工维护 N×N 矩阵。提供查看风险矩阵与高级覆盖入口。

风险来源优先顺序是用户已选且可用的冻结风险版本、当前明确请求的历史估计、显式导入。不是系统静默选择一个“最好看”的窗口。

没有 Proxy 历史数据时，允许显式外部／人工风险输入并清楚标记；不能用单位矩阵或零相关冒充自动估计。当前指数来源历史风险受现有 CmaRequest 约束限制，自动风险接入需要真实接口改造，不只是前端回填。

保留高级波动率、相关性或完整协方差覆盖；覆盖后重新验证 PSD 与来源。手动收益不需要伪装成统计显著性，缺少可估计不确定性时明示。

---

# 10. Bayesian CMA：NIW 从先验到单套有效参数

## 10.1 模型与使用条件

这里是一般贝叶斯参数估计，不是朴素贝叶斯分类。基础模型使用同频率多元正态似然和 Normal–Inverse-Wishart 共轭先验；可解析更新，不需要默认 MCMC。[R4]

在基础观测频率中：

\[
r_t\mid\mu_p,\Sigma_p\sim N(\mu_p,\Sigma_p),\quad
\Sigma_p\sim IW(\Psi_0,\nu_0),\quad
\mu_p\mid\Sigma_p\sim N(\mu_0,\Sigma_p/\kappa_0).
\tag{B1}
\]

要求 `kappa0>0`、`nu0>N+1`、Psi0 正定，才能按本文输出有限协方差期望。似然样本在该假设下条件独立；不把波动聚集自动当成已建模。

## 10.2 先验参数来源

| 参数 | 来源规则 |
|---|---|
| mu0 | 上一版已确认 CMA、显式人工长期锚、BL 先验或未来基本面／外部模型 |
| Sigma0 | 已确认风险矩、上一版 CMA 或不同证据窗口的历史风险 |
| kappa0 | 已版本化的均值先验信息量，不是 0–100% 置信概率 |
| nu0 | 风险先验信息量，使用 `nu0=N+1+n0_cov` 表达 |
| Psi0 | `n0_cov * Sigma0`，保证先验风险期望为 Sigma0 |

普通界面提供先验版本、证据窗口及弱／中／强模板，但每个模板必须显示其对应的 **基础频率等效观察数** `kappa0 / n0_cov`。没有已确认模板时，不伪造默认先验强度。

“24 个等效月”不是 24 条日观察；更不能仅把 kappa0 乘一个交易日数，就声称 NIW 先验在不同频率下完全等价。第一版采用 **先验拟合频率与似然频率一致** 的门禁。已有 NIW 后验续更锁定原频率；需要切换日／月频时，从已确认的长期中心参数按新频率重新建立先验版本，并显式选择该频率的强度模板。只转换收益矩的单位，不自动恢复原先验的信息量或分布。

若使用上一版 NIW 后验续更，新似然默认只包含上次样本截止后的新增证据。若需要衰减，使用明确 d∈(0,1]：

\[
\kappa_0=d\kappa_{old},\quad
n_{0,cov}=d(\nu_{old}-N-1),\quad
\Psi_0=n_{0,cov}E[\Sigma\mid D_{old}].
\tag{B2}
\]

这保留先验均值和风险中心，降低强度。d 的选择是版本化适应性策略，不是新事实。其他 CMA 没有 NIW 强度时必须提供先验构造模板，不能仅凭一个协方差还原不存在的后验分布。

先验和似然复用同一数据属于经验贝叶斯／数据复用，须标记并限制置信度解释；不宣称两份独立证据。若未来自动校准强度，用时间序列训练／验证划分优化预测得分，不使用最终测试集，也不在此运行成千上万次 SAA。

## 10.3 解析后验

令样本均值为 r_bar，离差和矩阵为 `A = sum((r-r_bar)(r-r_bar)')`，注意 A **不是除过 T−1 的样本协方差**。

\[
\kappa_T=\kappa_0+T,\quad \nu_T=\nu_0+T,
\quad\mu_T=\frac{\kappa_0\mu_0+T\bar r}{\kappa_T}.
\tag{B3}
\]

\[
\Psi_T=\Psi_0+A+\frac{\kappa_0T}{\kappa_T}(\bar r-\mu_0)(\bar r-\mu_0)^\top.
\tag{B4}
\]

\[
C_p=E[\Sigma_p\mid D]=\frac{\Psi_T}{\nu_T-N-1},\qquad
U_p=\operatorname{Var}(\mu_p\mid D)=\frac{C_p}{\kappa_T}.
\tag{B5}
\]

基础期后验预测协方差为 `C_p + U_p`。均值边际分布是自由度 `d_f=nu_T-N+1` 的 t 分布，尺度矩阵：

\[
T_\mu=\frac{\Psi_T}{\kappa_Td_f}.
\tag{B6}
\]

t 的尺度矩阵不等于协方差；`Cov(mu)=d_f/(d_f-2)*T_mu`。各资产边际可信半宽为 `t_quantile * sqrt(T_mu[ii])`；边际 95% 不是整个 N 维向量同时 95%。

## 10.4 年化输出与避免重复计量

采用年化算术参数时：

\[
\mu_a=a\mu_T,\quad \Sigma_a=aC_p,\quad U_a=a^2U_p.
\tag{B7}
\]

模型输出默认：`effective_returns=mu_a`、`effective_covariance=Sigma_a`、`posterior_mean_covariance=U_a`。

若未来 a 期中共享同一个未知均值，在条件独立、加法累计近似下，预测协方差是：

\[
\Sigma_{pred,a}=aC_p+a^2U_p,
\tag{B8}
\]

不是简单把单期预测协方差乘 a。复合收益预测还需第 13 节的模型，不能由 B8 自动得到。

Bayesian 生成器计算一次解析后验，SAA 消费一套参数。后验抽样如用于分布检验，也不是每个 draw 都必须做优化。

## 10.5 第一版实现与失败条件

2026-09-18 第一版已实现 `bayesian_niw` 契约、固定签名 NIW 内核、数值 t 分位数及先验选择器，并核对了 SciPy 参考值。当前支持同日频下重新构造先验与后验续更；基础强度由用户明确输入，不伪造弱／中／强模板；尚未实现上述全部衰减策略或跨频率适配。

数值更新可用矩阵乘法与稳定散点累积；不得显式求逆。先验缺失、非正定、样本频率不匹配、旧先验与新数据重叠未声明、可信区间尺度混淆均阻断正式发布。

t 分位数属于数值实现要求：可在受控预计算表／已验证固定签名函数中求取，并记录误差与支持范围；不能默默用 1.96 代替所有自由度。

---

# 11. Black–Litterman：参考均衡、观点与自动参数来源

## 11.1 基础数学及先验含义

经典 BL 使用均衡收益作为中性起点，再结合观点；它属于贝叶斯方法家族，但不等于一般 NIW。[R5]

\[
\Pi=\delta\Sigma w_{ref},\qquad
\mu_{prior}=r_f\mathbf1+\Pi.
\tag{L1}
\]

这是风险资产超额收益均值方差反向优化的构造。若使用任意机构基准、含现金代理或其他实施约束，称为 `benchmark/reference-implied prior`；不声称由真实全球市场唯一识别。

存在满仓约束及活跃上下界时，反向最优条件还包含约束乘子。给定一个参考权重并不能唯一证明 L1 是市场真实预期收益；本项目将 L1 作为明确的先验构造规则，而不是因果预测。

## 11.2 参数自动来源表

| 参数 | 自动准备规则 | 不能做的事 |
|---|---|---|
| Sigma | 同资产轴 Proxy 风险估计或明确风险版本 | 让用户默认手填；无数据时伪造相关性 |
| w_ref | 已发布中性基准；或通过完整规模门禁后按市值生成 | 用指数内部权重、ETF AUM、复合指数内部权重冒充跨资产市场组合 |
| delta | `RP_ref / (w_ref' Sigma w_ref)`；RP_ref 来源必须明确 | 把本次 Mandate 的个人／机构效用偏好冒充市场风险价格 |
| r_f | 同币种、同收益口径的现金／利率假设版本 | 今天隔夜利率自动代表未来十年现金收益 |
| tau | 版本化先验不确定性设置，保留当前 0.05 作为可选兼容起点 | 宣称它是唯一正确常数 |
| P、Q | 从绝对／相对观点自动构建 | 让一般用户手写矩阵 |
| Omega | 当前明确正 `view_std²` 的对角矩阵 | 把“80% 自信”直接当方差 0.8 |

无风险利率、参考权重及溢价来源不可得时，显示缺失并允许显式研究输入；“偏自动”不等于自动猜值。

若历史参考组合溢价用于估 delta，使用明确再平衡规则的组合历史超额收益与同口径方差。负／零估计溢价不满足当前 delta>0 契约时，应提示改用有依据的长期溢价或另建研究，不取绝对值修补。

## 11.3 市值权重的数据门禁

严谨市值构造要求同资产范围、同币种、可比较市场价值口径、无底层重叠、可得时点与更新日期可追溯。

- `index_weights_df.parquet`／Tushare `index_weight` 表示指数内部成分权重，不是股票、债券、黄金之间的权重。[R12]
- `index_daily_basic_df.parquet` 的指数市值可能覆盖重叠股票，不能直接相加。
- ETF 的 AUM 是包装产品规模，不是全体底层资产市值；可以作为 `aum_reference`，但必须明确不是 CAPM 市场组合。
- 股票基础目录不能自动提供所有股票研究日市值；债券与海外资产还需要各自规模、币种和日期口径。

这些是数据准备要求。本轮未重新盘点正式活跃数据快照；实现时须重新核对 `data/tushare_active.json`、实际 Parquet schema 与数据可得性。不能把文档中出现文件名当作当前数据已完整的证据。

## 11.4 观点与后验

将总收益绝对观点或相对收益差转换为超额观点：

\[
Q^e=Q-r_fP\mathbf1.
\tag{L2}
\]

例如三资产轴股票、债券、黄金：黄金总收益 5% 对应 P=[0,0,1]；股票比债券高 3 个百分点对应 P=[1,−1,0]。相对观点的 P1=0，所以无风险利率抵消。

令 `V=tau*Sigma`、`S_v=P V P' + Omega`，通过线性方程求 K，而非显式求逆：

\[
K=VP^\top S_v^{-1},\quad
\mu^e_{BL}=\Pi+K(Q^e-P\Pi),\quad
\mu_{BL}=\mu^e_{BL}+r_f\mathbf1.
\tag{L3}
\]

后验均值协方差使用 Joseph 形式：

\[
U_{BL}=(I-KP)V(I-KP)^\top+K\Omega K^\top.
\tag{L4}
\]

当前实现默认保留资产风险 Sigma 不变，U_BL 单独保存。无观点时总收益为 `r_f + Pi`，**不是仅 Pi**。

其他实现可以使用 Sigma+U_BL 表示后验预测风险；本项目不默认这样做，以避免风险语义变化及与均值惩罚重复。若未来开放，需明确 `covariance_role` 与下游去重规则。[R6]

当前 view_std 独立于 tau，所以不能直接引用“tau 会抵消”的某些特殊 Omega 构造结论。0–100% confidence 转 Omega 需明确转换算法；第一版继续以观点标准差为执行契约，不伪装已经接入 Idzorek 校准。

## 11.5 现金与奇异矩阵

当前模型要求各资产正方差。确定性无风险现金不应通过人为加一个极小方差绕过门禁。第一版可使用有实际波动的现金代理；若需要数学上确定性现金，应在新模型中显式分离风险资产块与现金，并验证完整资产轴回填，不回写旧模型。

S_v 在正观点噪声下可求解不意味着所有旧协方差门禁都支持奇异资产；具体支持范围以契约与测试为准。

---

# 12. Regime／Scenario CMA：历史占用与年度情景分开

## 12.1 责任边界与现有缺口

情景算法中心负责保存“哪些日期属于哪些状态”的不可变研究结果；CMA 中的桥接层负责用这些日期，对**本次所有大类的共同收益轴**估计状态参数。

当前 `conditional_stats` 主要针对单个评价目标，且带 `forward_1_period_from_signal` 语义，不能直接充当多资产状态协方差。

当前 LTCMA 的 `cma_evidence.py` 已提供证据桥接；历史中心与 CMA 保持单向引用：

```text
HistoricalRegimeRun + ReturnPanelVersion
→ RegimeEvidenceBuilder
→ 带基础频率的状态条件矩与占用率
→ 单一共享混合内核 + 明确期限适配
→ CmaVersion
```

基础版从事后识别导入；实时状态仍用于 TAA／监控。历史实时输出也可以作为另一个有独立定义的研究数据源，但不是本期默认桥接，更不以“今天熊市”覆盖未来十年。

## 12.2 状态轴、收益归属与有效样本

使用同一个市场状态定义切分全部资产。例如用权益市场牛／熊／震荡作为条件，研究债券、黄金在**权益牛市期间**的收益；不能分别筛每个资产自身牛市后拼成一行多资产样本。

先计算每个真实相邻观测的收益 r_t，再依据冻结规则把 r_t 分配给状态。第一版默认 `return_end_state`：区间 `(t−1,t]` 的收益归属于 t 对应的事后状态；跨状态边界的收益只计一次，并在诊断中披露。

不把分散在多年中的“熊市价格点”拼在一起重新求收益，不忽略中间状态形成跨期跳跃。若采用“必须两端同状态”的严格口径，会改变样本与占用率，须作为独立选项，并不再声称重构原全样本矩。

Unknown、缺失 Proxy、无效间隔从同一个多资产共同面板中排除，保留排除原因和覆盖率；不归入震荡、不填零。合法样本不足按质量策略阻断正式生成。

## 12.3 默认概率与用户覆盖

在当前有效共同面板上：

\[
p_s^{detected}=n_s/T,\qquad T=\sum_s n_s.
\tag{G1}
\]

这是时间占用率，不是状态段数量占比、HMM 某日置信度或转移概率。保存原运行全样本占用率与当前共同样本占用率的差异，避免用户误以为发生了参数错误。

显示“历史识别占用率／本次应用权重”两列；默认相同。修改后保留 detected、applied、原因并支持恢复。应用权重须非负且合计 1，禁止静默归一化。

零概率状态保留用于审计，但区分“合法、尚无样本”和“数据／状态本身非法”。桥接结果保存完整 `declared_states` 与 `unestimated_states`；数值混合只传入具有完整、合法参数的状态，并保存该执行子集与全状态轴的映射。

没有历史样本且应用权重为 0 的状态可以仅保留为 `unestimated`，不能伪造零均值／零协方差。其应用权重一旦改为正数，必须有显式先验或人工条件参数；否则阻断。已发生样本异常、状态编码非法或缺失覆盖，不能通过把概率设为 0 免除质量门禁。用户新增历史没有出现的情景时，也必须给出独立参数来源。

## 12.4 基础频率条件矩

对状态 s：

\[
m_s=\frac1{n_s}\sum_{t:z_t=s}r_t,\quad
C_s^{ML}=\frac1{n_s}\sum_{t:z_t=s}(r_t-m_s)(r_t-m_s)^\top.
\tag{G2}
\]

默认使用状态专属风险，而非所有状态共用同一个风险矩。小样本状态可明确选择对角收缩、NIW 条件估计或共享风险；共享风险并不能补出不存在的状态均值。

所有均值和协方差在**同一观测频率**统计。段数、完整段数、首尾截断段、持续时间、覆盖率和状态样本量一并输出。

## 12.5 历史占用模式：先混合，再年化

基础频率混合：

\[
m=\sum_s p_s m_s,
\quad C=\sum_s p_s C_s+\sum_s p_s(m_s-m)(m_s-m)^\top.
\tag{G3}
\]

采用独立增量近似时：

\[
\mu_a=a m,\qquad\Sigma_a=a C.
\tag{G4}
\]

**不能**先令 `mu_s=a*m_s`、`Sigma_s=a*C_s`，再套年度混合式。那会把状态间协方差放大为 a² 倍，而正确基础期混合再年化是 a 倍。

历史状态有持续性，严格多期风险还包含 U2 的跨期项。第一版应标为 `historical_occupancy_iid_annualization`，不是已校准的多年状态转移预测。需要保留原时序的 Bootstrap／Markov 路径时，必须另有转移、持续期、再平衡与期限模型，不能只换一个名称。

## 12.6 年度替代情景模式：现有混合公式继续适用

当每个情景描述的是**同一年度收益随机变量在不同世界下的条件分布**，且输入已经是可比的一年简单收益矩：

\[
\mu=\sum_s p_s\mu_s,\quad
\Sigma=\sum_s p_s\Sigma_s+\sum_s p_s(\mu_s-\mu)(\mu_s-\mu)^\top.
\tag{G5}
\]

这是现有 Scenario Mixture 的数学语义。年度世界概率与历史日／月占用概率不能默认视为同一件事。

建议 source kind 明确为：

```text
manual_same_period_scenarios
historical_regime_occupancy
```

当前已提取唯一频率无关的 `mixture_moments_kernel`，返回均值、总协方差、状态内协方差和状态间分歧。`scenario_mixture_kernel` 保留人工情景的输入验证；历史状态经 `cma_statistical_models.py` 在基础频率直接复用共享内核，再年化。两种包装不能交换概率及时间尺度语义。

## 12.7 一个必要的恒等式检查

若 p=n_s/T，使用 ML 条件协方差，未做收缩、覆盖或预测调整，G3 应恢复同一全样本的 ML 均值与协方差。

因此“先切牛熊震荡，再按原历史占用率原样混回去”可能与普通 Historical CMA 完全一样。此时价值是解释状态贡献，不是凭分组自动增加预测信息。

使用 `ddof=1` 时，重构全样本协方差须用：

\[
S_{all}=\frac{\sum_s(n_s-1)S_s+\sum_s n_s(m_s-m)(m_s-m)^\top}{T-1}.
\tag{G6}
\]

未来真正改变结果的来源包括：显式修改概率、状态条件均值／风险收缩、不同长期先验、经验证的状态转移路径等。必须标注改变发生在哪里，不能把机械重构包装为新预测。

## 12.8 发布、Horizon 与跨期信息

建议新增 `cma_reference` 引用用途，保存 run ID、定义 revision、状态映射、run hash、数据知识截止、共同样本、概率覆盖和统计参数哈希。

事后识别可以在当前研究日使用截至当前的信息作历史解释；如果按 2020 年 as_of 研究，不能裁剪一个看过 2026 年行情的识别结果冒充 2020 年知识。

状态中位持续几十天并不自动证明可作十年长期状态生成器。基础模式可作为历史对照并展示周期覆盖；结构性 LTCMA／多年路径资格需要独立 Horizon 证据，不靠把 `horizon_years` 改成 10 获得。

---

# 13. 分布适配、资金成功率及同步风险指标

## 13.1 必须显式增加分布适配层

CMA 的两个矩不能唯一确定 CVaR、回撤、资金短缺概率或多年财富分布。需要 `RiskDistributionSpec`：

```text
source_cma_ids / hashes
moment_semantics / input_frequency
engine / version
horizon / time_step
serial_dependence_assumption
parameter_uncertainty_mode
scenario_persistence_rule
fees / rebalance / cashflow_order
seed / path_count
calibration_quality / limitations
```

路径生成属于模型假设，不是由均值和协方差无条件推出的真相。选择模型原生分布时，必须有真实已实现的采样能力；只保存一个 distribution 名字不算完成。资本充足性使用确定性计算和 Monte Carlo 有公开方法依据，但公开私财富教材不直接替代家办／企业的治理政策。[R15]

## 13.2 与当前 Funding 内核的衔接

当前 `funding_paths_kernel` 把输入 mu、sigma 当作一年简单收益矩，先计算：

\[
v=\ln\left(1+\frac{\sigma^2}{(1+\mu)^2}\right),\quad
\ell=\ln(1+\mu)-\tfrac12v.
\tag{D1}
\]

月度对数增长均值 `(ell+ln(1-fee))/12`，方差 `v/12`。在 mu>−1 的条件下，这个独立对数正态代理可匹配所指定的一年简单收益矩。

若输入只是 `a*日均值` 与 `a*日协方差`，它不是原日过程精确复合后的年度分布。旧版本仍保持旧解释；新研究必须标为 `annual_moment_proxy_approximation`，不能悄悄声称精确。

目标增加 `base_period_moment_match_v2`，优先使用 CMA 保留的基础期矩。对组合：

\[
m_p=w^\top m,\quad c_p=w^\top Cw,\quad
v_p=\ln(1+c_p/(1+m_p)^2),\quad
\ell_p=\ln(1+m_p)-v_p/2.
\tag{D2}
\]

若每年 a 个基础期，按独立增量模型映射至一个模型月：

\[
\ell_{month}=\frac a{12}\ell_p+\frac{\ln(1-f)}{12},\quad
v_{month}=\frac a{12}v_p,\quad G_t=\exp(\ell_{month}+\sqrt{v_{month}}Z_t).
\tag{D3}
\]

适配器要求 `m_p>-1`、`c_p>=0` 且所有输入、对数和输出有限；不能对非法均值加 epsilon 绕过。现金流继续按 F3 的“收益增长 → 月末投入 → 必要支付”顺序。D2 匹配单基础期组合矩，是组合层的对数正态近似，并非宣称多只资产的真实组合必然对数正态。日历日数不等长的模型另存时间步长；基础版明确使用等长模型月。

D3 的年度复合均值与方差可由 lognormal 关系计算，不拿年化算术参数直接称为 CAGR。若缺少基础期矩，可显式使用 D1 近似；用户应看到适配方法和限制。

上述两个适配器委托同一个现金流递推内核，避免复制支付、资本补足与费用逻辑。新适配器是待实现能力，旧 artifact 不重算。

输入 `mean <= -1`、非法方差、非有限数或指数溢出必须返回明确失败，不裁剪为一个可模拟收益。确定性零波动可以在分布／现金递推层作为合法特例，不意味着现有 BL 的正方差输入门禁也应被绕过。

## 13.3 资金成功事件和统计

模拟路径 b 中发生一次未足额支付即记为失败，后续新增投入不能“抹掉”先前失败。禁止负余额作为隐含融资。

\[
I_b=\mathbf1\{\text{所有必要支付均完成且 }W_{H,b}\ge K\},\quad
\hat p=\frac1B\sum_{b=1}^B I_b.
\tag{D4}
\]

同时输出支付失败率、期末财富 P05/P50/P95、期末正缺口期望、未支付金额、市场净值回撤和资本补足。**期末目标缺口的均值不是收益分布的 CVaR。**

需要资本保全／途中余额保护时，必须有明确的货币金额底线或已确认政策规则，并加入 I_b；不能通过现金流自动“发现”一个用户从未授权的最大回撤百分比。

## 13.4 概率区间、选优与独立验证

沿用 Wilson 区间作为单个固定候选的诊断，z=1.959963984540054 对应常用双侧 95%：

\[
L_W=\frac{\hat p+z^2/(2B)-z\sqrt{\hat p(1-\hat p)/B+z^2/(4B^2)}}{1+z^2/B}.
\tag{D5}
\]

该区间描述独立模拟样本的抽样误差，不覆盖 CMA、分布和现金流模型误差。[R7]

新流程必须分两套随机样本：

1. `search_seed`：所有候选使用 common random numbers，比较并选定权重。
2. 冻结选定权重及模型后，使用独立 `validation_seed` 做确认；不能拿验证结果重新挑选另一个候选而继续沿用同一置信解释。

多个 CMA 的每个 95% 区间，不等于整个联合结论有 95% 覆盖。第一版可保留逐项 Wilson 诊断；需要联合采纳保证时，增加明确的有限样本保守门禁，例如对 K 个预先规定的检查、总误差 alpha：

\[
L_H=\max\left(0,\hat p-\sqrt{\frac{\ln(K/\alpha)}{2B}}\right),\quad
U_H=\min\left(1,\hat p+\sqrt{\frac{\ln(K/\alpha)}{2B}}\right).
\tag{D6}
\]

这是独立 Bernoulli 路径下 Hoeffding 界配合联合界的项目实现选择。[R13] K 是实际使用的**单侧界数量**：成功率用下界、失败率用上界各算一项；若同一个统计量两侧都要求联合覆盖，就算两项。模型间可共用随机数，联合界不要求模型间独立，但每个统计量的 B 条路径须满足自身抽样假设。

门禁方法、alpha、B 属于已确认的计算与治理策略；不能隐藏变化。旧记录保留 `wilson_95pct_lower_bound`；新规则生成新版本。反复试验／重复挑选须保留研究次数与验证数据使用记录，不能把固定 seed 的反复调参称为独立验证。

## 13.5 现金流自动建议的可执行算法

在 Universal 参考空间或实际 SAA 研究空间中：

1. 生成经过数值验证的前沿与其他明确候选；包括低风险、流动性约束后的前沿，而非只看五个代表点。
2. 过滤资产／分组／现金／波动／TE 等硬约束。
3. 用同一资金计划逐候选计算 D4–D5；保留全部候选结果。
4. 在搜索样本中选择满足政策门槛的最低风险候选；并列时按成功率、资金缺口、稳定 ID 等预定规则选择。
5. 冻结该候选并使用独立样本验证；通过后映射到 C1–C5，返回建议与已测试集合。
6. 无候选通过时返回 `no_validated_candidate_in_search`；除非另有不可行证据，不写“所有组合数学上无解”。

资金成功概率一般不是凸目标，也不必随收益或实际风险单调增加。均值方差前沿上的有限搜索不保证找到全空间最高成功概率方案。第一阶段明确限定为可复现的候选研究；不伪装成已实现动态规划最优财富策略。

如果授权上限提高，系统仍保留之前的低风险可行候选，保证 F8。仅在高风险区间内的失败不能被误读为更高授权上限失败。

## 13.6 资本补足与目标调整

在固定增长路径 `G_(1:t)` 下，可用折现现金流推导每条路径完成支付与期末目标所需的初始资本，再按采纳规则求足够成功路径对应的资本秩。保留现有 `capital_gate_kernel` 的并列值处理、向上取整到金额精度与同路径重新注入验证。

不得直接用插值分位数当作满足概率门槛的资金金额；不得给出精确金额却在实际递推中失败。

系统可以比较增加本金、延后支付、降低期末目标、调整时间等情景，但这些都是用户确认的新预算，不自动修改原目标。

## 13.7 CVaR、VaR 与回撤的具体计算

**统一约定：损失 L=−R，较大的正值代表较大损失；每个指标附期限、置信水平、币种／单位、模型与费用口径。**

对于同期限正态收益代理：

\[
VaR_\alpha=-\mu+\sigma\Phi^{-1}(\alpha),\quad
ES_\alpha=-\mu+\sigma\frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}.
\tag{D7}
\]

正态简单收益允许低于 −100% 的理论尾部；它只能作为标注清楚的参数参考，不能作为无杠杆本金保护保证。若 ES 为负，应保留数值含义；展示“损失不小于零”是另一个转换字段。

若使用 `1+R=exp(ell+sZ)` 的对数正态代理，令 q=Phi^{-1}(1−alpha)：

\[
VaR_\alpha=1-e^{\ell+sq},\quad
ES_\alpha=1-e^{\ell+s^2/2}\frac{\Phi(q-s)}{1-\alpha}.
\tag{D8}
\]

经验损失样本使用一致的分位数／尾部权重规则，或等价的 CVaR 表示：

\[
ES_\alpha=\min_\eta\left[\eta+\frac1{(1-\alpha)B}\sum_b(L_b-\eta)_+\right].
\tag{D9}
\]

同一实现用于显示和门禁，不因尾部样本数不是整数而取不同规则。CVaR 的优化表示有原始研究支持，但将资金成功事件改成凸约束并不是它自动提供的能力。[R8]

市场最大回撤：

\[
MDD=\max_t\left(1-\frac{V_t}{\max_{u\le t}V_u}\right).
\tag{D10}
\]

V 是剔除外部现金流影响的市场净值路径；不能用扣除大额支付后的财富余额直接计算“市场回撤”。历史、模拟和压力 MDD 分开标识，指定样本／模拟期限，不以历史一次最大值保证未来。

---

# 14. SAA 的共同输入与单 CMA 模式

## 14.1 同一授权，统一可行集

SAA 消费 MandateVersion、RiskScaleVersion 和本次 AssetClassVersion。所有候选满足：

\[
w\ge0,\quad\mathbf1^\top w=1,\quad l\le w\le u,\quad g_l\le Bw\le g_u.
\tag{S1}
\]

叠加现金／流动性、禁止项与相对基准。使用 Mandate 和本次研究约束的交集，下游只能收紧，不能静默放宽。

C1–C5 在默认业务里是**最高允许风险**，对应数值 `sigma_cap`；不设置强迫增险的最低波动约束。研究不同风险等级时，必须显示当前正式授权；高于授权的方案只能作为对照，采纳须新建授权版本。

## 14.2 单 CMA 的基础优化

对同口径绝对收益目标，基础可行集：

\[
\mathcal F_m=\{w\in\mathcal W: w^\top\Sigma_mw\le\sigma_{cap}^2,
\ \mu_m^\top w\ge r_{target}\}.
\tag{S2}
\]

相对目标把收益条件替换为 `(w−b)'mu_m >= excess_target`，并加入 U4 的 TE 上限。Funding Goal 不使用混口径的 r_required 代替收益条件，而是进一步做 D4 的现金流检验。

单模型基础候选保留：最小风险、风险上限下最大收益、名义效用、显式均值保守效用、风险预算匹配。默认推荐使用清楚的风险上限下目标，而不是要求业务用户填写 lambda。

效用候选：

\[
U_m(w)=\mu_m^\top w-\frac\lambda2 w^\top\Sigma_mw.
\tag{S3}
\]

显式不确定半宽 u、非负权重下，旧保守收益为 `mu'w − gamma*u'w`。这是一个所选不确定集合的处理方式；默认多 CMA 兼容模式可只报告模型内不确定性，不叠加惩罚。启用时记录 gamma 与 u 的来源，避免把相同误差重复计算。

## 14.3 风险预算不是普通线性约束

Euler 方差份额：

\[
RC_i(w)=\frac{w_i(\Sigma w)_i}{w^\top\Sigma w}.
\tag{S4}
\]

份额可能为负；零方差时未定义。当前系统基于有限候选最小化份额与目标预算的平方距离，应保持“有限搜索”的描述。

该目标一般不是普通凸 QP；不把所有已有五种候选都宣称为连续全局最优解。新共同配置的精确凸路径优先支持线性收益／凹二次效用，风险预算保留为明确的候选评分或受限研究模式。

## 14.4 可行性与最优性的分开输出

一个点满足全部约束可以证明“找到一个数值可行点”；没有找到不等于不存在。每个求解返回：

```text
status: converged | feasible_approximate | infeasible_certified |
        no_candidate_found | numerical_failure | unsupported
max_constraint_violation / objective_value / optimality_residual
objective_bound / optimality_gap (有证据时)
iterations / candidate_count / algorithm_version
```

发布等级校准前沿需要严格的端点和数值稳定性门禁；普通 SAA 可以在明确标注近似程度且满足业务硬约束时确认研究候选。不得用“已收敛”一词覆盖非凸全局最优的缺失。

---

# 15. 多 CMA 模式 A：参数融合后求一个 SAA

## 15.1 融合前的兼容性门禁

所有 CMA 必须满足：相同资产定义与轴、币种／FX 口径、收益和风险时间尺度、费用口径、目标预测期限，且源信息不晚于本次决策知识截止。`moment_semantics` 与 `covariance_role` 也必须一致：不能把资产实现协方差与已经含均值估计不确定性的预测协方差直接平均或同列声称为相同风险测量。需要变换时必须有来源明确的适配结果，不能通过修改字段名通过校验。

不同历史窗口允许存在；不同 as_of／vintage 不默认当作同一时点预测。允许用户显式采用已知的旧 vintage，但保存其年龄、用途和适用性；不可自动修改 as_of 使旧模型看起来新鲜。

Universal 标尺只提供统一授权阈值，不要求其资产轴和本次实际轴相同。若额外使用 Universal 协方差测量实际组合，必须先有明确暴露映射，不能拿不同维度直接相乘。

## 15.2 权重来源

\[
a_m\ge0,\quad\sum_{m=1}^M a_m=1.
\tag{E1}
\]

支持显式自定义权重、已确认的等权规则、已发布的组合研究权重版本。没有充分样本外证据时，不自动声称某个 CMA 更准确，也不把 UI 权重叫作贝叶斯后验概率。

基于预测得分学习权重是后续独立功能，需要时间序列验证、防止相关模型重复计权、训练截止与权重版本；不能用本次最终 SAA 回测表现倒推本次权重。

## 15.3 默认：参数平均，不添加模型间分歧项

用户选择 `aggregation_semantics=parameter_average`：

\[
\bar\mu=\sum_m a_m\mu_m,\quad
\bar\Sigma=\sum_m a_m\Sigma_m.
\tag{E2}
\]

这是一套明确的折中参数。PSD 矩阵的非负加权和仍为 PSD；统一 lambda 时，以 E2 优化 S3 等价于优化模型效用的加权平均。

不能使用 `sum(a_m²*Sigma_m)`：那隐含把不同模型当成独立投资资产，而它们其实是在描述同一组资产。

协方差平均、相关矩阵平均、波动率平均不是同一个操作。先按 E2 融合协方差，再从对角线和标准化得到波动及相关性。

## 15.4 可选：有概率含义的预测分布混合

仅当 a_m 明确作为同一随机变量、同一时间尺度下的模型／情景概率时：

\[
\Sigma_{mix}=\sum_m a_m\Sigma_m+
\sum_m a_m(\mu_m-\bar\mu)(\mu_m-\bar\mu)^\top.
\tag{E3}
\]

这是分布混合的全协方差公式，不是所有“参数融合”都必须加入的项。界面显示两者差别，并记录 `predictive_distribution_mixture`。基本参数加权需求默认采用 E2，不悄悄替用户改成 E3。

E3 的长期路径仍需指定模型状态是否跨期固定、每期重抽还是按转移切换；矩融合本身不提供这个答案。对历史状态占用率的频率错误，继续按第 12 节避免。

## 15.5 均值不确定性如何处理

多个模型通常共享数据，不能默认认为均值估计误差独立，进而使用 `sum(a_m² U_m)` 声称得到精确融合不确定性。

基础版保存每个 U_m 和模型分歧矩阵；融合不确定性标为未联合校准。若各模型有明确盒式半宽，可使用 `u_bar=sum(a_m*u_m)` 作为已声明集合下的保守界，并标注不是自动统计可信区间。

资产风险、预测风险、均值误差、模型分歧必须独立标记。原 CMA 已含预测均值不确定性时，不再叠加同一成分。

## 15.6 求解、验证与保存

```text
选择 M 个 CMA + 权重 + 融合语义
→ 生成派生有效参数与 lineage
→ 同一 Mandate 下执行单 CMA SAA
→ 展示每个原 CMA 对最终权重的交叉评估
→ 确认 SaaPolicyVersion
```

参数融合模式只保证最终权重满足**所选融合模型**的约束；不自动保证每个原模型都满足。原模型失败项必须可见，不能显示“全模型稳健”。用户需要所有模型同时满足时，应使用模式 B 或显式增加 all-model gate。

资金与尾部诊断也必须继承融合语义：E2 的 `parameter_average` 默认对融合后的矩使用 D1／D2 中已声明的分布适配器，**不能改为随机抽取某一个原 CMA 的路径**，因为后者通常对应 E3 的分布混合风险。采用 E3 时，需同时冻结模型选择概率与跨期持续规则；未实现该路径引擎时只能进行单期矩研究，或显式选用有局限说明的矩匹配代理，不声称完成原生混合分布资金模拟。

派生参数作为研究内不可变子产物保存；需要成为可单独复用的 CMA 时再显式发布 CmaVersion。M=1 必须退化为相同单模型结果，不引入额外偏差。

## 15.7 模式 A 本次实施边界

模式 A 以 E2 为可执行范围，沿用既有单 CMA 数值求解与资金诊断。模式 B 现已独立实现，见第 16 节；E3 原生混合路径、跨币种／跨频率及不同 vintage 适配仍是后续任务。

- **请求**：旧 `cma_id` 单模型调用保留；参数平均显式传 `mode=parameter_average` 与 `cma_refs[{cma_id,content_hash,weight}]`。不接受重复版本、非法权重、合计不为 1 或冲突的单模型引用，不自动归一化。
- **兼容**：新融合仅接受具有显式口径的 V2 CMA；研究日、币种、期限、收益语义、费用、FX、风险角色及完整资产定义须一致。统计窗口可以不同；代理定义、现金／流动性角色不能按资产同名猜测。不同 as_of 的旧 vintage 暂不开放，不能把未实现适配当成可用功能。
- **数值**：复用频率无关混合内核的均值和 `within` 协方差，`between` 单独保存为模型分歧；不对独立最优权重求平均，不平方模型权重。使用既有固定签名与启动预热，派生输出可分配小型数组，原大样本与冻结来源不复制成路径大张量。
- **不确定性**：保留各模型原有不确定性证据；融合半宽为显式盒式边界的权重和，标记未联合校准，不冒充独立误差或新统计置信区间。
- **交叉诊断**：每个最终候选在各原 CMA 下重新计算收益、波动、必要基准与资金检查。仅融合模型参与模式 A 的采纳门禁；原模型失败仍必须可见，不显示“全模型兼容”。资金模型继续明确使用融合矩的年度矩代理近似。
- **冻结与下游**：派生参数是政策内部子产物，保存全部源版本、hash、研究权重和有效参数，不自动发布新的独立 CMA，也不冒用第一个 CMA 的 ID。新政策与 TAA 同时读取冻结融合风险和原模型诊断；TAA 重算各源矩风险与 TE，`goal_check=null`，不表示已做剩余资金续算。历史单模型契约保持原义。
- **界面与失效**：成员、权重、模式、范围或 PIT 变化均使当前预览失效；全部来源读取成功后才允许计算。旧单选草稿显式适配，晚到响应不得恢复已失效候选；历史政策展示完整冻结引用。
- **资源门禁**：最多 20 个来源、30 个大类；资金计算按 `(来源数+1) × 候选数 × 路径数 × 月数` 检查 5000 万预算，中央／保守计算至多两倍。冻结证据使用与仓库相同的 UTF-8 紧凑 JSON 编码，超过 7,000,000 bytes 时在落盘前拒绝，给仓库 8 MB 上限保留余量；不自动减路径或丢来源。

单模型／参数平均的实际接口字段、实现限制和执行结果以最终代码及本次验收记录为准；上述实施边界不将第 16–18 节的全部目标能力提前标记为完成。

---

# 16. 多 CMA 模式 B：共同约束下寻找一套兼容配置

**2026-09-19 实现边界：** 已支持年度线性预期收益（相对目标为预期超额）的 minimax regret 与 maximin，M≤20、N≤30；每个模型分别执行收益、权重／分组、方差及基准 TE 约束。`risk_budget` 在模式 B 显式拒绝；没有静默忽略或替换原模式 A 的风险预算能力。资金模拟仅检查一个共同候选，并保留锚点交叉诊断。第 16.6 节五等级联合前沿、凹效用、CVaR 优化等仍为目标设计。证据见 [模式 B 验收记录](multi-cma-compatible-all-models-acceptance-2026-09-19.md)。

## 16.1 不是简单平均各自权重

先在同一授权和同一目标下，分别生成各 CMA 的 SAA 锚点／候选，再做交叉评估。**各自可行不代表它们的平均在所有模型下可行**；共同解也可能不等于任何一个模型的独立最优点。

共同可行集：

\[
\mathcal F_{all}=\bigcap_{m=1}^M\mathcal F_m.
\tag{C1}
\]

除 S1 外，对每个 m 同时施加：

\[
w^\top\Sigma_mw\le\sigma_{cap}^2,
\tag{C2}
\]

以及该目标需要的收益／主动风险／现金约束。所有模型面对同一个最终 w，不是每个市场状态分别持有一个事后最优组合。

所谓“兼容各种 CMA”，严格指通过所选有限模型集的验证，不代表对所有未知市场模型或未来实际结果有保证。

## 16.2 默认兼容目标：最小化最大收益机会损失

基础版先支持风险上限内最大收益的锚点。对于绝对目标用 `q_m(w)=mu_m'w`，相对目标用 `q_m(w)=(w−b)'mu_m`。

每个模型先求：

\[
V_m^*=\max_{w\in\mathcal F_m}q_m(w).
\tag{C3}
\]

然后求共同配置：

\[
\min_{w\in\mathcal F_{all},\rho\ge0}\rho
\quad\text{s.t. }V_m^*-q_m(w)\le\rho,\ \forall m.
\tag{C4}
\]

含义：在所有模型均满足授权的前提下，使相对于各模型自身最佳方案的最坏收益牺牲尽量小。所有 q_m 必须使用相同收益单位与目标定义。

若 C3 只得到近似锚点，保留其上下界，结果叫 `approximate_regret`；不把有限搜索最好值冒充精确 V_m*。也可以预先声明采用有限候选参照，但须明确搜索域。

## 16.3 可选目标：最大化最坏模型收益／效用

\[
\max_{w,t}t\quad\text{s.t. }w\in\mathcal F_{all},\ t\le q_m(w),\ \forall m.
\tag{C5}
\]

这是 maximin。它与 minimax regret 是两种不同权衡，前端明确选择，不把它们一起称为“平均融合”。

统一 lambda 的凹二次效用也可代替 q_m；参数须在全部模型中保持同一业务解释。每个模型用不同 lambda 后直接比较效用，会把偏好变化混入模型差异。

基础版不以不透明的权重距离评分作为最终风险保证。若用权重共识作并列解的次级规则，必须在满足共同可行集及各模型风险约束后执行，并记录不会突破主目标的容差。

## 16.4 计算算法：明确新增凸二次约束能力

当目标为线性收益或凹二次效用，风险矩 PSD，约束是线性与方差／TE 上限，问题是凸 QCQP／可转 SOCP。它**不是只有线性约束的普通 QP**。[R9]

当前已由 `compatibility.py → compatibility_solver.solve → compatibility_kernels → qp_numba.bounded_lp_kernel` 接通线性收益目标的共同约束求解。原 QP 内核继续作为唯一数值实现；新增乘子输出的内部求解及保留原 ABI 的薄适配。凹二次效用与 CVaR 连续适配尚未实现，不能据此宣称全部 QCQP 目标都已开放。

### 可执行的基础数值路线

1. 将 S1、收益约束、线性收益的 regret／maximin 约束写成线性主问题；采用凹效用时，将 `t−U_m(w)≤0` 或 `V_m*−U_m(w)−rho≤0` 写成凸不等式，再按下述支撑切面构造外近似。
2. 对所有真实二次约束 `g_m(w)=w' Sigma_m w−cap²`，在当前违反点 w_k 添加有效支撑切面：

\[
g_m(w_k)+2(\Sigma_mw_k)^\top(w-w_k)\le0.
\tag{C6}
\]

   因 g_m 凸，切面定义的是包含真实可行集的外近似。**满足切面并不代表已满足原二次约束**。
3. TE 约束用偏移变量 `w−b` 的对应梯度；凹效用约束同样使用其凸形式的支撑切面。
4. 每轮主问题使用共享固定签名 QP／LP 内核。新增切面可能使上次点不可行，必须先做线性 Phase-I，而不是把不合法初值交给要求可行初值的旧函数。
5. Phase-I：当前实现保留 simplex 与有限 box 为硬域，对其余归一化约束添加一个公共非负松弛，最小化最大松弛量。相较原草案的松弛和，零最优值的可行性含义一致，仅需一个额外变量；不把正松弛数值解释为原单位的总缺口。只有有界 LP 的对偶目标下界大于 1e-8 才报告不可行；其余失败均为未完成。
6. 逐轮计算全部真实 g_m、收益、权重约束。真实可行点给出可行目标界；主问题给出松弛问题目标界。两者差小于声明容差且真实残差通过，才认为凸求解完成。
7. 设置最大迭代、最大切面、超时、矩阵条件数与停止容差；预算耗尽返回近似／未收敛，不能退回简单平均并称为成功。

历史 `feasible_qp_kernel` 保留**首行为等式，其余行为 `A x >= b`，且要求可行初值**及原四项返回 ABI。当前源码也已有显式等式数、独立等式秩检查与矩阵 QP 入口，不能再记作缺失能力。模式 B 的 LP 只有 sum(w)=1 一条等式；切面已转换为 `>=`。新增 `bounded_lp_kernel` 在同一内核上取得乘子，对非负约束乘子与残余梯度的 box 最小值计算保守下界，并扣除浮点运算裕量。下界不依赖“残差小即最优”的推断。

多条产品预算等式仍需在产品实施层接入现有矩阵 QP 契约，不属于本次模式 B 的单预算 LP。重复等式应按线性相关性处理，不用微小松弛制造可行。无可核验目标界时只报告近似或未完成，不能依据残差推断全局不可行。

普通工具或第三方优化器不能绕过项目 NJIT 要求成为未登记生产路径。

最坏情形／切集优化有研究依据，但上述具体接口与停止规则是本项目设计，不是某家机构公开实现的复制。[R10]

## 16.5 Funding Goal：线性矩优化只是候选生成

对 Funding Goal，不把未知的资金成功率伪装成 C2 那样的凸二次约束。

生成模型锚点、联合兼容前沿与其他有明确来源的候选；对每个最终权重，在每个 CMA 的分布适配器下执行第 13 节资金模拟。

模式 B 的资金通过条件是：

```text
同一权重在所有选定 CMA 下：
  风险与授权通过
  资金成功率门槛通过
  支付／保护政策通过
  独立验证通过
```

找不到通过候选时，输出已搜索范围与失败模型；不能证明所有权重无解，除非硬约束子问题另有证书。第一版不加入无界的概率全局优化、MCMC 内层 SAA 或指数级后验组合。

## 16.6 C1–C5 比较页的含义

可为五个授权上限分别运行同一多 CMA 研究，得到每个上限下的兼容配置。但本次正式采纳不能超过 Mandate 的授权上限。

每张卡显示：授权上限、最终权重、每模型预期收益与风险、最坏 regret／收益、资金验证、约束状态、实际风险等级范围。模式 B 的汇总实际等级可展示各模型等级的最大值，但不丢弃逐模型结果。

如果只改变上限，较高上限的候选集应保留较低上限已找到的解；否则搜索实现可能制造假的非单调可行性。

## 16.7 必须展示的交叉矩阵

| 被评估配置 | CMA A | CMA B | CMA C | 是否全部通过 |
|---|---|---|---|---|
| A 的独立 SAA | mu／vol／TE／资金／状态 | 同左 | 同左 | 计算结果 |
| B 的独立 SAA | 同左 | 同左 | 同左 | 计算结果 |
| C 的独立 SAA | 同左 | 同左 | 同左 | 计算结果 |
| 最终兼容配置 | 同左 | 同左 | 同左 | 全列通过才显示 |

不能仅比较每个锚点在自己模型下的一格。若没有公共交集，返回具体冲突，不自动删除最悲观 CMA、调整概率、放宽 C 等级或提高预期收益。

## 16.8 可核对的二资产例子

设两模型收益分别为 `(10%,2%)` 与 `(2%,10%)`，同一风险矩 `0.04 I`，波动上限 16%，每个模型要求预期收益至少 5.8%。

各自最大收益锚点约为 `(76.4575%,23.5425%)` 及相反权重。它们在另一模型下收益仅约 3.8834%，所以均不兼容。

共同权重 `(50%,50%)` 在两个模型下均为 6% 预期收益、14.1421% 波动，满足约束。这说明共同解不应只在独立最优点中挑一个。

若两个模型收益下限都提高到 6.1%，第一模型要求资产 1 权重至少 51.25%，第二模型要求至多 48.75%，线性条件已经冲突。此时可以明确证明不存在共同解，不能靠平均权重解决。

## 16.9 求解和最终校验的能力边界

| 条件 | 在优化中的处理 | 正式确认前仍需复核 |
|---|---|---|
| 权重、资产／分组上下限、现金份额、同口径预期收益 | 线性约束 | 原始数值、完整资产轴与来源 |
| 每模型方差、相对基准 TE | 凸二次约束；共享 QP 子问题／外逼近 | 每一个真实二次约束，不能只查切面 |
| 固定单期收益情景下的 CVaR | 只有辅助变量 LP 适配已实现时才参与连续求解；否则为候选后验过滤 | 同一情景集、置信度、分位数和尾部规则 |
| 资金成功率、支付失败、路径 MDD／保护事件 | 基础版有限候选模拟与过滤，不宣称一般凸优化 | 冻结候选后的独立验证和明确政策门槛 |
| 真实产品赎回／准入、治理审批和期限资格 | 业务门禁，不伪装成收益参数 | 当前应用日期与相应证据 |

每项约束保存 `enforcement=solver_and_gate | candidate_filter_and_gate | diagnostic_only`。配置成硬约束却缺少对应评估引擎时返回 `unsupported_hard_constraint`，不能只在界面展示红黄标记后仍允许采纳。

模式 B 的执行顺序是“各模型锚点 → 交叉检查 → 联合约束求解／已声明的有限共同候选搜索 → 最终验证”。若仅启用有限候选路径，输出须明确 `search_domain=finite_candidates`；即使每个候选都经过全模型检查，也不称为连续域最优。连续联合求解器尚未完成时不能把简单平均权重当作其替代实现。

## 16.10 保存与计算预算

SaaPolicyVersion 保存一套最终权重，并同时冻结：

```text
mandate_id / risk_scale_version_id / numeric_limits
asset_class_version_id / ordered_asset_ids
cma_ids[] / hashes[] / decision_as_of
mode: single | parameter_average | compatible_all_models（均已接通）
aggregation_semantics / weights (模式 A)
compatibility_objective / reference_optima (模式 B)
primary_evaluation_spec / all_model_risk_contract
cross_model_metrics / funding_validation
solver_status / residuals / bounds / limitations
```

M 个模型、G 个网格点的锚点研究是约 M×G 个基础问题，加一次或若干联合问题；不是把各模型后验样本数相乘。实际耗时仍取决于约束、收敛和模拟预算，不承诺“无论怎样都很轻量”。

模式 B 冻结 `compatibility`（目标、各锚点、最终状态、残差、上下界和限制）、全部原 CMA 及逐模型结果。政策内部等权矩仅为展示兼容字段，显式记录 `effective_moments_role=display_reference_only` 与 `primary_evaluation_spec=each_frozen_source_model`；不能以其替代采纳或 TAA 的逐模型风险。指标汇总按列取最不利值，不对应单一分布。确认时对每个源模型使用与探索独立的资金随机流；历史政策读取不重新拟合。

运行限制为 1–128 轮（默认 64）、每个主问题 500 步、最多 2,048 个切面，锚点与共同求解共享 30 秒轮间检查预算；单次 LP／资金内核不可抢占，因此不是进程级硬实时中断。资金路径月预算为 `M×(M+2)×paths×months ≤ 50,000,000`，中央／保守至多两倍；继续执行 7 MB UTF-8 元数据上限。目标界差≤1e-7 且真实风险误差≤1e-10 才报告收敛；预算耗尽不制造后备权重。

SAA 多模型功能不能只把 `cma_id` 改成列表；`cma_application`、policy gate、冻结对象和 TAA 消费也必须同步更新。当前旧单模型请求与记录保持原形状、预览 hash 和风险语义；模式 A 新政策使用内部 `schema_version=2.0`、`mode=parameter_average` 与完整 `multi_cma` 证据。缺失融合证据时失败关闭，不按人工单模型继续。

---

# 17. TAA：继承最终 SAA，不丢失多模型授权

## 17.1 统一大类层战术研究

\[
w^{TAA}=w^{SAA}+\Delta w,\quad\mathbf1^\top\Delta w=0,\quad w^{TAA}\ge0.
\tag{T1}
\]

目标权重始终包含现金，满足资产／分组上下限与现金用途要求。目标设计允许未启用 TAA 时以 Delta=0 直接引用 SAA 进入产品实施，不应为了交接强造战术观点。**当前产品桥接只接受已保存的合法 TAA decision，直接 SAA 交接尚未接通**；现阶段仍需保存零偏离的 TAA 决策并通过原有门禁。

保留当前 momentum、manual、published regime、composite，以及训练／验证、walk-forward、信号成熟期、执行滞后、最低持有期、换手、交易成本、preflight 等研究能力。

实时状态只有在相应定义、数据可得性、发布用途、延迟和失效门禁通过后才能用于交易时点研究；事后标签不能因设置一个 lag 就变成实时信号。

## 17.2 产品映射不是大类 TAA 的必要前置

目标架构中，指数 Proxy 已有合法收益序列即可做 class-level TAA。没有产品映射时允许保存大类研究，状态为 `class_research_only`；不得声称已有实际交易产品或执行资格。

当前战略来源存在实施映射门禁。实现时应区分 class research 与 product application，而不是直接删除现有 `require_policy_application` 或绕过安全检查。产品应用仍必须验证映射、真实持仓、成本、日期与准入。

## 17.3 风险检查必须继承 SAA 模式

- 单 CMA：复核冻结的单模型风险契约。
- 参数融合：复核冻结融合参数，继续显示原模型交叉诊断；未启用 all-model gate 时不声称全部兼容。
- 兼容模式：对每个选定 CMA 复核绝对波动、TE、目标收益及其他声明为持续约束的条件。

TAA 相对 SAA 的主动风险为：

\[
TE_m^{TAA}=\sqrt{\Delta w^\top\Sigma_m\Delta w}.
\tag{T2}
\]

不能只拿一个平均协方差替代模式 B 的全部风险矩。新增 CMA／替换参数需要新的 SAA 研究和政策版本，不能在 TAA 中悄悄重定义长期假设。

Funding Goal 的原始成功率不是未来所有战术决策的自动保证。实际交接时应基于最新确认的余额、剩余支付计划与当前目标重新做资金诊断，并标注“假定未来保持当前目标／指定再平衡策略”；没有模拟未来 TAA 策略时不声称完成策略级资金保证。

## 17.4 历史回测与前瞻参数

即使信号本身完全因果，使用今天挑选的资产、今天估计的 CMA 和今天优化的 SAA 回测过去，也不自动构成历史可交易结果。

正式历史回放必须按每个决策日加载当时可得的资产／CMA／政策版本；缺少这些版本时标为事后固定方案回看，不伪造历史上线日期。walk-forward 不能只滚动信号、却固定使用未来形成的资产范围。

---

## 17.5 资金诊断续算的最小交接契约

新 TAA／产品交接使用 `FundingContinuationContext`，至少包含：

```text
original_mandate_id / original_cash_budget_hash
valuation_as_of / confirmed_investable_balance
completed_payment_ids / remaining_cashflows / remaining_terminal_target
remaining_periods / model_calendar / fee_basis
cma_evaluation_refs / horizon_applicability
future_weight_rule: static_target | explicit_rebalance_rule
search_or_validation_spec / content_hash
```

以已确认的**当前可投资余额**为新的起点，只投影尚未发生的现金流；不能继续使用最初本金，也不能再次扣除已经单独剔除的组合外储备。已发生支付须依据业务事实核对，不仅按日期假定已支付。

原模型使用等长月；第一版续算也在可核对的模型月边界执行。不在模型边界的日期，必须有明确的首期剩余时间与支付排序适配，否则返回 `cashflow_calendar_adapter_required`，不能直接四舍五入剩余年数。参考 CMA 需覆盖剩余期限；使用平稳延展时保留第 19.5 节的明确假设。

续算结果是“当前余额 + 剩余计划 + 指定未来权重规则”下的新诊断，不覆盖原 SAA 的资金证据。没有已实现的未来信号／调仓路径仿真，就不显示“动态 TAA 资金成功率已验证”。当前代码的 `goal_diagnostic_scope=strategic_plan_only_not_tactical_probability_guarantee` 不能只通过改标签升级。

# 18. 产品映射与配置：从大类预算到实际权重

## 18.1 两种代理模式在此分叉

**产品代理模式：** 使用原冻结产品池和授权限制。允许在该范围内研究不同实施产品及类内权重，但“代理成分及权重”不必等于“最终实施成分及权重”。改变代理定义需要新大类／CMA，改变实施权重则需重新验证跟踪误差和实际风险。

**指数代理模式：** 用户选择一个或多个已发布产品池，或全市场候选库。全市场仅扩大搜索来源，不代表全部产品自动获得配置资格。筛选后的候选与当时准入规则需冻结为 ImplementationUniverseVersion。

没有合适产品的大类保留预算缺口，不能自动删除、把预算分给剩余产品，或偷偷改成现金。

## 18.2 暴露匹配与产品优选分开

先按资产类别、投资合同、基准、币种、产品类型、准入和数据质量过滤，再使用统计暴露证据。

单基准超额收益回归：

\[
r_{product,t}-r_{f,t}=\alpha+\beta(r_{benchmark,t}-r_{f,t})+\varepsilon_t.
\tag{I1}
\]

同口径计算 Beta、R²、相关性、tracking difference 和 tracking error：

\[
TE_{hist}=\sqrt a\operatorname{sd}(r_{product}-r_{benchmark}),\quad
TD_{arith}=a\operatorname{mean}(r_{product}-r_{benchmark}).
\tag{I2}
\]

采用复合 TD 时另存定义，不能混用算术和几何差异。R² 或波动为零时输出未定义，不填成 1 或 0 来获得高分。

被动产品重点看基准一致、Beta 接近目标、TE／TD、费用、规模、流动性、折溢价与复制方式；主动产品另看因子暴露、风格稳定性、样本外表现。Alpha 不决定资产类别，也不自动成为下一期 CMA 的预测增益。

规则阈值来自已确认的 MappingPolicyVersion，不是 UI 隐藏的统一“R²>某值”真理。采用时间上独立的验证窗口与滚动稳定性诊断；只有样本内拟合时明确 `in_sample_only`。

## 18.3 类别归属矩阵不是 Beta 矩阵

令 x 为 K 个产品权重，Q 为 N×K 类别预算归属矩阵。基础版每个产品唯一归类，Q 为 0／1 且每列合计 1：

\[
Qx=w^{target},\quad x\ge0,\quad\mathbf1^\top x=1.
\tag{I3}
\]

统计 Beta 可以为负、超过 1 或跨大类，不能直接当成 Q 分配资本。跨资产主动基金需要明确、可验证的穿透归属；基础版无法证明时列为未匹配，不靠回归系数归一化获得看似正确的预算。

现金余额如作为实施项，必须具有明确类型和现金收益口径；不是缺失产品的占位符。

## 18.4 产品权重求解

在同一历史／风险研究窗口估计产品 P 与大类代理 A 的联合风险矩：

\[
\begin{bmatrix}C_{PP}&C_{PA}\\C_{AP}&C_{AA}\end{bmatrix}.
\]

基础产品实施目标为最小化相对目标大类组合的跟踪风险：

\[
\min_x\ x^\top C_{PP}x-2x^\top C_{PA}w^{target}
+(w^{target})^\top C_{AA}w^{target},
\tag{I4}
\]

subject to I3、产品最大权重、池限制、现金／流动性条件。该二次目标的常数项可在求解中省略，但展示 TE 时需还原。

线性交易费用或相对现持仓的 L1 换手可通过辅助变量增加；固定开仓费、产品数量整数限制会改变问题类别，不在基础 QP 中假装精确支持。

类内简单等权、自定义权重和现有风险方式继续保留，但都要经过同一个预算与实际风险门禁。

## 18.5 从 CMA 风险到实际产品风险的桥接不能省略

`Qx=w` 只证明资本预算匹配，不证明产品组合和代理具有完全相同风险。产品有跟踪误差、主动暴露和费用，必须重新评估。

可复用风险模型中心的已发布多因子回归，明确产品超额收益模型：

\[
r^P-r_f\mathbf1=\alpha+B(r^A-r_f\mathbf1)+\varepsilon.
\tag{I5}
\]

若残差与因子零协方差假设经过说明，逐 CMA 产品风险估计为：

\[
\Sigma^P_m=B\Sigma_mB^\top+\Sigma_\varepsilon.
\tag{I6}
\]

必须估计产品间的残差协方差，而不是未经声明地只取对角线。使用共同样本、PSD 验证、样本外暴露稳定性和足够覆盖；若假设不成立，采用完整联合风险模型或返回不支持，不能漏掉交叉项。

预期收益桥接可采用：

\[
\mu^P_m=r_f\mathbf1+B(\mu_m-r_f\mathbf1)+\alpha_{assumed}-cost_{incremental}.
\tag{I7}
\]

其中 alpha_assumed 需要明确来源。基础默认可采用 `alpha_mode=zero_active_premium`、`alpha_assumed=0`，意思是“本次不预测主动增益”，不是“已经证明产品未来 Alpha 为零”。使用非零 Alpha 必须引用独立预测研究，不直接外推历史回归截距。被动产品的预期跟踪差、费用已含于 NAV／上游 CMA 的部分必须避免重复扣减。

对模式 B，最终产品权重 x 应在每个模型下复核 `sqrt(x' Sigma^P_m x)`、收益及必要资金条件。没有足够产品风险映射证据时可保存研究草稿，不能宣称已经实现“所有 CMA 下兼容的最终产品组合”。

## 18.6 产品择时与组合合成

产品择时研究默认只是引用证据，不自动改变目标权重。如果明确应用入场／退出规则，必须指定未投入资金的现金去向，并重新检查类别预算、交易成本、资金支付和风险，不以“择时开关”绕过 I3。

最终产品目标、TAA 大类目标、SAA 长期中枢均保存独立版本。产品无法实施时反馈为实施缺口，由用户决定扩大实施范围或重新研究，不逆向修改历史 SAA。

---

## 18.7 统计匹配与残差风险的执行细节

I1／I5 不是仅写“做 Alpha/Beta 归因”就完成。明确构造同日期、同币种、同频率的设计矩阵 `X=[1, asset_excess_returns]`，因变量为产品超额收益。一次联合回归得到每个产品对所有所选代理的暴露；不能把多次单因子回归的 Beta 拼接成已控制共线性的多因子模型。

在训练窗口解最小二乘 `min ||Y-X Theta||²`。复用当前 `backend/sensitivity/service.py::_fit()` 的标准化、归因内核和验证组织；需要的秩、条件数、残差数组与联合风险输出属于新增契约。使用经过验证的 QR／SVD 或等价稳定解法，不显式计算 `(X'X)^(-1)`。秩不足、常量因子或严重共线性时返回诊断；需要正则化则新建有明确强度与验证证据的模型版本，不静默增加 ridge 后继续声称是原 OLS。

同一训练窗口内，产品残差矩阵为 `E=Y-X Theta`，各产品具有相同的有效日期轴。用于 I6 的残差样本协方差可明确采用：

\[
\Sigma_\varepsilon=\frac{a}{T-1}(E-\bar E)^\top(E-\bar E).
\tag{I8}
\]

这里采用中心化残差样本协方差的口径；它不声称是经过回归自由度修正的无偏预测风险。若采用 `T-rank(X)` 自由度修正，必须作为独立估计器记录分母、模型假设及验证，不能与 I8 混用。测试集使用训练系数计算样本外 R²、TE 与暴露稳定性；不能在同一测试样本上重拟合后仍称样本外。

发布的风险模型还须声明因子－残差交叉协方差假设。训练内 OLS 的正交不等于未来正交；若保留非零交叉矩阵 `D=Cov(r^A, epsilon)`（N×K），则产品风险包含 `B D + D' B'`，需有完整联合 PSD 风险模型支持。缺少该模型时不把未知交叉项填成“已证明为零”。I5–I7 的基础版本将同期间 r_f 视为确定性参考；若把 r_f 建模为随机现金收益，则需显式计入其协方差／暴露，不能仍套确定性截距公式。

I4 使用的是共同历史联合风险矩，可作为实施跟踪优化的目标；I6 则把大类 CMA 传导到产品风险。两者来源不同但可以并存，页面必须显示“用哪套风险优化、用哪几套风险验收”。所有模型风险约束参与连续优化时需要第 16 节的联合求解能力；仅做候选过滤时按其真实搜索范围报告。

## 18.8 当前产品交接契约需要显式扩展

当前 `backend/tactical_allocation/portfolio_bridge.py::validate_allocation_source()` 要求产品属于原 TAA 基线的同一可投资域与冻结归属，而且策略类型必须为 `manual`。因此，指数模式下 TAA 后新增产品映射、无 TAA 的直接 SAA 实施以及类预算约束下的产品 QP，都不是改页面按钮即可接通。

新版本应明确支持：

```text
allocation_source.kind: saa_policy | taa_decision
allocation_source.id / content_hash / target_class_weights
implementation_universe_id / implementation_mapping_id
implementation_method: manual_with_class_budgets | class_budget_optimization
risk_evaluation_contract / final_constraint_audit
```

旧 `kind=taa` 的已保存记录保持原验证和读取语义；新方法在共同门禁核验预算、准入、目标来源与多模型风险。不能把自动优化伪装成手动输入，也不能构造不存在的 TAA ID 绕过来源检查。

“无杠杆”不仅校验产品权重非负且合计 1；已明确为杠杆／反向敞口的产品应由准入规则排除。底层杠杆信息不完整时记录未核实，不能仅凭组合没有融资就宣称穿透敞口也已验证无杠杆。

# 19. 前端、接口、版本与工程组织

## 19.1 前端顺序与操作

投前导航的**目标形态**按“投资目标 → 大类代理 → CMA → SAA → TAA → 产品实施 → 验证”组织。Settings 单独提供“全局风险等级”。辅助历史实验、压力测试和指标诊断以侧栏／上下文工具呈现，不与主步骤混排成无先后清单。

截至 2026-09-18，风险等级配置中心和投资目标列表保持现有职责；LTCMA 已提升为投前一级节点 `/pre-investment/ltcma`，支持“研究输入 → 结果与确认”。`StageLayout` 快捷流程已加入 LTCMA。`StrategicAllocationWorkspace` 改为选择冻结 LTCMA，保留来源资格和原 SAA→TAA 门禁，不再重复编辑模型。

| 页面 | 默认展示 | 高级／详情 |
|---|---|---|
| 全局风险等级 | Universal 输入、前沿、自动／人工 C1–C5、发布 | 分段参数、端点、失败点、稳定性、指标口径 |
| 投资目标 | 事实、三种成功标准、手动／资金建议等级、确认 | 原始数值、治理来源、模拟设置、诊断限制 |
| 大类代理 | 产品／指数来源、成分、类内规则、共同历史 | 日历、PIT、收益口径、异常与数据排除 |
| CMA | 五类方法、自动来源、少数业务参数、结果与保存 | mu／Sigma／U、先验、概率覆盖、方法限制 |
| SAA | 一个／多个 CMA；参数融合／兼容配置；统一等级上限 | 融合语义、交叉矩阵、求解证据、无解诊断 |
| TAA | 冻结中枢、观点／规则、候选、回测与交接 | 多模型风险契约、时点、成本与资金重检 |
| 产品实施 | 来源范围、匹配证据、目标权重与缺口 | Q、Beta、残差、产品风险、池限制与费用 |

CMA 在 SAA 页面只显示选择器、摘要和“复制为新 CMA 研究”，不重复完整编辑器。修改任何上游输入后，预览与采纳状态失效；正在返回的旧请求不能覆盖新状态。

中途进入允许选择任意符合条件的历史成果。用户有旧版本时，应先读取冻结结果，不打开页面就重新训练或覆盖。

## 19.2 建议接口契约

以下同时记录**当前已实现接口**与后续目标扩展，避免继续把已落地能力写成待开发。

| 操作 | 当前状态 | 路径／目标扩展 | 输入与输出 |
|---|---|---|---|
| 全局前沿与分级预览 | **已实现** | `POST /api/strategic-allocation/risk-scales/preview` | 资产、参考输入、约束、分段规则 → 前沿、等级、诊断、hash |
| 全局标尺版本治理 | **已实现** | `/risk-scales/confirm`、`/{id}`、`/{id}/activate`、`/{id}/retire`、`/compare`、`/defaults`、`/drafts`，同前缀 | 不可变版本、默认绑定、退役、比较、草稿；active 指针不覆盖历史 |
| 风险标尺参考输入 | **已实现** | `/reference-inputs/catalog`、`/preview`、`/confirm`、`/{id}` | 可追溯参考输入版本及冻结证据 |
| 投资目标 | **Mandate 2.0 已实现** | 现有 `/mandates/preview`、`/mandates/confirm`、读取／退役及资金回显接口 | 事实、现金预算、RiskScale 授权、参考诊断 → 不可变 MandateVersion |
| 独立 CMA | **第一版已实现** | 现有 `/api/strategic-allocation/cma/preview`、`/cma`、`/cma/{id}`；新增能力、研究选项、草稿及停止新引用接口 | 单一 CmaResearchService；V2 请求、幂等发布和生成类型；旧版本按原语义读取 |
| 历史状态证据 | **已集成到 CMA 预览，不另建重复接口** | `/cma/preview` 的 `historical_regime_occupancy` 方法 | run 引用＋共同收益面板＋区间 → 状态矩／占用率／应用概率／lineage；专门的独立证据发布接口仍非本版范围 |
| 单／多 CMA SAA | single、模式 A E2、模式 B 线性收益基础版 **已实现** | 现有 `/policy/preview`、`/policies` | 单模型 `cma_id`；A 使用带权 `cma_refs`；B 使用无权 `cma_refs`、`compatibility_objective` 和有界 `solver_max_iterations`；返回锚点、共同候选、状态／目标界、逐模型诊断与冻结政策 |
| 实施匹配与配置 | **目标扩展待实现** | 扩展现有 implementation-map／portfolio 研究接口 | 候选范围、匹配规则、目标预算 → 映射与产品权重 |

带固定名称的子路径须在通用 `{id}` 路由前注册，避免 `regime-evidence` 被当作 ID。请求和响应 schema 由同一契约生成前后端类型，不靠字符串猜测方法。

共同预览输出：

```text
request_echo / resolved_refs / data_fingerprints
result / diagnostics / limitations
mathematical_status / business_status
publication_eligibility / application_eligibility
execution_audit / preview_hash
```

确认请求带 preview_hash、显式确认与引用。服务端重新核验冻结输入／结果一致性；源数据或参数变化时返回 conflict，不自动保存另一份结果。只读预览不得保存正式成果。

## 19.3 统一目录与调用职责

当前复用 `backend/strategic_allocation/` 作为 CMA／目标／政策业务边界：

```text
cma_model_contracts.py / cma_models.py / cma_model_kernels.py
cma_application.py                   冻结参数唯一消费入口
cma_service.py / cma_store.py        已有：研究编排、草稿与版本治理
cma_evidence.py                      已有：共同数据准备及事后状态桥接
cma_statistical_models.py            已有：统计模型编排
cma_statistical_kernels.py           已有：固定签名统计／NIW 内核
risk_scale_contracts.py
risk_scale_service.py
risk_scale_kernels.py
mandate_diagnosis.py                 已有：目标和标尺联动，不再另建重复实现
contracts.py                        已有：PolicyCmaRef 与 single／parameter_average 请求
multi_cma.py                        已有：融合来源校验、冻结矩与原模型诊断编排
multi_cma_kernels.py                已有：固定签名声明半宽聚合；矩融合复用 cma_model_kernels
goal_kernels.py / planning.py        现金流与资金诊断复用
policy_gate.py                      单／多模型统一资格门禁
```

共享数值求解留在 `backend/optimizer.py`／`backend/qp_numba.py` 及适当拆分的共用内核中。历史收益 → 矩的准备，与冻结 CMA → 矩的读取分开，随后共用求解能力；不从前瞻研究调用历史 HTTP 接口伪装同一数据来源。

RiskLevels 和独立 LTCMA 工作台均已落地。原工作树已移除旧 SAA 内独立假设编辑、有效结果及战略前瞻表单的未使用组件；当前统一从 LTCMA 研究并在 SAA 引用。本次 `MultiCmaSelection`／`CrossModelResults`／`CompatibilityResults` 已接通 E2 参数平均、共同约束、收益目标选择、锚点交叉诊断和可行／不可行／未收敛展示。模式或成员变动均废弃旧预览；B 转回 A 需重新确认研究权重。

## 19.4 版本依赖与失效

| 上游变化 | 新研究如何处理 | 历史结果 |
|---|---|---|
| active RiskScale 更新 | 提示选择原版本或升级后重新诊断 | 原 Mandate 继续引用原标尺 |
| Mandate 改风险／目标／币种／期限 | SAA、TAA、产品当前草稿待重算 | 不删除旧 CMA／SAA |
| Proxy 成分／权重／日历变化 | 新 AssetClass／Proxy 版本，相关 CMA 重新计算 | 不修改旧收益面板 |
| CMA 或融合权重变化 | SAA 预览失效，不能沿用旧采纳证据 | 旧政策保持冻结 |
| SAA 模式 A 改模式 B | 重新求解与逐模型门禁，不只换标签 | 保留旧模式与哈希 |
| 产品池新版本或撤销准入 | 新实施范围需重检；当前应用资格单独更新 | 历史成员不回写 |
| 算法／参数模板更新 | 创建新定义和结果版本 | 旧定义按原语义只读 |

`allocationJourney` 只保存当前选择与草稿身份；需要扩展 cma_refs、risk_scale_ref、SAA study/policy 等引用，但不把 localStorage 当作模型证据数据库。

## 19.5 时点、期限与系统冷启动

参考 CMA 的预测期限与本次目标期限不一致时，不自动重命名。全局风险轴可以按年化波动定义，但参考收益和资金模拟必须有明确的期限适用规则。

可允许经确认的 `stationary_parameter_extension` 在指定期限范围内做参考投影；记录“长期参数固定延展”的假设，不能把十年预测自动称为一年精确预测。正式多 CMA 融合默认要求同一预测期限，其他期限需独立、经验证适配。

算法 readiness 按进程与固定签名检查。没有预热／来源不可用时 fail closed，不能首个请求临时编译、用旧缓存蒙混或回退 pandas／Python。

## 19.6 数值与内存预算

N 保持与当前大类契约一致的上限 30；情景数上限 60；网格上限 200。模式 A 当前请求硬上限 M=20，已提供最大轴数值与内存基准脚本 `backend/tests/test_multi_cma_performance.py`；该小矩阵基准不代表整次求解或资金模拟耗时。路径数沿用已确认 Mandate 的预算。

当前资金预览硬门禁为 `(M+1) × (4 或 5 个候选) × paths × months ≤ 50,000,000`；加 1 包含融合模型，中央与保守计算最多两倍。超限时用户须显式减少来源或重新确认路径预算，不能静默丢弃 CMA。源证据逐个读取／冻结、预览完成和最终保存前均检查 7,000,000 bytes 紧凑 JSON 预算；最终检查包含名称、理由及政策，早于正式写入。

不得分配完整 `M×候选×路径×月份×资产` 大张量。复用只读收益面板，分块生成随机数和路径，累计指标；保存足以确定性重现的随机数算法、种子与分块规则。common random numbers 是比较方差控制工具，不意味着各候选的真实市场收益具有该人为耦合。

NumPy 数组及 mmap 在边界规范化后复用；协方差、Beta、成分索引等使用稳定 dtype。必要复制说明位置与原因，用 `np.shares_memory`、峰值内存和数值等价测试验证，不凭 `copy=False` 宣称零拷贝。

---

## 19.7 能力注册、统一门禁与可解释的失败

后端目录需明确返回本进程可执行的 `supported_methods / supported_objectives / supported_constraints / supported_moment_semantics`。一个方法已经在设计中出现，不代表能在当前后端执行；任何尚未预热的方法、未支持的跨频率 NIW、指数 TAA 或多 CMA 联合求解均须显示具体原因，不回退到另一种方法。当前日频 NIW 已实现，不属于笼统未开发能力。

统一约束结果至少包含：

```text
constraint_id / scope: reference | strategic | tactical | product
model_id / evaluated_version_ids / evaluated_weights_hash
metric / value / unit / horizon / confidence_level
limit / comparison_operator / tolerance
enforcement / status: passed | failed | unavailable | not_applicable
source / reason / execution_version
```

`unavailable` 不等于 passed；`not_applicable` 要有适用性规则。仅能计算波动的引擎不能将 CVaR 护栏显示为已执行。最终 ResearchPackageVersion 收集这些记录，明确哪些是数学／模型证据、哪些是机构治理确认，不把二者合成一个无解释的绿色分数。

数值职责按 AGENTS.md 拆分：收益构造、样本统计、均值／风险来源、混合、分级映射、验证和展示分别可追溯；NIW、BL 和整体投资流程不新增不可展开的大黑盒算子。执行层可复用统计量和融合计算，但不隐藏中间结果。

真正耦合的内核保留明确边界：QP／联合约束求解维护权重、活跃约束及切面状态，按残差与预算终止；前沿分段 DP 维护阶段与终点最优状态，按有限网格回溯；资金路径递推维护余额与曾经未支付标记，按固定期间终止。各自的取数、参数选择、结果统计和发布在内核之外，预热与只读数组契约分别验收。

前端交互遵循 `docs/frontend-design-guidelines.md`：复用现有 UI 原语和 ECharts；图表失败点保留缺口；跨模型表格显示单位、阈值和失败原因；加载／空态／错误／禁用原因齐备。版本选择、参数编辑与计算结果分区，窄屏按区域切换，不通过缩小字体塞入全部矩阵。本文不据此宣称已经做过浏览器验收。

# 20. 算法、模型和关联流程的闭环矩阵

## 20.1 每条业务边都必须有执行定义

| 关联流程 | 算法／契约 | 输出与失败含义 | 验收编号 |
|---|---|---|---|
| 资金事实 → 现金流计划 | F1–F3、月度时间轴 | 现金流、名义目标；非法日期／金额失败 | Q01–Q03 |
| 现金流 → 所需收益 | F3–F4、可行性二分 | required effective return；超范围不是已求解 | Q04 |
| 现金流 → 流动性／现金 | F5–F6、现金资格规则 | 资金金额／比例；不能超过本金后裁剪 | Q05 |
| 参考 CMA → Universal 前沿 | R1–R4、QP／端点验证 | 连续目标的数值前沿；保留失败点 | Q06–Q08 |
| 前沿 → C1–C5 | R5–R8、加权角度 DP／人工阈值 | 有版本的风险标尺；退化／不稳定可见 | Q09–Q12 |
| 标尺 → Mandate | 第 5 节、上限解析 | 数值硬上限，不是强制最低风险 | Q13–Q15 |
| 资金计划 → 风险建议 | D2–D6、有限候选与独立验证 | 最低已测试可行等级，不是最高承受意愿 | Q16–Q19 |
| 产品／指数 → 大类收益 | P1–P3、权重时点与日历 | ReturnPanelVersion；缺成分不静默再归一化 | Q20–Q23 |
| 历史区间 → Historical CMA | H1–H2、U1 | 基础及年化矩、样本诊断 | Q24–Q26 |
| 手动观点 → Manual CMA | M1、自动风险与覆盖 | 带来源的 mu／Sigma | Q27 |
| 先验＋证据 → Bayesian CMA | B1–B8 | posterior mean／asset covariance／U | Q28–Q31 |
| 参考组合＋观点 → BL CMA | L1–L4 | posterior total returns、独立 U | Q32–Q34 |
| 状态结果 → Regime CMA | G1–G6 | 占用率＋同频矩混合＋期限适配 | Q35–Q39 |
| 多 CMA → 融合参数 | E1–E3 | 参数平均或同期间分布混合，不混含义 | Q40–Q42 |
| 单 CMA → SAA | S1–S4 | 声明精确／有限搜索性质的候选 | Q43 |
| 多 CMA → 共同 SAA | C1–C6、联合二次约束 | 一套全模型兼容权重或明确失败状态 | Q44–Q47 |
| CMA／权重 → 分布与资金 | D1–D10 | 模型指定的概率、尾部、回撤；不保证现实 | Q48–Q50 |
| SAA → TAA | T1–T2、信号时点、继承原 SAA 模式的门禁；模式 A 为融合风险＋原模型诊断，模式 B 已按全部冻结模型复核收益、波动及 TE | 同一授权下战术目标 | Q51–Q53 |
| 大类 → 实施候选 | I1–I2、合同筛选与样本外检验 | 候选及未匹配原因 | Q54 |
| 大类预算 → 产品权重 | I3–I4、线性约束 QP | 预算守恒、成本与 TE | Q55–Q57 |
| 产品 → 实际风险复核 | I5–I7、多模型评估 | 无风险映射证据时只读研究、不能正式应用 | Q58–Q59 |
| 所有结果 → 保存／下游引用 | 第 19 节、hash／版本／资格 | 预览、确认、当前应用分离 | Q60–Q63 |
| 绝对／相对目标 → 现金保障 | 第 5.9 节、同一预算 F1–F6 与支付事件 | 成功标准不抹掉现金流事实，不重复金额 | Q64 |
| TAA／实施更新 → 剩余资金诊断 | 第 17.5 节、当前余额与剩余模型日历 | 新诊断，不继承历史成功率 | Q65–Q66 |
| 产品预算 → 多等式求解 | 第 16.4／18.8 节、秩检查与通用等式适配 | 不把单等式 QP 当作多预算已支持 | Q67 |
| 方法选择 → 执行能力 | 第 16.9／19.7 节、能力目录与统一门禁 | 未支持硬约束阻断，不只显示警告 | Q68–Q72 |

上述矩阵是设计验收要求，不表示新增算法已经运行在项目中。缺少证据时的 `unsupported / pending / not_found` 是明确的业务状态，不用虚假默认值填补。

## 20.2 数学与边界测试要求

- **Q01**：资本、外部储备、内部现金不重复扣减；总资金等于储备时拒绝。
- **Q02**：实值现金流和期末目标分别按对应时点转名义；非整数目标期限按支持契约拒绝，不静默取整。
- **Q03**：同月收益、投入、支付顺序固定；一次未支付不能被未来资金抹除。
- **Q04**：无现金流解析 required return 与二分结果一致；下界已满足和超上界独立状态。
- **Q05**：累计净现金流全负时流动资本为零；股票 liquid 标签不满足现金保全。
- **Q06**：二维 GMV 与解析值一致，低于最低单资产波动的例子可重现。
- **Q07**：收益相同／协方差奇异的端点处理、最高收益面最小方差正确。
- **Q08**：目标网格未收敛、点数预算耗尽、不连续数据不得伪造连续前沿。
- **Q09**：DP 代价前缀和与直接计算一致，小规模与穷举切点一致。
- **Q10**：竖直切线、近直线、重复点、零跨度与稀疏前沿处理确定。
- **Q11**：自动／人工分档无重叠；边界相等、低于参考下界和高于 C5 有明确输出。
- **Q12**：改变网格密度、CMA 与约束时显示边界变化；冻结版本不漂移。
- **Q13**：授权 C3 可接受实际 C1／C2，不添加下界增险条件。
- **Q14**：风险上限增加时保留已有低风险可行点，数值搜索不制造错误非单调。
- **Q15**：手动风险选择与资金建议独立；覆盖不能突破机构最高授权。
- **Q16**：资金所需有效收益不直接传给算术收益硬约束。
- **Q17**：资金成功率可随实际风险非单调，但 cap 可行集应嵌套。
- **Q18**：选优 seed 与确认 seed 分开，候选冻结后再做验证。
- **Q19**：Wilson 单项与联合门禁有不同解释；零观察失败不代表失败概率为零。
- **Q20**：固定权重／买入持有／定期再平衡代理分别满足 P1–P2。
- **Q21**：复合指数与产品组合使用相同收益组合语义，不靠价格点相加。
- **Q22**：FX 报价方向、费用净额、复权和 total-return 口径可追溯。
- **Q23**：共同样本先求合法单期收益，不把缺日后跨期收益当单日。
- **Q24**：Historical 窗口以 as_of 解析，1Y 不等价于固定 252 条。
- **Q25**：ddof、固定收缩和原始样本结果可核对；不足／零方差明确失败。
- **Q26**：短样本警告不禁止用户有意开展对照研究，也不伪装长期可靠性。
- **Q27**：Manual 风险覆盖清除原认证并保存覆盖来源。
- **Q28**：NIW 解析后验与独立参考一致；离差和与样本协方差区分。
- **Q29**：资产 covariance 的 a 与均值 covariance 的 a² 换算正确。
- **Q30**：t 尺度矩阵与方差、边际区间与联合区间不混淆。
- **Q31**：先验续更只用新增似然；重叠数据须显式经验贝叶斯标记。
- **Q32**：BL 无观点输出 rf+Pi；弱观点趋向先验。
- **Q33**：相对观点不减 rf；绝对观点正确转换；tau 和 view_std 各自作用正确。
- **Q34**：BL 的 posterior U 不自动替代资产 Sigma，确定性现金不加假噪声绕门禁。
- **Q35**：状态概率基于共同有效时间占用，不用段数或转移概率。
- **Q36**：状态收益归属包含跨边界一次且仅一次；Unknown 不等于震荡。
- **Q37**：未调整状态 ML 混合恢复总体 ML 矩；ddof=1 使用 G6。
- **Q38**：基础期混合再年化与年度世界混合分开，禁止状态间项多乘 a。
- **Q39**：概率覆盖、零权重、新情景、未来信息与历史 run hash 全部校验。
- **Q40**：参数融合是 sum(a*Sigma)，不是 sum(a²*Sigma)。
- **Q41**：预测分布混合包含 between term；不被自动套用到参数平均。
- **Q42**：M=1 的退化、模型顺序置换、相同数据相关性与不确定性缺失行为正确。
- **Q43**：普通 QP、有限风险预算候选与非凸指标各自标注最优性范围。
- **Q44**：最终共同权重对所有 CMA 交叉复核，不只对锚点自评。
- **Q45**：第 16.8 节共同解和线性冲突样例复现。
- **Q46**：支撑切面通过但真实二次约束失败的点不得采纳；Phase-I 初值合法。
- **Q47**：外逼近预算耗尽、残差、目标界和无解证书严格区分。
- **Q48**：D1 与 D2 的年度／基础期语义不同，不默认为精确等价。
- **Q49**：正常、极端、负收益条件下 lognormal／normal ES 与数值积分一致。
- **Q50**：市场回撤剔除外部现金流，历史 MDD 不等于预测上限。
- **Q51**：多 CMA SAA 的风险集被 TAA 完整继承。
- **Q52**：无产品映射的指数 TAA 只能大类研究，不获得实际应用资格。
- **Q53**：因果信号不掩盖未来资产／CMA 的回测泄漏。
- **Q54**：主动／被动、样本内／外、零波动与无匹配候选行为清晰。
- **Q55**：类别 Q 与 Beta B 不混用，产品预算守恒。
- **Q56**：原产品池限制与全市场准入都在最终写入前复核。
- **Q57**：实施不可行时保留预算缺口，不静默重分配。
- **Q58**：纯代理复制、零残差时产品风险恢复原 CMA；非零残差不被忽略。
- **Q59**：产品跟踪差和费用不重复计量，历史 Alpha 不自动当预测收益。
- **Q60**：旧版本只读，新的 active 标尺／CMA 不改变历史成果。
- **Q61**：晚到的网络结果和旧 hash 不能覆盖新草稿、不能确认。
- **Q62**：预览无持久化，确认重验来源，当前应用资格独立检查。
- **Q63**：固定签名、启动预热、只读／非连续数组、共享内存、无 Python fallback，以及前后端真实交互回归。

- **Q64**：三类目标可以共享一份预算；旧 `funding_plan` 适配不重复本金／现金流，切换成功标准不静默丢失预算。
- **Q65**：续算只使用确认余额和剩余支付；过去支付不重复扣除，组合外储备不再扣一次。
- **Q66**：非模型月边界、缺少剩余期限适配、未来 TAA 规则未模拟，必须显示对应限制。
- **Q67**：产品多条预算等式与总权重等式的秩正确；冲突等式、等式号／不等号方向和可行初值都有反例测试。
- **Q68**：E2 参数平均与 E3 分布混合不共用未声明的随机模型选择路径。
- **Q69**：改变 NIW 采样频率不能直接沿用旧强度；正概率无样本状态必须提供合法条件参数。
- **Q70**：配置硬 CVaR／MDD／资金门槛但无评估能力时，不可采纳；仅诊断指标不伪装硬护栏。
- **Q71**：指数 TAA 后产品映射、无 TAA 直接 SAA 实施与类预算优化使用新显式来源契约，不能伪装手动或伪造 TAA ID。
- **Q72**：联合产品回归、残差跨产品协方差、因子－残差交叉项和确定性 r_f 假设可追溯；样本内正交不冒充未来保证。

## 20.3 现有测试的复用位置

已核对存在的重点测试文件：

```text
backend/tests/test_cma_models.py
backend/tests/test_cma_model_integration.py
backend/tests/test_frontier_grid.py
backend/tests/test_mandate_diagnostic_integrity.py
backend/tests/test_investment_mandate.py
backend/tests/test_tactical_allocation_bridge.py
frontend/src/pages/InvestmentObjectivesWorkspace.test.tsx
frontend/src/pages/StrategicAllocationWorkspace.test.tsx
frontend/src/components/strategic-allocation/CmaModelEditor.test.tsx
```

全局风险分段、NIW 与历史状态桥接已有对应测试（`test_risk_scale_*`、`test_ltcma_statistics.py`、`test_ltcma_evidence.py`、`test_ltcma_lifecycle.py`）；参数融合及下游继承的本次证据另行记录，模式 B 联合二次约束、独立参考解、不可行证据和下游全模型门禁已有专门测试（见模式 B 验收记录）。参考求解器可用于隔离数值验证，但生产路径仍须满足 AGENTS.md 的固定签名 NJIT 要求。

---

# 21. 实施依赖、审核范围与本轮数值自查

## 21.1 按依赖实施，避免只有页面没有算法

| 阶段 | 当前状态（2026-09-19） | 必须完成 | 验收重点 |
|---|---|---|---|
| P0 统一契约 | **部分完成，持续治理** | 资产／日期／收益／风险语义、现有只读兼容、数据质量 | 旧结果不变，错误口径不能进入新链路 |
| P1 共用代理与 CMA | **Product-first／Strategic-first 两条上游路径及 LTCMA 第一版均完成；通用代理与频率适配仍有后续范围** | 双路径已汇合到统一 LTCMA；独立中心、五类方法族、冻结证据及单 CMA 交接已实现。指数来源语义、跨币种／跨频率、统一代理独立生命周期按实际需求继续增强 | 已有结果不重算；原产品链路保留；战略研究不因缺实施产品而丢失资产；日历缺失或来源不完整不得静默缩短 |
| P2 全局标尺 | **已有实现，本次相关回归通过** | 维护已实现 RiskScaleVersion、参考输入、前沿、分段、版本治理和 Mandate 消费链 | 无客户 Mandate 也可初始化；历史版本不可变；默认版本切换不改写旧研究 |
| P3 投资目标 | **已有实现，本次相关回归通过** | 维护 Mandate 2.0、三类成功标准、独立现金预算、RiskScale 授权、资金建议／参考诊断及版本生命周期 | 所需风险、承受政策与正式授权分开；预算不重复；修改不原地覆盖历史 |
| P4 多 CMA SAA | **模式 A E2 与模式 B 线性收益基础版已实现** | 同轴同口径融合；共同约束、两种线性收益目标、锚点交叉矩阵与冻结读取已接通；五等级联合前沿、凹效用与 E3 仍待实现 | 一个最终 w；不能只增加多选框或平均独立最优权重 |
| P5 TAA／产品继承 | **三种模式风险下传已接通；完整目标仍待开发** | 尚需类级 TAA 与实际应用分开、剩余资金续算、Q／Beta、多个预算等式与实施风险 | 原模型诊断不冒充全部通过，TAA 不冒用原 SAA 资金成功率 |
| P6 统一验证 | **部分能力已有，统一闭环待后续** | 数值、PIT、现金流、压力、成本、回归与可复现发布 | 未验证项不显示通过，不把模拟当保证 |

前端重组可与各阶段并行，但不可先显示“已支持”再用静态示例补结果。涉及前端代码时完整遵守项目设计准则和真实浏览器验收。

## 21.2 明确延后的能力

ALM、保险偿付能力、养老金负债过程、杠杆／做空、固定交易费整数优化、完整多期随机动态资产配置、全局成功概率最优、完整自动市场市值采集、数据驱动的 CMA 权重学习，均不因本文出现关联名词而算作第一阶段已支持。

这些功能的缺失通过能力门禁暴露；不能以“默认参数”伪装可执行。已有原功能和旧数据契约不因本设计收敛而擅自删除。

## 21.3 本轮已执行的独立数学样例

以下保留 2026-09-16 原设计审核时的数学证据：当时使用两组隔离的 NumPy／SciPy 数学参考脚本完成 **38 项检查，全部通过**，其中基础公式与联合配置样例 26 项，分段 DP／穷举、现金流、残差和边界补充 12 项。输入全部为固定合成样本，不读取真实投资数据，不导入 BetterSaaTaa 的生产实现。**2026-09-18 文档同步及 2026-09-19 本次开发均未重新运行这 38 项脚本；本次实现验收另见第 21.4–21.5 节。**

这些是**文档算法核对**，不是新功能已开发、生产 NJIT 已执行或全量 pytest 已通过的证据。以下表格汇总主要关系，不要求表格行数等于断言数量；数值误差按复核容差记录，不把某次浮点末位误差当作算法精度保证。

| 自查项 | 结果 |
|---|---|
| 状态 ML 混合重构总体矩 | 均值与协方差均通过 1×10^-14 绝对容差核对 |
| 状态间项年化顺序 | 日频 a=252 时，错误顺序使 between term 比正确结果多放大 252 倍 |
| 状态 ddof=1 重构 | 与整体样本协方差一致，通过 1×10^-14 绝对容差核对 |
| NIW 频率换算 | 月转年资产风险乘 12，年化均值协方差乘 144 |
| NIW t 尺度与协方差 | 按 df/(df−2) 还原一致 |
| BL 三资产参考 | 先验总收益约 6.6541%／2.6755%／2.7022%；观点后约 5.9334%／2.7239%／3.9569% |
| 共同配置与单模型锚点 | 第 16.8 节锚点交叉失败，50/50 共同配置通过 |
| 参数平均与分布混合 | 两者风险矩不同；各自 PSD |
| GMV 与单资产下界 | 波动 10%／20%、零相关时，80/20 组合波动约 8.9443% |
| 有效收益与算术收益 | 100 五年变 150、年费 1% 时所需扣费前有效收益约 9.5426%；不与算术均值混用 |
| 零模拟失败的区间 | 2000/2000 成功的 Wilson 95% 下界约 99.8083%，不是 100% |
| 授权上限的嵌套性 | 放宽波动上限保留原可行权重集合 |
| 连续分段 DP 与穷举 | 30 条边、五段约束的样例均得到切点 6／12／18／24，代价差为 0 |
| 弧长加权前缀代价 | 与直接加权角度误差一致，通过 1×10^-12 绝对容差核对 |
| 直线前沿退化处理 | 101 点样例的等弧长边界为索引 20／40／60／80 |
| 对数正态 ES | 年均值 8%、标准差 20% 的一年简单收益代理，ES95 约 27.1242%；与数值积分误差约 6.66×10^-16 |
| 支撑切面与真实约束 | 构造了切面通过而原二次约束仍违反的样例，证明必须复核真实约束 |
| 产品完整复制及残差跟踪 | B 为单位矩阵、残差为零时公式恢复大类风险；非零残差样例的分解式与完整联合式一致 |
| 参数平均与随机选模型路径 | 同样的模型内方差 0.04、均值 ±10% 时，参数平均方差为 0.04，概率混合为 0.05，不能交换语义 |
| 产品预算的等式秩 | 两个类别预算加总权重约束可能线性相关，不能把重复等式全部当独立 KKT 条件 |
| 类别归属与统计暴露 | 资本预算满足，不代表 Beta 暴露完全相同；两种矩阵不能替代 |
| CVaR 非整数尾部 | 通过含分位点原子、需要部分尾部权重的样例；不使用不同的取整规则 |
| 切面通过并不等于风险通过 | 两资产样例切面残差 −0.0016，但真实二次约束残差 +0.0016，必须拒绝采纳 |

BL 样例的可重现输入：权重 `(0.55,0.35,0.10)`，delta=2.5，rf=0.02，tau=0.05；Sigma 行分别为 `(0.0324,0.00189,0.00135)`、`(0.00189,0.0049,−0.000525)`、`(0.00135,−0.000525,0.0225)`；观点为黄金总收益 5%（std 3%）与股票减债券 3%（std 2%）。

## 21.4 本轮完成与尚待验证

截至 2026-09-19，设计之外已经形成以下代码里程碑：

1. **P2 风险等级配置中心已有实现。** RiskScaleService、参考输入、草稿／预览／确认、不可变版本、默认版本、启用／退役、比较及前端工作台已经形成真实链路。
2. **P3 投资目标与约束已有实现。** Mandate 2.0 已接入 RiskScale 授权，现金预算与三类成功标准分离；列表、新增、修改为替代版本、删除／退役、实时资金回显、参考诊断和确认流程已经落地。

3. **P1 的两条 LTCMA 上游路径与独立 LTCMA 中心第一版已完成。** Product-first 已形成“产品池／产品 → 大类 → LTCMA”，Strategic-first 已形成“独立战略大类 → 研究 Proxy → LTCMA”；`applyScope()` 统一接收 `allocation:*` 与 `universe:*`。LTCMA 本身包括五类方法族、草稿与发布、冻结证据、人工／统计结果展示、SAA 只读选择和相关回归。实现与验收分别见 `ltcma-center-implementation-design-2026-09-18.md`、`ltcma-center-acceptance-2026-09-18.md`。

4. **P4 模式 A 的 E2 参数平均本次已实现。** 包含契约校验、固定签名 NJIT、原模型交叉诊断、不可变来源与融合结果、预览失效、历史恢复及 TAA 风险消费。验收范围与执行结果见 [模式 A 验收记录](multi-cma-parameter-average-acceptance-2026-09-19.md)。模式 B 现已有独立的连续联合求解与全模型门禁，见第 21.5 节；最终权重由共同问题求出。

仍未完成：跨频率／跨币种的通用证据适配、自动机构基准及利率来源选择、全部 NIW 先验模板与衰减策略、完整多期状态生成器、模式 B 五等级联合前沿与凹效用、E3 原生混合路径、class-level TAA 与产品应用解耦、产品实施新来源契约及全投研闭环。本次验收属于本地离线研究能力的技术验证，不代表真实市场预测已经验证或系统已部署。

## 21.5 共同约束模式 B 的本次交付（2026-09-19）

- 代码链路：冻结 CMA → 独立收益锚点 → 全源交叉评估 → 共同约束连续求解 → 全源最终检查 → 独立资金验证 → 不可变政策 → TAA 逐模型门禁。
- 数值实现：复用唯一 active-set QP 内核，新增带有限 box 目标下界的 LP、Phase-I 与风险支撑切面；固定 float64 只读签名，按 worker PID 预热，禁止请求时编译与 Python 数值回退。
- 页面交付：第三种模式、无权重模型选择、两种目标、求解状态／不可行下界、全部锚点交叉矩阵、共同候选复核、历史冻结来源与 TAA 诊断。
- 当前完成范围及实测记录：[模式 B 验收记录](multi-cma-compatible-all-models-acceptance-2026-09-19.md)。未完成的广义能力仍按 P4／P5／P6 表保留，不把本次基础版写成全投研闭环已完成。

## 21.6 下一实施顺序（2026-09-19）

两条 LTCMA 上游路径、独立 LTCMA 和 SAA 的 single／模式 A／模式 B 基础链路已经形成。后续不再优先扩写上游页面，按主流程阻塞程度推进：

1. **P5：拆开 class-level TAA 与产品应用门禁。** 大类层 TAA 可以在没有最终实施产品时继续研究；只有进入真实产品应用时才要求完整、有效的 Implementation Mapping。
2. **P5：收口产品实施契约。** 统一 SAA／TAA → Implementation Mapping → ProductAllocation 的来源、预算守恒、Q／Beta／残差、费用与实际风险复核，同时保留 Product-first 旧链路兼容。
3. **P6：统一验证与研究包。** 将 Mandate、RiskScale、LTCMA、SAA、TAA、产品权重、压力／成本／PIT 证据汇总为可复现的 ValidationReport／ResearchPackage，并明确通过、失败和未验证。
4. **增强项后置。** 模式 B 五等级联合前沿、凹效用、E3 原生混合分布，以及 LTCMA 跨币种／跨频率／统一 Proxy 生命周期不阻塞当前主链，按实际研究需求再扩展。


## 21.7 产品实施与统一研究包的本次交付（2026-09-20）

第 21.4、21.6 节为前一轮时间点记录。当前已新增 `backend/pre_investment`：直接冻结 SAA／已保存 TAA 承接、显式后置映射、类别预算与统计暴露分离、联合残差风险、single/A/B 产品门禁、逐边自融资费用、近期现金日历及任意剩余月份续算。共享原资金递推，保留存量整数年 ABI；未将原始 SAA 概率继承成当前余额的保证。

四个投前下游页面已接入真实研究包：产品与资金、证据汇总、候选锁定验证、具名研究定稿。不可变版本冻结来源、数组、代码及环境指纹，旧报告不能用于新候选；导出、复制重研及当前资格重新核验均已接通。SAA、TAA、产品历史构建及产品择时原入口保留。

本轮提供明确同月结算条件下的 ETF／现金产品路径、声明费率敏感性、复用已发布情景；不把静态产品回放称为动态 TAA 产品执行。实际全期交收／成交容量、基金未来持有期费率表、PIT 和认证独立审批仍未验证。Class-level TAA 与产品映射彻底解耦、模式 B 产品联合 QCQP、原生混合路径等增强项未由本轮基础版冒充完成。

详细实现、数值契约、里程碑修正及最终命令见[开发验收](pre-investment-implementation-acceptance-2026-09-19.md)。

---

# 22. 核验后的主要外部依据

引用用于支持具体方法，不用于给项目自定义政策背书。动态机构页面记录的是本轮核验时的内容；算法分档阈值、五方法产品分类、UI／接口名称和实施阶段均是本项目设计。

以下外部资料沿用早期设计审核的引用，2026-09-19 本次未重新检索。本文本次新增的实现状态和验收结论来自本地源码及执行证据，不以旧外部页面快照证明当前产品能力。

**[R1] CFA Institute — Capital Market Expectations, Part II: Forecasting Asset Class Returns（2026 curriculum）。** 支持统计、DCF、风险溢价及估计误差的区分，不代表“五类工具”是官方分类。
https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/capital-market-expectations-part-ii

**[R2] BlackRock Investment Institute — Capital Market Assumptions。** 本轮核验页面展示 August 2026、data as of 30 June 2026；公开方法区分均值不确定性与实现风险，并结合情景及模拟。不能据此断言其使用本文 C1–C5 分段或必须五套 CMA。
https://www.blackrock.com/institutions/en-global/institutional-insights/thought-leadership/capital-market-assumptions

**[R3] Vanguard — Vanguard Capital Markets Model forecasts。** 说明以动态风险因子与模拟形成分布；其公开长期收益展示可能采用几何年化，不能未经适配直接并入本项目算术参数。
https://corporate.vanguard.com/content/corporatesite/us/en/corp/vemo/vemo-return-forecasts.html

**[R4] Kevin P. Murphy — Conjugate Bayesian Analysis of the Gaussian Distribution（2007），第 9 节。** 核对 NIW 后验、t 尺度和预测分布；参数化须与本文定义一致。
https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

**[R5] Fischer Black、Robert Litterman — Global Portfolio Optimization（1992）。** 均衡起点与观点更新的原始研究。
https://rpc.cfainstitute.org/research/financial-analysts-journal/1992/faj-v48-n5-28

**[R6] PyPortfolioOpt 官方文档 — Black-Litterman Allocation。** 用作公开实现语义对照，尤其 posterior risk、tau／Omega 与 confidence；不代表本项目已经使用该库或其默认值。
https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html

**[R7] Edwin B. Wilson — Probable Inference, the Law of Succession, and Statistical Inference（1927）。** 成功比例区间的方法来源。
https://www.jstor.org/stable/2276774

**[R8] R. T. Rockafellar、S. Uryasev — Optimization of Conditional Value-at-Risk（2000）。** CVaR 辅助变量／尾部损失优化表示；不等同于资金成功概率问题。
https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk

**[R9] Stephen Boyd、Lieven Vandenberghe — Convex Optimization。** 凸性、二次约束、对偶界和数值停止条件的基础依据。
https://web.stanford.edu/~boyd/cvxbook/

**[R10] Almir Mutapcic、Stephen Boyd — Cutting-Set Methods for Robust Convex Optimization with Pessimizing Oracles（2009）。** 最坏情形与切集方法的研究依据；本文有限模型 QCQP 的实现仍需单独验证。
https://web.stanford.edu/~boyd/papers/prac_robust.html

**[R11] J.P. Morgan Asset Management — Cash Investment Policy Statement。** 企业现金流盘点、流动性目标与现金分层的业务依据；没有宣称可由现金流唯一求得风险容忍概率。
https://am.jpmorgan.com/gb/en/asset-management/liq/insights/liquidity-insights/cash-investment-policy-statement/

**[R12] Tushare — 指数成分和权重 index_weight。** 指数内部权重的数据含义，不是大类市场权重。
https://tushare.pro/document/2?doc_id=96

**[R13] Wassily Hoeffding — Probability Inequalities for Sums of Bounded Random Variables（1963）。** 独立有界变量的概率界；本文 D6 是其 Bernoulli 特例配合联合界。
https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830

**[R14] CFA Institute — Basics of Portfolio Planning and Construction（2026 curriculum）。** IPS 的收益、风险、期限、流动性与约束结构；不是以问卷或某个固定 C 级替代数学可行性。
https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/basics-of-portfolio-planning-and-construction

**[R15] CFA Institute — Overview of Private Wealth Management（2026 curriculum）。** 资本充足性可用确定性预测与 Monte Carlo 评估；这里只引用其数学方法，不把私人客户流程直接当成家办／企业治理模板。
https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/overview-private-wealth-management
