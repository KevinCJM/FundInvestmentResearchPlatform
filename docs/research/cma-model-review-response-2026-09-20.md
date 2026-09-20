# CMA 五模型审核：逐项判定、最小修复与验收

日期：2026-09-20。输入：根目录 `cma-model-review-2026-09-20.md`。基线：当前工作树（包含已有未提交 LTCMA／Multi-CMA／产品实施工作），本次不提交、不推送、不修改正式研究数据。

## 1. 判定与本次范围

| 编号 | 事实与判断 | 本次处理 |
|---|---|---|
| P0-1 | Box 半宽求和确实不使用均值协方差，但这是合法的区间稳健模型，不是数值错误；椭球可选项合理。 | 保留 box 默认和 ABI；为单 CMA 增加显式椭球选项，消费冻结均值协方差；缺来源或未支持的多 CMA 组合失败关闭。 |
| P0-2 | 只收缩非对角线属实；“必然低估真实风险”“常相关或 LW 自动保守”不成立。 | 不改变收缩目标、默认值或 RiskScale 算法；明确收缩方法和样本风险方向，增加正／负相关反例。 |
| P1-1 | 当前只是历史占用率混合并做日频年化，不是多年 Markov 模型。状态持续不必然产生收益正自相关。 | 增加相邻状态转移、持续期和可识别性诊断；保留单期矩和年化语义，明确没有多年状态路径。核验事后标签的知识截止。 |
| P1-2 | 单一对数正态代理没有保留原生情景尾部属实；成功率偏差方向不固定。年度情景每月重抽会改变模型。 | 强化资金分布披露；不从单期矩擅自补出时间转移或分布族。原生路径作为独立契约扩展，不改旧资金递推。 |
| P2-1 | 单资产／两两观点限制属实，是表达力缺口。 | 新增篮子观点，保留旧观点格式及数值；统一进入同一个 BL 更新内核。 |
| P2-2 | BL 的 M 已冻结但不进入 box 半宽属实，是明确的既有取舍。 | M 接入可选椭球；不强制覆盖人工半宽或把 M 加进资产风险；多 CMA 显示半宽来源及混合语义提醒。 |
| P2-3 | 缺少直观的 δ／τ／Ω 诊断属实。 | 输出隐含市场 Sharpe、逐观点方差比、逐资产更新贡献与先验／后验变化；不把某个 Sharpe 阈值当硬性金融规律。 |
| P2-4 | 短窗口可搭配长预测期限属实；“业内从不用历史均值”、任意三年硬门禁不成立。现有结果表已经展示半宽。 | 增加样本／预测期限和均值标准误披露；NIW 区分新增证据批次与先验信息。保留短窗口敏感性研究，不擅自取消历史方法的采纳资格。 |
| P3-1 | ddof=0／1 的区别属实，但直接替换状态分母会破坏现有 ML 混合恒等式。 | 保留状态 ML，明确与历史样本无偏协方差的区别，补重构测试；不制造“统一 ddof 后就等于全样本无偏估计”的假结论。 |
| P3-2 | 缺估值／收益组成模型属实，属于产品范围。 | 记录独立 building-block 需求，不把第六模型和新数据采集混入修复；Manual、BL、情景可承载前瞻判断，不能说所有现有模型都无法生成前瞻假设。 |

## 2. 数理边界与反驳

### 不确定性

Box：`haircut = penalty * sum(abs(w_i) * u_i)`；其半宽可以是人工区间或模型边际参考，不是组合的联合 95% 置信保证。连续分布中“角点概率为零”不能用来否定整个不确定集。

椭球：`haircut = kappa * sqrt(w' U w)`，U 是均值估计协方差，不是资产收益协方差。高维联合半径也可能比 box 更大，不能承诺必然少扣收益。

令 d 为 U 中严格非零方差的坐标数，不是通过特征值阈值估计的秩；确定性零方差坐标不计入 d。

- BL：在声明的高斯均值模型下使用 `kappa² = chi2(d, confidence)`。非零方差子块正定时为该模型的联合可信域；奇异时 d 可能大于实际秩，采用保守半径并明确提示，不通过数值阈值删除小方差方向。
- NIW：边际均值为多元 t，要求均值协方差正定，此时 d=N。以协方差 U 表示时，`kappa² = d * (df - 2) / df * F(d, df; confidence)`，`df = nu_n - N + 1`，不得把 t 的协方差误当尺度矩阵。
- Historical：未收缩、独立正态样本且非零方差子块正定时使用 Hotelling `kappa² = d*(T-1)/(T-d)*F(d,T-d;confidence)`；要求 T>d。使用收缩或奇异样本矩阵时不宣称精确覆盖，须明确确认高斯插件近似。全零 U 使用半径 0，不把样本零波动宣称为未来无风险。
- 缺失均值协方差不得用资产风险代替；多 CMA 的均值误差交叉协方差尚未定义，本次明确不支持多 CMA 椭球，不擅自平均协方差。

### 协方差收缩

相对原样本：`Delta variance = -2*lambda*sum_{i<j}(w_i*w_j*S_ij)`。正的加权交叉项使风险下降，负的交叉项使风险上升；都不能推出相对未知真实风险的偏差方向。常相关不是逐组合风险上界，Ledoit–Wolf 的矩阵损失优化也不等于保守风险估计。审核建议的 `r_bar/rho_ij` 写法在零相关处无定义，不采用。

### 状态持续与多期分布

`Var(sum_{t=1}^H r_t) = H*Gamma_0 + sum_{k=1}^{H-1}(H-k)*(Gamma_k+Gamma_k')`。波动率聚集不等于收益正自相关；如果各状态条件均值相同且创新不相关，状态可以很持续而收益自协方差仍为零。不从转移矩阵单独推算“十年风险低估百分比”。

相邻状态统计只使用真实相邻的已知状态；未知状态是断点。无出边或不可确认唯一平稳分布时，输出不可用原因，不补自循环、不编造平稳概率。经验持续期是一步 Markov 解释下的观察期数，不直接冒充月数。

年度情景矩只规定单期分布的两个矩；未定义情景内分布、情景选择时点及跨期规则。按月重抽年度情景并不能保留年度混合风险。单因子对数正态与左偏情景分布的成功率差异取决于目标阈值和现金流，验收不能预设所有例子都同方向。

### ddof 与样本

状态 ML 矩按 `n_s/T` 混合，在原占用率、无收缩下重构全样本 `ddof=0`。全样本无偏协方差需组合 `sum((n_s-1)*S_s + n_s*(mu_s-mu)*(mu_s-mu)')/(T-1)`；只替换每状态分母不等价。

审核附录 A 的历史均值协方差写成 `Sigma_annual/T`，遗漏 252；当前代码 `252*Sigma_annual/T` 才是日频样本年化均值的协方差。

## 3. 工程设计与四问

1. 本次新增选项、观点与诊断均直接对应审核条目；building-block、统一 Markov／情景资金路径、改变 RiskScale 收缩目标不属于必要修复。
2. 不改会阻塞的内容：篮子观点表达、冻结均值协方差的显式稳健消费、参数／样本／分布诊断；不阻塞内容不借机重构。
3. 复用唯一候选搜索、唯一 BL 线性求解、现有状态转移和 Beta 数值能力；新算子仅处理独立的椭球半径、均值误差及诊断，不复制完整投研算法。
4. Box 默认、旧 absolute／relative 数值、冻结历史、RiskScale、Mandate 和资金现金递推保持原契约。新字段明确选择，旧请求序列化省略新默认字段。

数值输入沿用 readonly 任意 stride ABI；小型矩／系数工作缓冲区允许分配，收益面板不按状态复制。新内核固定签名、禁止请求时编译、按进程预热。历史记录只读，不调用新模型重算。

## 4. 里程碑与验收记录

- M1：BL 篮子、参数诊断；原观点数值、轴／日期／近共线／缺失测试。
- M2：冻结均值协方差与单 CMA 椭球；独立 SciPy 参考、box 位级不变、只读／stride、缺失来源拒绝、多 CMA 拒绝、保存后原样恢复。
- M3：状态／窗口／资金分布与前端；转移断点、ddof 恒等式、诊断数值、输入失效、真实浏览器与响应式。
- M4：相关后端、前端全量、类型、构建、设计／语言、契约及路由检查。

### 4.1 实际修改位置与原因

| 变更组 | 主要文件 | 为什么／怎么改 |
|---|---|---|
| 单 CMA 均值椭球 | `backend/strategic_allocation/kernels.py`、`uncertainty.py`、`uncertainty_kernels.py`、`contracts.py`、`service.py` | 为已冻结的 U 提供显式消费渠道，复用唯一候选搜索；Box 默认、旧 ABI 和旧请求回显保持原义。覆盖半径按模型分别校准。 |
| 冻结证据与一致性 | `cma_application.py`、`cma_models.py`、`multi_cma.py` | Historical U 提升为持久化数组；mmap 读取、来源与结果 hash 校验；不补算旧版本，也不从资产风险伪造 U。多模型半宽来源可见，未定义的多模型椭球明确拒绝。 |
| BL 表达与诊断 | `cma_model_contracts.py`、`cma_model_kernels.py`、`cma_models.py` | 增加显式篮子观点及校验，全部委托同一高斯更新；输出隐含 Sharpe、观点方差比和逐资产更新贡献，不自动重设 δ／τ／Ω。 |
| 状态、样本与资金限制 | `cma_statistical_kernels.py`、`cma_statistical_models.py`、`cma_service.py`、`planning.py` | 新增转移／持续期诊断、样本期限比及标准误披露；保留 ddof 与年化口径。资金路径明确标记矩匹配代理及未保留的原生情景信息。 |
| 前端真实交互 | `BlackLittermanViews.tsx`、`MeanUncertaintyFields.tsx`、`LtcmaModelDiagnostics.tsx`、`LtcmaResults.tsx`、`CmaModelEditor.tsx`、`PolicyCandidates.tsx`、`StrategicAllocationWorkspace.tsx`、`MandateResults.tsx` | 篮子可编辑；椭球显式选择覆盖水平和近似确认；显示诊断及冻结依据。同步服务类型、生成契约和中英文词条，输入变化废弃旧预览。 |
| 回归与治理 | `backend/tests/test_cma_review.py`、`test_cma_uncertainty.py`、`test_cma_diagnostics.py`；前端 `CmaReview.test.tsx`、`e2e/ltcma.spec.ts`；`docs/pitfalls.json` P23 | 独立数值参考、反例、旧版本、真实调用和浏览器验证；记录 Box／椭球及时间／统计语义不能混用的契约。 |

### 4.2 2026-09-20 本轮实测

以下均为重新打开工作区后实际执行，日志前缀为 `.tmp_cma_model_review_20260920/resume-`；此前日志不重复计作本轮证据。

| 检查 | 实际结果 | 日志 |
|---|---|---|
| CMA 专项基线 | 94 项通过 | `resume-baseline.log` |
| 补齐边界后的 CMA 专项 | 105 项通过；其中新增 11 项覆盖坐标维度、奇异 BL、无效 U、NIW 冻结消费与内存共享 | `resume-specialist.log` |
| 后端关联回归 | 41 个测试文件、905 项通过，包含上述专项，不相加计数 | `resume-backend-full.log`；文件清单 `resume-backend-files.json` |
| 前端全量单测 | 160 个文件、1297 项通过 | `resume-frontend-full.log` |
| LTCMA 浏览器 | 18 项通过，320／768／1440；包含篮子、椭球、发布和历史恢复 | `resume-browser.log` |
| 多 CMA 浏览器 | 12 项通过，覆盖模式 A／B、约束冲突、冻结读取及 TAA 门禁 | `resume-multi-browser.log` |
| 原战略流程浏览器 | 10 项通过，覆盖原历史前沿、资金目标、SAA→TAA | `resume-strategic-browser.log` |
| 旧数值兼容 | Git `0401b01` 的 Box 候选与旧 BL 输出，24 组逐字节一致 | `resume-legacy-parity.log` |
| 类型／构建／契约／设计／语言 | TypeScript、Vite、生成契约检查、design:check、i18n 全部通过 | `resume-types.log`、`resume-build.log`、`resume-contracts.log`、`resume-design.log`、`resume-i18n.log` |
| AI Hermes 路由治理 | 全工作树 `validate_ai_routing.py` 通过；本轮 8 个路径 coverage 全部登记；R87 路由 smoke 通过 | 当前工作树实测；稳定新增路径已进入 Git 暂存候选，不使用 allowlist 绕过 |
| **仅 staged snapshot 验证** | 从 `HEAD + git diff --cached --binary` 建立独立 worktree；CMA/LTCMA 专项 **229 项通过**，前端关键回归 **6 文件 / 61 项通过**，TypeScript、Vite build、`validate_ai_routing.py` 均通过 | 证明提交索引本身包含调用方与新增符号，不再依赖未暂存工作树；41 文件后端全量在隔离路径因 Numba 冷缓存超过单次 300 秒工具上限，未把超时记作通过或失败 |

旧数值参考从 Git 对应文件读入隔离的内存模块，只关闭参考模块的编译缓存；未把旧实现复制到生产源码。Box 使用三个固定种子，BL 覆盖无观点、绝对、相对及多观点。此证据只覆盖列明样例，不宣称对所有可能输入作过穷举证明。

性能 smoke：30 个资产、5000 个候选、预热后 5 次中位数，Box 约 5.37ms、椭球约 10.24ms；Python 跟踪分配峰值分别 4312／24040 字节，签名增长 0、Python fallback 0。椭球增加一次均值协方差二次型，不宣称比 Box 更快。U 使用只读跨步视图并验证共享内存；此为单机内核测试，不包含启动／I/O，跟踪分配不代表全部原生内存。详见 `resume-performance.log`。

已查看本轮桌面 BL 诊断、手机椭球结果截图；浏览器测试同时覆盖布局边界、文本对比度和真实 API 交互。截图保存在 `resume-browser/`。既有 Numba 连续性提示、Starlette 弃用提示、部分 React act 提示和前端包体积警告均非本轮测试失败，未扩大范围修改。

### 4.3 验收边界与未采纳项

- 当前代码／数值／浏览器验证通过，不表示真实市场预测或资金成功率已获模型有效性认证。
- 没有实现原生情景／Markov 多期资金引擎，没有改变现有现金递推，也没有把年度情景每月重抽伪装为等价修复。
- 椭球仅支持具备可校准冻结 U 的单 CMA；多 CMA 的均值误差依赖结构没有擅自推断。
- 没有强制替换对角收缩，没有直接把状态 ddof 改为 1，没有新增任意三年采纳硬门禁，也没有取消原 Historical／Manual 等方法。
- Building-block 仍为独立规划需求；本次不下载新数据或实现第六模型。
- 修正本回复文档曾写成 `rank(U)` 的描述：代码按严格非零方差坐标数校准，奇异时披露保守／近似语义，并补充测试防止文档与代码再次分叉。
- AI Hermes：已把核验过的行为边界补入现有 P23，并把本轮 8 个新增测试／LTCMA 前端／回复文档路径登记到 `strategic_allocation_policy`。原 LTCMA／Multi-CMA／产品实施中已被 routing 稳定引用但尚未跟踪的源码、测试、配置及验收材料已纳入 Git 暂存候选；没有通过 artifact allowlist 或关闭 tracked-reference 检查规避治理。当前全工作树 `validate_ai_routing.py` 通过，本轮 coverage 无遗漏，R87 路由 smoke 通过。
- 未提交、推送或部署；原始审阅文档只读，已有并行任务和正式研究数据未改写。

## 5. 方法依据

- CFA Institute, Capital Market Expectations Part II：统计、DCF、风险溢价方法并存。https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/capital-market-expectations-part-ii
- Ledoit & Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices：矩阵损失意义的收缩估计，不是投资组合风险上界。https://doi.org/10.1016/S0047-259X(03)00096-4
- Ceria & Stubbs (2006), Incorporating estimation errors into portfolio selection：估计不确定性与稳健选择。https://link.springer.com/article/10.1057/palgrave.jam.2240207
- Kevin Murphy (2007), Conjugate Bayesian analysis of the Gaussian distribution：NIW 后验和多元 t 的尺度／协方差。https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
- Penn State STAT 505, Hotelling T-square：未知协方差下有限样本 F 校准。https://online.stat.psu.edu/stat505/Lesson07
