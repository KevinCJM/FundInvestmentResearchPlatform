# 情景算法中心：历史定义、实时识别和可信度闭环

日期：2026-09-14。分支：`ISSUE2609/BetterSaaTaa`。

本文是本次实施的设计契约，不代表验收已经完成。实际命令、结果和未完成项另见同目录验收记录。遵守 AGENTS.md 和 `docs/frontend-design-guidelines.md`；不修改 AGENTS 的通用规则。

## 1. 范围和原则

把一个页面中的实时/事后开关改为两个明确的业务入口，共享唯一计算图与执行器：

`历史状态定义 → 冻结历史参考 → 实时识别 → 独立验证/校准 → 有条件的下游应用`。

保留情景模拟与压测、全球历史事件库、指标中心连接、画布/公式/向导、全部现有算法、不可变研究版本和已有薄兼容入口。系统级 PIT 直接复用，不建立新的截止日默认值管理。CMA/SAA 主体、数据下载和新识别模型（如 Student-t HMM）不属于本次范围。

本次不提交、不推送、不合并。已有未提交的沪深300事后模板及图表原样保留，不回退或重新调参。

### 1.1 不制造重复功能

Regime Definition 本身是可执行规则：数据、区间生成、指标、条件和状态输出；运行即产生历史区间。因此不额外开发第二套 Historical Labeling 编辑器。已有 RegimeDefinitionV2、版本库、正式运行及发布机制继续为唯一实现。新业务元数据用可选、版本化契约扩展；旧数据缺失字段时保持原始序列化/hash，不自动改写。

### 1.2 可信度的严格含义

校准目标是 `P(参考版本在该观察日的状态 = 实时模型当时输出的状态 | 当时可得证据)`，不是市场真理、未来收益为正的概率，也不是投资获利概率。

分开显示：原始证据、判断稳定性、参考匹配表现、校准概率。不得将几个诊断随意加权得到“76%可信”。未验证、标签未成熟、样本不足、参考不匹配、数据过期必须明确显示，不能填零或绿色通过。

## 2. 前端：以用户任务组织

### 2.1 情景算法中心入口

保留一个中心，四个简短入口：

| 入口 | 说明 | 主操作 |
| --- | --- | --- |
| 历史状态定义 | 用规则划分历史，作为参考 | 生成历史区间 / 保存为历史参考 |
| 实时状态识别 | 只看当时信息识别已定义状态 | 运行识别 / 验证识别能力 |
| 情景模拟与压测 | 假设冲击对产品和组合的影响 | 沿用现有操作 |
| 全球历史事件库 | 可重叠的真实事件及研究窗口 | 沿用现有操作 |

使用当前 ScenarioCenters 路由和懒加载保留草稿。切换入口不串用 definition/revision/template/mode 查询参数。旧 `center=historical` 继续打开历史定义；旧带 mode=realtime 的深链接定位实时入口；不维护两份图编辑器。

### 2.2 历史状态定义工作台

固定事后模式，不显示能误切实时的模式按钮。名称、状态集合、数据来源、规则、运行按钮首先可见。保留内置模板、新建、自定义子图和公式。对业务分类增加可选 family（市场趋势、增长/通胀、风险、金融条件、自定义）及说明；它们是分类语义，不强制所有模型使用同一组状态。

结果继续使用现有真实走势图、区间、证据及未分类区域。确认保存才发布参考；预览不成为正式参考。参考锁定定义修订、运行 hash、观察序列、频率、状态字典和数据血缘。后续引用不能根据同名模板或最新数据重新拟合。

参考质量展示边界稳定性、参数敏感性和覆盖；不加入无依据的综合 Definition Fitness 百分比。首尾未知不能叫震荡。人工历史事件允许重叠，不能充当单一互斥状态的默认训练标签。

### 2.3 实时识别工作台

固定实时模式，共用原有编辑器。非因果祖先在算子目录说明不可选，服务端仍检查全部依赖，不能靠隐藏按钮保护。

顶部显示“要识别的历史参考”，选择已发布的精确版本。说明范围、日期、状态和频率；更换参考即使名称相同也使旧验证结果失效。可以未绑定先探索，明确提示“选择历史参考后才能评价准确率”。

三个任务结果区域：识别结果、历史验证、可信度。默认只显示当前状态/数据日期、相对参考的一致程度、区间重合、转折延迟、是否可用。详细矩阵、参数和 hash 折叠。验证前要保存当前模型，不能误用上次保存版本验证未保存草稿。

验证配置只暴露：参考版本、验证截止/分段日期、训练最少样本、校准/测试分割、误报容忍和拒识阈值等真正生效参数。高阶块长度、最小类别数、概率方法放高级选项；不写只展示不生效的参数。

图表采用现有 ECharts/结果适配器，实际日期轴对齐，概率图旁有表格；颜色不是唯一信息。可靠性图显示每箱预测概率、实际匹配率、样本数。类别统计和独立区间数量必须可见。

### 2.4 交互和视觉验收

使用 ui.tsx Button/Badge 等原语，accent 主按钮、slate 中性、最小12px、控件≥40px，数字 tabular-nums。每个加载/空/错误/禁用态解释原因并提供下一步。取消、请求乱序、模型/参考切换、PIT 切换不能展示旧结果为当前。320/768/1440px、键盘页签、焦点、表格/图表文字对比度在浏览器验收。保留画布窄屏区域切换，不在主屏塞入大段技术文案。

## 3. 后端领域契约

### 3.1 定义的可选研究契约

在现有图定义中增加可选 `study`（或等价单一扩展契约），包含：

- `purpose`: historical_reference / realtime_recognition；旧定义无字段保持现有含义。
- `family`: market_trend / macro_growth_inflation / risk / financial_conditions / custom。
- `reference`: 精确 run_id、publication_id、content_hash、definition_id/revision；不能只存名称。
- `state_mapping`: 实时状态到参考状态的显式映射；默认完全相同状态 ID，禁止按颜色、顺序或中文名猜测。
- `calibration_id`: 可选已冻结校准制品引用，不作为历史事实倒灌到早期日期。

没有 `study` 的旧版本序列化不得新增 null/default 字段改变 hash。实时引用的参考是评价目标，不加入其图节点依赖；避免未来标签污染特征。

### 3.2 历史参考

复用正式 retrospective run + 已确认 publication。参考目录是上述成果的只读投影，不复制巨量状态数组或新建一个第二定义库。验证时检查：immutable、快照 hash、发布记录 hash、非人工重叠事件、状态字典、范围和频率。首次确认可复用 enable_research_version 的研究发布路径。新“作为参考”用途必须有明确服务端门禁，不能把任意试算当发布结果。

状态生效区间使用现有观察索引契约。参考定位与交易执行定位分开：比较当前状态用 observation_date；真正交易信号仍按 recognized_at/effective_date 处理。

### 3.3 验证与校准产物

独立不可变的 ValidationReport / CalibrationArtifact，引用原始预测、参考、数据、算法、状态映射和实际时间切分。复用 AtomicJsonStore 的原子版本机制，大时序使用既有制品目录与校验；禁止反写已有 run。

报告至少包含：

- identity：输入的精确 hash、规则/内核版本、随机种子、实际范围。
- sample：输入数、匹配数、参考未分类数、预测拒识数、未成熟数、各状态数、完整参考区间数、转折事件数。
- classification：confusion matrix、per-state precision/recall/F1、balanced accuracy、macro F1、strict accuracy。
- intervals：每状态 IoU、区间等权聚合，明确与逐日时间加权 IoU 区别。
- transitions：同方向转折的一对一匹配、延迟中位数/P90、漏检/误报数；窗口内多次切换不能重复命中一个参考转折。
- stability：复用已有参数扰动、prefix invariance、flip 指标；没有实际执行的检验不能标 passed。
- probability：raw score type、Brier/logloss、reliability bins、ECE（诊断而非唯一门禁）。
- calibration：方法/参数/训练截止/验证截止、测试表现、可用日期、样本充足性、过期规则。
- eligibility：可研究、证据不足、仅回顾性评分、校准可用/不可用及原因。

不要用 `accuracy=0` 表示不能计算；无分母指标为 null + 原因。

## 4. 原始证据与最终状态关联

| 来源 | 实际含义 | UI/计算处理 |
| --- | --- | --- |
| 规则命中后的 one-hot | 确定性编码 | 显示“规则判定，未校准”，不能当100%可信 |
| HMM/GMM/Markov posterior | 模型内部后验 | 显示“模型概率，未校准” |
| Ensemble vote | 加权投票比例 | 显示“模型一致票占比” |
| 变化强度/阈值 | 信号强弱 | 显示得分，不叫概率 |
| Frozen calibrator output | 对指定参考的经验映射 | 显示“校准后的参考匹配概率”，带状态与样本依据 |

证据必须跟随最终状态输出的真实祖先。经过 confirmation / component_map / confidence_gate 后，不能把上游 HMM 的牛概率无解释地绑定到已经被改成熊或拒识的输出。保留 raw_state 与 final_state 区别；计算最终状态可信度取 `q[final_state]`，不是一律 max(q)。未知状态不生成有效置信度。

旧 `series.confidence` 数值不被静默重算；新协议添加明确 provenance/nullable calibrated 字段，读旧数据时只作语义标记，不改保存数据。下游新策略不能继续把原始字段当校准结果。

## 5. 时间与样本契约（最高风险）

### 5.1 三种日期

- observation_date：历史上该标签描述哪一天。
- label_known_at：该标签计算实际使用的信息最晚可得时间。
- published_at：成果真实发布的时间；不能倒填。

训练/校准要求标签在训练截止日已成熟。很多全样本峰谷、PS筛选、平滑HMM的全部标签都依赖全输入；不能简单用“峰值 + 右窗口”假装成熟。直接继承现有 temporal audit / recognized_at / full_input 信息。

### 5.2 两种验证必须明确区分

**回顾性参考评分**：用今天已成熟参考评价历史因果输出。可以报告参考一致性，但不能宣称校准器在历史时点已可使用。

**可部署时序检验**：每个训练边界只用当时可得数据和已成熟标签拟合；校准使用训练样本之外的历史预测；再在之后未用样本测试。若只有当前 full-input 参考、无历史成熟版本，则不伪造可部署校准检验。可以发布今天训练的实验性校准器并保守显示验证不足，但不能授权过去 TAA 回测。

实时识别模型有拟合步骤时走 expanding walk-forward：训练窗拟合标准化/模型/状态映射后，只输出测试窗。规则固定时可直接因果执行，但参数搜索、参考规则选择仍必须披露是否在同一历史期间看过结果。

建议显式 train / calibration / validation / final-test 时间块。实现可先提供前向分段的两阶段评分，但任何校准方法、门槛选择不能利用最终测试块。去掉未知标签和 warmup 后核对每类/每阶段是否足够；不足就说明，不偷偷随机重分。

### 5.3 对齐

默认相同观察频率与同一研究对象，日期严格交集。需要月频参考对应日频预测时，必须显式区间映射，不能概率线插值；该功能未实现时明确阻断跨频率并建议同频模型，不自动 forward-fill。

参考边界模糊区只作补充分析；全样本严格指标、剔除边界比例必须同时保留。首次参考区间左截断、末段右删失，不算完整转折；未知不能被算命中。

## 6. 统计定义

混淆矩阵按真实参考状态作行、预测状态作列，额外拒识列。拒识是未接受预测，不能先删除再报全样本准确率。每类 recall 分母含被拒识样本，precision 只针对该预测类。Balanced Accuracy 为实际出现参考类的 recall 平均，缺少必要类的发布门禁单独失败；Macro F1 明确类别集合。

IoU_k = 同属状态k的样本 / 至少一方为k的样本。区间匹配使用同状态一对一最大交叠；每个参考段最多匹配一段预测，孤立误报仍统计。转折匹配必须相同有序状态对，不能把转牛匹配转熊。延迟按观察间隔而非不明确的自然日；日/月频在 UI 明示。

多类 Brier = mean(sum_k((q_k - 1[y=k])²))，不是准确率；LogLoss = -mean(log(q_y))。概率 clip 仅用于数值对数保护，要公开 epsilon。报告未校准、校准和简单训练类别基准；基准只能用过去样本估计。

熵和margin只描述模型分离度，不进入人为“综合可信概率”。只投票/后验高不证明与参考一致。

## 7. 校准、拒识和不确定性

### 7.1 概率模型

先采用单参数 temperature/power scaling：q_k ∝ p_k^(1/T)，固定正值有界搜索由 NJIT 完成。选择 T 仅使用 calibration 样本，最终测试不参与。对 one-hot 不执行 temperature（加 epsilon 后软化不等于找到了额外信息）。映射后的类别轴与 final state 一致，概率和严格校验。

### 7.2 规则模型

缺少连续得分时采用明确的保守基线：从校准段估计 `P(reference_class | predicted_class)`，可带事先声明的简单平滑计数。显示“同类历史平均”，不是当天独立证据；无该预测类样本则不可用。不用 one-hot 的1作为 sigmoid 输入制造精确概率。

未来可复用连续规则margin+sigmoid/isotonic；本次不要为一个原型引入无法验证的复杂元模型。标签-only基线和temperature都是同一校准服务的有类型方法，不复制一套图模型。

### 7.3 门禁

可配置最少总样本、每类样本、参考完整区间数、最大校准年龄、拒识门槛。默认值是平台安全政策而非行业真理，文档说明并锁定版本。样本数不能只看日数：十年日线不等于2500个独立周期。

校准拟合成功 != 验证通过。要求独立后续样本且有明确基准比较；少样本、没有某一状态、参考后来才产生的标签进入早期训练等均不给正式可用标记。UI 不把泛化证据不足合并成普通错误。

### 7.4 区间与拒识

报当前匹配概率时同时显示其来源样本和报告状态；不默认给“95%置信区间”。依赖时序下区间使用明确块长度/固定种子的块 bootstrap，检查足够区间和边界；若未实现或样本太少则显示“不足以估计”。块自助仅是平稳/弱依赖近似，不保证金融变动下有效。

拒识以 final_state=unclassified/decision_status=abstained 明确表达，raw_state/probabilities可供诊断；保存 accepted coverage 与 accepted error，不能只展示过滤后的高准确率。

## 8. 模块化实现

在 `backend/historical_regimes/` 下建立小型 reference/reliability 子域：contracts、service、alignment、metrics/kernels、calibration、routes（按实际职责可合并小文件）。继续调用 RegimeGraphV2Service 的现有 prepare/executor/版本/序列hydrate；不要复制模型拟合、PIT、图解析或状态生成代码。公共哈希/运行校验用唯一 helper，不在数个服务抄写。

已有 _validation_reports 的工程稳定性检查保持用途；新的 reference report 不把 full_sample_comparison 冒充准确率。若抽出 walk-forward runner，原稳定性与新可靠性服务共用唯一实现，参数/时点行为以回归固定。

新增数值核明确固定 readonly任意stride ndarray签名、启动warm、disable_compile、execution audit。对齐用整数日期轴/indices，避免每折复制输入。所有 confusion、IoU、transition、Brier/logloss、temperature、bootstrap数学在NJIT，不使用pandas滚动或sklearn数值fallback。

允许分配必要输出和小型K×K工作区。记录关键np.shares_memory测试、常量K/时间轴最大长度、折数/实验数/样本量预算；无边界的任意用户函数不允许进入核心。

### 8.1 API 意图（命名随现有路由风格统一）

| API | 职责 |
| --- | --- |
| GET /api/historical-regimes/references | 返回可选已确认历史参考目录，不加载全部大数组 |
| POST /api/historical-regimes/references | 确认一个精确事后运行作为参考，校验血缘 |
| POST /api/historical-regimes/reliability/preview | 精确模型+参考+policy，运行因果回放并产生未保存报告 |
| POST /api/historical-regimes/reliability/confirm | 显式确认报告/校准制品，检查请求与预览hash |
| GET /api/historical-regimes/reliability/reports/{id} | 读取不可变报告 |
| GET /api/historical-regimes/reliability/catalog | 可用报告/校准器摘要，关联精确模型参考 |

重任务复用已有作业/取消/预算机制；不把全历史训练塞进无边界请求。设计若采用同步受控首版，必须限制输入/折数和报告耗时，且不冒充异步就绪。

发布/确认接口由服务端重新解析模型和参考，不信任客户端上传的“已校准数组”“passed”或hash。预览不保存业务产物；确认可以重新计算，或引用有TTL及请求hash的不可变服务器预览，不接受前端自造报告。

## 9. TAA接入与兼容

新协议 realtime 模型用于 TAA 时，使用统一函数解析 calibrated reference confidence；无校准、过期、模型/参考/状态轴不符、计算日期早于可用日期 => 明确原因并回归 SAA，不抛弃整个研究功能。

老的冻结研究/回测结果保持只读原样。若旧运行仍供旧契约回放，可由显式协议版本进入有测试的薄适配；不得让缺少协议版本的新调用自动绕过新门禁。manual/momentum等非regime策略不被要求参考。

只校准 max confidence 不够：TAA按状态概率倾斜时，应使用映射到参考状态轴的校准概率向量和其匹配目标，不能拿“预测正确率76%”替代三个状态概率。此行为需要与现有state_tilts轴检查一起回归。

## 10. 迁移与清理

旧深链接继续定位正确工作区，原显式双模式兼容入口仅作为实际旧链接薄适配，主工作流不再显示双模式。旧定义在目录按已保存mode/明确元数据分类；不根据名字猜测。复制旧定义到新流程需用户确认，不自动换hash。

更改定义、参考、参数、校准策略应产生新版本并使未保存结果过期；发布历史不跟随最新引用改变。删除被本次新交互替代的无用模式选择UI、未引用导入，不删除仍有使用的算法内核。

## 11. 里程碑与自审自测

- **M0设计**：核对规范、数据/时点边界和前后端接口，不宣称设计已实现。
- **M1结构与引用**：独立历史/实时入口、可选study、正式参考血缘。回归旧hash、模式服务端约束、参考篡改/跨轴/旧链接/草稿保留。
- **M2参考验证**：真实因果执行+状态/区间/转折统计，缺失、拒识、同日轴、非连续日期、边界测试；验证数学参考与NJIT一致。
- **M3校准与展示**：独立时间块、标签成熟门禁、规则平均/temperature、报告保存、UI四态/乱序。固定随机种子、概率sum、空类、0/1、只读stride测试。
- **M4下游**：TAA新协议calibrated gate、过期/未可得回归SAA、旧冻结结果不变、非regime不受影响。
- **M5总验收**：全部相关后端+前端，类型/构建/design/i18n，真实浏览器320/768/1440/键盘/对比度。审查实际调用路径、死代码和重复逻辑。路由记忆按验证事实最小更新。

每个里程碑实施后马上运行对应测试并记录实际结果；失败先修，不把后续全回归代替中间检查。

### 必须包含的反例

1. one-hot为1也没有校准可信度；无参考的实时运行仍可研究。
2. 参考未分类不能补震荡；预测拒识不被从全样本分母删除。
3. 不同状态ID不能靠顺序自动映射；改变映射hash使校准失效。
4. 校准与测试分开；修改测试标签不能改变拟合T/计数。
5. full-input参考在最后时点才成熟，不能拿早期observations训练过去校准器。
6. 所有概率来源沿实际最终依赖；确认期改变final state时不能误取max概率。
7. 预测/参考日期错位、跨频率、缺失参考段/首尾截断明确处理。
8. 不可变成果篡改、参考发布不符、过期、请求乱序均失败关闭。
9. TAA未校准时回归SAA，旧历史结果不重算，非regime不强绑。
10. 视图共享、readonly/stride、内核未warm/新dtype拒绝，不出现request-time编译/Python fallback。

## 12. 研究依据

下列文献支持方法，不构成这些模型在本项目市场上有效的证明：

- NBER Business Cycle Dating：https://www.nber.org/research/business-cycle-dating 。回顾性周期确认与当前时点判断不同。
- Guo et al. (2017), On Calibration of Modern Neural Networks：https://proceedings.mlr.press/v70/guo17a.html 。单参数temperature及校准思想，原实验不是金融市场认证。
- scikit-learn Probability calibration：https://scikit-learn.org/stable/modules/calibration.html 。模型与校准的独立样本、概率定义；本项目数值实现仍遵守NJIT，不直接复制第三方拟合路径。
- Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation：https://doi.org/10.1198/016214506000001437 。概率评分。
- Politis & Romano (1994), The Stationary Bootstrap：https://doi.org/10.1080/01621459.1994.10476870 。相关时序块重采样的理论及适用假设。
- Gangrade et al. (2021), Selective Classification via One-Sided Prediction：https://proceedings.mlr.press/v130/gangrade21a.html 。拒识与风险/覆盖权衡。

所有示例门槛属于配置策略，不称为行业统一标准。最终只有实际执行的测试可以进入验收记录。

## 13. 实施收敛与本轮审核补充

以 `regime-reference-confidence-final-audit-2026-09-14.md` 为本轮交付记录。核心研究版已实施；前瞻部署验证、依赖时序置信区间、新报告内的统一参数/窗口/种子稳定性汇总仍未完成，不能以设计存在替代实现。

新增校准样本门禁同时检查：有效预测—参考配对总量、参考类别样本、预测类别样本。Temperature 只计入有效概率行；只看参考标签数量不够。未分类、拒识、缺失概率不得被转换成有效校准样本。所有检查仅使用校准时间段，未来测试标签或预测不能提高校准段的充足性。

数值颗粒度复核：`calibration_support_kernel` 只回答“这次校准真正拥有多少可用配对及类别支持”。输出计数可以独立审阅，门槛属于上层验证政策；它不是用户图中的新黑盒分类模型。现有分类统计保留拒识分母，不能直接替代需要按校准方法筛除无效概率的配对统计。统一一次只读扫描维护两个类别轴，进一步拆成逐条比较节点没有业务价值；不更改现有识别算法、阈值或状态输出。

固定用途工作区的模板徽标展示当前用途；复用实时可用模板作为历史定义，不代表混入实时任务。原双模式兼容入口的行为保留。旧不可变成果和先前保存的 hash 不因以上修正被覆盖。
