# 市场状态：验证、校准与前瞻资格

本专业契约区分历史参考质量、回顾性识别分数和实际前向资格。主要操作入口见[市场状态研究](README.md)。历史报告不可回写，当前可用资格须按消费者决策时间重验。
## 后端领域契约

### 定义的可选研究契约

在现有图定义中增加可选 `study`（或等价单一扩展契约），包含：

- `purpose`: historical_reference / realtime_recognition；旧定义无字段保持现有含义。
- `family`: market_trend / macro_growth_inflation / risk / financial_conditions / custom。
- `reference`: 精确 run_id、publication_id、content_hash、definition_id/revision；不能只存名称。
- `state_mapping`: 实时状态到参考状态的显式映射；默认完全相同状态 ID，禁止按颜色、顺序或中文名猜测。
- `calibration_id`: 可选已冻结校准制品引用，不作为历史事实倒灌到早期日期。

没有 `study` 的旧版本序列化不得新增 null/default 字段改变 hash。实时引用的参考是评价目标，不加入其图节点依赖；避免未来标签污染特征。

### 历史参考

复用正式 retrospective run + 已确认 publication。参考目录是上述成果的只读投影，不复制巨量状态数组或新建一个第二定义库。验证时检查：immutable、快照 hash、发布记录 hash、非人工重叠事件、状态字典、范围和频率。首次确认可复用 enable_research_version 的研究发布路径。新“作为参考”用途必须有明确服务端门禁，不能把任意试算当发布结果。

状态生效区间使用现有观察索引契约。参考定位与交易执行定位分开：比较当前状态用 observation_date；真正交易信号仍按 recognized_at/effective_date 处理。

### 验证与校准产物

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

## 原始证据与最终状态关联

| 来源 | 实际含义 | UI/计算处理 |
| --- | --- | --- |
| 规则命中后的 one-hot | 确定性编码 | 显示“规则判定，未校准”，不能当100%可信 |
| HMM/GMM/Markov posterior | 模型内部后验 | 显示“模型概率，未校准” |
| Ensemble vote | 加权投票比例 | 显示“模型一致票占比” |
| 变化强度/阈值 | 信号强弱 | 显示得分，不叫概率 |
| Frozen calibrator output | 对指定参考的经验映射 | 显示“校准后的参考匹配概率”，带状态与样本依据 |

证据必须跟随最终状态输出的真实祖先。经过 confirmation / component_map / confidence_gate 后，不能把上游 HMM 的牛概率无解释地绑定到已经被改成熊或拒识的输出。保留 raw_state 与 final_state 区别；计算最终状态可信度取 `q[final_state]`，不是一律 max(q)。未知状态不生成有效置信度。

旧 `series.confidence` 数值不被静默重算；新协议添加明确 provenance/nullable calibrated 字段，读旧数据时只作语义标记，不改保存数据。下游新策略不能继续把原始字段当校准结果。

## 时间与样本契约（最高风险）

### 三种日期

- observation_date：历史上该标签描述哪一天。
- label_known_at：该标签计算实际使用的信息最晚可得时间。
- published_at：成果真实发布的时间；不能倒填。

训练/校准要求标签在训练截止日已成熟。很多全样本峰谷、PS筛选、平滑HMM的全部标签都依赖全输入；不能简单用“峰值 + 右窗口”假装成熟。直接继承现有 temporal audit / recognized_at / full_input 信息。

### 两种验证必须明确区分

**回顾性参考评分**：用今天已成熟参考评价历史因果输出。可以报告参考一致性，但不能宣称校准器在历史时点已可使用。

**可部署时序检验**：每个训练边界只用当时可得数据和已成熟标签拟合；校准使用训练样本之外的历史预测；再在之后未用样本测试。若只有当前 full-input 参考、无历史成熟版本，则不伪造可部署校准检验。可以发布今天训练的实验性校准器并保守显示验证不足，但不能授权过去 TAA 回测。

实时识别模型有拟合步骤时走 expanding walk-forward：训练窗拟合标准化/模型/状态映射后，只输出测试窗。规则固定时可直接因果执行，但参数搜索、参考规则选择仍必须披露是否在同一历史期间看过结果。

建议显式 train / calibration / validation / final-test 时间块。实现可先提供前向分段的两阶段评分，但任何校准方法、门槛选择不能利用最终测试块。去掉未知标签和 warmup 后核对每类/每阶段是否足够；不足就说明，不偷偷随机重分。

### 对齐

默认相同观察频率与同一研究对象，日期严格交集。需要月频参考对应日频预测时，必须显式区间映射，不能概率线插值；该功能未实现时明确阻断跨频率并建议同频模型，不自动 forward-fill。

参考边界模糊区只作补充分析；全样本严格指标、剔除边界比例必须同时保留。首次参考区间左截断、末段右删失，不算完整转折；未知不能被算命中。

## 统计定义

混淆矩阵按真实参考状态作行、预测状态作列，额外拒识列。拒识是未接受预测，不能先删除再报全样本准确率。每类 recall 分母含被拒识样本，precision 只针对该预测类。Balanced Accuracy 为实际出现参考类的 recall 平均，缺少必要类的发布门禁单独失败；Macro F1 明确类别集合。

IoU_k = 同属状态k的样本 / 至少一方为k的样本。区间匹配使用同状态一对一最大交叠；每个参考段最多匹配一段预测，孤立误报仍统计。转折匹配必须相同有序状态对，不能把转牛匹配转熊。延迟按观察间隔而非不明确的自然日；日/月频在 UI 明示。

多类 Brier = mean(sum_k((q_k - 1[y=k])²))，不是准确率；LogLoss = -mean(log(q_y))。概率 clip 仅用于数值对数保护，要公开 epsilon。报告未校准、校准和简单训练类别基准；基准只能用过去样本估计。

熵和margin只描述模型分离度，不进入人为“综合可信概率”。只投票/后验高不证明与参考一致。

## 校准、拒识和不确定性

### 概率模型

先采用单参数 temperature/power scaling：q_k ∝ p_k^(1/T)，固定正值有界搜索由 NJIT 完成。选择 T 仅使用 calibration 样本，最终测试不参与。对 one-hot 不执行 temperature（加 epsilon 后软化不等于找到了额外信息）。映射后的类别轴与 final state 一致，概率和严格校验。

### 规则模型

缺少连续得分时采用明确的保守基线：从校准段估计 `P(reference_class | predicted_class)`，可带事先声明的简单平滑计数。显示“同类历史平均”，不是当天独立证据；无该预测类样本则不可用。不用 one-hot 的1作为 sigmoid 输入制造精确概率。

未来可复用连续规则margin+sigmoid/isotonic；本次不要为一个原型引入无法验证的复杂元模型。标签-only基线和temperature都是同一校准服务的有类型方法，不复制一套图模型。

### 门禁

可配置最少总样本、每类样本、参考完整区间数、最大校准年龄、拒识门槛。默认值是平台安全政策而非行业真理，文档说明并锁定版本。样本数不能只看日数：十年日线不等于2500个独立周期。

校准拟合成功 != 验证通过。要求独立后续样本且有明确基准比较；少样本、没有某一状态、参考后来才产生的标签进入早期训练等均不给正式可用标记。UI 不把泛化证据不足合并成普通错误。

### 区间与拒识

报当前匹配概率时同时显示其来源样本和报告状态；不默认给“95%置信区间”。依赖时序下区间使用明确块长度/固定种子的块 bootstrap，检查足够区间和边界；若未实现或样本太少则显示“不足以估计”。块自助仅是平稳/弱依赖近似，不保证金融变动下有效。

拒识以 final_state=unclassified/decision_status=abstained 明确表达，raw_state/probabilities可供诊断；保存 accepted coverage 与 accepted error，不能只展示过滤后的高准确率。

## 历史参考、稳定性与块自助接口

以下保留 API 的字段与统计定义。Horizon Profile 同时统计观察数和日历天，完整独立 Episode 排除首尾删失及 unknown 断点；样本日数不等于独立周期数。
## Historical quality

- `POST /api/historical-regimes/reference-quality/preview`
- `POST /api/historical-regimes/reference-quality/confirm`
- `GET /api/historical-regimes/reference-quality/reports/{report_id}`
- `GET /api/historical-regimes/reference-quality/catalog` → `{items: [...]}`

Preview request:
```json
{"definition_id":"saved-id","revision":1,"mode":"retrospective","as_of":null,"compile_token":null,"policy":{"stability":{"enabled":true,"max_variants":6,"perturbation":0.1,"parameters":true,"windows":true,"seeds":true,"truncation":true},"include_price_returns":true}}
```
`mode` only accepts retrospective. `policy` is required; its fields have the above defaults.
`as_of` is an optional ISO date. No client graph, labels, report, truth flag, or provenance assertion is accepted.
Confirm request is `{request: <preview.request>, preview_hash: <preview.preview_hash>}`.
Preview returns `{preview_hash, request, report}`; confirm additionally returns
`{id, created_at, immutable:true, content_hash}`. Report id prefix is `reference-quality-`.
Catalog item: `{id,created_at,definition_id,revision,status}`. Preview does not publish,
save reference arrays, or persist business artifacts. Confirmation verifies saved revision,
actual source bytes and report hash again. Existing research publication remains separate.

Report:
- `schema_version: "1.0"`, `kind: "historical_reference_quality"`, `status: "diagnostic_only" | "insufficient_evidence"`.
- `sample: {input,classified,unknown,coverage,head_unknown,tail_unknown,first_date,last_date}`.
- `segments: {total,transitions,per_state:[{state_id,observations,segments,min_length,median_length,max_length,mean_length,price_return_samples,mean_price_return,price_return_reason}]}`.
- `price_returns: {status:"available"|"unavailable"|"disabled",reason,semantics}`.
- `stability`: shared diagnostic schema below.
- `lineage`: exact definition hash, execution snapshots and evaluation snapshot.
- `warnings: string[]`, `execution`: warmed NJIT audit.

Lengths/distances count observations, not calendar days. Returns are decimal simple
endpoint returns within each contiguous state segment, only on a verified price/NAV
source, at least two observations, and positive finite prices at every observation; no interior missing price is bridged. Unknown states split segments. No price
semantics means returns are null with reason. Unknown head/tail count consecutive
unclassified observations; an all-unknown series has both equal to input length.

## Shared stability

Policy `stability` is optional on reliability policy and defaults as in the example.
`max_variants` 1..12; `perturbation` >0..0.5. Flags choose parameter/window/seed/truncation
families. Deterministic declared variants only; seeds fixed server-side. Total budget
120 seconds, 20000 observations/input, 64 required nodes, 4 fitted models, 100 iterations/model.
Each variant obtains a matching preparation plan; realtime fitted variants reuse frozen
baseline expanding-fold cuts. State IDs are never permuted to improve agreement.

Result: `{status:"completed"|"partial"|"disabled"|"not_applicable",variants:[],seed_status,limits,reason}`.
Each variant: `{kind:"parameter"|"window"|"seed"|"truncation",status:"completed"|"failed"|"budget_exceeded",changes:[],graph_hash,reason,agreement,comparable_observations,classification_coverage,boundary_distance,boundary_distance_unit:"observation_steps",...}`.
Changes are `{node_id,parameter,before,after}`. Failed/unavailable numerical results
are null. Agreement denominator is paired classified observations on the same date axis;
classification coverage counts classified candidate rows over all baseline dates retained for that comparison, including unpaired/missing rows. Boundary distance
is symmetric nearest boundary distance for the same ordered state pair; null if no
matching boundaries. This is sensitivity, not accuracy. Rule-only seed_status is
`not_applicable`; no fabricated seed accuracy. Truncation removes the last 10% of baseline
observation dates and compares only the retained prefix (recorded in variant metadata).

Reliability nests this under `report.stability.parameter_sensitivity`; existing
`report.stability.temporal_audit` remains.

## Paired moving-block bootstrap

Optional reliability `policy.bootstrap` defaults:
```json
{"enabled":true,"replicates":200,"block_length":10,"confidence_level":0.95,"minimum_blocks":5,"minimum_cycles":3,"minimum_valid_replicates":100}
```
Bounds: replicates 20..500; block_length 2..250; confidence_level 0.8..0.99;
minimum_blocks 2..100; minimum_cycles 1..100; minimum_valid_replicates 10..500 and <= replicates.
Fixed server seed 1729. Uses only holdout, or final test when validation_end exists.
No calibration/validation resampling, no fitting in bootstrap; missing observations
retain their positions inside contiguous blocks. All metrics share sampled indices.

`report.confidence_interval`:
`{status:"available"|"partial"|"unavailable"|"disabled",reason,method:"paired_moving_block",scope:"holdout"|"test",conditional_on:"fixed_model_reference_and_calibrator",confidence_level,block_length,replicates,seed:1729,samples,full_blocks,complete_cycles,metrics:{...}}`.
Metrics: `accuracy`, `accepted_coverage`, `accepted_error`, `brier`, `paired_brier_improvement`.
Each is `{estimate,lower,upper,valid_replicates,reason,unit:"fraction"|"brier_score"}`.
Accuracy counts abstention as mismatch against known reference. Accepted coverage/error
use calibrated confidence_floor. Brier uses calibrated probabilities. Paired improvement
is class-baseline Brier minus calibrated Brier on exactly the same valid rows. Insufficient
full blocks/cycles/valid replicates returns null bounds with reason. Finite point estimates
remain visible; unavailable estimates are never coerced to zero.
Intervals describe historical metrics, never today's hidden state or economic truth.

Each reliability point adds `probability_evidence`: null for rules/unverified sources,
or `{selected_probability,top_probability,second_probability,margin,entropy,entropy_unit:"nats"}`
from the verified raw model posterior on the mapped reference axis. This evidence is
independent of fitted calibration and does not imply deployment eligibility.

## Templates and deployment

New editable template IDs: `historical_hmm_risk_v1`, `historical_gmm_volatility_v1`,
`historical_window_mean_change_v1`, `historical_trend_ensemble_v1`, version 1 (existing integer version contract).
Existing template APIs instantiate their full editable graphs, with default
`study.purpose=historical_reference` and `mode=retrospective`.
No BOCPD/PELT claim. Existing PS/CSI300/macro identities remain unchanged.

Historical quality alone does not establish deployment qualification; actual forward qualification is defined below. These historical
quality and reliability APIs do not create eligibility;
client booleans/cutoffs cannot qualify reports. Old immutable reports remain readable.

## 执行与分母的补充约束

- Template version is integer `1`, following the existing catalog contract.
- Integer diagnostic perturbations move by at least one observation when rounding
  would otherwise erase the change; existing experiment defaults keep their prior rounding.
- Variant agreement/boundary outputs also carry `agreement_reason` and
  `boundary_distance_reason`. Coverage is the fraction of classified candidate rows;
  unpaired/missing rows remain in that denominator. No label permutation is performed.
- Seed perturbations apply only to fitted nodes using `initialization_strategy=random`.
  Quantile/explicit initialization is deterministic and reports `not_applicable`.
- At most 8 expanding folds and 8 evaluation targets. Model dimension/iteration work is
  admitted under a 100,000,000-unit conservative budget, including baseline and variants.
  120 seconds is a cooperative deadline checked between executions; running NJIT calls
  are bounded by input/iteration dimensions and are not forcibly interrupted mid-kernel.
- Bootstrap metric objects additionally expose `samples` (their actual denominator)
  and `full_blocks`. Full blocks are nonoverlapping blocks on the original time axis
  with no missing reference; Brier/improvement additionally require valid probability
  pairs throughout each block. `complete_cycles` counts sequences of fully bounded,
  contiguous state segments visiting every state; missing labels reset a cycle. Segment
  cycles and full blocks are eligibility checks, not a claim of independence.
- Quality segment returns require **every** observation in a segment to have positive
  finite price, and at least two observations. No interior missing price is bridged.
- Preparation uses the existing graph machinery with transient manifests disabled;
  newly created temporary graph plans are released after the diagnostic execution.

## 语义边界

注册把一个已保存且 fitted 的可靠性报告冻结成候选，不修改旧报告的 `deployment_eligible=False`。注册后的**实际捕获**才是前向预测证据。历史 expanding replay、已保存报告的历史 points、后来重新运行的历史预测都不能进入该日志。

当前状态资格契约：部分状态获得资格时，chosen 未获资格仍整体回到 SAA；chosen 已获资格时，显示和置信度保留完整校准分布，两个 TAA 入口只累计 qualified_states 的 tilt，其他状态对应零偏离，概率质量留在 SAA。新 TAA 数值内核保留归一化概率契约，通过屏蔽未授权 tilt 行实现；审计同时提供完整 `probabilities` 与 `allocation_probabilities`。

发布与消费边界：study和非study均必须先通过正式发布门禁，未发布直接拒绝，不以全SAA回退替代发布。已发布后的资格验证失败仍回退并记录原因。TAA偏离配置采用精确绑定的参考状态轴，原始模型状态留给校准映射；realtime_recognition study模板仅声明realtime模式。历史验证见[情景验收纪要](../verification/regimes.md)。

日精度采用保守 UTC 时钟：捕获截止为服务端昨天；观察日期必须严格晚于注册当天。请求不接收时间、起始日期、报告、概率或 eligible 标志。参考的日期精度不证明日内可交易时点。

注册固定一个观察窗口（默认252个参考轴位置），不按表现挑日期。窗口未完整、成熟标签或实际捕获未出现时返回 pending；完整窗口完成首次终局评估后关闭协议。不能不断增大窗口或换门槛直到通过；同一 calibration_id 重复注册相同策略幂等，更换策略冲突。该控制不禁止用户重新选择其他模型，因此只消除本协议后续窗口的回顾性选择，不证明全部模型选择无偏。

## Python 类型和方法

导入位置：`historical_regimes.reliability.prospective`（项目根导入时使用相同包前缀规则）。

```python
class ProspectiveService:
    def __init__(self, graph, reliability,
                 *, clock: Callable[[], datetime] | None = None): ...
    def warm(self) -> dict: ...
    def register(self, payload: RegisterRequest | dict) -> dict: ...
    def capture(self, protocol_id: str) -> dict: ...
    def assess(self, protocol_id: str, payload: AssessRequest | dict) -> dict: ...
    def get_protocol(self, protocol_id: str) -> dict: ...
    def catalog(self) -> dict: ...
    def get_qualification(self, qualification_id: str) -> dict: ...
    def verify_qualification(
        self, qualification_id: str, calibration_id: str,
        model_binding_hash: str, reference_definition: dict | None = None,
        *, decision_at: str | datetime | None = None,
    ) -> dict: ...
```

- `graph` 为原 `RegimeGraphV2Service`，`reliability` 为其原 `ReliabilityService`。不创建第二个模型执行器。
- clock 仅供服务构造及隔离测试注入，必须返回带时区 datetime；不可绑定 HTTP 参数。
- `RegisterRequest = {calibration_id: str, policy: ForwardPolicy}`；policy 缺省采用默认值；未知字段拒绝。
- `AssessRequest = {reference: Reference}`；Reference 沿用 `{run_id, publication_id, content_hash}` 精确三元组。
- capture 无业务输入，HTTP 允许空请求体或 `{}`，额外字段拒绝。
- `decision_at` 供父消费者逐决策时点验证，不是写入时间。ISO 日期按 UTC 00:00；完整时间须带时区；省略则用服务器当前时间。

### 冻结策略

| 字段 | 默认值 | 范围/含义 |
| --- | --- | --- |
| observation_window | 252 | 60–2000，首个固定参考日轴窗口 |
| minimum_observations | 120 | 60–2000，有效预测/标签配对 |
| minimum_class_observations | 10 | 5–1000，参考与预测两轴逐类都满足 |
| minimum_class_complete_regimes | 2 | 1–100，每个参考类的完整区间数 |
| block_size | 20 | 5–252，固定不重叠日轴块 |
| minimum_complete_blocks | 5 | 3–100，完整有效块 |
| minimum_coverage | .95 | .8–1，有效配对/完整窗口轴位置 |
| minimum_agreement | .6 | 0–1，全局参考一致率的诊断参考线；不再单独否决已通过逐状态门槛的状态 |
| minimum_state_precision | .65 | 0–1，证据充分状态取得资格所需的高置信 Precision |
| minimum_brier_improvement | .01 | (0,2]，每个完整块均达到的 base−model Brier 改善 |
| max_capture_lag_days | 3 | 1–7，捕获时间距观察日期的日历天数 |
| max_observation_gap_days | 7 | 1–62，大于该间隔会破坏完整块/区间 |
| expires_after_days | 30 | 1–730，资格有效期限，同时受最近评分日约束 |
| maximum_protocol_days | 730 | 90–3650，注册后最长有效天数 |
| criterion | all_complete_blocks_improve_no_significance_claim | 固定枚举；不声称统计显著性 |

窗口至少容纳最少样本和最少块。窗口、样本与门槛只能在注册前选择。没有标签或没有前向记录时不会制造零置信度。

上述默认值为日频。省略 policy 时按模型频率选择：周频窗口/最少样本 120/120、block_size=10、捕获滞后7天、观测间隔14天、资格90天、协议1460天；月频为60/60、block_size=5、捕获滞后7天、观测间隔45天、资格365天、协议3650天。其他统计门槛不变。显式 policy 原样验证并冻结，间隔必须容纳对应频率，不能靠放宽期限替代实际捕获；月频60个有效观察需要至少约5年。周/月轴只取服务端截止日前已闭合的观察桶。

## 冻结内容与来源更新

注册冻结：原制品完整 hash、模型定义 id/revision/hash/binding hash、执行器及相关 NJIT 库的已加载代码指纹、原始精确参考三元组、参考定义 id/revision/hash/family/states/target/frequency、映射、完整校准方法/参数/class_base、策略、服务端时间、初始源前缀字节 hash、源快照和模型训练审计。执行器/内核代码指纹改变会使旧协议失效；它规范化字节码、常量、参数等 code 字段并排除源路径/行号，不使用引用标记不稳定的 marshal 输出。Python 版本变更仍可能改变指纹，跨部署迁移应重新验证。

注册时当前源 fingerprint 必须与原候选执行 fingerprint 相同；候选保存之后输入已变则要求重新保存候选。注册后允许**同一绑定源新增数据**，每次捕获冻结本次精确 snapshot/fingerprint，并核对上次捕获（首笔为注册）所见的整个输入前缀：日期、可得时间、值、dtype、shape 和字节都保持。修订/删除旧数据、重排或重新拟合起点变化会失败关闭。前缀 hash 是内存视图的字节序列化；必要 hash bytes 分配不是数值计算副本。

参考可以有新的run/data coverage；默认要求原定义修订、hash、族、状态、频率及target完全一致。同对象同频的单一日/周/月频指数可通过签名source_version追加记录批准仅数据绑定变化的参考修订；评分只允许原身份或这些明确批准的完整身份。算法、参数、状态和映射的变化不能作为数据续接。snapshot hash、发布三元组和数组checksum继续由原resolver/hydrate校验。

`create_definition`仍自动冻结指数snapshot/checksum。注册时的forward_source_mode保持原始不可变语义；后续是否已续接由progress.current_source_version说明，不改写注册记录。新增的sources/preview与sources/confirm仅支持模型和参考均为同对象同频的单一日/周/月频指数：从当前已验证源取绑定、验证完整历史前缀、显式确认后写签名续接记录并通过原版本服务创建参考的仅数据修订。原模型/文件不被覆盖，未确认不接入，确认不发布参考。上传、宏观、多源、跨频续接仍不支持。已有growing_legacy_index继续按既有严格前缀契约运行。

进度投影增加current_source_version和reference_definitions。前者包含批准的model_bindings、model_binding_hash及reference_definition；后者列出原始及批准的参考身份。界面打开精确后续参考修订，由用户执行并确认发布。所有捕获/确认仍使用服务端时钟，客户端不传来源路径或数据。

### 推断配方

- 固定规则直接沿原图因果推断。
- 既有 HMM/GMM/Markov 使用原图声明的 `initial_train_size` 初始训练规则，再因果过滤/分类；不是扩展到新标签的自适应拟合。
- 原可靠性报告的 expanding replay 是历史评估配方；前向捕获是同一代码/参数绑定下声明的初始窗口推断。两者分别记录，不能把 expanding replay 的历史分数称作该前向模型已验证。
- 初始历史前缀锁定防止初始训练集、标准化及标签排序随源修订改变；不增加自己的模型数值实现。结局不能用于调整模型、映射或温度。
- temperature 只用原可验证后验轴，class_frequency 使用冻结 counts。所选状态置信度始终为 `q[chosen]`，不是 `max(q)`。

支持同频的日/周/月轴、最多32个实际依赖节点、最多2个 index/upload 源、最多1个原标准拟合模型（100迭代以内）；模型节点允许 threshold、range_threshold、drawdown_cycle_realtime 和原 HMM/GMM/Markov。仅支持能由既有 target_identity 确定唯一对象的轴；evaluation_targets 只允许与单一源同对象、同数据绑定和同频，其他评价轴、不受支持源/模型/频率明确拒绝。每个源和输出最多20000条，同一定义最多检查128个参考运行；捕获总检查45秒、时点探针15秒。超时与行数在调用边界检查，不是可抢占的进程硬中断或源解码前的内存上限。

## 日志与评估

只有显式 capture 写预测。成功记录包含 raw point/hash、chosen、原始及校准概率、置信度、模型/校准绑定、源前缀、源快照和时点/拟合审计。仅捕获当前截止的最新可得点；旧点重复是 duplicate，变化冲突，缺失/过期/未知返回明确状态，不覆盖或补录。已经发布同一定义参考标签的日期不能新捕获；后来的重新发布不能洗掉先看标签的问题。

评估只 join 日志中注册后、评估前捕获的日期。标签观察日、recognized_at 和 data_available_at 需合法且成熟；日精度标签须在服务器今天之前可得。发布需真实发生且晚于捕获，不能借 observation_date 倒填知识时间。

窗口按参考完整日轴保留缺失捕获、未知标签和非法概率，不先删除后拼接。每笔捕获还冻结上次捕获后见过的日期轴元数据；新参考缺少这些日期时仍保留为未知标签。完整区间要求两侧都有有效状态转换，中间没有无效配对或超长日历间隔；逐类要求至少若干这样的区间。块在固定原轴上分割，缺失使该整块无效，不移除缺失再拼成假连续块。coverage 门槛不证明缺失随机，也不消除人为选择何时捕获的偏差。

输出 `status` 为 pending/rejected/qualified，并增加 `outcome=pending|qualified|partially_qualified|insufficient_evidence|failed`、`state_evidence[]`、`qualified_states[]`、`fallback_states[]`。每个状态分别记录配对样本、高置信预测、完整独立 Regime、Precision/Recall 与原因。

状态门槛与全局门槛分开：总样本、捕获覆盖、完整时间块、每块 Brier 改善和有效期仍是协议级门禁；状态自身则按 `minimum_class_observations`、`minimum_class_complete_regimes` 和 `minimum_state_precision` 独立取得资格。某个 Rare Regime 只有一个完整区间时应是 `insufficient_evidence`，不会把其他证据充分状态一起判失败。全局 accuracy / minimum_agreement 保留为诊断，不再用来覆盖逐状态结论。**无 bootstrap、无置信区间、无统计显著性声明。**

当至少一个状态通过且协议级门禁通过，assessment 的 `status=qualified`；如果并非全部状态通过，则 `outcome=partially_qualified`。消费端必须同时核验 `qualified_states`：当前识别状态不在其中时返回 `calibration_state_not_qualified`，TAA 应回退既有 SAA 基线路径，不能把未验证状态概率重新分给已验证状态。

所有 assessment 包括 pending 都是不可变记录；qualification_id 就是 assessment 的 id。终局窗口不能换参考 vintage 重考。可审计读取会重算签名捕获、冻结概率、`outcome/qualified_states/fallback_states/state_evidence` 与评估门槛，不信任单独保存的 qualified 字段。

## 持久化与安全边界

复用 AtomicJsonStore：`<reliability.root>/prospective/journal.json`，单个原子受锁追加日志；内容 hash 链加服务端 HMAC 签名防止编辑 eligible 或重算普通 hash 伪造记录。签名密钥单独保存为权限0600的 `signing-key.json`；读 API 不返回密钥。全日志20000条、单协议最多2000笔成功捕获，超限明确失败。

适用于受控本地工作区存储，不是外部可信时间戳/防管理员篡改装置。拥有文件及密钥写权限的管理员、整份工作区回滚、恶意提供者或伪造真实市场数据不在此信任边界内；父部署须保护目录和系统时钟。已有 AtomicJsonStore 提供跨进程文件锁与原子替换，不复制一套持久化服务。

## 用户操作

在实时识别的已保存报告/前瞻验证中，提供“检查新数据版本”。服务读取系统当前已验证快照，展示原/新快照、原/新观察数和历史前缀校验结果。只有显式“确认续接数据”才写入前瞻追加日志，并通过原有定义版本服务创建同一历史参考的仅数据版本修订。后续参考仍需用户进入历史状态定义、执行并确认发布；续接不发布标签、不补录预测、不自动采集。

模型参数、状态字典、映射和校准器全部不变。原模型和原报告不改写；执行时从签名的已确认续接记录取得精确源绑定。旧引用继续读取旧快照。采用通过检验的资格时，将已确认源绑定及资格一起形成模型新修订，避免验证新数据却运行旧快照。

## 边界和拒绝条件

续接支持单一指数来源、模型与参考同一对象/同频（日、周或月），复用现有 source.index、active snapshot resolver、PIT及prepare/executor。上传、宏观/多源和跨频仍明确不可续接，不能删除checksum来规避。

同源 evaluation_targets 与 source.index 同步重绑精确快照；recipe_hash 仅排除这些数据绑定字段，指数对象、字段、频率、模型参数仍固定。参考身份的 frequency 从实际输出推导。跨对象、跨频和算法变化继续拒绝。

新版本必须有新增观察，且注册/上一实际捕获或续接记录覆盖的原始日期、可得时间和数值前缀完全一致。缺行、插入旧日期、修订数值/可得时间、不同指数/字段/参数/状态/源类型、截短、同名换文件都拒绝。整个快照文件仍按既有校验和验证；不能只校验图输出。

预览为进程内有界TTL、不可变请求摘要；确认重做来源与前缀校验，并核对原来的最后续接/捕获位置。预览后活跃快照或历史定义改变则要求重新预览，不自动使用新值。并发确认复用签名journal锁；重复确认幂等。引用定义新修订与journal不是跨文件事务，重试仅可复用字节相同的已创建修订，不覆盖用户编辑。

## 模型和参考血缘

每个 source_version 事件冻结实际 model source bindings、model_binding_hash、原始前缀、数据指纹及新参考的 definition_id/revision/hash。来源解析与数值执行仍委托唯一图服务。

前瞻评分只允许注册时的参考定义或经显式续接批准的参考数据修订。任意修改算法后发布的同名参考不可用于检验。捕获前检查这些参考版本是否已发布该日标签；已知标签不能补录预测。已保存资格重放只使用资格生成之前的续接事件，不让后来的版本改变历史评分。

TAA对新源修订的放行仍必须通过真实前瞻资格核验：报告、模型数学、当前批准数据绑定、状态映射、有效时点均匹配。没有资格不放宽旧hash门禁；采用前瞻资格不改写历史报告 deployment_eligible。

## API

- POST /api/historical-regimes/prospective/{id}/sources/preview：空请求，无客户端日期、路径、数据或可信标志。返回版本差异及 preview_hash。
- POST /api/historical-regimes/prospective/{id}/sources/confirm：仅 preview_hash；重验后返回不可变source_version事件。不会自动发布参考。
- GET .../protocols/{id}/progress：增加current_source_version及允许使用的reference_definitions投影。

## 验收

实际临时Parquet/图执行：原快照保留、跨版本新增观察能捕获、继续发布参考并评分、原数据/日期/PIT修订拒绝、错误对象/参数拒绝、确认幂等/过期/乱序、旧资格不倒灌、TAA当前源hash匹配。前端空/错误/确认/重载/请求乱序、三个宽度、原有质量/统计/事件库回归。测试只写隔离目录。

## 确认与历史兼容

参考质量、可靠性及校准预览不写正式制品；确认携带规范请求与 preview_hash，后端重算来源／修订／hash。参考三元组、同对象同频、明确状态映射及结果真实祖先缺一不可。旧无 study 定义的序列化不增加默认字段改变 hash。

Realtime 概率不再作为 LTCMA 证据。历史参考供条件估计，实时报告为 recognition_evidence；旧报告可以阅读，不能因此取得新应用资格。

最新状态验收、前瞻日志和源码指纹修复见[情景纪要](../verification/regimes.md)。HMAC 只保护受控本地工作区，不能证明外部可信时间戳或防管理员改写。
