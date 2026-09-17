# 独立前向验证域 API

日期：2026-09-15。分支：`ISSUE2609/BetterSaaTaa`。服务、路由、每进程预热、Study、共享TAA消费者及前端已接线；单指数数据续接已完成，最新验收见 [收口验收](regime-source-continuation-acceptance-2026-09-15.md)。

## 语义边界

注册把一个已保存且 fitted 的可靠性报告冻结成候选，不修改旧报告的 `deployment_eligible=False`。注册后的**实际捕获**才是前向预测证据。历史 expanding replay、已保存报告的历史 points、后来重新运行的历史预测都不能进入该日志。

2026-09-16 修复补充：部分状态获得资格时，chosen 未获资格仍整体回到 SAA；chosen 已获资格时，显示和置信度保留完整校准分布，两个 TAA 入口只累计 qualified_states 的 tilt，其他状态对应零偏离，概率质量留在 SAA。新 TAA 数值内核保留归一化概率契约，通过屏蔽未授权 tilt 行实现；审计同时提供完整 `probabilities` 与 `allocation_probabilities`。

第二轮修复：study和非study均必须先通过正式发布门禁，未发布直接拒绝，不以全SAA回退替代发布。已发布后的资格验证失败仍回退并记录原因。TAA偏离配置采用精确绑定的参考状态轴，原始模型状态留给校准映射；realtime_recognition study模板仅声明realtime模式。验证见[第二轮审核报告](scenario-audit-followup-2026-09-16.md)。

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

当至少一个状态通过且协议级门禁通过，assessment 的 `status=qualified`；如果并非全部状态通过，则 `outcome=partially_qualified`。消费端必须同时核验 `qualified_states`：当前识别状态不在其中时返回 `calibration_state_not_qualified`，CMA/TAA 应回退既有 Base/非 Regime 路径，不能把未验证状态概率重新分给已验证状态。

所有 assessment 包括 pending 都是不可变记录；qualification_id 就是 assessment 的 id。终局窗口不能换参考 vintage 重考。可审计读取会重算签名捕获、冻结概率、`outcome/qualified_states/fallback_states/state_evidence` 与评估门槛，不信任单独保存的 qualified 字段。

## 持久化与安全边界

复用 AtomicJsonStore：`<reliability.root>/prospective/journal.json`，单个原子受锁追加日志；内容 hash 链加服务端 HMAC 签名防止编辑 eligible 或重算普通 hash 伪造记录。签名密钥单独保存为权限0600的 `signing-key.json`；读 API 不返回密钥。全日志20000条、单协议最多2000笔成功捕获，超限明确失败。

适用于受控本地工作区存储，不是外部可信时间戳/防管理员篡改装置。拥有文件及密钥写权限的管理员、整份工作区回滚、恶意提供者或伪造真实市场数据不在此信任边界内；父部署须保护目录和系统时钟。已有 AtomicJsonStore 提供跨进程文件锁与原子替换，不复制一套持久化服务。

## 路由安装与父集成

```python
from historical_regimes.reliability.prospective import ProspectiveService, install

# 构造；不得在 capture 中惰性创建/预热。
graph.prospective = ProspectiveService(graph, graph.reliability)

# 每个 worker 的 lifespan，原图与可靠性内核完成 warm 后：
graph.prospective.warm()  # 失败即 readiness 不通过

# 沿用已有历史情景 router/getter/error adapter：
install(router, lambda: get_graph_service().prospective, call)

# 在现有 prepare 显式流程为该已保存模型准备图计划；重启后亦须恢复准备。
graph.prepare(graph.get_definition(model_id, revision))

# TAA 新协议分支：按每个决策时间调用，资格仅用于有效时间之后。
qualified = graph.prospective.verify_qualification(
    study.qualification_id, study.calibration_id,
    model_binding_hash(definition),
    reference_definition=expected_reference_identity,
    decision_at=decision_date,
)
```

端点：POST `/api/historical-regimes/prospective/register`、POST `/{protocol_id}/capture`、POST `/{protocol_id}/assess`；GET `/catalog`、`/protocols/{protocol_id}`、`/qualifications/{qualification_id}`（后五个均共用上述 prospective 前缀）。异常由父 `call` 保留现有领域错误 HTTP 映射。

已集成的共同契约：

1. Study已增加可选qualification_id，缺省序列化保持旧hash；model_binding_hash仅排除calibration_id与qualification_id，不排除参考、参数、数据绑定或映射。采用已验证且续接过的候选时，前端同时写入批准的四项源绑定；后端要求实际模型hash等于协议当前批准的hash。
2. 两个 TAA 消费者共用上述 verifier，单独检查原有正式发布/时点/模型契约；旧无 study 路径照旧。验证通过的新分支使用 qualification 的独立资格及有效时间，并在每个决策点检查识别状态是否属于 `qualified_states`；未取得状态资格就回退，不因同一模型其他状态通过而放行。概率仍使用已绑定冻结参数，不把旧 calibration.deployment_eligible 改 True。
3. verifier 验证指定校准/模型/参考定义、签名日志及当前模型/原始参考 lineage，重算评分，并核查当前图输入前缀。当前时钟和决策时间都受限：`available_from < decision_at <= server_now < expires_at`。严格晚于实际评估发布时间；日精度采用下一天 `available_from_date`，永不授权过去 TAA。
4. 资格 expiry 取评估时间+策略有效期、最后评分日+有效期及协议 deadline 的最早值。读取仍可看已过期的历史记录，verifier 拒绝使用。
5. 不自动采集；父调度若需要，必须把 capture 作为用户授权的显式任务，不在质量预览/报告加载时写预测。

app/router/consumer/frontend/lifespan均已接入，未改变旧无study回放或非Regime策略。失效旧协议不会得到资格；预热可将其列入blocked而不阻断其他研究功能，真实数值预热或journal完整性失败仍失败关闭。真实注册协议需要真实后续数据和参考成熟，软件测试不表示任何真实模型已取得资格。

get_protocol和catalog返回冻结注册记录；GET /protocols/{protocol_id}/progress返回实际捕获数、最近观察日、最新检验及批准源版本。注册记录不会被资格或续接改写。

新增数据续接端点（同prospective前缀）：

- POST /{protocol_id}/sources/preview：空请求，返回preview_hash、old/new/added_observations、prefix_unchanged、精确model_bindings等；有界进程内TTL，不存正式报告。
- POST /{protocol_id}/sources/confirm：仅preview_hash，重新核验，返回签名source_version，重复确认幂等；不自动发布参考、不捕获预测。

确认后源码/参数未变但数据版本变化，采用资格必须使用批准源hash；不允许通过省略qualification_id绕过报告的原始绑定门禁。具体并发重试和接口边界见regime-source-continuation-design-2026-09-15.md。
