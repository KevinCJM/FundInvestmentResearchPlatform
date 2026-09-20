# 市场状态研究

> [全部文档](../README.md) · 本页为主题入口，专业契约按下表读取。

| 专业契约 | 何时读取 |
| --- | --- |
| [连续实时市场状态](continuous-state.md) | 连续状态与概率序列 |
| [历史事件库与时点能力](events.md) | 事件类型、时间能力及区间消费 |
| [峰谷定界法：事后牛熊震荡区间识别](peak-trough.md) | 峰谷识别、确认及边界 |
| [风险模型、情景模拟与压力测试](risk-models.md) | 发布风险模型、敏感性与压力期限 |
| [市场状态研究流程与平滑算子](smoothing.md) | 状态平滑与迟滞规则 |
| [市场状态：验证、校准与前瞻资格](validation.md) | 参考置信度、实时可靠性与前瞻资格 |

本页是市场状态主题入口。情景算法中心分为市场状态、全球历史事件、模拟压测三个模块；它们的产物与时点语义不同。

- 历史参考 → 实时识别 → 验证识别能力，是同一研究流程的三个步骤。
- [验证与前瞻资格](validation.md)负责统计、校准、捕获和部署门禁。
- [事件库](events.md)负责可重叠历史事件。
- [风险与模拟](risk-models.md)负责传导、模拟和压力测试。
- [方法决定与研究证据](../research/regime-methods.md)保留算法选择及否决理由。
“历史状态定义”和“实时状态识别”不再作为两个平级一级页签暴露给用户。

### 市场状态研究的人类流程

页面顶部固定展示三步：

```text
① 定义历史参考
        ↓
② 建立实时识别
        ↓
③ 验证识别能力
```

#### 第一步：定义历史参考

用户看到的是“先定义什么叫这个市场状态”，而不是技术上的 retrospective mode。

核心操作：

- 选择或创建 Historical Reference 算法；
- 查看完整历史区间；
- 检查状态支持、独立 Episode 和 Horizon Profile；
- 保存为不可变 Historical Reference；
- 直接点击“下一步：建立实时识别”。

#### 第二步：建立实时识别

核心操作：

- 选择精确 Historical Reference 版本；
- 构建只使用当时可得信息的识别模型；
- 运行 realtime replay；
- 保存模型版本；
- 直接进入“验证识别能力”。

Historical Reference 与 Realtime Model 是不同对象，不做隐式复制或覆盖。

#### 第三步：验证识别能力

第三步复用第二步同一份实时模型草稿，不重新初始化算法。

展示：

- 与 Historical Reference 的状态级匹配；
- calibrated probability；
- 每状态独立 Episode；
- `verified / insufficient_evidence / failed`；
- prospective qualification。

### 用户体验原则

- 一级模块少：用户只需理解“市场状态 / 历史事件 / 模拟压测”。
- 三步研究关系一直可见，不需要来回跳两个工作区。
- 桌面、平板、320px 手机使用同一逻辑。
- Step 切换支持鼠标、键盘左右键、Home、End。
- Historical 草稿与 Realtime 草稿分别保留。
- Realtime → Validation 复用同一草稿。
- 跨 Historical / Realtime 时，不携带另一个对象的 `definition/revision/template`，避免错误加载。
- 旧 `center=historical`、`center=realtime` 深链接继续兼容到新流程。

底层 immutable artifact 不融合：

```text
Historical Definition / Run / Publication
Realtime Definition / Run / Publication
Reliability Report
Prospective Qualification
```

仍保持各自版本、hash 和血缘。

---

## LTCMA / TAA 消费边界

正式边界：

```text
Historical Regime ──→ LTCMA Research
Realtime Regime   ──→ TAA / Product PIT / Monitoring
```

Realtime Reliability 新结果使用 `recognition_evidence` 语义，不再把 realtime 报告称为 CMA evidence。

LTCMA 后续应消费 Historical Reference 侧的：

- reference identity / hash；
- independent episode count；
- duration statistics；
- state occupancy；
- conditional-estimation readiness。

而不是 realtime confidence。

---

## Horizon：不能由用户声明，必须有证据

### Historical Market State Horizon

Historical Regime 的 Horizon 由完整独立 Episode 实际测量，不使用“用户填 long-term / short-term”作为事实。

质量报告输出：

- observation frequency；
- complete independent episodes；
- duration observations P25 / Median / P75；
- calendar duration P25 / Median / P75；
- state occupancy；
- annualized transition rate；
- classified coverage。

首尾未完成区间、Unknown 隔断区间不进入完整 Episode Duration 估计。

因此消费者可以基于真实持续期判断适用性，而不能靠命名绕过门禁。

### Scenario / Stress Horizon

未来情景没有已经发生的 Episode，因此不能用 Historical Regime 的方式“测出真实持续时间”。

系统目前明确拆成两件事：

1. `frequency × horizon`：只定义模拟网格长度；
2. `Horizon Evidence`：检查路径在终点是否仍存在冲击。

预览现在展示：

- 频率 × 期数；
- market path 是否在终点回到 baseline；
- baseline tail periods；
- `economic_horizon_status = not_empirically_established`。

如果终点仍有冲击：

> 当前模拟窗口可能截断了传播或恢复过程。

即使终点回基线：

> 也只能说明“模型路径在窗口内结束”，不能证明现实经济冲击必然持续相同时间。

未来若要把 Scenario Horizon 升级为“经济持续期已验证”，还需要额外历史类比、衰减模型、流动性/传导证据；本轮没有伪造这一结论。

---

## 定义、图与状态类型

Historical Definition／Run／Publication 与 Realtime Model／Run／Publication 独立不可变，页面融合不合并对象身份。普通数值、条件、枚举状态、事件及区间有明确类型；unknown 不等于 neutral 或 false。

画布、表单、规范公式共用一个定义。模板生成真实可编辑节点和连线；节点单独预览，多个分支同范围对比共用同次输入。保存前保留草稿，异步旧预览不能覆盖新定义。图布局与数学身份分离。
## 颗粒度审计

| 对象 | 职责与中间结果 | 处理 |
|---|---|---|
| 三状态阈值 | 比较与条件选态可单独使用；score 是原输入，编码与时点是独立输出适配 | 组合模板 |
| 双指标四象限 | 两组阈值比较与条件选态可复用 | 组合模板 |
| PS 整体 | 定位、筛选、分段、统计与震荡合并可单独替换 | 组合模板 |
| PS 联合约束筛选 | 删除一个峰谷会改变相邻阶段、完整周期及首尾约束，必须重复检查至不再删除 | 保留耦合内核；每轮至少删除一个拐点，保证终止 |
| 小波段震荡合并 | 从最早起点寻找最长完整合格波段，幅度、方向结构和路径效率相互约束 | 保留独立耦合内核；返回合并区间及同一次选择的证据 |
| 滞回、连续确认 | 需要维护活动状态、候选计数及保持期 | 保留状态递推内核；不能当无状态过滤串联 |
| HMM / Markov / GMM | 拟合与推断为完整模型求解；特征准备和业务标签映射属于外围 | 保留模型核心，不拆为用户操作的 EM 循环 |
| 趋势特征包、滤波状态方案、结构突变 | 仍有可独立替换的特征、条件和递推边界 | 保留审计标记；仅在对应等价回归完成后开放展开，不能假称已原子化 |

多输出不是重型算子的判据：峰谷筛选返回位置与相应价格属于一个事件检测问题；把无关指标放在一起则不成立。

## 数值与类型契约

1. 新条件端口采用独立名义类型 `condition<int64>`：1 真、0 假、-1 缺失。不能连接到数值或状态端口。逻辑组合严格传播缺失；缺失不是假，更不是震荡。
2. 条件选态只接条件与可选状态分支，缺失条件输出 -1；只使用被选中的分支。默认状态码由参数给定，校验范围 -1 至当前状态数减一。
3. 原三状态阈值保留 `>= upper`、`<= lower`；区间阈值保留严格 `>`、`<`。不统一等号口径。
4. 确定性状态编码输出是 one-hot 隶属编码，有效项指示值 1 不代表预测正确率或校准概率。UI 标签应说明用途，不新增统计解释。
5. PS 阶段方向使用独立 `phase_codes<int64>`（0 上行、1 下行、-1 未完成），不能当成三态市场标签。震荡合并输出才映射至定义中的市场状态。
6. 保留所有原始缺失、非正价格断点、平台最早极值、半开区间 `[start,end)`、完整终点价格与未完成尾部。区间起止必须来自同一分段节点，不允许混接。
7. 阶段数以输入观测间隔计，不替换为自然日。大幅变动例外只放宽最短阶段，不放宽完整周期。
8. 所有新节点在输出可得时点上合并全部输入依赖；PS 的全样本知识范围随下游传播，不能因末端变成普通比较而获得实时资格。

## 拆分结构

### 三状态阈值

输入 → 与上界比较（含等号） → 条件选态（牛 / 下游结果）

输入 → 与下界比较（含等号） → 条件选态（熊 / 震荡）

score 引用原输入；需要旧 probability/confidence/recognition/reason 输出时接独立编码与时点节点。最终图只执行实际依赖，不为每个分类默认生成无用多输出。

### 四象限

增长 ≥ 增长界线、通胀 ≥ 通胀界线 → 两个通胀分支 → 按增长条件选态。保持旧 0、1、2、3 的象限顺序。原 score 仍为增长减通胀。

### PS

原始价格 → 峰谷定位 → PS 联合约束筛选 → 相邻峰谷分段。

同一分段 → 阶段方向 / 区间涨跌幅 / 峰谷连线。

阶段方向 + 原始价格 + 同一分段 → 小波段震荡合并 → 最终状态及合并证据。

提取后的旧 PS 包装函数也调用上述内核，避免两套数学逻辑。现有独立日频模板不冒充 PS 等价模板，因为它本来就没有 PS 周期筛选和多波段震荡规则。

## 前后端实现约束

- 展开接口只接受一个注册且有展开契约的节点。校验节点版本、参数、状态数和展开后节点/边预算；允许尚未连接完成的草稿，但不允许把已存在的非法引用当作成功迁移。
- 所有输出都有明确映射，包含非主输出、显式边、exposed nodes 和命名输出。保留其他节点、图表元数据、数据源绑定及布局；新节点以原节点位置为锚排列。
- 不自动覆盖最终输出或用户的状态名称/颜色，不隐式增加数据源，不持久化任何转换结果。
- 组合目录给出步骤概览；尚不支持展开的复合节点显示真实状态和原因，不伪标为基础算子。
- 新增普通数值节点使用固定 dtype、连续数组和固定签名 NJIT；导入/启动完成预热后禁用新签名编译。纯图转换保持 Python 编排，不进入数值路径。
- 服务实际执行路径与审计中的 kernel_id 同步注册。性能验证记录绝对耗时和额外中间数组成本，不在没有测量时声称加速。

## 来源与日期

指数、ETF、公募、上传和宏观使用显式来源类型。字段、频率、快照／checksum、产品身份与价格口径冻结，不能因为名称相同就重绑。ETF 市价与净值分别读取；后复权、前复权和复权净值不混用，缺因子不填 1，前复权锚点在截止日过滤后确定。

宏观 PMI/CPI 发布日缺失保持未知，不以观测月份猜公告日。美林时钟选择的数据别名与标准输入映射显式校验；请求失败保留原因而不是悄悄显示旧图。来源不足时不能把历史研究模板升级成严谨 realtime 模型。

上传使用受控文件解析，数值与日期明确绑定，不接受任意后端路径。多序列先按日期和频率对齐，不把两个数组的相同位置当成相同时间。

## 保存与消费

普通运行和预览不等于发布；用户明确“保存为研究版本”才形成可供其他页面引用的不可变发布。前端携带精确 run/publication/hash，打开旧成果不重算、不覆盖。

Realtime 的新证据命名为 recognition_evidence；旧 `cma_research_ready` 仅可能作为历史报告兼容字段读取。实时 cma_evidence 入口已明确拒绝 REALTIME_CMA_EVIDENCE_REMOVED，不能恢复旧“实时报告直接授权 LTCMA”的关系。

历史未来依赖沿实际祖先传播；confirmation、滞后或字段提取不能洗掉事后性。实时产品／TAA 消费还需核验发布、实际可得时点与对应资格，绝不凭 raw confidence 放行。

## 专业契约与验收

[峰谷算法](peak-trough.md)、[平滑方法](smoothing.md)、[因果审计](../indicators/causality.md)独立保留。历史技术验收归入[情景纪要](../verification/regimes.md)。
