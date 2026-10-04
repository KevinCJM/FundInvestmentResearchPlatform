# 因果性审计模块设计（未来函数 / 数据泄露检验）

## 0. 一句话

PIT 封版管住了**能看到哪些数据**；本模块管住**公式有没有偷看**。两者正交，缺一不可：
一个把未来值用于当时决策的 `shift(-1)` 公式，不能因数据已PIT封版而获得时点资格。明确事后标签或评估目标可以使用未来数据，但不得再当作历史决策输入。

适用范围：当前注册表中的内置算子、用户在行情指标中心自建的公式及情景研究中心定义。2026-10-04静态读取 `backend/causality/baseline.json` 含120项 operators 记录；这不等于本轮重新执行全部算子审计，注册表别名数与去重算子数也须分别统计。

---

## 1. 判据

只有一条，其它全是它的推论：

> **因果性**：输出在时点 t 的取值，只能是输入 `x[0..t]` 的函数。
> 等价地——**改变 t 之后的输入，不允许改变 t 及之前的输出**。

违反 = 未来函数 / 数据泄露。

注意「窗口右端」这个概念区分了两类完全不同的东西：

| | 说明 | 例子 | 是不是缺陷 |
|---|---|---|---|
| **读取未来点** | 输出 t 依赖 x[t+1..] | `shift(-1)`、`ZIG` | 若声称t时已可得或用于当时决策则违反要求；事后标签另行标明 |
| **消费整窗** | 输出是整段窗口的函数 | `mean_time`、`max_value` | 否——取决于窗口是谁切的 |

`MEAN(过去 36 个月)` 正确，`MEAN(全期)` 错误，**同一个算子**。所以算子级审计
无法对归约类给出通过/失败，只能给出**分类**；真正的判决必须发生在公式级。

---

## 2. 三个探针

### 2.1 尾部扰动 P1（主探针）

```
x' = x.copy(); x'[t+1:] = 另一组完全不同的值
断言:  f(x')[0..t] == f(x)[0..t]
```

这是 §1 判据的**直接翻译**，也是首选探针。相比截断法的关键优势：**序列长度不变**。
截断会同时改变历史长度和预热条件，因此本项目优先用同长度扰动来隔离这一变化。Freqtrade分别提供lookahead与recursive分析，可作为区分问题的参考；没有本项目对比实验时，不宣称误报率更低或推断对方拆分命令的原因。

扰动值需覆盖不同变化并保持被测函数定义域；当前实际采用下述三组正系数扰动。早期负系数示意不用于净值等要求正数的输入。任何有限扰动仍可能产生数据相关的假阴性。

扰动方式用**三种**（倒序×1.9、放大×6、倒序缩小×0.013；掩码则取反 / 全真 / 全假），
不是一种。单一扰动会产生**数据相关的假阴性**：`max_consecutive_true` 只在最长
真值段恰好落在头部时才「看起来」不依赖尾部，换一种扰动就露馅。实测中它正是
先被误判为因果、加上多扰动后才纠正的。所有系数取正以保号——净值类输入变负会让
`log`/`sqrt`/`drawdown_series` 直接抛错，审计结果退化成一片 UNKNOWN。

### 2.1b 尾部依赖（P1 的对偶，用于无时间轴的输出）

输出没有时间轴时 P1 无从比较前缀，但「无时间轴」**不等于**「消费整窗」：
`first(x)` 只用 `x[0]`，改动尾部它纹丝不动。所以这里实际测一次——改了未来，
输出变不变：

* 不变 → `CAUSAL`（本次有限探针未观察到尾部依赖，不证明所有输入都只用前段）
* 变了 → `WINDOW_CONSUMING`（真的吃整窗，需在公式层确认窗口右端）

不做这一步的代价是 `first`、`max_consecutive_true`、`last` 全被塞进同一档，
真正需要确认窗口右端的算子淹没在噪声里。

### 2.2 截断重算 P2（决策日语义）

```
断言:  f(x[0..t]) == f(x)[t]        # 序列输出
断言:  f(x[0..t]) == "t 日应有的值"  # 标量输出
```

P1 管不到标量输出的算子（整窗归约按定义依赖全窗）。P2 补上这一段：它问的不是
「算子因不因果」，而是**「把数据砍到决策日，公式还给不给同一个答案」**——这正是
指标计算中心的真实语义（每个 as-of 日出一个值）。

**实现状态**：截断的机制（`SyntheticPanel.context_asof`，含 `T`/`L` 长度差的
唯一处理点）已落地并有测试，但 `audit_expression` 目前不用它下裁决。当前采用P1、尾部依赖及作用域审计；有限样本、截点和扰动下未发现反例不构成全部输入的因果性证明。截断还会改变预热条件，需独立解释。`context_asof` 留作滚动求值接入时的接缝（行情指标中心按 as-of 日逐日算值的
那条路径），不是死代码，但现在还没有调用方。

### 2.3 预热敏感性 P3（不是泄露，是可复现性）

```
对 k in (20, 40, 80, 160, 320):
    比较 f(x[-k:])[-1] 与 f(x)[-1] 的相对偏差
```

`recursive_smooth`（EMA 类，`shape_rule` 就叫 `causal recurrence`）按设计就会在这里
亮灯。**这不是缺陷**，是一个必须被记录的性质：它意味着回测值与实盘值会因历史长度
不同而分叉。单独归类、单独报告，绝不与泄露混为一谈。

---

## 3. 裁决

```python
class Verdict(StrEnum):
    CAUSAL            = "causal"             # 已执行的适用探针未发现反例，不是形式证明
    WINDOW_CONSUMING  = "window_consuming"   # 归约类，按定义吃整窗；公式层需确认右端
    LEAK              = "leak"               # 前缀被改写，附首个失配 t 与数值差
    WARMUP_SENSITIVE  = "warmup_sensitive"   # P3 超阈；可复现性风险
    UNKNOWN           = "unknown"            # 探针无法执行
```

**`UNKNOWN` 绝不静默降级为通过。** 探针跑不动（类型不匹配、输入退化、算子需要特殊
定义域）就如实报 `UNKNOWN` 并说明原因，由人来裁。把不确定性悄悄变成通过，是这类
工具唯一不可原谅的失败模式。

### 浮点容差（真正的误报来源）

截断后浮点求和顺序会变，`sum`/`variance` 可能产生 1e-15 级差异。双阈值：

| 比较规则 | 裁决 |
|---|---|
| 所有比较点满足 tight：`rtol=1e-9, atol=1e-12` | 一致 |
| 不全满足 tight，但全部满足 loose：`rtol=1e-6, atol=1e-9` | `UNKNOWN`（灰区，人工看） |
| 至少一点不满足 loose | `LEAK` |

`probes._closeness` 使用逐元素 `np.isclose(a,b,equal_nan=True)`，有限值条件为 `abs(a-b) <= atol + rtol*abs(b)`；不是单独按相对误差划三段。微小泄露可能低于容差，未触发行为也可能漏检；报告必须保留样本、截点、扰动和容差，不能声称阈值与结论无关。

---

## 4. 内置检验数据集

不落盘、不联网、不依赖行情。`data/fixtures/` 目前并不存在，也就不必去维护它。

```python
CAUSALITY_DATASET_VERSION = "1.0.0"

def synthetic_panel(periods: int = 512, assets: int = 6,
                    seed: int = 20260908) -> SyntheticPanel
```

确定性生成（固定 seed + 版本号），每条设计约束都是为了**让泄露显影**，不是为了像真实行情：

| 约束 | 理由 |
|---|---|
| 逐点唯一，无重复值 | 「把未来某点抄到过去」这种泄露，在有重复值时会被巧合掩盖 |
| 末段 15% 注入结构性突变（跳空 + 波动放大） | 任何偷看尾部的算子都会被放大显影；平稳噪声是弱探针 |
| 净值路径恒正、收益率 ∈ (-0.5, 0.5) | `log`/`sqrt`/`drawdown_series` 定义域安全；`prod(1+r)` 不塌 |
| N 个资产用不同因子载荷 → 协方差满秩 | `covariance`/`solve`/`matmul`/`quadratic_form` 不退化 |
| `asset_weights` 非负且和为 1 | DSL 的 `_weight_vector_sum_kernel` 会校验 |
| **`adjusted_nav` 长度 = T + 1** | 它的类型符号是 `L` 而非 `T`（净值路径比收益率多一点）。截断到决策日 t 时必须 `returns[:t+1]` 配 `adjusted_nav[:t+2]`，否则整套探针静默失效 |

### 对照组（negative controls）

没有对照组的检测器不可信。数据集同时提供**已知答案的样本函数**：

```python
KNOWN_CAUSAL = {"rolling_mean_20", "cumulative_return", "drawdown_series"}
KNOWN_LEAKY  = {"shift_minus_1", "full_window_zscore", "normalize_by_global_max",
                "rank_over_time", "peek_last_value"}
```

CI 断言：喂 `KNOWN_LEAKY` 必须报 `LEAK`，喂 `KNOWN_CAUSAL` 必须报 `CAUSAL`。
探针自己坏掉时，这一条会先响。

另供 `hostile_variants()`：常数序列、单调序列、含 NaN 段、超短序列——验证探针在
退化输入下报 `UNKNOWN` 而不是误报 `LEAK`。

---

## 5. 算子级审计（L1，一次性 + CI 守门）

早期记录曾为141个注册键、按 `operator_id` 去重后104个算子；这组数量是历史基线。当前数量应从指定版本的注册表/基线读取并区分别名，不能继续作为现行覆盖统计。

输入不靠解析 `signature.inputs` 那些人类可读的字符串（`'series<time>[T] | vector<asset>[N]'`
这种带联合类型的），而是拿**类型系统自己当预言机**：从内置数据集构造一个候选
`ValueType` 池，笛卡尔积枚举，用 `spec.infer_output()` 筛出该算子接受的组合。
类型推断说行就行，不用维护第二套解析规则。

候选池里同一个结构会重复出现（收益率 / 净值 / 无量纲），因为 `ValueType` 的相等性
**刻意忽略**语义量纲，而算子的类型推断却会检查它——净值类算子只接受 `price_basis`
非空的输入。池的顺序也有讲究：带时间轴的排在前面，否则 `add` 这类多态算子会被
标量组合抢先占满配额，时间轴上的行为反而测不到。

分流规则最终只看**输入输出的轴**，不看 `shape_rule` 文本：

| 情形 | 探针 | 裁决 |
|---|---|---|
| 输入无时间轴 | — | `CAUSAL`（不存在时间方向的泄露） |
| 输出有时间轴 | P1 | `CAUSAL` / `LEAK` |
| 输入有、输出无时间轴 | 尾部依赖（§2.1b） | `CAUSAL` / `WINDOW_CONSUMING` |
| 输出有时间轴且只有一个时间轴入参 | 附加 P3 | 另一个字段，**不并入裁决** |

驱动用 `TypedOperatorSpec.evaluate` —— 这个字段的 docstring 写明就是给一致性测试用的
NumPy 参照实现（生产路径走 NJIT，不受影响）。这只描述普通算子参考路径；当前 `rolling_scope.py` 会调用 `compile_numba_plan` 对完整滚动子图执行已编译P1，不能再把整个审计模块描述为“不碰Numba或编译器”。本轮仅静态核对调用链，未重新执行数值审计。

结果落 `backend/causality/baseline.json`：算子 id → 裁决 + 人工核定说明。CI 比对，
**新增算子无裁决即测试失败**（fail-closed，与现有 NJIT 预热约定一致）。这条守门
在开发过程中真的响过一次：多扰动修复让 `max_consecutive_true` 从 `causal` 变成
`window_consuming`，基线比对直接把测试打红。

---

## 6. 公式级审计（L2，运行时）

```python
def audit_expression(
    expression: str,
    *,
    variable_types: Mapping[str, ValueType] | None = None,
    decision_dates: Sequence[int] | None = None,   # default_decision_dates 默认 count=8
) -> ExpressionReport
```

当前 `audit_expression` 编译表达式后逐个检查滚动作用域之外、具有时间轴的中间节点，用 `_probe_plan` → `run_tail_probe` 检查未来扰动是否改变历史前缀；滚动作用域另由编译子图审计。无时间轴根输出不会被直接当作前缀比较，整窗归约需保留 `WINDOW_CONSUMING` 和窗口右端解释。

- P1 出现超出容差的前缀失配时报告节点与差异；当前是逐节点审计，不是二分定位。
- `SyntheticPanel.context_asof` 的截断机制有独立测试，但当前生产公式审计不调用P2。旧稿的“同时跑P1/P2、P2失配即LEAK”不是当前实现，截断差异也可能来自预热，不能直接等同泄露。
- 将来如接入按决策日截断重算，须明确其窗口/预热语义和批准范围，并单独补运行证据；本次不替实现新增这一能力。

**覆盖缺口待专项复验**：当前 `audit_expression` 的 `max_nodes=32`，达到节点预算后写入“其余节点未审计”warning并停止循环，汇总仍可能为 `CAUSAL`。因此调用方不能只看该枚举就宣称全图完整审计。是否将覆盖不完整纳入强制UNKNOWN/阻断须另行决定并验证实际入口；本次静态发现不等于已经证明某生产入口可绕过，也未修改裁决代码。

报错文案给**具体位置**，不给布尔值：「第 128 个观测点的历史值在数据延长后从 0.0312
变为 0.0455，责任节点：`mean_time(returns)`」。「有偏 / 无偏」这种结论对用户没有用。

---

## 7. 模块结构

```
backend/causality/
    __init__.py       # 公开 audit_operators / audit_expression / Verdict
    synthetic.py      # §4 内置数据集 + 对照组
    probes.py         # §2 三个探针 + §3 容差判定
    audit.py          # §5 算子扫描 + §6 公式审计
    baseline.json     # 带版本的算子裁决基线；数量从文件读取
backend/tests/test_causality.py
```

与 `backend/pit/` 平级：同为横切关注点，被行情指标中心和情景中心共用，不属于任何一个。

---

## 8. 接入

### 指标计算中心
保存自定义指标时同步跑 `audit_expression`（512×6 的合成面板，毫秒级）：
- `LEAK` → **拒绝保存**，报出责任节点。
- `WINDOW_CONSUMING` → 允许保存，前端提示「此公式消费整个时间窗口，请确认窗口右端
  不晚于决策日」。
- `WARMUP_SENSITIVE` → 允许保存，记录所需最小预热长度。

裁决写进 `custom_indicators.json`，前端列表挂徽章。

### 情景研究中心
情景发布前审计一次，裁决写入发布元数据，与 PIT 封版引用并列：
**PIT 版本回答「用了哪些数据」，因果裁决回答「公式有没有偷看」**，两者共同构成
该情景可复现的完整凭据。已发布情景的裁决不可变——重算即新版本。

### CI
`test_causality.py` 三件事：对照组必须报对（§4）；`baseline.json` 必须与实际扫描一致；
新算子无裁决即失败。

---

## 9. 明确不做

- **静态代码扫描**（LeakageDetector / NBLyzer 那一路）。用户公式是自研 typed DSL，
  最终降为 NJIT 内核，本项目当前不接入这些外部静态分析工具；现有行为探针只覆盖已执行样本和扰动，不声称与静态方法等价或覆盖所有缺陷。
- **接入 Freqtrade**。它是交易机器人不是库：`lookahead-analysis` 要求完整的
  `IStrategy` + 回测引擎 + ccxt 行情，数据模型是「每交易对一张 OHLCV」，表达不了
  本系统的 asset 轴与矩阵算子。方法论照抄（P1/P3 的分层来自它），代码不接。
- **真实行情夹具**。当前审计优先采用可控合成数据，便于构造反例、离线复现和调整T/N；这不证明其对所有真实行情缺陷更强，也不替代后续实际数据场景验证。

---

## 历史决策与后续验收

### 移位方向与时间轴

移位算子按以下受控方向工作；这不代表组合公式或输入数据天然没有前视：

* `_validated_periods` 拒绝负数，所以 `lag(x, -1)` 根本编译不过；
* `lag(x, n)` 返回 `x[:-n]`——丢弃**最新**的 n 个点，而不是前移；
* `difference(x, n)` 返回 `x[n:] - x[:-n]`，第 j 项属于时点 `j + n`。

后两条合起来说明本 DSL 的缩短序列是**右对齐**的。这也是本模块最容易出的假阳性：
按左对齐比较，`difference` 会被判成读取未来。`compare_prefix` 因此按
`offset = 输入长度 - 输出长度` 右对齐，并有专门的回归测试守着。

### 组合公式也必须审核

算子干净，不等于公式干净。实测抓到的两条都是**教科书级**的作用域错误：

```
(returns - mean(returns)) / std(returns)     → LEAK
    节点 #2 `returns - mean(returns)` 在决策日 64 的输出，
    被未来数据从 0.00093 改成了别的值

adjusted_nav / max_value(adjusted_nav)       → LEAK
```

`mean` 与 `max_value` 自身都是合法的 `WINDOW_CONSUMING`；错的是把一个整窗归约
广播回**每一个历史点**。这正是必须有公式级审计、光有算子基线不够的原因。

对照之下正确分类的：

| 公式 | 裁决 |
|---|---|
| `rolling_mean(returns, 20, 1) - rolling_mean(returns, 60, 1)` | `CAUSAL` |
| `difference(adjusted_nav, 1) / lag(adjusted_nav, 1)` | `CAUSAL`（L 轴偏移正确） |
| `drawdown_series(adjusted_nav)` | `CAUSAL` |
| `mean_asset(asset_returns)` | `CAUSAL`（截面聚合） |
| `mean_time(asset_returns)` | `WINDOW_CONSUMING` |
| `mean(returns) / std(returns)` | `WINDOW_CONSUMING`（2 处） |

无法执行或缺少充分证据时返回 UNKNOWN，不能把数值错误当作通过或已证实泄露。

算子目录、注册版本和接入程度随代码演进；初始交付顺序不再作为现状列表。历史基线及后来跨中心作用域修复见[工程纪要](../verification/engineering.md)，发布门禁另见[情景验证](../regimes/validation.md)。

### 方法核验边界（2026-10-04）

有限行为测试可以发现反例，但没有穷尽所有输入、节点和数据时点。[Freqtrade lookahead caveats](https://www.freqtrade.io/en/stable/lookahead-analysis/#caveats)也明确提示未触发情形可能带来假阴性；[recursive analysis](https://www.freqtrade.io/en/stable/recursive-analysis/)讨论历史长度敏感性。这些来源支持分别处理问题和保留限制，不证明本项目探针完备或实现已通过。
