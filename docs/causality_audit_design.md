# 因果性审计模块设计（未来函数 / 数据泄露检验）

## 0. 一句话

PIT 封版管住了**能看到哪些数据**；本模块管住**公式有没有偷看**。两者正交，缺一不可：
一个用了 `shift(-1)` 的公式，跑在最严格的 PIT 封版下，依然是错的。

适用范围：141 个内置算子 + 用户在指标计算中心自建的公式 + 情景研究中心的情景定义。

---

## 1. 判据

只有一条，其它全是它的推论：

> **因果性**：输出在时点 t 的取值，只能是输入 `x[0..t]` 的函数。
> 等价地——**改变 t 之后的输入，不允许改变 t 及之前的输出**。

违反 = 未来函数 / 数据泄露。

注意「窗口右端」这个概念区分了两类完全不同的东西：

| | 说明 | 例子 | 是不是缺陷 |
|---|---|---|---|
| **读取未来点** | 输出 t 依赖 x[t+1..] | `shift(-1)`、`ZIG` | 是，永远是 |
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
截断法会把「预热不足」和「未来函数」混成同一个信号——Freqtrade 正是因此不得不把
`lookahead-analysis` 和 `recursive-analysis` 拆成两个命令。扰动法天然分开，误报低得多。

扰动值的选取要**对抗性**：取 `x'[t+1:] = -3 * x[t+1:] + 7`（保号性破坏 + 量级改变 +
排序改变），而不是置零或置常数——置常数会让 `max`/`argmax` 这类算子在巧合下仍然一致。

扰动方式用**三种**（倒序×1.9、放大×6、缩小×0.013；掩码则取反 / 全真 / 全假），
不是一种。单一扰动会产生**数据相关的假阴性**：`max_consecutive_true` 只在最长
真值段恰好落在头部时才「看起来」不依赖尾部，换一种扰动就露馅。实测中它正是
先被误判为因果、加上多扰动后才纠正的。所有系数取正以保号——净值类输入变负会让
`log`/`sqrt`/`drawdown_series` 直接抛错，审计结果退化成一片 UNKNOWN。

### 2.1b 尾部依赖（P1 的对偶，用于无时间轴的输出）

输出没有时间轴时 P1 无从比较前缀，但「无时间轴」**不等于**「消费整窗」：
`first(x)` 只用 `x[0]`，改动尾部它纹丝不动。所以这里实际测一次——改了未来，
输出变不变：

* 不变 → `CAUSAL`（只吃了窗口前段）
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
唯一处理点）已落地并有测试，但 `audit_expression` 目前不用它下裁决——P1 加
尾部依赖已经是决定性的，而截断会把预热条件一起改掉，反而引入 P1 专门避开的
混淆。`context_asof` 留作滚动求值接入时的接缝（指标中心按 as-of 日逐日算值的
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
    CAUSAL            = "causal"             # 通过全部适用探针
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

| 相对差 | 裁决 |
|---|---|
| ≤ `rtol=1e-9, atol=1e-12` | 一致 |
| 1e-9 ~ 1e-6 | `UNKNOWN`（灰区，人工看） |
| > 1e-6 | `LEAK` |

真实泄露是**宏观**的（值整个变掉），不是 1e-10 级的，所以结论对阈值不敏感。灰区
留给人，是因为灰区里真的可能藏着病态数值问题——那也值得知道。

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

`get_typed_operator_registry()` 有 141 个键，但其中含别名；按 `operator_id`
去重后是**104 个算子**。不去重会把算子审两遍，基线里出现重复条目。

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
NumPy 参照实现（生产路径走 NJIT，不受影响）。**不碰 numba、不碰编译器、不要夹具。**

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
    decision_dates: Sequence[int] | None = None,   # 默认在预热后均匀取 12 个
) -> ExpressionReport
```

走已有入口，不需要改编译器：

```python
runtime = TypedIndicatorRuntime.from_expression(expression, variable_types=..., output_contract=...)
full  = runtime.compute(panel.context())
for t in decision_dates:
    sliced = runtime.compute(panel.context_asof(t))   # 按 T/L 语义一致地截断
    ...比对...
```

`panel.context_asof(t)` 是全模块唯一处理 `T` / `L` 长度差的地方——集中一处，
新增变量时不会有人忘记。

同时跑 P1 与 P2：
- **P2 失配** → 归约窗口伸到了决策日之后 → `LEAK`，报出首个失配的 t 与两个值。
- **P1 失配** → 公式里有算子直接读了未来点 → `LEAK`，并回溯到具体是哪个 DAG 节点
  （`TypedExpressionPlan` 的节点可逐个求值，二分定位到最小出问题的子表达式）。

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
    baseline.json     # 141 个算子的已核定裁决
backend/tests/test_causality.py
```

与 `backend/pit/` 平级：同为横切关注点，被指标中心和情景中心共用，不属于任何一个。

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
  最终降为 NJIT 内核，静态分析既没有现成工具认得，行为测试也已经覆盖同样的缺陷。
- **接入 Freqtrade**。它是交易机器人不是库：`lookahead-analysis` 要求完整的
  `IStrategy` + 回测引擎 + ccxt 行情，数据模型是「每交易对一张 OHLCV」，表达不了
  本系统的 asset 轴与矩阵算子。方法论照抄（P1/P3 的分层来自它），代码不接。
- **真实行情夹具**。合成数据的对抗性强于真实行情，且不联网、可复现、可任意调 T 和 N。

---

## 10. 交付顺序

1. `synthetic.py` + 对照组 + `test_causality.py` 的对照组断言 —— **先证明探针可信**。
2. `probes.py`（P1/P2/P3 + 容差）。
3. `audit.py` 算子扫描 → 生成 `baseline.json` 首版 → 人工核 `LAST`/`lag`/线性代数那几个。
4. `audit_expression` + 指标中心接入。
5. 情景中心接入 + 发布元数据。

1–3 是自洽的一步，可独立验收：CI 绿 + 104 个算子全部有裁决。**已完成**（见 §11），
公式级审计（第 4 步的算法部分）也已可用，尚未接入前后端。

---

## 11. 实测结果

`python -m causality.audit`，104 个算子，2.4 秒：

| 裁决 | 数量 |
|---|---|
| `CAUSAL` | 62 |
| `WINDOW_CONSUMING` | 42 |
| `LEAK` | **0** |
| `UNKNOWN` | **0** |

另有 7 个算子标记预热敏感（**不是**泄露）：`cumulative_max/min/product/return/sum`、
`drawdown_series`、`recursive_smooth`。前六个是扩张窗口，最新值依赖全部历史；
`recursive_smooth` 是 EMA 类递归。共同含义是回测值与实盘值会因可得历史长度不同
而分叉，需要声明最小预热长度。

### 一个值得记录的结论

**typed DSL 结构上就拿不到未来数据。** 0 个算子泄露不是运气：

* `_validated_periods` 拒绝负数，所以 `lag(x, -1)` 根本编译不过；
* `lag(x, n)` 返回 `x[:-n]`——丢弃**最新**的 n 个点，而不是前移；
* `difference(x, n)` 返回 `x[n:] - x[:-n]`，第 j 项属于时点 `j + n`。

后两条合起来说明本 DSL 的缩短序列是**右对齐**的。这也是本模块最容易出的假阳性：
按左对齐比较，`difference` 会被判成读取未来。`compare_prefix` 因此按
`offset = 输入长度 - 输出长度` 右对齐，并有专门的回归测试守着。

### 泄露只可能来自组合

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
| `rolling_mean(returns, 20)`（无 min_periods） | `UNKNOWN`——运行时拒绝含 NaN 的结果 |
| `returns / rolling_max(adjusted_nav, 60, 1)` | `UNKNOWN`——`T` 与 `L` 无法逐元素运算 |

最后两行是**如实报告**而非缺陷：平台本身就算不出这两条公式，探针没有证据力时
就该说 UNKNOWN。

### 测试

`backend/tests/test_causality.py`，56 个测试，5.5 秒。第一组是对照组——6 个已知
因果函数必须放行、5 个已知泄露函数必须抓住、5 种退化输入不许误报——**它是其余
所有断言的前提**：探针自己坏掉时这一组先响。

全量后端 **1908 passed / 1 failed**；那 1 个失败在移除本模块的测试文件后依然存在
（`test_custom_indicator_routes.py::test_time_series_builder_requires_fixed_constants_and_runtime_cannot_override`），
是既有问题。
