# CMA 五模型算法审阅与问题清单

- **审阅日期**：2026-09-20
- **审阅范围**：`backend/strategic_allocation/` 下 CMA（长期资本市场假设）生成与应用链路
  - `cma_model_contracts.py` / `cma_models.py` / `cma_model_kernels.py`
  - `cma_statistical_models.py` / `cma_statistical_kernels.py`
  - `cma_application.py` / `multi_cma.py` / `multi_cma_kernels.py`
  - 下游消费点：`kernels.py`（政策候选打分）、`goal_kernels.py`（资金路径）
- **审阅性质**：只读代码审阅，**本文档不改动任何代码**。所有修复建议均需单独获得授权后实施。
- **对照基准**：Black & Litterman (1992)、He & Litterman (1999)、Meucci (2010)、Gelman *BDA* ch.3、Pástor (2000)、Barberis (2000)、Avramov & Zhou (2010)、Ledoit & Wolf (2003/2004)、Tütüncü & Koenig (2004)、Ceria & Stubbs (2006)、Hamilton (1989)、Ang & Bekaert (2002)、Kritzman/Page/Turkington (2012)；行业口径参照 JPM LTCMA、BlackRock CMA、Invesco、Vanguard VCMM、Research Affiliates 公开方法论。

---

## 0. 结论摘要

五个模型的**核心数学公式全部验算通过**，NIW 共轭更新与情景混合矩的实现质量高于多数商业系统。问题集中在三类：

| 类别 | 问题数 | 说明 |
|---|---|---|
| 🔴 数值口径错误/偏差 | 2 | 会系统性影响 SAA 候选权重，方向可判定 |
| 🟠 模型前提与实现自相矛盾 | 2 | 模型自身假设否定了它使用的年化/分布规则 |
| 🟡 表达力与可审性不足 | 4 | 不产生错误结果，但限制使用场景或削弱复核能力 |
| ⚪ 范围缺口 | 1 | 缺 building-block 收益模型，属产品定位问题 |

**优先级总表**

| 编号 | 优先级 | 问题 | 影响面 |
|---|---|---|---|
| [P0-1](#p0-1) | 🔴 P0 | 稳健化用 box/L1 收缩，已算出的 `Σ_μ` 全协方差未被使用 | 全部五个模型的 `conservative_return` / `robust_utility` |
| [P0-2](#p0-2) | 🔴 P0 | `shrinkage` 只压非对角线，长仓组合风险被系统性低估 | `historical_statistics`、`historical_regime_occupancy`、风险参考 |
| [P1-1](#p1-1) | 🟠 P1 | regime 模型承认状态持续，却按 iid ×252 年化 | `historical_regime_occupancy` |
| [P1-2](#p1-2) | 🟠 P1 | 情景混合的偏度/厚尾在资金测算前被对数正态抹平 | `scenario_mixture` + 全部模型的资金成功率 |
| [P2-1](#p2-1) | 🟡 P2 | BL 的 P 矩阵被限制为单资产/两两观点，无法表达组合观点 | `black_litterman` |
| [P2-2](#p2-2) | 🟡 P2 | BL 算出后验均值协方差 `M` 但不接入稳健化通道 | `black_litterman` |
| [P2-3](#p2-3) | 🟡 P2 | BL 缺 δ/τ/Ω 一致性诊断（隐含市场 Sharpe、观点收缩权重） | `black_litterman` |
| [P2-4](#p2-4) | 🟡 P2 | 估计窗口与 `horizon_years` 完全脱钩，无门禁 | 三个统计模型 |
| [P3-1](#p3-1) | ⚪ P3 | regime 的 `ddof=0` 与 historical 的 `ddof=1` 不一致 | `historical_regime_occupancy` |
| [P3-2](#p3-2) | ⚪ P3 | 缺 building-block / 估值锚定收益模型 | 产品范围 |

**不要动的地方**见[附录 A：已验证正确、修复时不得破坏的契约](#附录-a已验证正确修复时不得破坏的契约)。

---

<a id="p0-1"></a>
## 🔴 P0-1 稳健化用 box/L1 收缩，已算出的 `Σ_μ` 全协方差未被使用

### 问题是什么

政策候选的"保守收益"和"稳健效用"两个指标，用的是**逐资产边际半宽的 L1 加总**：

```
conservative_return = E[r] − penalty × Σᵢ |wᵢ| · uᵢ
robust_utility      = E[r] − penalty × Σᵢ |wᵢ| · uᵢ − ½·aversion·w'Σw
```

这是 **box / L∞ 不确定集**下的稳健均值方差（Tütüncü & Koenig 2004 的 L1 对偶形式）。它隐含假设：**所有资产的真实均值同时朝最坏方向偏离到各自 95% 边际置信上界**。

在 10 个以上资产的组合里，这是一个联合概率≈0 的事件。后果是惩罚量级被严重夸大。

**数量级验证**（用项目自己的默认参数）：

- 5Y 日频窗口（T≈1260），年化波动 20% 的权益类
- `historical_estimate` 给出 `uᵢ = t₀.₉₇₅ × σ/√(T/252) ≈ 1.96 × 0.20 / √5 ≈ 17.5%/年`
- `uncertainty_penalty` 默认 **1**（`contracts.py:227`、`contracts.py:372`）
- 一个 60/40 组合的 haircut ≈ `0.6×17.5% + 0.4×5%` ≈ **12.5%/年**

这会把整个风险溢价吃光，`conservative_return` 恒为大幅负数，`robust-utility` 候选退化为"接近最低风险组合"。实际使用中用户被迫把 `penalty` 调到远低于 1——**于是"95% 置信"这个标签就失去了统计含义，惩罚强度变成一个纯粹的拍脑袋旋钮。**

### 关键事实：全协方差已经算出来了，但没用

| 模型 | 已计算的 `Σ_μ` | 存放位置 |
|---|---|---|
| `historical_statistics` | `mean_covariance = Σ_annual × 252 / T` | `cma_statistical_kernels.py:190`，写入审计 `cma_statistical_models.py:31` |
| `bayesian_niw` | `252² × Ψₙ/((νₙ−N−1)·κₙ)` | `cma_statistical_kernels.py:227`，返回为 `posterior` |
| `black_litterman` | Joseph form `M`（见 [P2-2](#p2-2)） | `cma_model_kernels.py` 内 `posterior`，透传到 `posterior_mean_covariance` |

也就是说**联合置信域所需的全部信息都已经算出并冻结在 artifact 里，只是打分时只取了对角线再做 L1 加总。**

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/kernels.py` | 81 | `portfolio_moments_kernel(weights, means, covariance, uncertainty, aversion, penalty)` 签名 |
| `backend/strategic_allocation/kernels.py` | 88 | `variance, expected, haircut = 0.0, 0.0, 0.0` |
| `backend/strategic_allocation/kernels.py` | **97** | **`haircut += abs(weights[i]) * uncertainty[i] * penalty`** ← 问题点 |
| `backend/strategic_allocation/kernels.py` | 110–112 | `metrics = np.array([expected, √variance, expected − haircut, ..., expected − haircut − ½·aversion·variance])` |
| `backend/strategic_allocation/contracts.py` | 227 / 372 | `uncertainty_penalty: Number = Field(default=1, ge=0, le=5)` |
| `backend/strategic_allocation/cma_application.py` | 34–37 | `arrays["mean_uncertainty"]` 与 `arrays["posterior_mean_covariance"]` 的装配边界 |
| `backend/strategic_allocation/mandate_kernels.py` | 25 / 57 | `reference_candidate_checks_kernel` 复用同一个 `portfolio_moments_kernel` |

### 业内/学术对照

- **Tütüncü & Koenig (2004)** — box 不确定集，即当前实现。适用于**少数资产**、或者不确定性来自区间型专家判断（而非统计估计）的场景。
- **Ceria & Stubbs (2006)**, *Incorporating estimation error into portfolio selection* — 椭球不确定集 `κ√(w'Σ_μ w)`。三点优势：
  1. 它**是**一个真正的联合置信域，κ 与置信水平有明确对应（`κ = √(χ²_{N,α})` 或 N 大时用正态近似）；
  2. 惩罚随组合分散度自然下降——分散化会降低估计误差暴露，这是正确的金融含义，box 形式完全看不到；
  3. 仍是凸的，且 `√(w'Σ_μw)` 与现有 `√(w'Σw)` 同构，求解器改动极小。
- 行业稳健优化实现（Axioma Robust、MSCI Barra 的 robust 模块）默认使用椭球形式。

### 建议修复

**方案（推荐）**：新增椭球稳健度量，与现有 box 形式**并存**，由契约字段显式选择，不删除现有实现。

> ⚠️ AGENTS.md 存量代码治理条款：这里**不是**"新旧双实现"。box 与椭球是两种不同的不确定集语义，属于两个可选算法而非同一功能的历史版本。必须在契约上显式区分（`uncertainty_set: Literal["box", "ellipsoidal"]`），且各自有独立测试覆盖，否则就会构成违规的双实现。

1. **契约层**：`contracts.py` 的两处政策请求各加一个判别字段
   ```
   uncertainty_set: Literal["box", "ellipsoidal"] = "box"   # 保持旧默认，避免改变存量政策的数值
   ```
   `ellipsoidal` 时要求 CMA artifact 具备 `mean_estimation_covariance` / `posterior_mean_covariance`；缺失则报 `SAA_UNCERTAINTY_SET_UNAVAILABLE`，**不得回退到 box 静默降级**。

2. **内核层**：`kernels.py` 新增固定签名 njit 内核，不修改 `portfolio_moments_kernel` 的现有 ABI
   ```python
   @njit((V, V, M, M, float64, float64), cache=True, nogil=True)
   def portfolio_moments_ellipsoidal_kernel(weights, means, covariance, mean_covariance, aversion, kappa):
       # haircut = kappa * sqrt(w' Σ_μ w)，其余与 portfolio_moments_kernel 完全一致
   ```
   - 必须 `disable_compile()` 并纳入 `warm()`，否则违反"生产请求期间禁止临时编译"。
   - `Σ_μ` 以只读视图传入，不复制。

3. **应用层**：`cma_application.py:34-37` 已经把 `posterior_mean_covariance` 放进 `arrays`，但 `historical_statistics` 的 `mean_estimation_covariance` 目前只进了审计 dict（`cma_statistical_models.py:31`），没有进 `arrays`。需要把它提升为与 `covariance` 同级的持久化数组，才能被 `frozen_numeric_inputs` 以 mmap 方式读回。

4. **κ 的取值**：不要复用 `uncertainty_penalty` 的语义。建议独立字段 `uncertainty_confidence: Literal["68", "90", "95"]`，由它推出 κ，让置信水平重新变成可解释的。

5. **多 CMA 的边界**：`multi_cma.py:151` 目前用 `weighted_half_width_kernel` 做半宽的加权平均，这在椭球形式下不成立（半宽不能线性平均成协方差）。融合模式下要么加权平均 `Σ_μ` 本身，要么明确禁用椭球模式并报错。**不能猜。**

### 验收要求

- 椭球内核与受控 Python 参考实现的数值一致性（含 NaN/Inf、空区间、单资产、零方差现金）
- 证明 box 路径的数值**逐位不变**（存量政策不得漂移）
- `np.shares_memory` 验证 `Σ_μ` 未被复制
- 启动预热覆盖新签名，`execution_audit()` 的 `python_fallback=0` / `request_time_compilation=0` 仍成立
- 回归：`backend/tests/test_strategic_allocation.py`

---

<a id="p0-2"></a>
## 🔴 P0-2 `shrinkage` 只压非对角线，长仓组合风险被系统性低估

### 问题是什么

样本协方差的收缩实现为：

```python
cov[i, j] *= periods * (1.0 if i == j else 1.0 - shrinkage)
```

即**保持方差不变、把相关阵朝单位阵压**：`R_shrunk = (1−λ)R + λI`。

数学上没有错误（正定性有保证，特征值 `(1−λ)eᵢ + λ ≥ λ > 0`），审计字段也老实标注为 `fixed_diagonal_shrinkage`，没有冒充 Ledoit-Wolf。

**但金融含义方向是反的：**

> 大类资产之间的平均相关系数为正 → 压低相关 → 长仓组合的 `w'Σw` **系统性变小** → SAA 在 `max_volatility` 约束下误以为还有风险预算 → 实际组合比模型显示的更危险。

默认 `shrinkage = 0.1`（`cma_model_contracts.py:164`）。对一个平均相关 0.5 的股债商品组合，10% 的相关收缩会让组合波动率低估约 2–4%（相对），且**危机期（相关趋近 1）低估最严重**——恰好是风险预算最该生效的时候。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/reference_evidence_kernels.py` | 61 | `def annual_moments(returns, shrinkage, periods)` |
| `backend/strategic_allocation/reference_evidence_kernels.py` | **82** | **`cov[i, j] *= periods * (1.0 if i == j else 1.0 - shrinkage)`** ← 问题点 |
| `backend/strategic_allocation/kernels.py` | 26 | `def historical_risk_kernel(returns, shrinkage, periods)` |
| `backend/strategic_allocation/kernels.py` | **43** | **`covariance[i, j] *= periods * (1.0 if i == j else 1.0 - shrinkage)`** ← 同一问题的第二处 |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 181 | `historical_estimate` 调用 `annual_moments(returns, shrinkage, 252)` |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 270 | regime 模型的同向收缩：`value *= 1.0 - shrinkage`（仅 `i != j`） |
| `backend/strategic_allocation/cma_model_contracts.py` | 164 | `HistoricalCmaRequest.shrinkage` 默认 **0.1** |
| `backend/strategic_allocation/cma_model_contracts.py` | 190 | `RegimeCmaRequest.shrinkage` 默认 0.0 |

### 业内/学术对照

| 方法 | 收缩目标 | 强度 | 对组合风险的方向性 |
|---|---|---|---|
| **本项目** | 单位阵（仅相关） | 人工固定 | **系统性低估**（长仓） |
| Ledoit & Wolf (2003) | 常相关模型 / 单指数模型 | 解析最优 λ* | 中性 |
| Ledoit & Wolf (2004) | `μI`，μ = tr(Σ)/N | 解析最优 λ* | 中性偏保守（方差也向均值收缩） |
| Elton & Gruber 平均相关 | 全体平均相关 | 人工或最优 | 中性 |

关键差异：LW2004 的 `μI` 目标**同时收缩方差**（把各资产方差朝横截面平均方差拉），所以不产生"相关变小但方差不变"这种单边效果。本项目只动非对角，正好落在最不利的那一侧。

### 建议修复

有三个可选方向，按改动量从小到大：

**方案 A（最小改动，推荐先做）——把目标改成常相关**
```python
# 先算横截面平均相关 r̄（排除对角），再朝 r̄ 而非 0 收缩
cov[i, j] *= periods * (1.0 if i == j else ((1.0 - shrinkage) + shrinkage * r_bar / rho_ij))
# 等价写法：先在相关空间做 R' = (1−λ)R + λ(r̄·(J−I) + I)，再乘回波动率
```
这是 Ledoit-Wolf(2003) 的目标，保持方向中性，且不引入最优 λ 估计的复杂度。

**方案 B——保留现有内核，加保守性门禁**
不改数学，但在政策预览时**同时计算 λ=0 的组合波动率**，若 `σ(λ) / σ(0) < 0.97` 则在 `warnings` 中显式提示"当前收缩使组合风险下降 X%"，让低估变得可见、可审。改动最小，但不解决根因。

**方案 C——实现 Ledoit-Wolf 解析最优 λ**
学术最正统，但与本项目"显式声明假设、不做数据驱动自动调参"的治理哲学冲突。**不推荐**：CMA 是受治理的研究制品，λ 应该是分析师的声明而非样本的函数。

> 倾向方案 A + 方案 B 组合：改目标解决方向性偏差，加门禁解决可审性。

**⚠️ 兼容性红线**：`annual_moments` 与 `historical_risk_kernel` 被**风险参考（Risk Scale）链路**共用，不只服务 CMA。修改前必须用 CodeGraph 跑一遍 `callers`，确认所有消费方；存量已冻结的 CMA / 风险标尺版本**不得重算**，只能对新版本生效。这需要一个 schema 版本位或收缩方法判别字段。

### 验收要求

- 对同一样本，新旧两种收缩目标的组合波动率对比表（至少覆盖平均相关 0.2 / 0.5 / 0.8 三档）
- 证明 λ=0 时两种实现完全一致
- 存量冻结 artifact 的 `content_hash` 校验仍通过（说明历史未被重算）
- 回归：`backend/tests/test_strategic_allocation.py` + 风险标尺相关测试

---

<a id="p1-1"></a>
## 🟠 P1-1 regime 模型承认状态持续，却按 iid ×252 年化

### 问题是什么

`historical_regime_occupancy` 的计算链是：

```
逐状态 Welford → 条件均值 μₛ、条件协方差 Σₛ
pₛ = nₛ / Σnₛ                              # 历史占用频率
(μ_base, Σ_base) = mixture(p, μₛ, Σₛ)      # within + between，正确
μ_annual = μ_base × 252
Σ_annual = Σ_base × 252                    # ← 问题点
```

单期混合矩的计算**完全正确**（复用了情景混合核，`between` 项被正确计入）。问题在最后一步的年化。

**矛盾在于**：regime 模型存在的全部理由就是"市场状态是持续的、会聚集的"。而 `×252` 的 iid 缩放假设**日收益序列无自相关、无波动率聚集**——这恰恰否定了 regime 的前提。

后果：
- **均值端可辩护**。遍历马尔可夫链的样本占用率是其平稳分布的一致估计量。对 10–30 年 LTCMA 用平稳分布是正确的目标对象。
- **方差端不成立**。状态持续 → 多期收益存在正自相关（熊市连着熊市）→ H 期方差 **显著大于** H × 单期混合方差。低估幅度随状态持续性（转移矩阵对角元）单调上升。

代码把它标注为 `annualization_method="historical_occupancy_iid_annualization"`（`cma_statistical_models.py:89`）——**诚实地命名了一个被模型自身前提所否定的假设**。这是好的披露，但不能替代修复。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_statistical_kernels.py` | 241 | `conditional_state_moments(returns, states, count, shrinkage)` — 逐状态 Welford，正确 |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 278 | `occupancy_probabilities(counts)` — `pₛ = nₛ/Σnₛ`，要求总样本 ≥20（行 284） |
| `backend/strategic_allocation/cma_statistical_kernels.py` | **293** | **`annualize_moments(mean, covariance, periods)` — `return mean*252, covariance*252`** ← 问题点 |
| `backend/strategic_allocation/cma_statistical_models.py` | 73–75 | 调用 `conditional_state_moments` + `occupancy_probabilities` |
| `backend/strategic_allocation/cma_statistical_models.py` | 83–85 | `mixture_moments_kernel` 混合（正确） |
| `backend/strategic_allocation/cma_statistical_models.py` | **87** | **`means, covariance = numeric.annualize_moments(base_mean, base_cov, 252)`** |
| `backend/strategic_allocation/cma_statistical_models.py` | 89 | `annualization_method="historical_occupancy_iid_annualization"`（如实标注） |
| `backend/strategic_allocation/cma_statistical_models.py` | 80 | 正概率状态要求 ≥20 条共同样本（好设计，见附录 A） |
| `backend/strategic_allocation/cma_model_contracts.py` | 186–199 | `RegimeCmaRequest`：`run_ref` 绑定冻结的状态识别运行 |

### 业内/学术对照

- **Hamilton (1989)** — 马尔可夫状态转换的基础框架，核心对象是转移矩阵 `P` 而非占用频率。
- **Ang & Bekaert (2002)**, *International Asset Allocation with Regime Shifts* — 明确指出：多期配置下相关的不是无条件混合矩，而是**从当前状态出发的 H 步条件分布**；忽略持续性会系统性低估长期下行风险。
- **Kritzman, Page & Turkington (2012)**, *Regime Shifts: Implications for Dynamic Strategies* — 用转移概率驱动动态配置，并显示 regime 持续性对多期方差的放大效应是一阶量级。
- **Guidolin & Timmermann** 系列 — 多状态下的长期资产配置，强调 H 期矩必须由链的 H 步分布导出。

### 关键：修复所需的数据已经在手

`conditional_state_moments` 的输入 `states` 就是**完整的状态时间序列**（`cma_statistical_models.py:72` 的 `evidence["regime"]`）。估计转移矩阵 `P` 只需要在同一次遍历中统计相邻状态对的计数，**增量成本接近零**。

### 建议修复

**第一步（必须）——估计并暴露转移矩阵**

在 `conditional_state_moments` 同一次扫描里累加转移计数（或新增一个极小的 njit 内核）：
```python
@njit((I, int64), cache=True, nogil=True)
def transition_counts_kernel(states, count):
    # 返回 count×count 的相邻转移计数；state == -1 视为断点，不计入
```
把行归一化后的 `P`、平稳分布 `π`（`P` 的单位特征向量）、各状态期望持续期 `1/(1−Pₛₛ)` 全部写入 `audit`。**即使不改年化规则，这三个数字也应该让研究员看到**——它们直接决定当前年化假设的偏差有多大。

**第二步（推荐）——修正方差的年化**

两个可选路径：

- **路径 A（解析，改动小）**：用状态链的自协方差修正。对可遍历链，H 期方差为
  ```
  Var_H = H·Σ_within + Σ_between 的 H 期累积项
  ```
  其中 between 部分需要按 `Σₖ (H−k)·(Pᵏ − 1π')` 加权，而非简单 ×H。实现为一个固定签名内核，输入 `(P, μₛ, Σₛ, H)`。
- **路径 B（模拟，最直观）**：不改 CMA 的两阶矩输出，而是在**资金测算**里直接跑状态链（见 [P1-2](#p1-2)，两者可以合并实现）。

**第三步（治理）**：如果暂不修正，至少把 `annualization_method` 从审计字段提升为**界面上的显式警示**，并在 `limitations` 里量化——例如"当前状态平均持续期 X 个月，iid 年化可能低估 10 年期方差约 Y%"。

### 需要单独核实的风险点

`RegimeCmaRequest.run_ref` 绑定的是外部冻结的状态识别运行（`historical_regimes` 模块）。**如果该识别算法使用了全样本数据划分状态，状态标签本身带前视偏差**，那么条件矩就是事后结果。AGENTS.md 的时点语义条款（"需要后续或全样本数据的结果，即使下游只有普通比较，也仍是事后结果"）直接适用。

**本次审阅未追入 `historical_regimes` 内部，建议单独核一遍该模块的时点门禁。**

### 验收要求

- 转移矩阵内核与受控参考实现一致性；断点（`state == -1`）处理正确
- 平稳分布 `π` 与样本占用率 `p` 的差异在审计中可见（两者应接近，差异大说明样本不足或链不遍历）
- 若实现方差修正：对合成的两状态持续链，验证修正后的 H 期方差与蒙特卡洛一致
- 存量 regime CMA 版本不得重算

---

<a id="p1-2"></a>
## 🟠 P1-2 情景混合的偏度/厚尾在资金测算前被对数正态抹平

### 问题是什么

`scenario_mixture` 的单期矩计算**完全正确**：

```
μ = Σ pₛ μₛ
Σ = Σ pₛ Σₛ  +  Σ pₛ(μₛ−μ)(μₛ−μ)'
    └ within ┘   └──── between ────┘
```

`between` 项被正确计入——这是最容易做错的地方（大量实现只加权平均各情景协方差，丢掉情景均值本身的离散度，而这一项通常是情景框架下的主导风险）。

**但下游把它压扁了。**

一个"基准 / 繁荣 / 崩溃"三情景的混合分布是明显**左偏、厚尾**的。可是：

1. 只有 `(μ, Σ)` 传给 MV 优化器 —— 这是对的，MV 只需要两阶矩；
2. 资金测算再用融合后的 `(μ, σ)` 拟合一个**单因子对数正态**去抽路径 —— 崩溃情景的尾部在这里被彻底抹成了对数正态。

`goal_kernels.py` 的矩匹配本身是精确的（见附录 A），问题不在转换精度，而在**分布族选择**：对数正态是右偏的，用它去代理一个左偏混合分布，会**系统性高估资金成功率、低估尾部损失**。

代码 `limitations` 写了"两个矩不能唯一确定资金成功率或尾部损失"——准确，但这是一个可以修的问题，不只是可以披露的问题。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_model_kernels.py` | `mixture_moments_kernel` | within/between 分解，**正确** |
| `backend/strategic_allocation/cma_model_kernels.py` | `scenario_mixture_kernel` | 逐情景 PSD 校验 + 混合，**正确** |
| `backend/strategic_allocation/cma_models.py` | 104–118 | 情景矩装配与审计（`within_covariance` / `between_mean_covariance` 都已冻结） |
| `backend/strategic_allocation/goal_kernels.py` | 165 | `funding_monthly_parameters_kernel(mean, volatility, fee, method, periods)` |
| `backend/strategic_allocation/goal_kernels.py` | 174 | `log_variance = np.log1p((base_volatility / (1.0 + base_mean)) ** 2)` — 矩匹配，精确 |
| `backend/strategic_allocation/goal_kernels.py` | 176 | `drift = (frequency * (np.log1p(base_mean) − 0.5*log_variance) + np.log1p(−fee)) / 12.0` |
| `backend/strategic_allocation/goal_kernels.py` | 184 | `funding_paths_from_monthly_kernel(...)` — 路径主循环 |
| `backend/strategic_allocation/goal_kernels.py` | **213** | **`factor = np.exp(drift + scale * z)`** ← 单一对数正态因子，情景结构在此消失 |
| `backend/strategic_allocation/goal_kernels.py` | 154 | 资本门禁路径里的同一处理 |
| `backend/strategic_allocation/multi_cma.py` | — | `distribution_adapter: "annual_moment_proxy_approximation"`（如实标注） |

### 业内/学术对照

- 情景法 CMA（Mercer、WTW、Aon 的咨询实践，以及 Aladdin / Axioma 的情景叠加）的标准做法：**优化用两阶矩，压力测试与资金模拟用情景本身**。两条腿分开走，不把情景压成单一分布。
- Vanguard VCMM 明确使用非正态、状态相关的模拟引擎生成资金路径分布，而非从年化矩反推单一分布族。
- 目标导向投资（Goals-Based）文献（Chhabra、Das/Markowitz/Scheid/Statman）强调：达标概率对分布左尾极度敏感，两阶矩代理在这里是最不可靠的。

### 建议修复

**这是整个 CMA 栈里性价比最高的改进点** —— 所需数据已经全部冻结在 artifact 里（`within_covariance`、`between_mean_covariance`、逐情景 `μₛ`/`Σₛ`、概率 `pₛ`），改动集中在资金路径内核。

**方案：情景原生抽样**

新增一个固定签名 njit 内核，与现有对数正态路径**并存**（由 CMA 的 `method` 决定走哪条）：

```python
@njit((D, V, M, D, F, V, V, F, F, F, F), cache=True, nogil=True)
def funding_paths_from_scenarios_kernel(draws, probabilities, scenario_means,
                                        scenario_covariances, initial, inflows, outflows, ...):
    # 每个模型月：
    #   1. 按 pₛ 抽状态 s（用 draws 的一个额外因子维做逆变换抽样）
    #   2. 从 N(μₛ, Σₛ) 的组合投影抽收益，再做与现有一致的矩匹配对数正态月度化
    #   3. 复用 funding_payment_kernel 保持现金递推语义完全不变
```

关键约束：
- `draws` 的第三维当前固定为 1（`factors != 1` 直接报错，行 187）。情景抽样需要 2（状态 + 收益），**这是一个 ABI 变化**，必须新签名 + 预热，不能改旧内核。
- 现金递推、未付款判定、回撤统计全部复用 `funding_payment_kernel`，保证与现有报告口径一致。
- 种子与路径数走现有 `seed` / `paths` 契约，可重复性不变。

**同时适用于 regime 模型**：`historical_regime_occupancy` 的状态混合结构完全同构，如果配合 [P1-1](#p1-1) 估出的转移矩阵 `P`，就能抽**带持续性的状态链**而非独立同分布的状态——一次实现同时解决 P1-1 的方差年化和 P1-2 的分布适配。

**过渡期披露**：在实现前，应在资金测算结果上明确标注"分布代理：单因子对数正态；情景/状态的偏度与尾部未进入本次成功率"。

### 验收要求

- 对退化情形（单情景）验证新内核与现有对数正态路径数值一致
- 对三情景左偏混合，验证成功率显著低于对数正态代理（方向性验证）
- 路径可重复性：同 seed 同结果
- `MAX_DIAGNOSTIC_PATH_MONTHS`（`multi_cma.py:27`，5000 万路径月）预算核算需要更新——情景抽样不增加路径月数，但多一个因子维
- 回归：`backend/tests/test_strategic_allocation.py`

---

<a id="p2-1"></a>
## 🟡 P2-1 BL 的 P 矩阵被限制为单资产/两两观点

### 问题是什么

Black-Litterman 的 pick 矩阵被硬性限制为：**每行恰好一个 +1，至多一个 −1**。

```python
if positives != 1 or negatives > 1:
    raise ValueError("CMA_BL_VIEW_PICK")
```

这意味着只能表达两类观点：
- 绝对观点："A 资产年化收益 8%"
- 两两相对观点："A 比 B 多 2%"

**无法表达组合型观点**，例如：
- "新兴市场股票跑赢 50% 发达市场股票 + 50% 发达市场债券的组合 2%"
- "成长风格相对价值+红利的等权篮子超额 1.5%"
- 任何对多资产篮子的相对判断

而这恰恰是机构投研中最常见的观点形式。在 30 个资产（契约上限）的配置空间里，这个限制排除了大部分有意义的横截面观点。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_model_kernels.py` | 51 | `black_litterman_kernel(covariance, market_weights, picks, view_returns, view_std, delta, tau, risk_free_rate)` |
| `backend/strategic_allocation/cma_model_kernels.py` | 90–95 | pick 元素只允许 0 / +1 / −1，其余报 `CMA_BL_VIEW_PICK` |
| `backend/strategic_allocation/cma_model_kernels.py` | **97–98** | **`if positives != 1 or negatives > 1: raise ValueError("CMA_BL_VIEW_PICK")`** ← 限制点 |
| `backend/strategic_allocation/cma_model_contracts.py` | 38–58 | `BlackLittermanView`：只有 `asset_id` + 可选单个 `relative_to` |
| `backend/strategic_allocation/cma_model_contracts.py` | 51 | 绝对观点校验 |
| `backend/strategic_allocation/cma_model_contracts.py` | 53–55 | 相对观点校验：`relative_to` 必须是**另一个单一资产** |
| `backend/strategic_allocation/cma_models.py` | 88–93 | picks 矩阵装配：`picks[v, axis[asset_id]] = 1`；`picks[v, axis[relative_to]] = −1` |

### 业内/学术对照

- **Black & Litterman (1992)** 原文的 P 是任意 `k×N` 矩阵，行向量元素可以是任意实数。
- **He & Litterman (1999)** 的示例中，相对观点的行是 `(0, ..., +1, ..., −w₁/(w₁+w₂), −w₂/(w₁+w₂), ...)` —— 即**市值加权的篮子空头**，而非单一资产。
- **Idzorek (2005)** 的 confidence 映射同样建立在任意 P 上。
- 商业实现（Bloomberg PORT、FactSet、Axioma）均支持自定义篮子观点。

本项目的限制是一个**产品化简化**（UI 只需要两个下拉框），在算法层面没有必要——内核的矩阵运算本身对任意 P 完全成立。

### 建议修复

**内核层几乎不用改。** `black_litterman_kernel` 的数学（`cross`、`system`、`solved`、Joseph form）对任意 P 都正确。需要替换的只是行 97–98 的形状校验：

```python
# 替换单资产限制为通用的行有效性校验
row_sum, abs_sum = 0.0, 0.0
for i in range(count):
    p = picks[v, i]
    if not np.isfinite(p):
        raise ValueError("CMA_BL_VIEW_PICK")
    row_sum += p
    abs_sum += abs(p)
if abs_sum < 1e-12:                       # 全零行无意义
    raise ValueError("CMA_BL_VIEW_PICK")
# row_sum ≈ 0 → 相对观点（不减 rf）；row_sum ≈ 1 → 绝对观点（减 rf）
# 其余取值需要显式声明观点口径，不得猜测
```

**关键难点：`q_excess` 的 rf 调整**

现有实现用 `exposure = Σᵢ pᵥᵢ` 巧妙统一了两种情形（绝对 exposure=1 减 rf，相对 exposure=0 不减）。这对通用 P **仍然成立**，因为 `exposure` 就是该行的净敞口。所以 `residual[v] = q − rf*exposure − projected` 这一行**不需要改**。这是原实现的一个优点。

**契约层改动**（较大）：
```python
class BlackLittermanView(Contract):
    kind: Literal["absolute", "relative", "basket"]
    legs: list[ViewLeg]      # ViewLeg = {asset_id, coefficient}
    annual_return: Number
    view_std: Number
    observed_on: date
    available_on: date
    source: Source
```
需要校验：篮子系数之和为 0（相对）或 1（绝对）；所有 `asset_id` 在轴内；系数绝对值之和有上界防止病态。

**兼容性**：`absolute` / `relative` 两种旧 kind 必须保持**逐位相同**的数值结果，否则存量 BL 版本的 `content_hash` 会失效。建议保留旧 kind 作为 `basket` 的语法糖，在契约层展开成 legs，内核只见 legs。

### 验收要求

- 旧 kind（absolute / relative）在新实现下数值逐位不变
- 篮子观点与手工构造的等价 P 矩阵结果一致
- 病态 P（近似共线的多行观点）的 Cholesky 门禁仍生效（不得因放开而引入数值不稳）
- 存量 BL artifact 的 `content_hash` 校验通过

---

<a id="p2-2"></a>
## 🟡 P2-2 BL 算出后验均值协方差 `M` 但不接入稳健化通道

### 问题是什么

`black_litterman_kernel` 用 Joseph form 正确算出了后验均值协方差：

```
M = (I − KP)·τΣ·(I − KP)' + K·Ω·K',   K = τΣP'(PτΣP' + Ω)⁻¹
```

它被冻结进 artifact 的 `posterior_mean_covariance`。但**它不参与 `conservative_return` / `robust_utility` 的计算**。

原因在装配边界：

```python
"mean_uncertainty": result.mean_uncertainty if result.mean_uncertainty is not None
                    else np.asarray([a.mean_uncertainty for a in request.assets], dtype=np.float64)
```

`CmaModelResult.mean_uncertainty` 只有三个统计模型会填（来自 `half_width`）。**BL 和 `scenario_mixture` 落到用户手填的半宽上。**

而契约又明确规定：三个统计方法**禁止**手填半宽（`contracts.py:309`，"统计方法的均值不确定性由模型提供，不同时填写人工半宽"）。

也就是说系统刻意做了一个二分：
- 统计模型 → 模型提供不确定性，禁止手填
- BL / 情景 → 必须手填，模型算出的 `M` 不用

审计注释明确写了"均值后验协方差不作为资产风险，也不自动转为稳健半宽"——**这是有意为之的设计取舍，不是 bug。**

### 但为什么值得重新讨论

1. `√diag(M)` 是 BL 框架下**最自然**的均值不确定性来源。让用户手填，等于在一个已经给出严格贝叶斯后验的模型上，再叠一层主观数字。
2. 造成**模型间不可比**：同一组资产，NIW 路径的 `conservative_return` 有严格统计含义，BL 路径的没有。而 `multi_cma` 的 `weighted_half_width_kernel`（`multi_cma.py:151`）会把这两种语义不同的半宽**线性加权平均**，得到一个语义混合的结果。
3. 与 [P0-1](#p0-1) 的椭球化改造直接相关：`M` 正是 BL 下的 `Σ_μ`，接通后椭球形式立刻可用。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_models.py` | 34 | `CmaModelResult.posterior_mean_covariance: np.ndarray \| None` |
| `backend/strategic_allocation/cma_models.py` | 38 | `CmaModelResult.mean_uncertainty: np.ndarray \| None = None` |
| `backend/strategic_allocation/cma_models.py` | 100 | BL 审计：`"posterior_mean_covariance_method": "joseph_form"` |
| `backend/strategic_allocation/cma_models.py` | 122 | BL / 情景的 `CmaModelResult(...)` 构造 — **`mean_uncertainty` 未传，取默认 `None`** |
| `backend/strategic_allocation/cma_models.py` | 74 | 统计模型的构造 — 传入 `half_width` |
| `backend/strategic_allocation/cma_application.py` | 27–28 | `if result.mean_uncertainty is not None: asset["mean_uncertainty"] = ...` |
| `backend/strategic_allocation/cma_application.py` | **34–35** | **回退到 `request.assets[i].mean_uncertainty`（人工声明）** ← 分叉点 |
| `backend/strategic_allocation/cma_application.py` | 36–37 | `posterior_mean_covariance` 单独进 `arrays`，但无人消费 |
| `backend/strategic_allocation/contracts.py` | 266 | `mean_uncertainty: Number = Field(ge=0, le=1)` |
| `backend/strategic_allocation/contracts.py` | 309–310 | 统计方法**禁止**手填半宽 |
| `backend/strategic_allocation/multi_cma.py` | 151 | `weighted_half_width_kernel` 线性平均异质语义的半宽 |

### 业内/学术对照

- **He & Litterman (1999)** 用 `Σ + M` 作为优化输入，把估计不确定性直接加进资产风险。
- **Meucci (2010)** 指出 `Σ + M` 混淆了市场风险与估计风险，主张两者分开处理——**本项目正是这么做的，方向是对的**。
- **Ceria & Stubbs (2006)** 的椭球稳健框架里，BL 的 `M` 就是标准的 `Σ_μ` 输入。也就是说：把 `M` 送进稳健化通道（而不是加进 `Σ`）**恰好是 Meucci 立场 + 稳健优化的正确组合**。

所以当前实现只差最后一步：`M` 已经与 `Σ` 分开了，但分开之后没有送到它该去的地方。

### 建议修复

**方案（与 P0-1 合并实施）**：

1. `cma_models.py:122` 的 BL 分支传入 `mean_uncertainty=√diag(M)`：
   ```python
   return CmaModelResult(..., posterior, audit, kernels.execution_audit(),
                         model.model_dump(mode="json"),
                         readonly_float64(np.sqrt(np.diag(posterior)) * kappa_from_confidence, 1))
   ```
   注意：`√diag(M)` 是**标准差**，不是 95% 半宽。要与统计模型的 `half_width` 语义一致，需要乘一个分位数（正态下 1.96）。**必须在审计里写明用的是哪个分位数**，不能默认。

2. 契约层：把 `contracts.py:309` 的禁止手填规则**扩展到 BL**，保持"模型能算的就不许手填"的一致原则。`scenario_mixture` 没有 `Σ_μ`（情景概率是信念而非估计），保持手填，但应在审计里标注 `uncertainty_status: "declared_not_estimated"`。

3. `multi_cma.py:151`：融合时若来源的 `uncertainty_status` 不一致（estimated vs declared），应报错或显式警示，**不能静默线性平均**。

4. **兼容性**：这会改变 BL 政策的 `conservative_return` 数值。存量已冻结的政策**不得重算**（`frozen_policy_assumptions` 会校验血缘）。需要 schema 版本位区分。

### 验收要求

- 无观点时（`views` 为空）`M = τΣ`，验证 `√diag(M)` 与解析值一致
- 观点极确定（`view_std → 0`）时该资产的半宽趋近 0
- 存量 BL artifact 的 `content_hash` 与 `frozen_assumptions` 校验通过
- 融合模式下混合语义半宽的拦截生效

---

<a id="p2-3"></a>
## 🟡 P2-3 BL 缺 δ / τ / Ω 一致性诊断

### 问题是什么

BL 的三个关键参数全部是自由用户输入，彼此之间**没有任何一致性校验或展示诊断**：

| 参数 | 契约约束 | 缺什么 |
|---|---|---|
| `delta`（风险厌恶） | `gt=0` | 没有隐含市场 Sharpe 的反算与展示 |
| `tau` | `gt=0` | 没有与 Ω 的量级关系提示 |
| `view_std`（→ Ω） | `gt=0` | 没有"这个观点实际有多大话语权"的展示 |

具体风险：

1. **δ 可以随便设**。经典做法是由市场组合反解 `δ = (E[rₘ] − rf)/σₘ²`。这里用户设 `δ=10` 就能得到一个隐含 40% 的股票先验收益，系统只做 `−50% ~ 200%` 的范围校验（`validate_effective_returns_kernel`）。审计里输出了 `prior_excess_returns`（π）算部分缓解，但**没有一个数字告诉研究员"你这个 δ 隐含市场 Sharpe = 1.8，不合理"**。

2. **τ 与 Ω 相互独立**。用户可以设 `τ=0.05` 配一个极小的 `view_std`，让观点几乎完全覆盖先验，而界面上没有任何提示。He-Litterman 的 `Ω = diag(PτΣP')` 靠构造保证二者同量级；本项目选择让用户显式给 Ω（更透明，这个取舍本身合理），但透明的代价是**必须补上诊断**。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_model_contracts.py` | 67–69 | `delta: Number = Field(gt=0)`；`tau: Number = Field(gt=0)`；`risk_free_rate` 范围 `[−0.5, 2]` |
| `backend/strategic_allocation/cma_model_contracts.py` | 43 | `view_std: Number = Field(gt=0)` |
| `backend/strategic_allocation/cma_model_kernels.py` | 62–64 | δ/τ/rf 的有限性与正性校验（仅此而已） |
| `backend/strategic_allocation/cma_model_kernels.py` | 73–80 | `prior[i] += delta * covariance[i,j] * market_weights[j]` — π 的计算 |
| `backend/strategic_allocation/cma_model_kernels.py` | 102 | `omega[v] = view_std[v] * view_std[v]` |
| `backend/strategic_allocation/cma_model_kernels.py` | 123–129 | `solved = np.linalg.solve(system, cross.T.copy())` — 即 `Kᵀ`，**收缩权重信息在这里，但没有输出** |
| `backend/strategic_allocation/cma_models.py` | 95–103 | BL 审计字段，已有 `prior_excess_returns`，缺一致性诊断 |

### 建议修复

**纯增量，不改任何算法。** 全部是审计字段的补充：

1. **隐含市场 Sharpe**（`black_litterman_kernel` 内，π 算完后即可得）
   ```
   market_return = Σᵢ w_mktᵢ · πᵢ
   market_vol    = √(w_mkt' Σ w_mkt)
   implied_sharpe = market_return / market_vol        # 等价于 delta * market_vol
   ```
   写入 `audit["implied_market_sharpe"]`。研究员看到 >1.0 就知道 δ 需要复核。合理区间一般 0.2–0.5。

2. **逐观点的实际收缩权重**。`solved`（即 `Kᵀ`）已经在内核里算出来了，直接输出：
   ```
   audit["view_influence"] = [Σᵢ |solved[v,i]| for v in views]
   ```
   或更直观的：每个观点对每个资产后验均值的边际贡献 `solved[v,i] * residual[v]`（这就是 `means[i] += solved[v,i] * residual[v]` 那一项）。

3. **τΣ 与 Ω 的量级比**
   ```
   audit["view_precision_ratio"] = [system_without_omega[v,v] / omega[v] for v in views]
   ```
   即 `(PτΣP')ᵥᵥ / Ωᵥᵥ`。>> 1 表示观点几乎完全覆盖先验，<< 1 表示观点基本被忽略。这是最直接的"我这个观点到底起没起作用"的答案。

4. **先验/后验对比**：审计已有 `prior_excess_returns`，再加 `posterior_shift = means − (prior + rf)`，界面上直接显示"观点把每个资产的预期收益推动了多少"。

**零算法风险**：这些全部是已有中间量的导出，不改变任何数值结果，不影响 `content_hash` 的语义（虽然会改变 hash 值，需要 schema 版本位）。

### 验收要求

- 无观点时 `posterior_shift` 全为 0，`view_influence` 为空
- `implied_market_sharpe` 与 `delta × market_vol` 解析一致
- 不改变 `effective_returns` / `effective_covariance` 的任何数值

---

<a id="p2-4"></a>
## 🟡 P2-4 估计窗口与 `horizon_years` 完全脱钩

### 问题是什么

三个统计模型的估计窗口与 CMA 的预测期限之间**没有任何约束关系**：

- `CmaWindow.kind` 允许 `"1Y"`（`cma_model_contracts.py:124`）
- 内核最低样本要求 **20 条日频观测**（`reference_evidence_kernels.py:68`、`cma_statistical_kernels.py:198`）
- `horizon_years` 允许到 **30 年**（`contracts.py:93` / `contracts.py:280`）

于是可以用 **1 个月的日数据**生成一份**30 年期**的长期资本市场假设，系统不会拦截。

**统计上为什么不成立**：均值的标准误 ≈ `σ/√(窗口年数)`，与样本频率无关（加密采样不改善均值估计精度，这是 Merton 1980 的经典结论）。

| 窗口 | 20% 波动资产的均值标准误 |
|---|---|
| 1Y | 20.0%/年 |
| 5Y | 8.9%/年 |
| 10Y | 6.3%/年 |
| 30Y | 3.7%/年 |

即便用 30 年数据，均值的 95% 置信区间仍有约 ±7.3%/年的宽度。用 1Y 窗口时点估计本身就没有信息量。

代码的 `limitations` 写了"日频均值和协方差采用独立增量年化近似；预测期限不等于历史估计窗口"——**只披露、未拦截**。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/strategic_allocation/cma_model_contracts.py` | 123–135 | `CmaWindow`，`kind` 默认 `"5Y"`，允许 `"1Y"` |
| `backend/strategic_allocation/cma_model_contracts.py` | 137–157 | `StatisticalCmaContext`，只校验 `window.end_date ≤ as_of` |
| `backend/strategic_allocation/contracts.py` | 93 / 280 | `horizon_years: int = Field(default=10, ge=1, le=30)` |
| `backend/strategic_allocation/reference_evidence_kernels.py` | 68 | `if rows < 20 ...` — 最低 20 条 |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 197-198 | NIW 同样 `rows < 20` |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 284 | regime：`if total < 20` |
| `backend/strategic_allocation/cma_statistical_models.py` | 80 | regime 正概率状态 ≥20 条 |

### 业内对照

真实 LTCMA 行业（JPM、BlackRock、Invesco、Vanguard、Research Affiliates）**从不用历史样本均值做预期收益**。历史数据只用于估计协方差（这里样本量确实有帮助），收益端一律用 building block + 估值锚定（见 [P3-2](#p3-2)）。

所以行业里根本不存在"用多长窗口估长期均值"这个问题——因为没人这么做。

### 建议修复

**方案（软门禁 + 强制披露，不阻断研究）**：

1. **窗口/期限比例告警**。在 `StatisticalCmaContext` 的 validator 或服务层加：
   ```
   若 窗口年数 < max(3, horizon_years × 0.3)：
       warnings.append("估计窗口 {X}Y 相对 {H}Y 预测期限过短；均值点估计的标准误为 {SE}%/年。")
   ```
   **只告警不阻断**——研究场景下短窗口有合法用途（比如做敏感性对比）。

2. **在政策采纳环节硬拦**。研究可以随便做，但 `publish_policy` / `policy_gate.check_policy` 应该拒绝采纳一个窗口年数 < 3 年的统计 CMA 作为正式 SAA 政策，除非有显式的人工复核记录（可复用 `institution.review_blockers` 的既有机制）。

3. **把均值标准误提升到界面一级指标**。`half_width` 已经算出来了（`cma_statistical_kernels.py:191`、`:228`），但目前埋在审计里。它应该和 `annual_return` 并排显示——**一个 8% 预期收益 ± 17% 半宽的数字，研究员看一眼就知道该怎么用它**。

4. **区分收益窗口与风险窗口**。协方差从更长窗口估计是有收益的（样本量真的有帮助），均值不是。可以考虑允许 `window` 拆成 `mean_window` / `risk_window` 两个字段——但这会增加契约复杂度，**建议先做 1–3，观察实际使用再定**。

### 验收要求

- 告警文案中的标准误数值与 `half_width / t₀.₉₇₅` 一致
- 政策采纳门禁对短窗口 CMA 生效，且错误码可追溯
- 存量政策不受影响（门禁只对新采纳生效）

---

<a id="p3-1"></a>
## ⚪ P3-1 regime 的 `ddof=0` 与 historical 的 `ddof=1` 不一致

### 问题是什么

两个统计模型的样本协方差自由度不同：

| 模型 | ddof | 代码 |
|---|---|---|
| `historical_statistics` | **1** | `covariance_1d` 返回 `total / (lhs.size − 1)` |
| `historical_regime_occupancy` | **0** | `value = (scatter[s,i,j] + scatter[s,j,i]) / (2.0 * counts[s])` |

regime 的每状态最低样本量是 20，所以 ML 估计相对无偏估计低估 `(n−1)/n = 5%`（方差口径），波动率口径约低估 2.5%。

审计字段 `covariance_ddof: 0` 有如实记录，**可追溯**。但这个不一致没有必要——它会让同一组资产在两个模型下的风险数字出现一个无意义的系统性差异，干扰模型间对比（而模型间对比正是 `multi_cma` 的核心用途）。

### 代码位置

| 文件 | 行 | 内容 |
|---|---|---|
| `backend/cal_indicators/typed_numba_kernels.py` | 1248–1256 | `covariance_1d` — `return total / (lhs.size - 1)`，**ddof=1** |
| `backend/strategic_allocation/cma_statistical_kernels.py` | 241 | `conditional_state_moments` |
| `backend/strategic_allocation/cma_statistical_kernels.py` | **270** | **`value = (scatter[s,i,j] + scatter[s,j,i]) / (2.0 * counts[s])`** — **ddof=0** |
| `backend/strategic_allocation/cma_statistical_models.py` | 89 | `covariance_ddof=0`（如实标注） |
| `backend/strategic_allocation/cma_statistical_models.py` | 32 | historical 侧标注 `covariance_ddof=1` |

### 建议修复

改成 `/ (counts[s] - 1)`，并在 `counts[s] < 2` 时保持现有的排除逻辑（`estimated = np.flatnonzero(counts >= 2)` 已经保证了分母 ≥ 1）。

```python
denominator = 2.0 * (counts[s] - 1)      # 原为 2.0 * counts[s]
```

同步把审计的 `covariance_ddof` 改为 1。

**注意**：这是数值口径变更，属于**新算法版本**而非纯重构（AGENTS.md："改变数学口径属于新算法版本，不得伪装成纯架构重构"）。存量 regime CMA 不得重算，需要版本位。

**也可以选择不修**——只要在界面上明确标注两个模型的 ddof 不同即可。考虑到 5% 的方差差异在 LTCMA 语境下不是一阶问题，**优先级定为 P3，可以合并到 [P1-1](#p1-1) 的 regime 改造里一起做**。

### 验收要求

- `counts[s] = 2` 的边界情形不产生除零或负方差
- 与 `numpy.cov(..., ddof=1)` 逐位一致
- 存量 artifact 不重算

---

<a id="p3-2"></a>
## ⚪ P3-2 缺 building-block / 估值锚定收益模型

### 问题是什么

**这不是 bug，是范围缺口，需要产品层面决策。**

对照真实 LTCMA 行业方法论，缺一整条前瞻收益生成路径：

```
预期收益 = 无风险利率
         + 期限溢价（由收益率曲线形态推出）
         + 信用利差（由当前利差与违约/回收假设推出）
         + 股权风险溢价（股息率 + 实际盈利增长 + 估值均值回归 + 通胀）
         ± 汇率对冲收益（利率平价）
```

现有五个模型里：
- 三个统计模型 = **回看**历史
- BL = 在市场均衡（δΣw）基础上叠加观点，但**观点数字必须从系统外带入**
- 情景混合 = 情景收益**必须从系统外带入**

**没有一个模型能在系统内生成前瞻收益。**

### 其他相关缺口

| 缺口 | 说明 |
|---|---|
| 无期限结构 | 行业 CMA 通常分别发布 10Y / 30Y，building block 构成不同（短期估值回归主导，长期增长主导）。本项目 `horizon_years` 只是一个标签，不影响任何计算 |
| 无通胀/实际收益通道 | `moment_semantics` 只有名义算术；通胀只出现在资金测算（`funding_schedule_kernel` 的 `real_basis`），CMA 本体没有实际收益口径 |
| `fx_hedging_basis` 只是标签 | 融合时校验它必须匹配（`multi_cma.py:91`），但**不参与任何计算**。跨币种资产的对冲收益差没有被建模 |

### 建议

**这是产品定位决策，不是技术债。** 两个合理选项：

**选项 1（推荐，短期）——明确定位为"机器而非预测"**

在产品文档和 UI 上明说：平台提供 CMA 的**生成机制、治理、融合与应用**，前瞻收益判断由分析师带入（通过 BL 观点或情景设定）。

配套动作：
- 把 `historical_statistics` 在 UI 上**降级**为"基线/对照"，不作为可直接采纳的 LTCMA 推荐项（这与 [P2-4](#p2-4) 的政策门禁一致）
- BL 和情景混合作为主推路径，因为它们是唯二能承载判断的入口

**选项 2（中长期）——新增第六个模型 `building_block`**

契约形态大致：
```python
class BuildingBlockRequest(CmaModelContext):
    method: Literal["building_block"]
    risk_free_curve: ...              # 期限结构输入
    blocks: list[ReturnBlock]         # 每资产的构成分解，每块须带 source
    valuation_reversion: ...          # 均值回归假设与回归期
    risk_source: CmaVersionRef        # 协方差仍从统计模型来
```
关键设计原则：**收益与风险解耦**——building block 只生成 `μ`，`Σ` 引用一个已冻结的统计 CMA 版本。这与项目现有的"显式来源、可追溯"哲学一致。

工作量较大（契约 + 内核 + 前端编辑器 + 数据依赖：收益率曲线、股息率、盈利数据），**建议作为独立需求单独立项**，不混在本次修复里。

---

## 附录 A：已验证正确、修复时不得破坏的契约

以下内容经过逐条验算确认正确。**任何修复都不得破坏这些行为**，验收时需要回归验证。

### A.1 数学公式（全部验算通过）

| 模型 | 公式 | 位置 | 结论 |
|---|---|---|---|
| Black-Litterman | `π = δΣw_mkt` | `cma_model_kernels.py:73-80` | ✅ |
| Black-Litterman | `μ_post = π + rf + τΣP'(PτΣP'+Ω)⁻¹(q_ex − Pπ)` | `cma_model_kernels.py:106, 123-128` | ✅ 标准 BL |
| Black-Litterman | `q_excess = q − rf·exposure` 统一处理绝对/相对观点 | `cma_model_kernels.py:105` | ✅ 巧妙且正确 |
| Black-Litterman | Joseph form `M = (I−KP)τΣ(I−KP)' + KΩK'` | `cma_model_kernels.py:124-137` | ✅ PSD 稳定形式 |
| 情景混合 | `Σ = Σpₛ Σₛ + Σpₛ(μₛ−μ)(μₛ−μ)'` | `cma_model_kernels.py` `mixture_moments_kernel` | ✅ 全方差定律，**between 项正确计入** |
| NIW | `κₙ = κ₀+n`，`νₙ = ν₀+n` | `cma_statistical_kernels.py:209` | ✅ |
| NIW | `μₙ = (κ₀μ₀ + n·x̄)/κₙ` | `cma_statistical_kernels.py:219` | ✅ |
| NIW | `Ψₙ = Ψ₀ + S + κ₀n/κₙ·(x̄−μ₀)(x̄−μ₀)'` | `cma_statistical_kernels.py:222-224` | ✅ |
| NIW | `E[Σ] = Ψₙ/(νₙ−N−1)` | `cma_statistical_kernels.py:225-226` | ✅ |
| NIW | μ 边际后验 `t(νₙ−N+1)`，scale `Ψₙ/(κₙ·df)` | `cma_statistical_kernels.py:210-211, 228` | ✅ |
| NIW 先验标定 | `Ψ₀ = Σ_d × s`，`ν₀ = N+1+s` ⟹ `E[Σ] = Σ_d` 精确居中 | `cma_statistical_kernels.py:233-237` | ✅ 优雅且正确 |
| 历史统计 | `Σ_μ = Σ_annual/T`，`half_width = t₀.₉₇₅·√Σ_μ[ii]` | `cma_statistical_kernels.py:185, 190-191` | ✅ |
| 样本协方差 | ddof = 1 | `typed_numba_kernels.py:1256` | ✅ |
| 对数正态矩匹配 | `s² = log(1+σ²/(1+μ)²)`，`m = log(1+μ) − s²/2` | `goal_kernels.py:174-176` | ✅ **精确矩匹配** |
| Wilson 区间 | 成功率的 Wilson score 区间 | `goal_kernels.py:96-105` | ✅ |
| Student-t 分位数 | 正则化不完全 Beta + 二分，**非 1.96 近似** | `cma_statistical_kernels.py:27-101` | ✅ |

### A.2 口径与治理（行业水平以上）

1. **算术/几何陷阱真的避开了**。MV 优化用算术矩（正确口径），多期资金路径用精确矩匹配的对数正态（正确处理方差拖累）。这个陷阱搞砸了绝大多数自研 CMA 系统。**修复 [P1-2](#p1-2) 时绝不能破坏这个分工。**

2. **拒绝修补协方差矩阵**。`covariance_repaired: False` 贯穿全部路径，PSD 失败直接 raise，不做 Higham nearest-PSD 投影。反行业惯例（多数系统静默修补），但对受治理的研究制品是对的——修补后的矩阵已经不是分析师声明的那个假设。

3. **NIW 的数据复用治理**。`prior_mode="continue"` 强制新数据起始日 > 先验样本截止日；`prior_mode="recenter"` 窗口重叠时必须显式 `data_reuse_acknowledged=True`。**显式标记经验贝叶斯的双算**，这个纪律在商业 CMA 工具里基本见不到。

4. **BL 观点的时点语义**。`observed_on ≤ available_on ≤ as_of` 三重校验。多数 BL 实现对观点**完全没有**时间语义。

5. **regime 拒绝为稀有状态编造矩**。正概率状态要求 ≥20 条共同样本（`cma_statistical_models.py:80`），`counts < 2` 的状态直接排除而非补零。这正是朴素 regime CMA 翻车的地方。

6. **regime 的自我批判精确到位**。limitation 写"原样按历史占用率混合、且无收缩时等同于共同分类样本的 ML 矩；不自动增加预测信息"——**数学上完全正确**（全方差定律的直接推论）。

7. **情景概率不自动归一化**。`probabilities_normalized: False`，和不为 1 直接报错。`risk_mode` 强制 shared 或 per-scenario 二选一，不允许隐式混用。

8. **全链路指纹与血缘**。`content_hash` 自校验、`frozen_assumptions` 血缘验证、融合来源校验 `config_hash / nav_hash / strategic_universe_hash / implementation_mapping_hash`，历史不可补算。

9. **融合模式的诚实设计**。参数平均明确排除 `between`（模型权重不是校准概率，`multi_cma.py:150` 取 within，`between` 单独存为 `model_disagreement`），并另建 `compatible_all_models` 模式：每个模型独立求解、全部通过才可采纳。这是该立场下智识上诚实的替代方案，而且真的实现了。

10. **NJIT 纪律**。所有内核固定签名、`disable_compile()`、启动预热、`execution_audit()` 断言 `python_fallback=0` / `request_time_compilation=0` / `object_mode=0`，只读任意 stride ABI。**修复时新增的任何数值路径必须遵守同一套约束。**

### A.3 修复的通用验收清单

任何本文档中的修复，提交前必须满足（依据 AGENTS.md）：

- [ ] 新增数值路径全部走固定签名 `njit`，纳入 `warm()`，`execution_audit()` 断言通过
- [ ] 输入以只读 NumPy 视图传入，`np.shares_memory` 验证无意外复制
- [ ] 与受控参考实现的数值一致性测试：NaN/Inf、空样本、边界窗口、单资产、零方差现金、dtype、确定性
- [ ] 存量冻结 artifact 的 `content_hash` / `frozen_assumptions` / `frozen_policy_assumptions` 校验仍通过（**历史不得重算**）
- [ ] 数学口径变更按新算法版本处理，带 schema 版本位，不伪装成重构
- [ ] 被替代的实现在同一变更中删除，同步清理导入、路由、配置、测试、文档
- [ ] 回归：`backend/tests/test_strategic_allocation.py` 及 `docs/repo_map.json` 中 `strategic_allocation_policy` 模块的 `minimum_regression`
- [ ] 审计字段（`model_audit`）如实反映新方法名与口径，`limitations` 同步更新

---

## 附录 B：问题与代码位置速查

| 编号 | 主要代码位置 |
|---|---|
| P0-1 | `kernels.py:97`（haircut）、`contracts.py:227,372`（penalty=1）、`cma_application.py:34-37` |
| P0-2 | `reference_evidence_kernels.py:82`、`kernels.py:43`、`cma_statistical_kernels.py:270`、`cma_model_contracts.py:164` |
| P1-1 | `cma_statistical_kernels.py:293`（annualize）、`cma_statistical_models.py:87,89` |
| P1-2 | `goal_kernels.py:213,154`（单因子对数正态）、`goal_kernels.py:165-176`（矩匹配，本身正确） |
| P2-1 | `cma_model_kernels.py:97-98`、`cma_model_contracts.py:38-58`、`cma_models.py:88-93` |
| P2-2 | `cma_models.py:122`（未传 mean_uncertainty）、`cma_application.py:34-35`、`contracts.py:309` |
| P2-3 | `cma_model_kernels.py:63-65,118-127`、`cma_models.py:96-103` |
| P2-4 | `cma_model_contracts.py:124,137-157`、`contracts.py:93,280`、`reference_evidence_kernels.py:68` |
| P3-1 | `cma_statistical_kernels.py:270`、`typed_numba_kernels.py:1256` |
| P3-2 | 无（范围缺口） |

---

## 附录 C：勘误与结论修订（2026-09-20 追加）

本节由原审阅者在收到整改回复 `docs/research/cma-model-review-response-2026-09-20.md` 后追加。**正文不作修改**，以保持整改方引用的输入记录可追溯；以下为经独立验算确认的错误与结论降级。

### C.1 确认的错误（我方）

| 位置 | 错误 | 正确内容 |
|---|---|---|
| 附录 A.1 历史统计行 | 写作 `Σ_μ = Σ_annual/T` | 应为 **`Σ_μ = 252·Σ_annual/T`**。因 `μ_annual = 252·r̄_daily`，故 `Var(μ_annual) = 252²·Σ_daily/T = 252·Σ_annual/T`。代码 `cma_statistical_kernels.py:190` 实现正确，是本文档抄录时漏了 252 因子。已数值验证。 |
| P0-2 建议方案 A | 给出 `(1−λ) + λ·r̄/ρ_ij` | **该式在 `ρ_ij = 0` 处未定义**。正确做法只有文中并列的第二种写法：在相关空间做 `R' = (1−λ)R + λ·R_target`，再乘回波动率。第一式作废。 |
| P3-1 建议 | 提议把状态分母改为 `counts[s] − 1` | **建议错误，已撤回**。见 C.2。 |

### C.2 撤回的结论

**P3-1 完全撤回。** 每状态 `ddof=0` 是**必需**的，不是疏漏。以占用率 `n_s/T` 混合 ML 状态矩，可精确重构全样本 ML 矩（经验分布的全方差定律恒等式）。改成 `ddof=1` 会破坏该恒等式——而这个恒等式正是模型那条关键披露（"原样按历史占用率混合、且无收缩时等同于共同分类样本的 ML 矩；不自动增加预测信息"）的数学依据。数值验证：

```
每状态 ddof=0: |混合 − 全样本ML| max = 5.4e-20   ✓ 恒等式成立
每状态 ddof=1: |混合 − 全样本ML| max = 5.0e-07   ✗ 恒等式被破坏
```

若确需无偏口径，正确形式是 `Σ((n_s−1)·S_s + n_s·(μ_s−μ)(μ_s−μ)')/(T−1)`，**不是**单独替换各状态分母。整改方保留原算法并补恒等式测试的处理是正确的。

### C.3 降级的结论

| 编号 | 原表述 | 修订后 |
|---|---|---|
| **P0-2** | "长仓组合风险被**系统性低估**" | **过强。** 方向由加权交叉项符号决定：`Δ(w'Σw) = −λ·Σ_{i≠j} wᵢwⱼSᵢⱼ`。含负协方差（股债负相关等）时组合风险**上升**。数值反例已验证：全正协方差组合 0.019780→0.019172（降），含负协方差组合 0.008820→0.008988（升）。<br>且"相对样本估计下降"≠"相对未知真实风险低估"——收缩本就是以偏差换方差的估计量改进。**保留的有效部分**：目标选择（单位阵 vs 常相关）是有意义的设计决策，固定 λ=0.1 无最优性保证，方向应对研究员可见。 |
| **P1-1** | 以"波动率聚集"论证多期方差被低估 | **因果推理有误。** 放大项来自**状态间条件均值离散度 × 持续性**，与波动率聚集无关：若各状态条件均值相同且创新不相关，状态可以极持续而收益自协方差仍为零。正确表达式 `Var(Σ₁ᴴrₜ) = H·Γ₀ + Σₖ(H−k)(Γₖ+Γₖ')`。<br>**保留的有效部分**：当状态间均值确有差异且链持久时，iid ×252 确实低估多期方差。但放大倍数需由 `P` 与均值离散度共同决定，不能凭持续期单独推算。整改方"只加转移诊断、不自动放大"是稳健处理。 |
| **P1-2** | "**系统性**高估资金成功率" | **方向不固定。** 等矩条件下左偏分布中位数高于右偏对数正态，达标概率的偏差方向取决于目标阈值相对分布位置与现金流结构。<br>另：我提出的"每月按 pₛ 抽状态"会把年度情景改成月度重抽，**隐含改变了模型**（年度情景未定义情景内分布与跨期规则）。该建议作废。**保留的有效部分**：单因子对数正态确实丢失了原生情景尾部信息，属实且应披露。 |
| **P2-4** | 建议"窗口 < 3 年禁止采纳"硬门禁 | **会误伤 NIW。** `prior_mode="continue"` 的正当用法就是在累积后验上叠加一个较短的新证据批次；新增样本长度不代表全部先验信息量。统一硬门禁会阻断合法的序贯贝叶斯更新。降级为**披露 + 标准误可见**，不设硬门禁。 |
| **P2-4 / P3-2** | "真实 LTCMA 行业**从不**用历史样本均值做预期收益" | **过于绝对。** CFA Institute *Capital Market Expectations* 将统计方法、现金流折现、风险溢价模型并列为资本市场预期的三类方法。修订为：**主流公开发布 LTCMA 的机构（JPM、BlackRock、Vanguard、Invesco、RA）以 building-block / 估值锚定为收益端主力，历史统计更多用于协方差**；但统计方法本身是被认可的方法族，不应据此取消其采纳资格。 |
| **P0-1** | "椭球形式**远没那么苛刻**" | **不能承诺。** 高维下按联合置信校准的椭球半径可能比 box 更大，是否更少扣减取决于 κ 校准与组合集中度。**保留的有效部分**：已冻结的 `Σ_μ` 未被消费属实，提供显式椭球选项合理。 |

### C.4 整改方优于原建议之处

- **椭球半径校准**：原建议笼统写 `κ = √(χ²_{N,α})`，这对**协方差需估计**的情形是错的。实现按模型分别校准——BL 高斯 `χ²(d)`、NIW 多元 t `d(ν−2)/ν·F(d,ν)`、Historical Hotelling `d(T−1)/(T−d)·F(d,T−d)`——并明确区分 t 的协方差与尺度矩阵。见 `backend/strategic_allocation/uncertainty_kernels.py:73-99`。这比原建议严谨。
- **多 CMA 椭球明确拒绝**而非猜测跨模型均值误差协方差，与原文 P0-1 第 5 点的担忧一致且处理更保守。

### C.5 独立复核结果（原审阅者执行）

| 项 | 声明 | 独立复现 |
|---|---|---|
| 前端全量单测 | 160 文件 / 1297 项 | ✅ 160 / 1297 通过，exit=0 |
| 后端关联回归 | 41 文件 / 905 项 | ✅ 905 项通过，exit=0（按 `resume-backend-files.json` 清单复跑） |
| CMA 专项 | — | ✅ 131 项通过，exit=0 |
| P0-2 方向性反驳 | 符号依赖 | ✅ 正负两组反例数值复现 |
| P3-1 ML 恒等式 | ddof=0 必需 | ✅ 恒等式验证通过 |

> ⚠️ **测试工具注意**：`vitest` 经 rtk 过滤时会在解析阶段 Rust panic（UTF-8 字符边界截断），输出不可信。前端测试必须走 `rtk proxy npx vitest run` 或直接落盘读取。

### C.6 仍然成立、未被整改覆盖的项

- **P1-2 原生情景/Markov 多期资金引擎** —— 未实现，整改方已明确记录为未完成，非缺陷。
- **多 CMA 椭球** —— 未支持，明确拒绝而非近似，处理正确。
- **P3-2 building-block 模型** —— 独立需求，未纳入本轮。
- **路由治理** —— `validate_ai_routing.py` 全工作树仍失败（既有未跟踪路径 + 本轮 8 个未登记路径）。整改方未标记该项通过，如实分开记录。**提交前必须解决。**
