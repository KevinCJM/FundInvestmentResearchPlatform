# 自动构建大类（auto-classification）设计

节点位置：投前决策 → 战略资产配置（SAA）→ 工具「自动构建大类」，路由 `/pre-investment/saa/auto-classification`。
第 1–6 节是设计，第 7 节记录本次已实现的部分及实现中对设计的三处修正。未实现的能力在第 7 节末尾单独列出，不写成现状。

---

## 1. 节点边界

| | 内容 |
|---|---|
| **拥有** | 从一组产品出发，自动把产品划分为 K 个资产大类，给出每类的代表产品、类内权重和可解释的分类依据 |
| **复用** | 产品净值与指标快照、指数日线、相关性计算（`fit.compute_rolling_corr`）、自定义指标引擎、产品选择器 |
| **不做** | 不决定大类之间的 SAA 权重（那是 `allocation-lab`）、不做产品准入（那是产品池）、不下单 |
| **交付给** | 「大类资产构建」(`ManualConstruction`) 与「配置与回测」(`ClassAllocation`)，即直接产出可保存的 `asset_alloc_info` 结构 |

### 与手工建大类的关系

`ManualConstruction` 已经实现了「人来定义大类 → 选 ETF 代理 → 拟合 → 相关性 → 保存」。
本节点是它的**上游自动化**：人给约束（几类、每类几个），机器给草案，人在 `ManualConstruction` 里改。
所以输出契约必须与 `asset_alloc_info.parquet` 完全对齐：

```
asset_alloc_name  分类方案名（= 本次自动分类版本名）
asset_name        大类名
etf_code / etf_name
etf_weight        类内权重，按大类归一化到 100
```

---

## 2. 输入

### 2.1 产品池（已接入）

本节点只接受**已锁定的不可变可投资域快照**：页面通过 `?universe=<snapshot_id>` 进入，
`POST /api/asset-classes/auto/preview` 必须带 `universe_snapshot_id`，产品清单由
`InvestableUniverseMembership.validate` 逐个核验；任何不在域内或当前不可用的产品都会以
`422 PRODUCT_OUTSIDE_INVESTABLE_UNIVERSE` 拒绝，并逐条列出原因。没有任意代码旁路。

快照同时携带**产品池限额** `max_weight`（(0,1] 的比例，跨池按 `min()` 合并），见 4.3。

### 2.2 用户约束

| 参数 | 说明 | 默认 |
|---|---|---|
| `k` | 大类个数；填 `auto` 时由系统建议 | auto |
| `size_min` / `size_max` | 每类产品数区间。**不建议做成精确值** | 3 / 8 |
| `unassigned_policy` | 装不下的产品：`force`（硬塞最近类）/ `park`（进「待观察」池） | park |
| `window` | 特征计算窗口，沿用 `slice_fit_data` 的 `all` / `rolling{N}` | rolling750 |
| `algorithm` | 见第 4 节 | `corr-hierarchical` |
| `features` | 特征集选择，见第 3 节 | 相关性 |
| `intra_class_weight` | 类内权重法：`equal` / `inv_vol` / `inv_var` / `affinity` | inv_vol |

**为什么 size 是区间不是精确值**：真实产品池里同类产品数量天然不均衡（宽基 ETF 几十只、商品 ETF 个位数）。强行等分会把不相关的产品塞进同一类，分类结论直接失真。区间 + 未归类池是更诚实的做法。用户坚持精确等分时，`size_min = size_max` 即可。

---

## 3. 特征体系

四类特征，可单选也可加权组合。全部先做**截面标准化（robust z-score，中位数/MAD）**，避免量纲和厚尾污染距离。

### F1 收益相关性（默认）

从 `etf_daily_df` / `fund_nav_df` 取复权净值 → 对数收益 → 相关矩阵 `C`。
距离：`d_ij = sqrt(0.5 * (1 - C_ij))`（López de Prado 的 HRP 距离，满足度量公理）。
可选二阶距离 `D_ij = ||d_·i - d_·j||`，刻画「与整体的关系是否相似」，对噪声更稳。

这是默认特征，因为**大类资产的经济本质就是同涨同跌**。分类依据是行为而不是标签，能自动发现「名字叫债基但走得像权益」的产品。

### F2 风险收益画像

直接读 `instrument_metrics_snapshot.parquet`（已有 38 列）：
`return_1m/3m/1y/3y, annual_volatility_1y, max_drawdown_3y, sharpe_1y, calmar_3y, premium_discount_latest, amount_avg_20d`
加质量列 `coverage_ratio_*` / `stale_days` 做样本准入门槛。

### F3 合同与基准标签（类别特征）

`etf_info_df` / `fund_info_df` 的 `fund_type, invest_type, benchmark, index_code, index_name, m_fee, c_fee, market`。
`benchmark` 字段形如 `沪深300指数收益率×80%+中债综合×20%`，可解析成基准权重向量，是**信息量最高的单一特征**——它是合同写死的资产暴露。
类别特征用 Gower 距离或 one-hot 后与数值特征加权（数值用欧氏、类别用汉明）。

### F4 风格暴露（RBSA）

对指数因子池（`index_daily_df` + `index_sw_daily_df` 的宽基/行业/债券/商品/海外指数）做 Sharpe 强式风格分析：

```
min_w  Var(r_fund - Σ w_i * r_index_i)
s.t.   w_i >= 0,  Σ w_i = 1
```

输出的暴露向量本身就是一条可读的分类依据（「该基金 78% 像中债综合」），也是最适合喂给树模型的特征。

---

## 4. 算法库

**核心架构决策：所有算法只负责产出亲和度矩阵 `A[n_products, k]`，大小约束由统一的指派求解器处理。**

这样新增算法不用碰约束逻辑，四类算法共用一套诊断、权重、版本化和 UI。

```
产品池 ──> 特征矩阵 X / 距离矩阵 D
              │
              ├── A. 规则映射     ──┐
              ├── B. 无监督聚类   ──┤
              ├── C. 回归/降维    ──┼──> 亲和度 A[n,k] ──> 容量约束指派 ──> 类别标签
              └── D. 树模型       ──┘                        (size_min/max)      │
                                                                                  ▼
                                                          类内权重 ──> asset_alloc_info 草案 ──> 诊断 ──> 人工确认 ──> 版本
```

### A 类：规则映射（基线，已实现三级分类）

按 `fund_type` / `invest_type` / `benchmark` / `index_name` / 产品名做确定性映射，`A[i,k] = 1` 或 `0`。
分类表在 `backend/fund_taxonomy.py`，**三级**，对齐国内已经收敛的行业口径：

| 层级 | 内容 | 依据 |
|---|---|---|
| 一级·资产大类 | 权益 / 固收 / 商品 / 货币 / 海外 / 混合 / 其他 | 银河证券公募基金分类体系的顶层，也是 SAA 真正配置的那一层 |
| 二级·细分类型 | 权益：规模(宽基) / 风格因子 / 行业 / 主题；固收：利率债 / 信用债 / 可转债 / 同业存单；海外：港股 / 美股 / 其他海外；商品：贵金属 / 能源 / 农产品 / 有色 | 易方达 2025 年发布的股票指数 ETF 四分法 + 交易所债券 ETF 口径 |
| 三级·风格/行业/主题 | 宽基：大盘 / 中盘 / 小盘 / 全市场 / 科创板 / 创业板 / 北证；风格因子：红利 / 低波 / 价值 / 成长 / 质量 / 动量 / 等权 / 基本面 / 自由现金流；行业八大：金融 / 医药 / 科技 / 消费 / 制造 / 周期 / 公用事业 / 房地产；主题六大：数字化与人工智能 / 高端制造 / 低碳转型 / 国企改革 / 人口趋势 / ESG；债券：国债 / 政金债 / 地方债 / 城投债 / 公司债 / 短融 / 科创债 | 中证/申万行业分类 + Smart Beta 常见因子 + 易方达八大行业六大主题 |

匹配是**有序关键词包含，首行命中即止**，不是模型：合同标签必须可复现、可解释、可上会。
关键的顺序设计（都有回归测试）：

- 风格因子排在行业和宽基之前 —— 「中证500低波动」是因子产品，不是中盘宽基。
- 主题排在行业之前 —— 主题天然跨行业（人工智能、光伏），行业是单一行业。
- 借用商品词汇的权益行业基金排在商品之前 —— **「黄金股 / 有色金属」是权益周期行业，不是商品**（旧的扁平表在这里是错的）。
- 债券词排在权益之前 —— 「科创债」是信用债，不是科创板。
- 长指数名排在短的之前 —— 「中证1000」也包含「中证100」。

**作用**：零配置基线、其它算法的对照组、下面 4.4 分层聚类的硬约束来源、D 类树模型的初始标签来源。

### B 类：无监督聚类

| 算法 | 用什么 | 特点 |
|---|---|---|
| **相关性层次聚类** ⭐ | D（F1 距离），ward / average linkage，切树到 K | 默认。无需初值、结果确定、树状图天然可解释「为什么这两个在一起」 |
| **K-medoids (PAM)** | D | 类中心是**真实产品** → 直接得到「代表产品」，正是 SAA 建大类要的东西 |
| **K-means** | X（F2/F4 标准化特征） | 快，但质心是虚拟点，且对量纲敏感；作为对照 |
| **GMM 软聚类** | X | 输出归属概率 → 天然的亲和度矩阵 + 边界产品提示（「该产品 45% 像权益、40% 像转债」） |
| **DBSCAN / HDBSCAN** | D | 不强制分配，专门用来**捞离群产品**（分级 B、打新、纯套利），进「待观察」池 |

推荐默认 `corr-hierarchical`，备选 `kmedoids`。

### C 类：数学回归 / 降维

- **RBSA 直接分类**：按 F4 暴露向量的最大分量归类，`A[i,k] = w_ik`。金融解释性最强。
- **PCA / 特征值分解**：对相关矩阵做谱分解，前几个主成分的载荷符号与大小分类；同时用 **Marchenko–Pastur 上界**统计显著特征值个数，作为 K 的建议值——这是把「该分几类」从拍脑袋变成有依据的关键一步。
- **去噪**：MP 去噪 / 收缩估计后的相关矩阵再进聚类，样本期短时明显更稳。

### D 类：树模型（第二阶段）

需要标签，因此排在 A/B/C 之后：用 A 类规则或人工确认过的历史 `asset_alloc_info` 作训练标签，训练 **决策树 / 随机森林 / 梯度提升**，用于：

1. 给新产品打类（增量分类，不用重跑全量聚类）
2. **输出可读规则路径**：`m_fee > 0.6% 且 annual_volatility_1y > 18% → 权益类`
3. 特征重要性，回答「到底是什么特征在决定分类」

树模型在这里的真正价值不是分类精度，是**把聚类结果翻译成投委会能审的规则**。

### 4.1 K 的自动建议

`k = auto` 时同时算并展示，让用户看着选而不是系统硬定：

- 轮廓系数（silhouette）峰值
- Gap statistic
- 相关矩阵显著特征值个数（MP 上界）
- 树状图最大合并高度间隔（elbow on merge distance）

### 4.2 容量约束指派

拿到 `A[n,k]` 后，求解：

```
max  Σ_ik  A[i,k] * x[i,k]
s.t. Σ_k x[i,k] <= 1                    每个产品最多进一类
     size_min <= Σ_i x[i,k] <= size_max  每类容量
     x ∈ {0,1}
```

这是**带容量的指派 / 最小费用流**问题，多项式可解。实现路径：

- 把每个类展开成 `size_max` 个槽位 → 退化为标准指派问题（`n × k*size_max`），
  用匈牙利算法或 auction 算法求解（balanced k-means 的经典做法）。
- 未被指派的产品按 `unassigned_policy` 处理。
- 规模考虑：手动池通常 `n < 500`，`k*size_max < 100`，O(n³) 完全可接受。

**这一步是把「用户想控制类数和类大小」这个产品需求，和「聚类算法本身不保证类大小」这个数学事实解耦的地方。**

### 4.3 产品池限额约束

产品池可以给单个产品设 `max_weight`（组合层面的权重上限）。本节点产出的是**类内权重**，
组合权重 = 大类权重 × 类内权重，而大类权重 ≤ 1，所以**把类内权重压在 `max_weight` 之内，就足以保证组合层面不越界**——这是保守且无需知道 SAA 权重的做法。

`apply_weight_caps_kernel` 分两种情形：

| 情形 | 处理 | 输出 |
|---|---|---|
| 类内限额之和 ≥ 100% | 注水法：触顶的产品钉在限额，剩余额度按原权重比例分给未触顶的，迭代至收敛 | 权重和 100%，逐个不超限；`max_class_weight = 100%` |
| 类内限额之和 S < 100% | 该大类无论如何都填不满而不越界，改为**按限额成比例**分配 | 权重和仍 100%；`max_class_weight = S`，即该大类的 SAA 权重上限 |

第二种情形的性质：令 wᵢ = capᵢ/S，则对任意大类权重 W ≤ S 都有 W·wᵢ ≤ capᵢ。
所以输出的不是"违规权重"，而是"带使用条件的权重"——条件即 `max_class_weight`，直接交给下游 SAA 使用。

实测：黄金 ETF ×2 各限 12% → 类上限 24%；沪深300 限 5% → 类内权重被压到 5%，释放的额度按原比例分给同类其它宽基。

---

### 4.4 合同分层 + 类内统计聚类（`block_by`，已实现）

纯统计聚类有一个结构性问题：**相关性窗口会骗人，合同不会**。2020 年以来黄金 ETF 和权益 ETF 有过长达半年的高相关期，一个只看收益的算法会把它们放进同一个大类；而在 SAA 语境下，这个大类是不能拿去做资产配置的。

因此加入国内 FOF 实务里的「先分层、再聚类」：

```
产品池 ──> 合同分类（一级/二级/三级，用户选）──> 硬分块
                                                 │
              ┌──────────────────────────────────┴──────────────────────────────┐
              ▼                                                                  ▼
      块内距离矩阵 D_b ──> 块内统计聚类（层次/K-medoids/K-means）──> 块内类标签 ──> 全局偏移
```

实现要点：

1. **K 在块之间的分配**：`allocate_block_clusters_kernel`（NJIT，定长签名）。每个非空块**至少保留 1 类**——分类层级是硬分区，宁可把 K 抬上去，也不允许为了凑 K 把两个合同块合并；剩余的 K 按「产品数 / 当前类数」最大者依次分配（highest averages），确定性、无需 tie-break。块容量为 `size // size_min`，K 超过总容量时被裁到总容量。
2. **自动 K 时按块单独判断**：块内轮廓系数超过 **0.25**（Kaufman/Rousseeuw 的「有实质结构」下限）才拆分，否则该块整体保持 1 类。没有这条阈值，5 只沪深300 ETF 会被切成两半互相像的类。
3. **容量指派不得跨块**：`capacity_assign_kernel` 会跳过非有限亲和度，所以跨块的 `A[i,k]` 直接置 `-inf`。这一条同时挡住了 `size_min` 回填和 `force` 策略——否则「强制归入最相近大类」会悄悄把债券 ETF 塞进权益类。
4. **诊断**：`diagnostics.blocks` 逐块返回 `block / size / k / silhouette`，前端单独一张表。分层后 `contract_deviations` 按构造为空，这是符合预期的。

`block_by = none` 时全部行为与分层上线前完全一致（有回归测试）。

## 5. 输出与诊断

### 5.1 分类结果

每个大类给出：类名、代表产品（medoid）、成员列表 + 类内权重、类内平均相关、与其它类的相关。

**自动命名**：取类内成员 `invest_type` / `index_name` 的众数 + 风险画像修饰，如「权益-宽基」「固收-利率债」「商品-贵金属」。命名永远可人工改写。

### 5.2 类内权重

| 方法 | 用途 |
|---|---|
| `equal` | 最朴素，先给这个 |
| `inv_vol` | 默认，`w_i ∝ 1/σ_i` |
| `inv_var` | 与 HRP 一致 |
| `affinity` | 按 `A[i,k]` 归一，越像这一类给越多 |

代理组合的净值直接送 `/api/fit-classes` 复算指标和相关性——这条链路已经存在，不重写。

### 5.3 诊断（决定这套东西能不能上会）

1. **可分性**：轮廓系数、类内/类间相关比、Calinski-Harabasz
2. **稳定性**：滚动窗口重跑，用 **调整兰德指数（ARI）** 衡量前后两期分类的一致性；bootstrap 重采样看成员归属的稳定概率
3. **时间漂移**：逐产品记录跨期换类事件，直接对接产品研究的「风格漂移」监控
4. **与合同标签的偏离清单**：行为分类 ≠ 合同分类的产品，逐条列出——**这是本节点最有业务价值的输出**，是研究线索而不是错误
5. **未归类池**：离群 / 数据不足 / 装不下的产品，附原因码

### 5.4 版本与治理

沿用产品池生命周期的模式：分类方案不可变版本化，记录算法、参数、特征集、数据截止日、样本窗口、人工覆盖记录。人工覆盖是一等公民，必须能一条条记原因并在下次重跑时保留。

---

## 6. 工程约束（AGENTS.md 的 NJIT 要求）

依赖里没有 sklearn，且 `AGENTS.md` 要求所有数值逻辑走 `njit`。因此：

| 计算 | 实现 |
|---|---|
| 收益矩阵、相关矩阵、距离矩阵 | njit kernel（`fit_numba` 已有 `returns_from_nav_kernel` / `rolling_correlation_kernel`，可直接复用） |
| 层次聚类 linkage | njit，nearest-neighbor chain，O(n²) |
| K-means / K-medoids Lloyd 迭代 | njit |
| RBSA 约束 QP | scipy 求解器（AGENTS.md 允许第三方黑盒），但前后的矩阵准备、残差与约束校验走 njit |
| 指派问题 | njit auction 算法，或 `scipy.optimize.linear_sum_assignment` |
| 轮廓系数 / ARI / Gap | njit |

启动阶段必须预热全部签名，禁止请求期编译。每个 kernel 配一致性测试（对照受控参考实现 + NaN/Inf/空样本/单产品/单类边界）。

---

## 7. 已实现（本次交付）

### 后端

| 文件 | 内容 |
|---|---|
| `backend/auto_class_numba.py` | 21 个 fixed-signature NJIT 内核：稳健标准化、winsorize、相关/距离矩阵、PCA 载荷与特征值、Lance-Williams 层次聚类与切树、K-means、K-medoids、亲和度、容量选择、轮廓系数、类内/类间相关、类内权重、产品池限额注水、**分层 K 分配** |
| `backend/fund_taxonomy.py` | A 股公募/ETF 三级合同分类表与匹配器（见 4.A），纯查表、可复现、可解释 |
| `backend/auto_asset_class.py` | 编排层：产品解析、特征装配、K 建议、分类、**合同分层**、命名、诊断与序列化 |
| `backend/services/auto_class_routes.py` | `GET /api/asset-classes/auto/meta`、`POST /api/asset-classes/auto/preview` |
| `backend/app.py` | 启动预热接入 `warm_auto_class_numba_kernels()`，readiness 覆盖新链路 |
| `backend/tests/test_auto_asset_class.py` | 130 条测试：内核对照参考实现、边界、确定性、编排、路由、**三级分类表与合同分层** |
| `backend/tests/conftest.py` | 统一 `sys.path`；并把测试的 `NUMBA_CACHE_DIR` 隔离到 `.numba_cache/tests` |

算法：`rule` / `hierarchical`（average、complete、ward）/ `kmedoids` / `kmeans`。
特征：`correlation` / `metrics` / `pca` / `blend`。
类内权重：`equal` / `inv_vol` / `inv_var` / `affinity`。
合同分类层级 `taxonomy_level`：`asset_class` / `category` / `detail`（决定大类命名与 `rule` 算法粒度）。
合同分层 `block_by`：`none` / `asset_class` / `category` / `detail`（统计聚类不可跨越的硬约束）。

### 与第 4 节设计的三处修正（实现中发现并改掉）

1. **容量约束从「全局带容量指派」改为「类内 top-N 选择」。**
   按原设计做全局指派时，权益类溢出的产品会被塞进还有空位的固收类，实测产出「国债 ETF + 银行 ETF」这类相关系数 0.02 的伪大类。
   现在的语义是：聚类结果不被推翻，每个大类按亲和度保留最有代表性的 N 个，溢出进待观察池。轮廓系数从 0.46 升到 0.83。

2. **`size_min` 回填加了相似度下限。**
   自然聚类出的单产品大类，若强行凑到 `size_min`，只会拉进当时恰好没被占用的无关产品。
   现在只接受比「池内平均距离」更近的候选；凑不满就如实报告「只有 N 个足够相似的产品，平台不会为了凑数塞入不相关产品」。

3. **聚类特征加了 25σ 稳健截尾。**
   实测 `511010.SH`（5 年国债 ETF）在 2026-06-02 存在 **−99% 的单日复权净值跳变**（复权净值由 1.00 塌到 0.012），年化波动率被算成 39%，与所有产品的相关性被打成 0，导致两只国债 ETF 无法归为一类。
   阈值取 25σ 是因为实测中国 ETF 真实肥尾最高约 18σ，而这类未复权跳变在数百 σ 量级（该点约 1371σ）。修复后两只国债 ETF 相关性 0.03 → 0.88，逆波动权重也从 5/95 恢复为合理的 56/44。

   **截尾只作用于聚类特征，展示链路仍走原始序列**——这是刻意的：不静默篡改对外报告的业绩。代价是 `/api/fit-classes` 会照原样算出「固收类年化 −5.20%、波动 21.82%、回撤 −55.24%」。
   因此第 ⑤ 节在渲染净值与指标前会点名受影响的大类和产品，明确提示这些数字不可直接采信。
   平台既有的 `series_quality` 质量检查其实已经标记了该产品（`adj_nav_anomaly_count=1`、`quality_reason_1y='adjusted_nav_anomaly'`，指标快照里相应字段为 NaN），但 `/api/fit-classes` 这条展示链路并不读取该标记，属于既有缺口。

另外，轮廓系数对单点簇按标准约定记 0 分而不是跳过——跳过会让「切成一堆单产品簇」得到虚高分数，把自动 K 建议一路推向最大值。

### 入口修复

`ProductPoolSelection` 锁定可投资域后的跳转按钮原先指向 `/pre-investment/saa/manual-construction?universe_snapshot=...`：
该路由不存在（真实路由是 `saa/asset-classes`），且查询参数名与页面读取的 `universe` 不一致。
两处都错的结果是点击后被通配路由兜回首页，锁定的可投资域被静默丢弃。已修正路径与参数名，并补上「进入自动构建大类」入口——在此之前本节点没有任何携带 `?universe=` 的入口，等于不可用。
原有测试只断言按钮存在、不断言跳转目标，所以没拦住；已补跳转断言。

同时修掉两处会在运行时抛错的可选字段直取：`universe.summary.eligible_count`、`snapshot.version_ids.length` / `snapshot.groups.length`（`summary`/`version_ids`/`groups` 在类型里都是可选，产品池流程返回的快照确实可能没有）。

### 已知运维坑

`pytest backend/tests` 把内核作为 `backend.<pkg>.<mod>` 导入，而 `uvicorn app:app`（在 `backend/` 下运行）导入同一份源码时叫 `<pkg>.<mod>`。
Numba 的 `cache=True` 会把导入时的模块名写进缓存环境，于是先跑测试、再起服务，会在 numba 缓存加载器深处抛出误导性的
`ModuleNotFoundError: No module named 'backend'`（堆栈指向 `portfolio_regime.py`，与真实原因无关）。
已通过给测试单独的 `NUMBA_CACHE_DIR` 隔离；根治需要统一全仓库的导入命名，属于打包约定层面的改动，不在本次范围。

### 大类净值拟合的执行证明校验（已修复）

「大类净值与指标计算失败：自动大类净值拟合未提供有效的固定签名 NJIT 执行证明」。

`/api/fit-classes` 的 `execution` 不是单个审计对象，而是两条并列通道：`{ fit_analytics, performance_metrics }`（见 `backend/fit.py::compute_nav_performance_payload`）。前端却用 `assertFixedNjitExecution()` 当成扁平对象校验，取 `execution_backend` 得到 `undefined`，必然失败关闭。两条通道各自都是合规的固定签名 NJIT 审计，缺的只是校验方式。

修复：`frontend/src/utils/fixedNjitExecution.ts` 增加 `assertFixedNjitExecutionLanes()`——扁平审计照旧单条校验，通道映射则逐条校验并在报错里带上通道名；空对象和数组一律拒绝，不放松任何一条原有判据。自动构建大类与手动构建大类两个调用点都改用它。

**手动构建大类的同一条链路也是坏的**（`ManualConstruction.tsx` 的「资产大类拟合」），一并修掉。它之所以没被发现，是因为 `ManualConstruction.test.tsx` 根本没有覆盖 `/api/fit-classes`；而 `AutoAssetClassification.test.tsx` 里这个接口的 mock 用的是并不存在的扁平形状。两处都已按真实响应形状改正，并在 `fixedNjitExecution.test.ts` 补了双通道用例。

### 可投资域快照读写不一致（已修复）

「运行自动分类」返回 `未找到指定可投资域快照。`，但同一个快照的 `GET /api/investable-universes/{id}/products` 正常返回 20 个产品。两个缺陷叠在一起：

1. **读错文件。** 快照由 `ProductPoolService` 追加写进 `data/product_pools.json` 的 `universe_snapshots`，而 `InvestableUniverseMembership` 的两个调用方（`services/auto_class_routes.py`、`custom_indicators/portfolio_service.py`）都去读 `data/investable_universes.json` —— 一个从来没有人写过的空库，目录里只留下了它的 `.lock`。
2. **读错形状。** 即便路径对了，落盘记录用的是 `products` 键；`members` 是 `ProductPoolService._decorate_universe_snapshot` 在读接口里现算的。`validate()` 直接读原始记录的 `members`，得到空列表，于是每个产品都会被判成 `not_in_universe`；就算命中了也因为缺 `eligible` 字段而被判 `not_eligible`。

修复：`product_pools/constants.py` 里用 `UNIVERSE_SNAPSHOT_STORE` 统一存储文件名，两个调用方都改用它；`InvestableUniverseMembership._members()` 兼容两种落盘形状，用领域层已有的 `member_eligibility()` 计算准入（快照里的产品必然已通过产品池准入，`research_status` 取 `approved`）；`_reference()` 在没有 `version_refs` 时用并列的 `version_ids`/`pool_ids` 还原血缘，否则真实快照的审计链会是空的。

**为什么测试没拦住：** `test_auto_asset_class.py` 和 `test_portfolio_research.py` 的夹具都直接写装饰后的 `members` 形状，没有任何一条用例走过「产品池服务真实落盘 → 下游校验」这条链路。已补 `_write_live_shaped_universe()` 夹具与三条用例（真实形状可运行、真实形状仍能拦住不可用产品、路由与产品池服务指向同一个存储文件）。

**遗留：** 真实快照没有 `content_hash`（`create_universe_snapshot` 不写），所以血缘引用里该字段为 `None`。属于写入侧的审计缺口，未改。

### 前端

`frontend/src/pages/AutoAssetClassification.tsx` 取代原静态占位页；`frontend/src/components/ClassFitPanel.tsx` 抽出净值走势、相关系数热力图、横向指标对比、收益风险象限与一致性表，与手动构建大类使用同一套展示。
分类完成后自动调用既有 `/api/fit-classes` 计算各大类虚拟净值与指标，结果可直接保存为大类配置或经 `sessionStorage` 交接到手动构建大类。

### 真实数据验证

20 只 2019 年前上市的 ETF、2020-01-01 起 1619 个交易日、`k=auto`：

| 大类 | 成员 | 类内相关 |
|---|---|---|
| 权益类 | 上证50、沪深300×2、消费80、中证500 | 0.85 |
| 固收类 | 5 年国债、10 年国债 | 0.80 |
| 商品类 | 黄金 ETF ×2 | 1.00 |
| 海外类 | 纳斯达克100 ×2 | 1.00 |

自动 K = 4，整体轮廓系数 0.734，单次运行 0.35–0.47 秒。**这四类完全由净值行为得出，没有使用任何合同标签。**

### 尚未实现（按第 4 节顺序）

RBSA 风格暴露特征、GMM 软归属、HDBSCAN 离群识别、树模型规则导出、滚动窗口 ARI 稳定性、分类方案版本化与人工覆盖记录。

## 参考

- [Returns-based style analysis (Sharpe 1992)](https://en.wikipedia.org/wiki/Returns-based_style_analysis)、[Sharpe: Asset Allocation — Management Style and Performance Measurement](https://web.stanford.edu/~wfsharpe/art/sa/sa.htm)
- [Hierarchical Risk Parity — QuantPedia](https://quantpedia.com/hierarchical-risk-parity/)、[An Empirical Evaluation of Distance Metrics in HRP](https://link.springer.com/article/10.1007/s10614-025-10848-w)
- [Clustering algorithms for Risk-Adjusted Portfolio Construction (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S187705091730772X)
- [Cluster-Based Mutual Fund Classification and Price Prediction for Robo-Advisors (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709763/)
- [Identifying investment fund cohorts through clustering — ESRB](https://www.esrb.europa.eu/pub/pdf/occasional/esrb.op30~18a83682c6.en.pdf)
- [Balanced K-Means for Clustering (Malinen & Fränti)](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_4)、[Balanced clustering](https://en.wikipedia.org/wiki/Balanced_clustering)
- [Determining the Optimal Number of Clusters (Datanovia)](https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/)
- [Machine Learning and Fund Characteristics Help to Select Mutual Funds (Oxford-Man)](https://oxford-man.ox.ac.uk/wp-content/uploads/2023/07/Javier-DGNS_MachineLearningMutualFunds.pdf)
