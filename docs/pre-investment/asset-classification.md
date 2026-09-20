# 自动构建资产大类

本主题负责 Product-first 路径中的产品分组、类内权重和诊断，向手工构建与 SAA 交接草案；不负责产品准入、跨大类 SAA 权重或交易。主流程见[投前研究](README.md)。

## 输入与输出契约

页面以 `?universe=<snapshot_id>` 进入，`POST /api/asset-classes/auto/preview` 必须提供不可变 `universe_snapshot_id`；产品通过 `InvestableUniverseMembership.validate`，不在域内或当前不可用的成员失败关闭。存储统一读取产品池真实快照，支持其原始 products 形状，不能要求只存在于 API 装饰响应中的 members。

API 当前默认 `algorithm=hierarchical`、`features=correlation`、`linkage=average`、`sizeMin=2`、`sizeMax=8`、`unassignedPolicy=park`、`weightMode=inv_vol`、`taxonomyLevel=asset_class`、`blockBy=none`；`k=null` 表示建议类数。`asOf/runMode/dataReleaseId` 未显式提供时由统一 PIT 上下文解析，不提前写默认值覆盖系统设置。枚举由引擎注册表提供。

输出与 `asset_alloc_info` 对齐：分类方案名、大类名、产品 code/name、按类归一到100的类内权重及类上限。分类草案可保存，或交给手工构建编辑；调用既有 `/api/fit-classes` 展示净值、风险与相关性。执行证明为 fit_analytics/performance_metrics 两通道，必须逐通道验证。

## 当前算法与容量

- 算法：rule、hierarchical（average/complete/ward）、kmedoids、kmeans、spectral、gmm。
- 特征：correlation、denoised、metrics、pca、blend；类内权重：equal、inv_vol、inv_var、affinity。
- 相关距离为 `sqrt(0.5*(1-C))`。MP去噪只改变聚类几何，原始相关与特征值诊断不改写；样本不足及全信号/全噪声边界按引擎处理。
- 规则按三级合同分类有序匹配；风格优先于行业/宽基、主题优先于行业，黄金股为权益、科创债为债券、长指数名优先，避免关键词碰撞。
- GMM 概率留在内核用于 argmax；亲和度仍为负距离。展示置信度须独立字段，不能把概率混入容量/权重的距离契约。
- 容量采用**每类按亲和度保留 top-N**，溢出进入待观察；最小数量回填须满足相似度门槛，不能为凑数把无关产品移入。早期全局带容量指派方案已被否决。
- 25σ截尾只用于聚类特征；原始展示序列不被静默修正，净值质量异常必须提示。单点簇的 silhouette 记0，避免鼓励大量单点簇。

### 产品池限额约束

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

### 合同分层 + 类内统计聚类（`block_by`，已实现）

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

## 运行与数据边界

编排位于 `backend/auto_asset_class.py`，数值位于 `backend/auto_class_numba.py`，路由及注册枚举位于 `backend/services/auto_class_routes.py`。固定签名启动预热后才可服务；测试隔离 NUMBA_CACHE_DIR，避免测试和运行导入命名不同造成缓存引用错误。

产品池成员限制、研究日、可得时点和 data_release_id 必须随输入传播；统计分类不能消除产品池本身的前视风险。普通 QP、指派和回归不因使用第三方库而豁免 NJIT；旧文档的 scipy 黑盒豁免说法已撤销，具体约束以 AGENTS 为准。

## 历史证据与未完成项

早期20只ETF、2020年起1619交易日的样本得到4类、silhouette 0.734，单次0.35–0.47秒；它只是一组历史数据验证，不证明普遍分类质量。原型修复涵盖跳转参数、可选字段、双通道执行证明和产品池真实落盘形状。

保留的缺口：RBSA暴露特征、GMM软归属展示、HDBSCAN、树模型规则导出、滚动ARI、分类方案版本化与人工覆盖记录。旧记录另指出 universe_snapshot 的 content_hash 缺失及 fit-classes 未消费 series_quality 标记；本次没有取得它们当前已关闭的证据，须专项复验，不能直接标完成。

稳定性、漂移和人工覆盖属于后续治理目标，不能把保存一份 asset_alloc_info 草案称为已经具备完整分类版本治理。
