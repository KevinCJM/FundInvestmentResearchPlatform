# 沪深300主趋势：实时识别与 CMA 研究校验

日期：2026-09-15。分支：`ISSUE2609/BetterSaaTaa`。本文记录当前最终实验口径；不是收益预测，也不代表模型已取得前瞻部署资格。

## 1. 固定历史参考

实时模型只识别既有 `market-trend-reference-csi300-v1`，不改参考标签：沪深300月频末值 → 局部峰谷 → Pagan-Sossounov 式阶段/周期过滤 → 完整波段收益；>15% 为牛，<-15% 为熊，其余为震荡，首尾未完成区间保持未分类。

实时模型只能读取当时已经闭合的月份，不使用未来峰谷、历史参考状态或参考概率作为特征。历史参考仅用于事后校验与概率校准。

## 2. 常数参数语义修复

`source.constant` 现在默认是可调研究参数，可参与参数扰动和稳定性检查。只有显式 `parameter_role=structural` 才固定，例如 `价格/SMA - 1` 中的常数 1。

因此：

- ±4% 牛熊门槛是可变参数；
- SMA 窗口是可变参数；
- 公式恒等常数 1 显式固定；
- 不再按“值恰好等于1”猜测它是否结构常数；
- `parameter_role` 只影响诊断范围，不改变数值、因果或时点语义。

## 3. 实时模型搜索协议

没有新增黑盒识别算子。候选全部由现有可编辑算子组成：

`指数 → 月频末值 → SMA → 价格/SMA → 减1 → 上下常数门槛 → 三状态分类`

对 SMA 4–12 月、对称缓冲 1.5%–5% 做有界搜索，共72个候选。使用三个扩展式时间折，每折的校准器只使用此前已经位于历史轴上的标签，再评价下一段：

1. 截至2014年校准 → 2015–2018评价；
2. 截至2018年校准 → 2019–2022评价；
3. 截至2022年校准 → 2023–2026评价。

固定 CMA 研究门槛：校准置信度 ≥60% 才接受状态；每折接受样本匹配率 ≥70%；每折接受覆盖 ≥35%；每折校准 Brier 必须优于同期历史类别基准。选型目标为 `最差折匹配率 × sqrt(最差折覆盖)`，避免只优化单一区间。

另测试现有 `model.trend_regime` 组合。它在部分时间折出现零接受样本或 Brier 恶化，未稳定超过简单 SMA，因此不新增更复杂算子。现有 `model.ensemble` 的旧概率契约在成员未分类时可能出现概率质量不足，可靠性服务会正确拒绝；本需求没有为了搜索结果改变该存量算子。

## 4. 最终模型

最终选中：**9个月 SMA + ±4% 缓冲 + 60% 校准门槛**。

公式：

`d_t = P_t / SMA_9(P)_t - 1`

- `d_t > 4%`：Bull；
- `d_t < -4%`：Bear；
- 其余：Sideways；
- 校准后 `P(reference = predicted state | predicted class) < 60%`：不接受当前状态。

三折结果：

| 时间折 | 高置信匹配率 | 高置信覆盖 | Brier 相对类别基准改善 |
| --- | ---: | ---: | ---: |
| 2015–2018 | 82.35% | 35.42% | +0.0764 |
| 2019–2022 | 75.76% | 68.75% | +0.0987 |
| 2023–2026 | 73.68% | 46.34% | +0.0919 |

最差折仍为 **73.68% 匹配率、35.42%覆盖，且三个折的 Brier 均改善**。这些滚动时间折用于候选选择与稳健性比较，因此只能说明 SMA9±4% 是当前较好的**因果实时候选**，不能再把同一批折作为独立最终验收并声称 CMA 已验证通过。

最终状态级验收单独使用 2023–2026 holdout，并按独立完整 Regime Episode 计数。该段 Bull / Sideways / Bear 只有 **1 / 1 / 0** 个完整独立区间，低于默认3个门槛，因此三个状态均为 `Insufficient Evidence`。这不是算法 Failed，而是最终独立状态周期不足。

## 5. CMA 使用约束

CMA 不应强制采用每个月的三分类。推荐契约：

`Realtime Regime → Calibrated Confidence`

- 只有 `State Verification = Verified`（真实生产还需 `Prospective qualified`）且 Confidence ≥60%：才允许使用对应状态的 regime-conditioned CMA；
- `Insufficient Evidence / Failed`、Confidence <60%、未知、冲突或校准失效：**回退到 Base / 无条件 CMA**；
- 未验证状态的概率质量不得重新归一化到其他已验证状态；
- 不把 one-hot 状态值1解释为100%可信；
- 不把历史匹配概率解释成“市场真实处于该状态的客观概率”；
- CMA 仍须保留原始长期假设与完整来源血缘。

当前最后完整月（2026-08-31）模型原始状态为 Sideways，但校准匹配概率约 **18.18%**，低于60%，因此当前应明确回退到无条件 CMA，而不是使用震荡条件化 CMA。

## 6. 稳定性

最终模型的默认稳定性检查：

- 上界 4% → 4.4%：状态一致率约98.44%；
- 下界 -4% → -4.4%：状态一致率约98.44%；
- SMA9 → SMA10：状态一致率约94.76%；
- 历史截尾：一致率100%；
- 结构常数1不参与扰动，并在报告中明确列出。

稳定性只说明小幅参数变化下结果不剧烈漂移，不等同于预测准确率。

## 7. 当前系统对象

当前系统已保存：

- 历史参考定义：`regime-9a005c98d024497da3fea853aea8546d` r1；
- 历史质量报告：`reference-quality-397d581e2299f84dadb3f9bc9394af01b01ea60b2fe6f227596447c59b479255`；
- 历史完整独立区间：Bull **6**、Sideways **3**、Bear **5**；三类达到默认3 Episode 的基础条件估计门槛，但 CMA 仍应对小样本做 shrinkage。
- 实时定义：`regime-461c1de9243f427787be0c412bb97627` **r3**；
- 名称：`沪深300主趋势 · 实时SMA9（4%缓冲，CMA研究）`；
- 状态级校准/验证报告：`reliability-36a8e68a1597457b72caa3286ab58f10c603b0223dd28b04674e0c70fbbcf850`；
- r3 正式研究运行：`regime-run-09dd77dc4b764d21a5e97f211a53e270`；
- r3 只更新精确 `calibration_id` 绑定，没有改变识别数学；
- 当前 `verification.status=insufficient_evidence`、`cma_research_ready=false`、`deployment_eligible=false`。

入口：**设置 → 情景算法中心 → 实时状态识别 → 我的工作区算法**。

前端现在允许一个新修订精确读取自己显式绑定的旧修订校准报告，但不会泛化读取其他旧报告；服务端仍负责完整性和血缘验证。

## 8. 前瞻边界

报告仍是 `retrospective_only`。真实实时生产使用还必须经过已实现的 Prospective Validation：登记冻结候选 → 后续真实捕获 → 后续成熟参考 → 固定窗口检验 → qualification。当前历史数据无法制造这份未来证据。

所以当前准确结论是：

- **历史参考算法可用于 CMA 的历史分组、参数估计研究与 shrinkage 设计；**
- **实时算法是可信的因果候选，但当前最终独立 Episode 不足，不能激活实时 regime-conditioned CMA；**
- CMA 现在可以接入 `regime_cma_evidence`：没有 Verified state 时自动全部回退 Base CMA；未来某些状态验证通过后只使用这些状态，其余质量仍回退。
- **不能宣称已经取得真实前瞻部署资格。** Prospective Validation 将按状态独立积累资格。

## 9. 验收

- 状态级 Reliability / Historical Quality / Prospective 核心回归：116项通过；
- 历史图谱相关回归：527项通过；
- TAA / 组合桥接相关回归：146项通过；
- 前端全量：145文件、1111项通过；TypeScript、design check、i18n、production build通过；
- Reliability 浏览器：18项通过；Prospective 浏览器：7项通过；
- 真实沪深300系统对象浏览器：320 / 768 / 1440 共6项通过，分别验证历史独立区间/CMA准备度及实时 `Insufficient Evidence` 展示；
- 新前瞻测试证明：Bull/Bear 可先获得状态资格，Rare Sideways 证据不足时只回退，不拖累已验证状态；消费端拒绝未在 `qualified_states` 中的状态。

研究脚本：`backend/scripts/csi300_recognition_search.py`。系统保存脚本：`backend/scripts/csi300_reference_study.py`。实验输出在忽略目录 `frontend/test-results/csi300-study/`，行情源未修改。
