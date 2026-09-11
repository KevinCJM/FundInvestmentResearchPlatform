# TAA 数值接口契约

`backend.tactical_allocation.numeric`，数组顺序由服务层固定并随快照保留。输入收益为小数，每期末日期对应 `period_start` 后一完整收益区间。Python 仅做输入边界、编排和输出封装，数值路径固定签名 NJIT，无请求期新签名或 Python 回退。

## 搜索

`evaluate_candidates(returns, probabilities, use_signal, base, state_tilts, min_weights, max_weights, max_abs_tilts, train_end_index, strengths, cost, periods_per_year, risk_penalty, max_tracking_error=1.0, max_turnover=1.0, objective='active_utility', selected_candidate_id=None, group_membership=None, group_min=None, group_max=None)`

- `returns`: `[N,A]`，`probabilities`: `[N,S]`，`use_signal`: `[N] uint8`，`state_tilts`: `[S,A]`。缺失预热用 `use_signal=0`，收益不能缺失或填零。
- `N>=40`，训练与留出分别至少20期。训练 `[0,train_end_index)`，留出 `[train_end_index,N)`；两段均从 SAA 起步并计入偏离成本。
- 搜索强度固定 `0,.25,.5,.75,1,1.25,1.5`；人工比较允许 `[0,1]`。偏离保持方向，每个状态统一缩放至逐资产与组合资产组约束；概率混合保持预算与边界，不独立截断后归一化。组输入为 uint8 `[G,A]` 与 float64 `[G]` 上下限，组可重叠且始终逐一检查。
- 训练目标：`active_utility` = 年化平均净主动收益 − `risk_penalty × tracking_error²`；另支持 `excess_return`（兼容别名 `net_excess`）相对 SAA 终值收益、`min_drawdown` 最大回撤（负数，越大越优）。训练主动风险和最大单期单边换手为候选可行性约束；零偏离保留为安全基线，明确豁免过严的战术换手限制，因为 SAA 自身漂移再平衡也可能产生换手。留出结果不参与选优。
- 返回 `{candidates,selected_id,selected_index,auto_selected_id,auto_selected_index,selected_path,selected_validation_path,selected_train_path,baseline_nav,selected_nav,selected_weights,train_baseline_nav,train_selected_nav,train_selected_weights,selected_state_tilts,execution,selection_policy}`。`auto_selected_*` 始终是训练选优；`selected_candidate_id` 允许用户选其他训练可行候选并保留自动选择结果。
- `candidates[]`: `{id:'scale-0',strength,feasible,validation_feasible,constraint_scales:[S],train:metrics,validation:metrics}`。ID按候选序号稳定；`selected_index`索引该数组。
- `metrics`: `{total_return,baseline_return,excess_return,total_return_difference,annual_volatility,max_drawdown,tracking_error,information_ratio,turnover,average_turnover,max_turnover,cost,baseline_turnover,baseline_cost,score,observations}`。`excess_return` = TAA终值/SAA终值−1，`total_return_difference` = TAA总收益−SAA总收益；波动使用样本标准差；换手为单边0.5×绝对权重差；成本是每单位初始净值的金额。
- `selected_path` / `selected_validation_path` 为同一留出段 ndarray `[N-train,A*2+14]`，`selected_train_path` 为训练段相同列结构；两段均从1起步且不拼接为连续业绩。列沿用唯一 `_taa_path_kernel`：目标权重[A]、交易前权重[A]、缩放、换手、成本率、成本金额、SAA毛收益、SAA净收益、TAA毛收益、TAA净收益、SAA净值、TAA净值、SAA换手、SAA成本率、SAA成本金额、毛主动收益。
- `baseline_nav/selected_nav/selected_weights` 为上述留出路径的共享视图，禁止写入。`selected_state_tilts` 为已约束的 `[S,A]`。

## 动量信号

当前算法版本为 `available-window-relative-momentum/2.0.0`，这是可得窗口选择与缺失口径的明确算法变更。版本进入 `audit.signal` 和预览哈希；旧冻结结果按原记录读取，不重算覆盖，旧预览不能用旧哈希保存为新结果。

`build_momentum_signals(returns,lookback,max_tilt,available_at=None,period_starts=None,as_of_day=None,period_ends=None,max_signal_age_days=31)` 返回 `{probabilities,use_signal,state_tilts,current_probabilities,current_use_signal,current_momentum,knowledge_verified,current_knowledge_verified,windows}`。`period_starts`、`period_ends`、`as_of_day` 为必需的明确日期，均为自1970年起日数；收益区间须有序、不重叠、期末晚于期初。`available_at` 为 int64 `[N,A]`，缺省/未知为 -1，不产生已知窗口。

- 每一期初选择观察期末不晚于该时点，且全部 `lookback` 个连续收益及所有资产端点都已经公布的最近窗口；不会删除未公布行再拼成窗口，不把 ann_date 前移。最新观察未公布时可使用更早的完整窗口。
- 有效期为 `cutoff - window_end <= max_signal_age_days`，等号仍有效；超过有效期的历史及当前信号均回 SAA。未知/预热/中性/过期与有信号分别保留语义。没有完整已知窗口时当前动量为 NaN，服务序列化为 null，不填零；可追溯的过期值可保留，但不发信号。
- 相对累计收益最强资产取得偏离，多资产并列概率均分，全部相同则中性。一个资产不产生战术偏离。状态偏离给赢家 `+max_tilt`，其余等额融资，合计 0；窗口知识有效与非中性信号是两项独立计数。
- `windows` 是 int64 `[N+1,5]`：收益起索引、止索引（不含）、最晚可得日、滞后日数、窗口状态（0预热/1尚无已知窗口/2过期/3可用）；无窗口的索引/日期/滞后为 -1。最后一行仅描述 `as_of_day` 的当前特征，不创建收益行、不进入回测。
- `momentum_windows_kernel` 只选择时点合格的窗口，使用滚动最大值和按公布日排序的最小堆处理乱序公布，O(N log N)。`momentum_signals_kernel` 以单调推进的窗口止索引维护累计对数收益，O(N×A)，不物化逐窗口收益；二者有独立输出和固定签名。窗口选择含队列/堆状态，拆成单向无状态步骤会改变释放顺序，因此保留此内核；相对强度、约束、回测仍分开。未知信息不可通过滞后、改名或分解绕过门禁。
- `signal_counts_kernel` 统计训练/验证中有效窗口期数与实际非中性信号期数；服务预检和计算均检查训练有效信号为 0 时禁止搜索，明确提供固定假设比较，不称零分并列为最优偏离。

服务保存逐期 `signal_timing`：期初知识截止、观察起止、最晚公布日、滞后、窗口状态与是否发信号。当前建议的信号日及到期日依据实际选中窗口；页面显示有效信号期数与当前信号来源。旧结果缺计数字段时不以新预检计数冒充历史结果。

`returns_availability_kernel(nav_available)` 输入 int64 `[N+1,A]`，返回 `[N,A]`：取两端NAV的最晚可得日，任一未知则保留-1。

`knowledge_window_status(available_at,cutoff,start=0,end=None)` 返回 `{future_cells,unknown_cells,verified}`，供服务对训练标签进行成熟度检查。已知在训练截止日后才可得的收益不能用于当时的候选选优；未知时点保持研究属性。

## 情景与权重联动

`stress_compare(returns,base,target,cost=0.0,periods_per_year=252)` 返回 `{baseline:metrics,target:metrics,excess_return,relative_excess_return,total_return_difference,baseline_nav,target_nav,baseline_contributions,target_contributions,excess_contributions,baseline_cost,target_cost,execution}`。单行资产冲击为假设情景，多行为历史窗口重放；每期回归同一目标且使用同一费用口径，不赋予概率或预测含义。`*_contributions` 是按期初财富链接的资产收益贡献；贡献合计减cost与终值收益对账。`excess_contributions`合计再扣双方费用之差，对账`total_return_difference`；相对收益`excess_return`/`relative_excess_return`另列，避免百分比与百分点混淆。

`compose_product_weights(class_weights,class_indices,within_weights)` 返回产品权重 ndarray；每个产品记录引用一个固定类索引，类内权重合计1，结果为大类目标×类内权重。同一实际产品跨类出现时，服务层拒绝应用并要求先明确唯一归属，不由名称猜测或擅自合并预算。

`aggregate_class_weights(weights,class_indices,class_count)` 将明确的产品权重按类索引汇总，用于下游防止预算漂移；无隐式类型取整。

`recommend_weights(probabilities,use_signal,base,state_tilts,lo,hi,max_tilt,strength,current_weights=None,group_membership=None,group_min=None,group_max=None)` 输出 `{weights,tilts,trade_deltas,turnover,constraint_scales,state_tilts,fallback_to_saa,has_deviation,is_saa}`。通过唯一推进内核的单期零收益特征计算得到当前目标，不将该零收益当作历史样本。`tilts` 相对SAA、`trade_deltas`相对current_weights，后者未提供时以SAA计算；服务应隐藏“实际调仓”展示并标明持仓尚未录入。`is_saa`根据实际目标相对SAA偏离判断，完全受限的非零候选同样为true，不能拿实际持仓换手判断是否偏离。当前建议换手超限由服务阻止应用。

`warm_tactical_allocation_kernels()` 返回 execution 审计。服务启动调用且失败关闭；只有当前PID真正执行全部预热后才标记complete/fully_warmed。未预热或fork后未再次预热的worker禁止进入数值业务wrapper，不能仅凭存在编译签名宣称ready。数组允许只读与任意步长视图；必要 dtype 转换仅在边界，候选共享原收益底层内存，仅分配输出和独占工作缓冲。

## 数值验收与测量

- 2026-09-11：新增数值测试覆盖训练/留出隔离、人工选择与自动推荐分别留痕、零偏离与同成本SAA完全相等、唯一旧内核逐值一致、资产与重叠组预算、主动风险参考值、信息可得边界、情景贡献对账及产品预算守恒。
- 初次新增数值套件 **32 passed（16.75s）**；对唯一旧TAA内核的兼容回归 **24 passed**，当时与29项新增测试联合 **53 passed（95.64s）**。后续只增加训练标签知识计数、worker实际预热门禁和展示用差额/is_saa字段，旧组合推进算法没有进一步改变。所有数据操作使用临时目录；唯一警告是已有Starlette/httpx弃用提示。
- 只读非连续输入通过 `np.shares_memory` 检查每次训练/留出调用共享原收益数据；负步长输入可用，无新增NJIT签名；输入未变、输出视图只读，删除结果容器后视图仍保有底层所有者。float32→float64 是明确边界复制，非连续float64不复制。中间目标、净值、候选指标与工作缓冲属于必要输出分配。
- 可重复合成微基准：固定 `np.random.default_rng(217)`，2400期×6资产、60期动量、训练1600期、7个候选强度、费用5bp。预热后3次完整候选评估中位数 **2.83ms**；输入115,200字节，保留的两段路径499,200字节，Python `tracemalloc`峰值890,898字节。该测量不等同于进程总RSS或生产吞吐；计时不含编译、I/O或API序列化。训练/留出及净值输出视图共享检查均为true。
- 默认测试缓存曾因既有 `historical_regimes` / `backend.historical_regimes` 包名缓存混用发生收集期循环导入；将 `historical_regimes/data.py` 对兄弟内核的导入改为相对导入后，仓库根目录与 `cd backend` 两种导入顺序均在原缓存通过。未删除或清理其他任务的缓存。
- 初次工作台服务、数据、应用桥接、新旧 TAA 与相关历史状态/产品回归联合 **132 passed（115.06s）**；证据见同目录工作台验收文档。


### 2026-09-11 可得窗口修正验收

- TAA numeric/service/data/bridge 定向回归 **91 passed，24.79s**。新增测试包括 T+1 取最近已公布窗口、等号边界、长滞后/未知且不拼接窗口、过期回 SAA、未来值改变不影响过去、乱序公布对穷举参考、只读步长共享与固定签名、空窗口/预热/非法日期与值、零训练信号预检门禁、算法版本改变导致旧哈希冲突和旧冻结结果不变。
- 同份真实隔离股债数据（510300.SH 60% + 511260.SH 40%，2017-08-04 至 2026-09-03，训练截止 2023-09-02，60期窗口、31天有效期、单边10bp），来源哈希 `ad4d480adf70e0945e74c6ad5a5eca91c4bceb8ef6f15dfeb0a48e66a83db897` 完全未变。训练有效信号从 0/1472 变为 1411/1472，验证从 66/731 变为 731/731；首个有效窗口2017-08-04至11-23，11-24公布并在11-24期初使用，滞后1日。非零候选训练相对净超额约 −0.361% 至 −1.577%，仍选择 SAA；这说明修正了比较机制，不说明趋势策略具有投资优势。没有修改任何数据/ann_date。
- 当前源码直接调用完整 preview（包括I/O与JSON封装）约111.46ms；没有替代主任务HTTP/浏览器验收。重复信号微基准 `default_rng(791)`、10次预热测量：2500期×2资产/L60 中位0.337ms、最大0.504ms；10000期×30资产/L756 中位8.416ms、最大8.579ms。只读非连续float64收益及int64元数据共享原底层输入，未增加编译签名。
- 必要分配包括 O(N) 滚动可得日、未知掩码、堆与窗口索引，O(L) 队列，O(A) 累计值及概率/状态输出。窗口选择工作与输出上界约65N+8L字节（不含输入与Numba分配头）；没有 N×L 收益副本。此处计时不含编译，工作缓冲说明不是进程RSS测量；不据此宣称整体零分配或生产吞吐。
- 可重复脚本与完整JSON暂存在 `/tmp/taa-mature-window-recheck.py`、`/tmp/taa-mature-window-recheck.json`；算法逻辑与固定夹具在上述测试文件中，不依赖临时脚本才能回归。

- 独立审核补充：800组合、40,438时点与194次未来/不可得单元扰动均与穷举参考相符；12组长期未知前缀（最多20,000期，L5,000）未产生伪信号，末窗累计值误差不超过约1.81e-13。此压力长度超过服务10,000期预算，只验证内核边界，不扩大API支持范围。启动预热使用同日已知输入，实际执行可得窗口释放与累计值分支；调整后numeric 36项再次通过。
