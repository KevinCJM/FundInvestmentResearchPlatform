# M2 独立 CMA 模型与编辑器验收

范围：`ISSUE2609/BetterSaaTaa` 当前 checkout；仅新增下列 7 个授权源码/测试文件及本报告。未修改既有源码、共享四份研究文档、路由 JSON、AGENTS.md 或正式数据，未执行 Git 写操作。本里程碑是可调用的独立实现，**服务、持久化、政策与启动链路的正式接入由协调者完成**。

## 交付

|文件|内容|
|---|---|
|`backend/strategic_allocation/cma_model_contracts.py`|`BlackLittermanRequest`、`BlackLittermanView`、`ScenarioMixtureRequest`、`CmaScenario`、带 method 判别的 `CmaModelRequest` / `CMA_MODEL_ADAPTER`。|
|`backend/strategic_allocation/cma_model_kernels.py`|固定签名 BL、情景混合矩、协方差诊断和收益范围检查；`warm()`、`require_ready()`、`execution_audit()`。|
|`backend/strategic_allocation/cma_models.py`|纯函数 `evaluate_cma_model`、`CmaModelResult`、只读 float64 边界转换。|
|`backend/tests/test_cma_models.py`|62 项参考数值、契约、只读内存、预热门禁及预览零文件 I/O 测试。|
|`frontend/src/services/cmaModelTypes.ts`|与后端对应的请求类型及输入完整性校验。|
|`frontend/src/components/strategic-allocation/CmaModelEditor.tsx`|受控方法选择、BL 观点、市场权重、风险矩阵、情景编辑与预览回调；无网络请求或持久化。|
|`frontend/src/components/strategic-allocation/CmaModelEditor.test.tsx`|10 项编辑、显示单位、空输入、概率、只读、过期上下文、加载、错误/重试测试。|

共用基础类型直接引用协调者新增的 `common_contracts.py`，避免以后由 `contracts.py` 引入模型时形成循环。风险矩阵校验调用既有 `cma_covariance_kernel`；没有复制风险预算或政策搜索。

## 精确接入接口

```python
from backend.strategic_allocation import cma_model_kernels
from backend.strategic_allocation.cma_models import evaluate_cma_model

cma_model_kernels.warm()  # 每个 worker 的启动阶段；失败必须阻止 readiness
result = evaluate_cma_model(
    model_request,  # dict 或已验证的 BlackLittermanRequest / ScenarioMixtureRequest
    asset_ids=resolved_asset_ids, as_of=research_date, currency=currency,
)
means = result.effective_returns       # readonly float64 [asset]
covariance = result.effective_covariance  # readonly float64 [asset, asset]
payload = result.to_payload()         # 仅转 JSON；不写文件、不生成确认记录
```

- `CmaModelResult` 是 dataclass，不是 dict；另有 `asset_ids`、`method`、`posterior_mean_covariance`、`model_audit`、`execution`、`definition`。资产顺序就是请求 `asset_ids`。三项可选 context guards 要由服务传入，精确验证顺序/日期/币种。
- 公共输入：`method / asset_ids / as_of / currency / source`，收益口径固定为 `annual_arithmetic_total_return`。
- BL：`covariance / risk_covariance_basis="input_covariance" / market_weights / market_weight_source / delta / tau / risk_free_rate / views`。每条观点有 `kind / asset_id / relative_to / annual_return / view_std / observed_on / available_on / source`。`relative_to` 仅相对观点使用。
- 情景：`risk_mode="shared"` 配一个 `shared_covariance`，或 `risk_mode="scenario_specific"` 配每条情景的 `covariance`；不允许混用。情景字段为 `id / probability / annual_returns / covariance / source`。字典收益、权重必须覆盖完整资产轴，矩阵必须按同一轴排列。
- `model_audit` 明确生成方法、风险方法、未修补矩阵、币种、收益口径及局限。BL 返回超额先验；情景返回概率、情景 ID、组内协方差与组间均值协方差。`effective_volatility / effective_correlation / min_correlation_eigenvalue` 可用于展示和现有 CMA 边界封装。
- 后端 Pydantic、`ValueError` 和 `np.linalg.LinAlgError` 必须由既有 API 异常封装呈现；未预热抛 `CMA_MODEL_NOT_READY` 的 `RuntimeError`。请求不得调用 `warm()`。

前端：`<CmaModelEditor context value onChange onPreview? busy? readOnly? disabledReason? error? onCopy? />`。`context={asset_ids,as_of,currency}`；`value=null` 表示继续现有人工表单；新方法的未填数字使用本地 `NaN`，预览前必须完成校验。`onPreview(value)` 只发出回调，调用者负责请求取消/代次、时点变化使旧结果失效、错误与确认流程。`onCopy` 由父级建立新研究，不修改旧版本。

收益/权重输入显示 `%`，相对收益差/观点标准差显示“百分点”，提交数值除以 100；协方差直接显示“年化小数收益的平方”，不做百分比缩放。空白市场权重、收益和矩阵不被自动填成等权、零收益或示例风险。

## 数学与边界审核

- `pi = delta * Sigma * w`；`q_excess = q_total - rf * P1`，相对观点 `P1=0`。零观点返回 `pi+rf`；弱观点趋向先验。风险仍为输入 Sigma，后验均值协方差单独返回，不自动加到风险、不转为稳健半宽。
- 使用 `np.linalg.solve`，不显式求逆。输入 Sigma 可奇异半正定；严格正的观点方差保证理论上的观点系统正定，机器精度下无法解算时失败，不加 jitter、不修补矩阵、不回退。
- 正确性细化：后验均值协方差采用与设计公式等价的 **Joseph form**，减少强观点时相减导致的数值消减；仅对最终均值协方差做对称舍入处理，不修改输入风险。此为数值实现细化，不改变数学模型。
- 情景矩包括 `sum(p * (Sigma_s + (mu_s-mu)(mu_s-mu).T))`。概率容差 `1e-8`，不归一化；概率为零的情景仍需完整有效。单情景退化到该情景。
- 范围：资产 1–30、BL 观点 0–60、情景 1–60；δ、τ 为有限正数，不任意限制 τ≤1。为接入既有 `AssetAssumption`，有效年化收益/绝对观点及无风险收益范围为 `[-0.5,2]`，相对观点为 `[-2.5,2.5]`，风险对角方差 `(0,9]`；超出下游范围明确失败，不裁剪。
- `view_std**2` 必须仍为有限正数，拒绝浮点下溢/溢出。保留观点观察日、可得日与来源，要求观察日≤可得日≤研究日；不因此宣称来源已被核验或历史 PIT 已获认证。

## 已执行验证

后端命令均使用 `PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12`。

|命令|结果|
|---|---|
|`-m pytest backend/tests/test_cma_models.py -q`|**62 passed**，最终 16.38s。|
|`-m pytest backend/tests/test_strategic_allocation.py -q`|**26 passed**；1 条已有 Starlette/httpx 弃用警告。|
|`npm run test --prefix frontend -- --run CmaModelEditor.test.tsx --maxWorkers=2 --minWorkers=2`|**10 passed**，最终 788ms。|
|`npm run design:check --prefix frontend`|通过，无回归。|
|`node scripts/check_i18n.mjs`|退出 0，`errors=[]`；不等于所有新增文案已翻译。|
|`npm run build --prefix frontend -- --outDir /private/tmp/bettersaataa-m2-build`|通过，6.16s；已有大 bundle 提示。构建目录不触碰正式 dist。|
|`node /private/tmp/bettersaataa-m2-browser/build.mjs`|正式组件＋共享 Tailwind 样式独立打包通过；产物 `standalone.html` 在同一临时目录。|
|`node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json`|最终复查通过，退出 0，日志为空。|

测试覆盖 NumPy 参考、绝对/相对/重复/无/弱观点、singular PSD、非 PSD、范围、NaN/Inf/bool、缺轴、未知资产、来源/日期、共用/逐情景风险、组间方差、单情景/零概率、确定性、只读非连续/负 stride、共享 owner 生命周期、alias 不污染、PID 改变/编译锁/预热失败，以及已验证对象被嵌套修改后重新校验。纯预览在禁止 `open/mkdir` 的测试下通过。

内存测量：`NUMBA_NRT_STATS=1 PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/bettersaataa-m2-memory.py`。固定 seed=42、30 资产、8 观点、1000 次 BL：0.09444s（94.44µs/次）；NRT 分配/释放均 31,000；只读非连续输入 `shares_memory=True`；逻辑协方差 7,200 bytes，owner 28,800 bytes。Python tracemalloc 峰值 41,384 bytes，进程峰值 RSS 335,462,400 bytes（含解释器、导入和预热，不是单次请求原生内存峰值）。这不是前后性能提升证明。

必要分配：JSON/Pydantic→float64 仅在模型输入边界；只读 float64 ndarray 使用保留 owner/stride 的视图。BL 分配小矩阵工作区与解算器必要连续缓冲；共用情景风险以 `[None,:,:]` 视图复用，不按情景复制。校验复用既有风险内核，其相关矩阵/特征值工作区允许分配；输出可以分配。没有大时序数据拷贝或跨进程共享功能的完成声明。

## 待协调者接入及未解决项

1. 在每个服务 worker 的启动链调用 `cma_model_kernels.warm()`，readiness 合并 `execution_audit()`；请求仅 `require_ready()`。当前独立文件不修改既有启动行为。
2. 将可选新模型嵌入 CMA 请求。省略模型时走原人工路径；新模型按 M1 已解析战略轴/币种/时点计算。把 `effective_returns/effective_covariance` 接入唯一政策求解、资金诊断及 TAA 风险检查；不要继续消费手工草稿旧均值，不把后验均值协方差当作风险。
3. preview 继续纯计算；confirm 服务端重算并核对 hash，再冻结原始模型和有效结果。旧 CMA 无模型输出时保持原数学口径及只读历史，不补算后验。模型本身不读写仓库或数据目录。
4. 风险预算评分由协调者加入已有候选搜索；本交付没有第二个搜索内核。
5. 在现有工作台挂接受控编辑器及结果展示，保留人工 CMA；父级处理请求代次、上游失效和不可变复制。当前未验收实际 API→政策→TAA 整条消费链。
6. 最终全仓 tsc 已通过，日志：`/private/tmp/bettersaataa-m2-tsc-final.log`。工作区仍有并发集成，协调者需对最终代码运行完整回归；本里程碑未修改其他工作者文件。
7. 已按明确 7 条 `--changed-file` 运行 `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --json`，退出 1：`backend/tests/test_cma_models.py`、`frontend/src/services/cmaModelTypes.ts` 尚未被路由覆盖。协调者登记到 `strategic_allocation_policy` 的相关文件/测试/minimum_regression 后运行 validate；本次不修改路由 JSON。
8. **浏览器验收尚未完成**：本地 Vite 监听被沙箱以 `listen EPERM 127.0.0.1:5188` 拒绝；浏览器连接失败，原生 Google Chrome 操作未获自动批准。未绕过限制。320/768/1440 布局、真实键盘交互和渲染对比度须由协调者补跑；静态打包及 Vitest 不作视觉验收证明。

报告仅认证上述独立范围与实际命令，不认证全仓测试、发布、投资有效性或正式历史 PIT。
