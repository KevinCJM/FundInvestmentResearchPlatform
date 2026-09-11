# TAA 工作台实施与验收记录

日期：2026-09-11。目标页面：`/pre-investment/taa`。设计见 [需求与设计](tactical-allocation-workbench-design-2026-09-11.md)，计算定义见 [数值契约](taa-numeric-contract.md)。

## 实现结果

| 用户任务 | 已实现行为 |
| --- | --- |
| 从 SAA 出发 | SAA 每项策略一键冻结真实大类权重、类内产品、资产/分组限制及来源指纹，带入 TAA；也可明确创建研究基准 |
| 决定如何偏离 | 趋势规则、人工零和观点、已发布市场状态三种入口；上下限、主动风险、单期换手、有效期均有明确含义 |
| 看懂当前建议 | 首屏直接比较 SAA、拟议权重、偏离百分点、参考持仓差额；手机使用卡片；信号中性或失效明确显示保持 SAA |
| 比较候选 | 固定强度搜索，始终包含零偏离；只按训练区选择，独立验证；净收益、净超额、回撤、波动、换手及成本可见 |
| 理解回测曲线 | 训练和验证从各自起点计算，曲线分段显示；两段不拼成虚假的连续实盘净值 |
| 复核极端情景 | 逐资产假设冲击及真实历史区间重演；SAA/TAA 同口径对照，贡献、费用和净收益对账 |
| 检查 PIT | 分开保留观测、可得、生效与研究时点；训练标签成熟度门禁；状态发布门禁复用；未知时点不冒充已验证 |
| 留痕与交接 | 保存前复算并核验指纹，历史版本不可变，可复制重算；将大类预算带入产品配置并保留来源 |
| 防止应用错误 | 两个服务入口共用应用校验，阻断过期、换手超限、验证不合格、可投资域变化和产品大类伪造 |

```mermaid
flowchart LR
    SAA[选择 SAA 策略] --> B[冻结基准与产品映射]
    B --> R[观点与偏离规则]
    R --> T[训练候选比较]
    T --> V[独立留出验证]
    V --> C[情景与时点复核]
    C --> D[保存不可变决策]
    D --> G[应用校验]
    G --> P[产品配置承接大类预算]
```

服务链为 `tactical_allocation_routes → TacticalAllocationService → Data / Repository / numeric`。动态回测与当前目标继续复用唯一 `_taa_path_kernel`，没有保留第二套数值模拟器。旧历史状态 TAA API 保留真实兼容调用与测试，未使用的旧前端 `taaBacktest.ts` 已删除。

## 自审核修复

- 所有参数变更立即使旧结果失效；异步预览、打开历史版本、情景结果均防止迟到响应覆盖新输入。
- 当前建议单独计算，不把最后一根回测权重或虚构收益行当成今天的建议。
- 训练收益若在截止日尚不可得，禁止用于选优；缺少可得时间继续标记研究限制。
- 过期判定取研究复核日与有效信号到期日较早者，应用时按当天重新检查。
- 导出和直接提交产品组合共用门禁；服务端真实映射决定大类归属，不信任前端标签。
- 对保存结果设置 7MB 预算，避免超过仓储读取上限后“保存成功但无法重开”。
- 每日净值拒绝重复日期、日内时间、未知资产、无效净值与不足样本；无前填或补零。
- 相对净值超额与收益百分点差分开标注，情景贡献显式扣除费用。
- 兄弟数值模块使用相对导入，修复既有 Numba 缓存下的双包名循环导入；原缓存、两种启动导入顺序均通过。
- 全量测试发现已有资产分类导航断言未等待路由完成；改为等待目标页面，不改变业务行为。

## 检查与证据

| 检查 | 结果与范围 |
| --- | --- |
| 后端联合回归 | **132 passed，2 warnings，115.06 秒**；新增 TAA 数值/数据/服务/桥接，旧 TAA、产品研究与历史状态 v2 共 7 个套件 |
| 前端全量 | **115 个文件、808 项通过**；包含 TAA、SAA 与产品配置交互 |
| TypeScript | `npx tsc --noEmit` 通过 |
| 生产构建 | `npm run build` 通过；保留现有大包体积提示 |
| 浏览器 | Chrome 桌面 1440px 与手机 390px **2 项通过**；观点输入、回测、冲击、保存、交接，无页面异常及页面级水平溢出 |
| NJIT / 内存 | 固定签名、实际 worker 预热、无 Python 回退；数值对照、只读/非连续视图、共享内存、输入不变与无签名增长均有覆盖 |
| 路由覆盖 | 显式核对本次 22 个改动路径，通过；`uncovered_files=[]` |
| 路由可复现性 | **未通过 Git 跟踪检查**：工作区中既有模块与新增 TAA 文件尚未纳入 Git。引用路径存在；未暂存、提交或放宽校验规则 |
| CodeGraph | 已手动同步，状态为 up to date：1,251 文件、19,634 节点、64,596 边；MCP 自动监听仍报告禁用，后续修改需再次同步。图谱仅用于链路定位 |

后端全部使用独立临时数据与缓存；服务测试读取固定 Parquet 输入并检查真实返回、持久化与产品预算链路。浏览器使用明确的离线 API 夹具，不证明线上数据或生产接口状态。没有下载正式数据、写入真实业务版本或部署。

可复查日志：

- 后端：`/tmp/taa-final-backend.IPSmEE/pytest.log`
- 前端：`/tmp/taa-final-vitest.log`
- 类型与构建：`/tmp/taa-final-typecheck.log`、`/tmp/taa-final-build.log`
- 路由：`/tmp/taa-routing-coverage.json`、`/tmp/taa-routing-validation.log`
- 图谱：`/tmp/taa-codegraph-final.log`
- 浏览器截图：`/tmp/taa-ui-evidence/`，桌面/手机各包含初始、回测、情景、保存四张截图。

复现命令：

```sh
validation_root=$(mktemp -d /tmp/taa-validation.XXXXXX)
CUSTOM_INDICATOR_DATA_DIR="$validation_root/workspace" HISTORICAL_REGIME_DATA_DIR="$validation_root/workspace" NUMBA_CACHE_DIR="$validation_root/numba" /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_tactical_allocation_numeric.py backend/tests/test_tactical_allocation_data.py backend/tests/test_tactical_allocation_service.py backend/tests/test_tactical_allocation_bridge.py backend/tests/test_historical_regime_taa.py backend/tests/test_portfolio_research.py backend/tests/test_historical_regime_v2_p1.py -q
npm run test --prefix frontend -- --run
npm run build --prefix frontend
npm run test:e2e --prefix frontend -- --config=playwright.taa.config.ts --workers=1
```

## 明确边界

1. **当前工作台输出研究结果。** 已实现时点校验与证据保留，但当前新建 SAA/规则不能倒签为历史已部署；缺少历史版本的数据不能被认定为正式 PIT 业绩。
2. **最优仅指训练区的候选强度最优。** 没有全空间或未来最优保证，也未实现自动滚动调参；反复查看留出区后修改规则会降低独立性，页面明确提示。
3. 回测按共同观察期每日目标再平衡，以 252 个观察期年化；数据有断档时不等于连续交易日实盘曲线，产品实际开盘/申赎时点另行验证。
4. 交接到产品层的是当前静态目标预算；既有产品层回测为毛收益回放，动态扣费 TAA 回测留在本页面，两者明确区分。
5. 情景不赋予虚构概率，不包含自动宏观因子传导或 VaR 认证；已有已发布风险模型通过真实产品组合入口联动。
6. 到期提示与重新研究由页面和应用门禁完成；没有后台监控、自动调仓或真实交易执行。
