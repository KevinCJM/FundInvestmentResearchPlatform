# 产品实施、资金续算与研究定稿开发验收

实施日期：2026-09-19—20。依据：[优化方案](pre-investment-implementation-validation-plan-2026-09-19.md)。本次代码修改保留原有未提交工作；没有 commit、push、部署或写入正式账户／行情数据。

## 已落地的操作流程

冻结 SAA／已保存 TAA → 产品与费用 → 余额与支付核对 → 保存研究包 → 锁定候选验证 → 具名研究定稿 → 导出／复制重研。SAA 可直接进入实施；独立战略范围允许显式选择同域后置映射。TAA 使用自身冻结映射，变更映射须重研。

原有产品历史组合、择时、SAA/TAA 页面保留。新四页承接同一研究包，不再使用原型结果。前端将产品和资金分成两步，费用详情按需展开，历史现金流每页 10 笔；报告优先显示失败／缺失证据，通过项折叠。新研究包使用自己的步骤与候选来源，不使用未选择的浏览器产品域提示其“尚未完成上一步”。

## 里程碑、自审核与修正

| 工作包 | 实现及复用 | 关键复核与自测 |
|---|---|---|
| E0 共同证据 | ArtifactRepository 复用；草稿、候选锁定、报告、定稿均为追加版本；revision、hash、幂等与失败尝试记录 | 初始 3 项存储测试通过；验证失败后同请求重试、160 字符操作标识、旧报告失效、定稿只读补充覆盖 |
| E1 产品风险 | 同收益区间对齐；复用现有标准化 OLS／时间留出；完整联合残差矩阵；训练矩阵秩／条件数诊断；Q 与 beta 分开 | 同一大类 beta=1.2、完全相关残差的组合波动为 13%；轴置换、非有限、PSD、秩退化、只读视图检查 |
| E2 成本与现金 | 唯一 self-financing 内核供首次调整、历史净成本和未来产品路径复用；每边费率；确知到账日现金日历 | 10 万元 50/50→60/40、每边 5bps 得约 9.9990001 元；不以半换手少扣费用；“已扣费用”必须已有目标持仓 |
| E3 资金续算 | 共享既有资金递推，新增任意剩余整月适配；原预算 occurrence hash、实际付款状态及真实价格基期 | 1/3/13/37 月、零本金后入金、零剩余期、部分付款、历史漏付、原整数年 ABI；真实 SAA 政策完成 13 月续算及逐产品现金路径 |
| E4 联合验证 | 类别预算 QP 复用 matrix_qp_kernel；single/A/B 原义下传；明确结算假设的产品路径；复用已发布情景影响计算 | 实际 SAA、TAA、A、B 和发布情景均有离线集成；未知费率／结算不作零值；可选现金预算不擅加成功概率门槛 |
| E5 工作台与定稿 | 四页真实 API；SAA/TAA 直接入口；当前资格复核；导出冻结输入、数组、报告、历史与尝试 | 前端状态测试覆盖空态、错误重试、输入失效、硬失败禁止定稿、当前来源失效和完整报告 hash；真实浏览器三尺寸走通填写→验证→定稿→导出→复制 |

自审核发现并修正：默认整数进入 float64 固定签名；日期字段名与类型同名；未执行模拟却标注独立随机流；零期误入模拟；未完成交易被声明费用已扣；非映射产品被当作普通缺证据；无支付保护的预算被新增概率门槛；数值轴不匹配与非有限；失败运行的幂等恢复；小屏表头竖排和金额负零显示。

## 计算与治理边界

- `backend/pre_investment/` 分为契约、来源、风险、费用、资金、路径、情景适配、存储及编排。已有 SAA/TAA、回归、QP、现金支付／Wilson、发布情景、受控 ArtifactRepository 均复用，没有复制第二套旧算法作为回滚。
- 数学内核固定 float64/int64 只读签名、`disable_compile()`，按进程预热；新计算加入应用 readiness。测试验证实际进入 nopython 签名，`python_fallback=0`、请求期新增签名为零。
- 共享资金内核支持任意 1–360 月及零起始本金；原整数年入口继续检查年数与正本金，作为有测试覆盖的薄适配，保持存量契约。
- 产品风险模型明确 D=0、前瞻主动 alpha=0；历史截距不当预测收益。矩阵病态／秩退化或留出未通过时停止风险桥接，不自动加岭或补系数。
- A 主模型采用原有融合协方差，原模型只作诊断；B 每个原模型都是硬约束。原模型资产风险、相对 SAA 总主动风险及适用基准约束均复核。
- QP 使用年度 TE 方差与单位买入成本比例的固定惩罚，返回系数及单位；只是类内候选生成，不是当前持仓交易成本最优、全模型 QCQP 全局最优或无解证明。实际自融资费用与风险仍按共同门禁复核。
- 产品路径是年度简单矩匹配的联合对数正态、月度独立增量；收益、比例年费、入金、调仓、现金付款按模型月末顺序。期内延期、暂停和盘中现金不可由“同月结算”证明。固定费率仅为显式敏感性假设，场外基金未来持有期费率表尚未支持，返回 unavailable。
- 单次产品随机数组不超过 64MB，逐模型路径工作量不超过 8000 万因子乘加预算；标量续算不超过 2000 万路径月。进程内计算准入一次一个任务。每个 worker 各自预热与准入，不声称跨进程全局限流。
- 只读／跨步长内核测试用 `np.shares_memory` 核验共享且输入未修改。严格日期对齐和已有回归边界需要一次数组物化，报告记录复制字节；没有宣称整个取数链路零拷贝。
- 验证固定 candidate hash；另用 validation_seed 的模拟才标记 independent_simulation，无资金预算或零期只标记候选验证。该名称不是独立审查员认证，也不是新独立市场留出样本。
- 定稿前重新检查来源生命周期、Mandate、CMA、映射、数据与代码指纹、Python/NumPy/Numba 环境、费用复核期；已定稿报告只读，过期资格单独呈现。代码指纹覆盖当前生产 Python 文件，并记录 Git HEAD 与脏工作树状态。
- 暂不认证完整实施资格：实际全期交收、成交价、容量、整数份额、历史 PIT 和机构独立审批仍明确缺证据。可作有限范围研究定稿；硬数值失败和研究完整性失败不能由备注豁免。

## 最终验证

2026-09-20 最终代码状态实测：

| 检查 | 结果 |
|---|---|
| 新增实施／资金测试和受影响 SAA、Mandate、multi-CMA、TAA、情景、前沿回归 | **327 passed**，90.53 秒；其中新增实施及续算用例 40 项 |
| 全前端 Vitest | **159 文件、1290 passed**，24.71 秒 |
| TypeScript `--noEmit` | 通过 |
| 生产构建 `npm run build` | 通过，6.41 秒 |
| i18n 与设计检查 | 通过；1895 个系统词条；未扩大原有设计债务预算 |
| Chrome 真实浏览器完整流程 | **3 passed**；1440×1000、768×1000、320×900；全页无横向溢出 |
| 新后端 Ruff、`git diff --check` | 通过 |
| 路由覆盖与结构／路径校验 | 通过；完整 Git 可复现性检查仍提示未跟踪的新文件，不通过暂存掩盖 |
| CodeGraph | 已同步，1265 文件记录、25332 节点、85580 边，up to date |

离线性能基准使用 37 个月、2000 条路径、3 个产品，在预热后连续运行 3 次，耗时为 **63.05／64.13／63.67 ms**。跨步长随机数组视图为 1,776,000 字节，实测与原数组共享内存；输入未改变，三次输出完全一致。执行审计确认 `python_fallback=0`、`request_time_compilation=0`。

基准输入和运行时准备完成后的 RSS 为 399,998,976 字节，2ms 采样观测峰值为 400,031,744 字节，增量为 32,768 字节。这是当前机器单次进程的观测结果，采样可能遗漏短暂峰值，不代表分配器精确上限、整个数据链路内存或相对旧版的提速结论。复现入口：`backend/tests/benchmark_implementation.py`，需已安装 NumPy、Numba、psutil。

最终命令（仓库根目录，Python 使用项目推荐解释器）：

```sh
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_implementation*.py backend/tests/test_funding_continuation.py backend/tests/test_strategic_allocation.py backend/tests/test_investment_mandate.py backend/tests/test_mandate_diagnostic_integrity.py backend/tests/test_multi_cma.py backend/tests/test_multi_cma_compatibility.py backend/tests/test_published_risk_models.py backend/tests/test_tactical_allocation_service.py backend/tests/test_tactical_allocation_bridge.py backend/tests/test_tactical_walk_forward.py backend/tests/test_frontier_grid.py -q
node frontend/node_modules/vitest/vitest.mjs run --root frontend
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run i18n:check --prefix frontend
npm run design:check --prefix frontend
npm run build --prefix frontend
npm run test:e2e --prefix frontend -- --config=playwright.implementation.config.ts
PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/tests/benchmark_implementation.py
```

浏览器证据：[桌面](screenshots/implementation-20260920/desktop-1440.png)、[平板](screenshots/implementation-20260920/tablet-768.png)、[手机](screenshots/implementation-20260920/mobile-320.png)。已人工查看截图，复核控件层级、缺失证据提示、表格滚动、文字可读性和定稿只读状态。空态、异常重试与旧报告失效另由前端测试覆盖；未声称所有错误状态均有浏览器截图。

非阻断的现有提示：Numba 非连续矩阵性能提示、Starlette 测试客户端弃用、React 测试 act 提示、Browserslist 数据较旧及现有大 bundle 提示。未为消除这些提示修改无关依赖或重构其他模块。路由完整检查的未跟踪文件问题包含本轮新文件及此前已有新文件；本轮没有提交授权，保持索引为空。


测试全部使用离线临时夹具，浏览器 API 仅绑定 localhost，临时数据在 `/private/tmp/implementation-e2e-*`。真实市场预测有效性、正式数据可得性及实际交易成本均未在本次认证。

## 2026-09-20 审核修复补充

研究包底层存储新增最终 UTF-8 文件容量校验，保持原有读取上限；超限请求明确失败，不登记不可读版本。真实候选的中文对账长文本、失败重试及既有研究包列表／历史读取均有回归。详细修复与本轮测试见 [分支审核修复记录](ltcma-center-acceptance-2026-09-18.md#6-2026-09-20-分支审核修复)。本轮相关后端 508 项通过；上文 327 项及当时未提交状态是首次实施的历史证据。

## PR 42 Bot 审核修复：复制来源保留

Bot 指出重新通过 `?package=` 打开复制包后，前端保存的 `copied_from_id=null` 会清空新版本的复制来源。新增 API 回归在修复前出现两项预期失败：编辑后来源为 null，以及不存在的复制来源仍返回 201。

后端保存时，新建复制必须引用存在且类型为研究包的具体 artifact 版本；更新只沿用服务端既有来源，客户端空值或替换值不能改变复制关系。原有 revision/CAS、幂等重放及不可变历史保持不变。此修复不重写既有历史记录，也不从缺失记录猜测来源。

修复后实施模块及资金续算后端 **43 passed**，前端实施／SAA／TAA 关联 **61 passed**，生产构建通过。Chrome 实际流程 **3 passed**（1440／768／320），新增复制→保存→重新打开→编辑→保存和历史来源断言。API 回归还覆盖创建与编辑重放、并发旧版本拒绝、错误类型／不存在来源、验证→定稿→ZIP 导出及原包不变。路由 validate／evolve、`git diff --check` 通过；CodeGraph 已同步。数值内核与前端业务代码未修改，上文更大规模测试属于前一提交的验证记录，不冒充本修复后全量重跑。
