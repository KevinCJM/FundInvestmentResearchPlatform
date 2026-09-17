# 独立前向验证域验收

> 最终集成补充：下文保留分工交付时的范围说明。父任务已完成真实服务、路由、预热、进度读取和TAA接线；相关测试与固定快照不能自动续接的边界见 [集成审核记录](regime-completion-final-audit-2026-09-15.md)。

日期：2026-09-15；分支：`ISSUE2609/BetterSaaTaa`。

## 交付范围

仅新增以下5个文件，没有修改既有后端模块、前端、AGENTS 或路由文件：

- `backend/historical_regimes/reliability/prospective.py`
- `backend/historical_regimes/reliability/prospective_kernels.py`
- `backend/tests/test_regime_prospective.py`
- 本文
- [API、策略及父集成说明](regime-prospective-api-2026-09-15.md)

未提交、推送、reset、stash；未运行网络测试、读取或改写生产数据；未启动子代理。保留工作区既有及其他工作者改动。

## 已实现的闭环

服务端注册冻结 fitted 候选 → 仅显式捕获当前可得的最新因果状态 → 同一定义的新发布参考提供成熟标签 → 在不可变预测日志上做固定窗口评估 → 验证独立资格供未来消费者使用。

原报告始终 `deployment_eligible=False`。本域输出单独的 qualified assessment；它不是投资获利证明，不授权其实际签发之前的 TAA，也不把历史回放重命名为前向证据。真实协议没有实际后续捕获/标签时保持 pending。

采用保守分块门槛，不依赖并行开发中的 bootstrap 接口，不宣称统计显著性。策略在注册时冻结；样本、两轴逐类、逐类完整区间、覆盖、一致率、最少完整块及所有完整块的 Brier 改善必须同时满足。终局窗口不允许追加数据后重考。

## 独立证据

隔离测试使用临时目录中的真实指数 Parquet、原 source binding/resolver、原图 prepare/executor/PIT 探针、原 reliability preview/confirm、原运行快照和参考 hydrate。可增长场景通过真实版本仓库存入既有的旧版未绑定指数定义契约；另测当前 create_definition 自动冻结源的契约。时钟通过 ProspectiveService 构造注入；测试参考发布时间也仅在临时 fixture 的发布时钟边界模拟。没有 patch 源门禁、mock 分类输出或在生产目录注入未来数据。

覆盖：

- 拟合/未拟合注册、制品及普通重算 hash 篡改、重复注册和不可换策略。
- 注册日排除、只取当前最新日期、漏捕不补录、重复捕获幂等、历史输入字节修订失败。
- 已发布标签拒绝新捕获，防止重新发布掩盖先看标签。
- 完整未来窗口上的 perfect classifier 取得独立资格；错选状态被拒绝，所选置信度确为 q[chosen] 而非 max(q)。
- 缺失捕获保留分母并使对应完整块失效；参考随后遗漏捕获时见过的日期也不压缩日轴。
- 未来窗口不足、无实际捕获、陈旧数据、不同参考定义、错误 snapshot hash、未来发布时间。
- 消费者的校准/模型/参考绑定、过去决策、资格过期、日志状态篡改、当前源历史修订和执行器代码变化。
- GMM 通过原图执行器进行真实捕获：声明的初始训练窗口、训练边界、初始化指纹和类别排序锁定；历史 expanding replay 与捕获配方分别记录。没有复制拟合算法。
- HTTP 请求禁止客户端注册时间/capture as_of，尚未预热时失败关闭。
- 新内核独立 NumPy oracle、正确及错误 Brier 改善、缺失/NaN/Inf/越界概率、空样本、长日历间隔、错误尺寸/日期轴/dtype、只读非连续视图及 PID 失效。

## NJIT 与内存边界

新增唯一内核 `forward_blocks_kernel`，计算完整配对块及完整区间的门槛统计；分类、Brier/class_base 和冻结校准应用复用原可靠性内核。新内核的固定签名允许 readonly arbitrary-stride；warm 后禁用编译，audit 同时校验 warmed PID、唯一 nopython 签名和编译禁用。

测试用 `np.shares_memory` 验证日期/概率视图共享底层数组；oracle 对照块均值，非法 dtype 不允许请求时新增签名。JSON/Parquet 解码、状态编码、hash 字节序列化、输出及块工作缓冲区属于明确边界分配。不宣称整条流水线零拷贝或已测峰值内存；本次未做性能加速声明。

算子治理：该内核只解决“原时间轴上有效配对的完整区间/块门槛”一个验证职责。概率映射、校准应用、分类与 Brier 指标独立复用现有能力。原轴缺失掩码、配对及区间边界必须共同处理，不能拆成丢失时序关系的独立筛选节点。本域不注册新的状态识别黑盒算子，不改任何已保存定义。

## 复现与结果

最终独立测试命令：**21 passed，1 warning，52.26 秒**。唯一告警为现有 Starlette/httpx 弃用提示。日志位于 `/tmp/prospective-tests-final.log`；前期有交叠的调试/定向测试次数不累计到此数。另对3个新增 Python 文件完成 AST 语法及尾随空白检查，全部通过。

```sh
CUSTOM_INDICATOR_DATA_DIR=/tmp/prospective-test-ci \
HISTORICAL_REGIME_DATA_DIR=/tmp/prospective-test-hr \
NUMBA_CACHE_DIR=/tmp/prospective-numba \
PYTHONPATH=.:backend \
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest \
  backend/tests/test_regime_prospective.py -q
```

调试阶段曾因合成夹具含零价格、过近阈值扰动、GMM 缺少可用类别配对而失败；修正的是测试数据/声明参数，没有降低前向生产门槛或改共享代码。代码指纹最初用 marshal 序列化时发现引用标记不稳定，已改为规范化不可变 code 字段和常量；对应血缘测试会覆盖注册后重复读取和执行器变化。

## 父任务仍需集成

按照 API 文档安装路由、服务和每 worker lifespan warm；增加可选 Study.qualification_id 并从 model_binding_hash 排除此附着字段；两个 TAA 消费者接同一 verifier，保留原有发布与时点门禁、旧无 study 契约及旧冻结报告。

已运行仅针对上述5个路径的 AI Hermes 覆盖检查：两个实现路径被已有 `backend_regime_research` 目录覆盖；新增测试和两份文档尚未登记，报告 `uncovered_files`。按独占范围要求没有修改路由文件，父任务应在统一集成时登记并验证。

限制：日频、受支持源/模型及预算见 API 文档；固定 snapshot/checksum 绑定不能自动滚动到新数据版本。当前新建指数定义自动冻结数据绑定，因此冻结上传及这类定义缺少未来数据时会 pending，不静默换源。旧版未绑定指数定义可以在同一定义下累积新观测；让当前新建模型支持数据续接和同定义参考的新 vintage 发布，仍需父任务明确整合来源契约，不能声称本模块已完成该来源扩展。HMAC 保证受控服务边界内的日志来源，不是抵御拥有目录/签名密钥写权限的管理员或整体工作区回滚的外部公证。当前验收没有证明任何真实市场模型已取得资格，也没有完成父级 TAA/前端上线验收。
