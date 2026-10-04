---
id: "cpp-local-coverage"
type: "claim"
title: "C++ AOT：局部标量适配不等于全平台迁移"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "显式单产品标量适配、当前平台NJIT装配及历史证据范围；不覆盖本次原生构建、性能或生产切换"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
evidence_kind: "mixed"
watch_globs: ["backend/**", "frontend/src/**", "backend/requirements.txt", "frontend/package.json", "frontend/package-lock.json", "Dockerfile", "docker-compose.yml", "deploy/**"]
dependencies: ["docs/indicators/cpp-aot-contracts.md::sha256:df25aeae098bcd6b50f42121b4f9052d0bd9f8aa3d9122a21f31dc47aab19732", "docs/verification/cpp-aot-contracts.md::sha256:6cd66af34b1a7fe836f50348357d12a82cf118eed7412441583ffc61bef8e3a4", "backend/cal_indicators/cpp_aot.py::sha256:b31d9f87e277bf429d2f6233a5b548b5416a6a793110cdad74eb7df80b554b6a", "backend/app.py::sha256:ed789f9e6f70b6ce5493608e0329cdfc94792bdddeea9bf73aee5de4eab64095", "backend/custom_indicators/service.py::sha256:2011a5cc40a1b3a21fe5340383bd6125134c7ab90afd730e0d8846e662a80e20", "backend/custom_indicators/typed_service.py::sha256:aabd173589399866bd4e2de2b5ca4d815f9c3d822ce1091c8322bd3bef84b2db"]
---

# C++ AOT：局部标量适配不等于全平台迁移

## 主张

平台提供 `CppIndicatorBatchPlan` 显式单产品标量批次适配并校验原生凭据；当前服务和启动仍保留NJIT路径。该适配及历史性能记录不能证明所有数值服务已迁移、NJIT预热已移除或整台后端启动耗时已解决。

## 证据

- [C++ AOT接入契约](../../indicators/cpp-aot-contracts.md)：范围、错误与数据、执行门禁章节均明确局部适配及既有路径保留。
- [cpp_aot.py](../../../backend/cal_indicators/cpp_aot.py)：`CppIndicatorBatchPlan.__init__` 第60行起拒绝非single_product及非scalar；使用 `calmetrics_engine.GraphCompiler`。`_ValidatedPreparedBatch`、`_validate_result`逐次校验图指纹、后端和方法对应生命周期。
- [app.py](../../../backend/app.py)：`lifespan` 第99行起仍调用优化器、typed及多条NJIT预热。
- [custom_indicators/service.py](../../../backend/custom_indicators/service.py)：`warm_numba_plans` 第898行起仍编译和缓存NJIT计划；[typed_service.py](../../../backend/custom_indicators/typed_service.py) 的 `infer_expression`仍调用 `compile_numba_plan`。
- 当前版本的backend文本检索仅在适配器定义和专项测试发现 `CppIndicatorBatchPlan`，未发现它自动成为既有业务服务后端。
- [历史验收记录](../../verification/cpp-aot-contracts.md)对应2026-09-20/21、calmetrics-engine 0.3.0、CPython3.12/macOS arm64等明确环境；其“范围结论”也没有宣称部署或全服务迁移。

## 四轴

- 批准目标：等价、错误隔离、所有权与可追溯执行凭据
- 当前实现：显式标量适配，原NJIT装配与预热仍在
- 已验证范围：本次静态；原生wheel、CTest、性能与浏览器数量属于历史记录
- 实际部署：本次未构建wheel、完整启动、测量性能或观察生产

## 限制

历史通过数不能合并为本次通过数；历史344与33等重叠组也不能重复相加。浏览器合成凭据不替代真实原生数学等价。V1、组合矩阵及时序服务不能由局部适配继承迁移完成状态。没有读取或认证外部CalMetricsEngine当前制品。

## 复核条件

依赖文件变化、服务后端选择、预热或适配范围变化必须复核。watch_globs覆盖backend、frontend/src及相关配置，以捕捉新增调用者；仅既有文件hash不变不能排除全树新增路径。外部引擎版本、工具链或制品身份变化也需人工触发复核。性能结论须按新版本和真实负载重测。
