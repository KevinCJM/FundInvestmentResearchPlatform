---
id: "developer-data-compute"
type: "topic"
title: "一份数据怎样成为可追溯的研究结果？"
domains: ["developer", "business"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/data/README.md::sha256:9b7139b71c37e06050207ae5e401a8f248a3e5aad054966be5f1fcc76b260d22", "docs/data/etl.md::sha256:6fd8b549c2a4e4336ef8752c73e4ede8b8381da9a09096bc5c5d1e163e717942", "docs/data/pit.md::sha256:3d2e36ef827cbc86b85dba45466af9b146d1c53619c44c41cb33c54d21a87abf", "docs/data/storage.md::sha256:09eb8831fc27645aed706fea1c5a66ac81c7f13f3f5dcbdbf345822dfb089353", "docs/indicators/README.md::sha256:b6ab4622f2dcb0dc08b2bdfccc5d6feec0125bc34f2f97770ec35fb3964220a9", "docs/indicators/computation-graph.md::sha256:18b81da52e84ea290344dd4f9e216d3a5189a93be4539d235b0c245b24fbb906", "docs/indicators/cpp-aot-contracts.md::sha256:df25aeae098bcd6b50f42121b4f9052d0bd9f8aa3d9122a21f31dc47aab19732", "docs/pre-investment/versioning.md::sha256:71ea867342ee951620957e96cc32b12b80d4ec936735e0726e02673b281568c3", "docs/governance/numeric-computing.md::sha256:ed5c01fbd46d8dd2048dfdb4922d853835a9a38f76f77ad9b14d817523ffdb7b", "backend/market_data.py::sha256:9c5e2a64adb399ffb9357f93d969ae95d2ad7094255424bfde834e4c680c40d3", "backend/market_data_validation.py::sha256:6f7f1b0fd9a041b7f5c0553ee4bdc9ea5b2800e86eb837370ef44e6de1279179", "backend/data_sources/etl_graph.py::sha256:7646c1352c1d48467d709dcf48c884f0aff5fc393cd1d58d349e3b1d7cebb31f", "backend/computation_graph/topology.py::sha256:de9715b25d4ed52d8a718e148de2f868752d1a286ad727d3cd408733f9bfbea9", "backend/computation_graph/series_numba.py::sha256:9b4c989633d8b94c8f9e6b4970f636dff8778ad9be507df5a734daf21c30f0b1", "backend/custom_indicators/parallel_engine.py::sha256:8d5de931cd62f3597131f96b0e253a77b6c70cd9ad470deac2a142c4d9041d2c"]
evidence_kind: "mixed"
aliases: ["计算", "数据与计算链", "数据生命周期", "PIT"]
---

# 一份数据怎样成为可追溯的研究结果？

[开发入口](../navigation/developer.md) · [系统分层](developer-architecture.md) · [运行门槛](developer-operations.md)

## 这页回答什么

“下载成功”“计算成功”和“可以应用”不是同一件事。本页把数据、执行和研究版本三条生命周期接起来，说明中间哪些门禁不能跳过。

## 核心认识

### 数据先形成有来源的候选，再讨论正式消费

数据接入先保存来源、接口与映射修订；运行冻结配置，在权限、配额、超时和范围约束下下载。原始响应经过标准映射，候选按业务键与可比口径进入多源取值。币种、价格/净值类型、复权基础不同的记录不能互补；合法但冲突的值也不能擅自平均。

候选产物保留来源、规则修订、输入hash、取值原因和质量结果。当前多源取值完成只表示统一候选生成，正式研究仍消费既有活跃快照，不能把这条候选链直接画成已经完成的正式读路径迁移。

市场快照由 `market_data.py` 统一解析，激活前经过完整性与指标抽样核验，原子更新活跃 manifest。这里还有真实兼容边界：`resolve_tushare_data_dir` 的非 strict 调用可回退根 data，`resolve_market_data_file` 缺少活动目录文件时也可回退。要判断某次运行实际用了什么，须检查消费者参数和血缘，不能仅凭“使用了 resolver”宣称严格物理快照锁定。

### 时点随真实依赖传播

PIT区分事件、可得、版本、决策四条时间轴。每个调仓日的可见输入都要约束，不能今天训练全样本再截短结果。未知可得日期、事后标签、旧版维表和人工决定要保留各自状态。

当前文档仍保留物理 vintage、各入口严格门禁传播等待核项；PIT元数据指纹不是逐行内容hash。快照、PIT和存储身份解决不同问题，三者不能互相代替。

### 可编辑定义与高效执行分层

指标从受限公式进入参数/类型校验、业务语法展开、原语DAG、共享节点和多根计划，再按所选结果裁剪依赖闭包。类型不仅是数组形状，还包括名义轴、语义、频率和价格口径。相同节点只有在版本、输入与参数绑定一致时才能共享。

数值执行使用准备好的固定签名 NJIT；稳定数组和只读视图沿链传递，必要复制留在明确边界。跨进程共享内存由所有者管理生命周期。“有切片”“一个HTTP请求”“一个njit函数”都不证明零拷贝或只扫描一次。C++ AOT 显式适配遵守另套真实执行凭据，并未自动替换现有服务。

ETL与历史情景共用画布、端口及拓扑结构，不共用领域状态模型。ETL `steps` 是真相，`inputs` 传数据、`after` 只控制顺序，布局不影响业务指纹；连线也不承诺并行。结构层的 `topology.py` 不做业务数学，但同目录另有序列数值内核，不能对整个包作“无数值代码”的概括。

### 保存的是带依赖的成果

投前产物按精确ID/hash引用上游：目标/范围 → LTCMA → SAA → TAA → 实施与研究包。上游被替代、删除或缺失时，下游保留历史内容并重新判断可用性，不把冻结结果悄悄绑定到最新版。数据存储迁移也不重写这些历史回执。

## 当前与边界

本页支持上述原文契约与静态执行结构概述，没有重跑下载、数学实验、内存基准或PIT反例。某阶段的历史通过不覆盖后续源码、数据代际和生产配置变化；候选发布、数据资格、研究应用与真实交易仍须分别判断。

## 依据与继续阅读

- [数据接入](../../data/README.md)、[ETL](../../data/etl.md)、[PIT](../../data/pit.md)、[存储](../../data/storage.md)：输入、来源与时点
- [快照解析和激活](../../../backend/market_data.py)、[快照校验](../../../backend/market_data_validation.py)：实际读路径与激活门槛
- [指标执行](../../indicators/README.md)、[公共图合同](../../indicators/computation-graph.md)、[计算规范](../../governance/numeric-computing.md)、[C++局部适配](../../indicators/cpp-aot-contracts.md)
- [ETL端口](../../../backend/data_sources/etl_graph.py)、[拓扑排序](../../../backend/computation_graph/topology.py)、[序列内核](../../../backend/computation_graph/series_numba.py)、[共享数组与worker](../../../backend/custom_indicators/parallel_engine.py)
- [投前版本与依赖](../../pre-investment/versioning.md)：冻结、替代与下游可用性

## 复核条件

来源映射、快照resolver、PIT消费者、类型/算子版本、计划缓存身份、worker所有权或产物引用规则变化时复核。新性能结论需要同环境可重复基准；新PIT结论需要具体决策日与反例，而非仅有选择器或版本号。
