---
id: "developer-decisions"
type: "topic"
title: "为什么采用这些架构决定，哪些旧设计已经被替代？"
domains: ["developer", "business"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["AGENTS.md::sha256:c691bc21980b359d49811d69517f864e66d680956b07bd2292a23f62892dee80", "docs/data/storage.md::sha256:09eb8831fc27645aed706fea1c5a66ac81c7f13f3f5dcbdbf345822dfb089353", "docs/indicators/computation-graph.md::sha256:18b81da52e84ea290344dd4f9e216d3a5189a93be4539d235b0c245b24fbb906", "docs/indicators/README.md::sha256:b6ab4622f2dcb0dc08b2bdfccc5d6feec0125bc34f2f97770ec35fb3964220a9", "docs/governance/numeric-computing.md::sha256:ed5c01fbd46d8dd2048dfdb4922d853835a9a38f76f77ad9b14d817523ffdb7b", "docs/governance/operator-contracts.md::sha256:c33cc8cb3037d908208906b64ad1135efdb323175d2d946cb640b812f5303d94", "docs/frontend/README.md::sha256:258fc38478f77af14b79f63bba270da51ea3c1382b874be4d58f28d62d611cb6", "docs/research/allocation-methods.md::sha256:8d2f0a79a54005ad165237a9adf504d1a96b36100361a922b4f2f62aad31bad6", "docs/research/regime-methods.md::sha256:04dbadaddf4a30b1c1fcb9a6a7a36b9c3f8980e1eca554e379b102f4d6ceb4fd", "docs/research/ai-functions-design.md::sha256:bfe87a5e27b2e0299df673270801b27476864974009edea4dbb23067e78f2fc6", "docs/research/ai-agent-reusable-architecture-design-2026-09-21.md::sha256:c3361c106abb7776ad8f676f1b944301d262daa8af90d106599acbcddb8a5bd4", "docs/research/ai-agent-context-compaction-design-2026-09-20.md::sha256:3f2ed6d5c48bcdfb378dd97eeb39a1b98ff6b588e87eb0ee3356d82ffb6450f8", "docs/research/portable-agent-platform-integration.md::sha256:aa7052eab7818c2bb91f0f4a1183c48026ad6e39e682f5980740d6ea393e32fe"]
evidence_kind: "mixed"
aliases: ["架构决定", "历史原因", "技术演进"]
---

# 为什么采用这些架构决定，哪些旧设计已经被替代？

[开发入口](../navigation/developer.md) · [系统分层](developer-architecture.md) · [研发交付](developer-delivery.md)

## 这页回答什么

本页保留跨模块的重要取舍及原文明确给出的原因，不把“看起来合理”补成未经记录的项目动机。决定仍归各专业契约，本页只连接问题、选择和代价。

## 核心认识

### 一个当前实现，历史交给Git

项目禁止为留档或回滚保留同功能新旧双实现。明确原因是当前调用、构建和发布只应指向一个版本；回滚由Git完成。有真实兼容合同且有测试的薄适配可以存在，但不能因此保留第二套数学实现。这项治理并不授权删除尚未获准变更的功能。

### 数据区整体解耦，不只移动最终Parquet

下载还产生原始响应、候选、检查点和临时文件，研究和凭据也有持久化要求。因此存储迁移面向整个data区，使用稳定逻辑入口、存储身份与共享锁。代价是切换须离线、掉盘失败关闭，且首次迁移与再次跨盘搬迁不能混为一谈：历史任务可能持有物理路径。

### 复用图基础，不把完整算法封进黑盒

公共画布和拓扑从既有历史情景能力抽取，避免再造外观相似的第二套实现。业务数学留在领域服务；可替换的特征、条件、分类与统计必须保留可编辑步骤。只有拆分会破坏递推或联合拟合语义的耦合内核才保留整体，原因须写明。

指标业务结果保持独立，内部执行可以共享DAG、回归状态和依赖闭包。这样复用计算而不把多个独立指标强绑成一种业务对象；共享也不能跨越数据轴、版本、参数或浮点顺序。

### 把准备成本放到明确阶段

固定签名NJIT和worker预热把正式请求限制在已准备的执行路径，避免首请求编译与静默Python回退。代价是启动/显式准备成本必须接受和观察。C++ AOT是经明确接入的另一路径，不能因它存在就声称全部服务已消除预热。

### 战略研究与产品实施分层，保留两种入口

原研究明确反驳“先产品池一定错误”。战略机会集回答希望持有哪些经济风险，产品池回答研究日有哪些合格工具；两者可以迭代。于是保留Product first，同时允许Strategy first及显式映射。缺产品的大类仍可研究，应用到交易产品前再核验，不能用改名称、填零或自动分配来掩盖覆盖缺口。

历史状态与实时信号同样分层：历史参考用完整区间定义经济语义，实时识别另作因果与独立样本验证。统计簇不能仅因数量合适就命名为牛、熊或复苏。

### 通用助手迁出，业务权威留下

Portable方案选择独立服务、固定版本、HTTP工具及同版Web Component。原文给出的直接原因是平台内部不再保留通用智能体实现；submodule只锁源码，不能自动提供接口适配或运行隔离。平台保留数据准入、权限、草稿与人工保存，防止通用框架拥有第二套业务事实。

前端也采用有意的分层：首页可展示品牌，工作台优先数据密度；阶段色不承担主操作语义，动画预算留给图表与表格。这些是原规范的项目取舍，不是通用视觉法则。

## 当前与边界

旧AI功能、Harness、压缩、共用架构文档保存迁移前行为及历史证据；当前装配读Portable迁移设计和现行代码。新设计文件的日期较晚并不自动证明获准、实现或部署。上述决定的实际成本没有在本页重新测量；未记录的更深层动机保持未知。

## 依据与继续阅读

- [AGENTS代码治理](../../../AGENTS.md)、[存储契约](../../data/storage.md)
- [公共计算图](../../indicators/computation-graph.md)、[指标执行](../../indicators/README.md)、[算子治理](../../governance/operator-contracts.md)、[计算规范](../../governance/numeric-computing.md)
- [配置方法与取舍](../../research/allocation-methods.md)、[状态方法与取舍](../../research/regime-methods.md)、[前端准则](../../frontend/README.md)
- 历史线索：[AI功能设计](../../research/ai-functions-design.md)、[上下文管理](../../research/ai-agent-context-compaction-design-2026-09-20.md)、[共用架构](../../research/ai-agent-reusable-architecture-design-2026-09-21.md)；当前接入：[Portable迁移](../../research/portable-agent-platform-integration.md)

## 复核条件

明确批准的新取舍、等价性反例或新的部署证据出现时，先更新原主题决定与替代关系，再更新本页。不能为使叙事一致而删除历史证据、重写未获准目标或擅自改变业务算法。
