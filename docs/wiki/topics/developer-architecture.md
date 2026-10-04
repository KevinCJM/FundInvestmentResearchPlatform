---
id: "developer-architecture"
type: "topic"
title: "这个平台怎样分层，哪些能力属于谁？"
domains: ["developer", "business"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["README.md::sha256:2ca8b5692848b7d1050a30730f12b5b1ea2e2490f7d8a6302c279bfa4e78477e", "backend/app.py::sha256:ed789f9e6f70b6ce5493608e0329cdfc94792bdddeea9bf73aee5de4eab64095", "frontend/src/App.tsx::sha256:87fd236147dda9c10d3936e4befcfed009313ed64e947b759c0be36d0072e922", "frontend/src/app/processRegistry.ts::sha256:b50d1466b8fdb1cf463f3fc5fddf78873b36885098fadae78f29bec6e337eda7", "shared/platform-navigation.json::sha256:781673cb6d0b7099bf116383d804247bd4c756b65b18d004670fdf2aa74a1a09", "docs/product/requirements.md::sha256:39e169152d837e65e8891d3489b6b67f3077fa4539a6a528e41926501dfc05c6", "docs/product/domain-language.md::sha256:f7d328f84c032a3b42a54cfc543a24f392b8781860ca8acc8dc5d8b379a258ce", "docs/indicators/README.md::sha256:b6ab4622f2dcb0dc08b2bdfccc5d6feec0125bc34f2f97770ec35fb3964220a9", "docs/indicators/cpp-aot-contracts.md::sha256:df25aeae098bcd6b50f42121b4f9052d0bd9f8aa3d9122a21f31dc47aab19732", "docs/research/portable-agent-platform-integration.md::sha256:aa7052eab7818c2bb91f0f4a1183c48026ad6e39e682f5980740d6ea393e32fe"]
evidence_kind: "mixed"
aliases: ["系统分层", "系统架构", "模块职责"]
---

# 这个平台怎样分层，哪些能力属于谁？

[开发入口](../navigation/developer.md) · [数据与计算](developer-data-compute.md) · [模块交接](developer-modules.md)

## 这页回答什么

平台不是一组孤立的基金页面，而是围绕研究输入、计算、冻结成果和下游应用组织的系统。本页解释当前可见调用链与外部边界，帮助先找对责任层，再读专业契约；不替代原需求、源码或 Hermes 路由。

## 核心认识

### 页面组织任务，业务服务持有计算与保存权威

浏览器以 React、TypeScript 和 Vite 为基础。`App.tsx` 装配路由与研究上下文，`processRegistry.ts` 从共享导航读取阶段、节点和能力标签。页面及服务客户端把用户选择送到 FastAPI；后端 `app.py` 同时装配模块路由与少量仍在入口文件中的业务端点，因此不能只搜 `services/` 判断某个 API 是否存在。

业务逻辑分布在指标、组合、配置、状态、因子、择时及风险等领域服务。服务负责对象校验、数据选择、版本与任务编排；生产数值路径主要进入固定签名 NJIT 内核。输入和结果不是靠页面展示文本定义，公式、类型、时点、缺失及保存规则各有后端契约。

数据层以本地 Parquet、版本化 JSON、SQLite 配置/业务记录和不可变研究制品组成。生产构建由 FastAPI 同源托管 `frontend/dist`。这描述源码的基础部署形态，不证明某个实际环境已经按它部署。

### 横切基础连接领域，不替领域做决定

存储身份与锁决定能否安全读取数据；PIT 决定研究日能看见什么；执行凭据说明数值后端与准备状态；冻结版本说明结果来自哪些输入。这几层共同约束研究，但任何一层通过都不能替代投资资格或真实业务授权。

[模块与交接](developer-modules.md)按产品研究、投前决策和共用研究能力展开；[数据与计算](developer-data-compute.md)说明它们如何共享输入与执行，而不重复建设相似算法。

### 独立项目的边界必须逐项看

Portable Web Agent 负责通用会话、模型调用、恢复和聊天组件。平台保留薄接入、`research_access` 业务权限/投影、草稿和人工确认保存；助手经 HTTP 调用原业务服务，不成为第二套计算引擎。

CalMetricsEngine 的 C++ AOT 已有显式单产品标量适配契约，但既有服务路由与 NJIT 预热没有因此整体切换。行情/持仓指标的目标归属 CalMetricsCenter 仍是规划：原文明确该项目仅初始化，本仓现有行情指标尚未迁出。

## 当前与边界

- **当前实现**：研究计算与版本链已有核心能力；共享导航以 `available`、`partial`、`prototype` 表达入口范围，标签本身不是本轮功能测试结果
- **规划与原型**：真实组合运营、投中执行、核算和完整投后大量入口仍为原型或局部能力；页面、数据模型定义和演示数值不能当成实盘闭环
- **历史证据**：旧 AI Harness、旧组件和旧测试记录用于追溯迁移基线，当前装配应看 Portable 接入及现行源码
- **实际部署**：本页没有运行服务、外部框架制品或生产身份链证据，具体门槛见[运行与部署](developer-operations.md)

## 依据与继续阅读

- [项目入口](../../../README.md)、[产品范围](../../product/requirements.md)与[领域语言](../../product/domain-language.md)：平台做什么、研究和运营怎样分界
- [前端装配](../../../frontend/src/App.tsx)、[导航注册](../../../frontend/src/app/processRegistry.ts)、[共享导航事实](../../../shared/platform-navigation.json)：入口与能力状态
- [后端装配](../../../backend/app.py)：`lifespan`、路由、`health` 与静态托管
- [Portable 职责和版本集成](../../research/portable-agent-platform-integration.md)、[指标中心](../../indicators/README.md)、[C++ AOT 范围](../../indicators/cpp-aot-contracts.md)：独立项目及迁移边界

## 复核条件

入口装配、共享能力状态、领域服务归属、助手部署协议或指标后端切换时重读相关调用链。若要把“源码支持”升级为“已运行”或“已部署”，必须补对应提交、环境、输入与实际结果，不能只更新本页日期。
