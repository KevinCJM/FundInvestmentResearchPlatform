---
id: "developer-operations"
type: "topic"
title: "怎样启动与部署，哪些门槛必须单独验证？"
domains: ["developer"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["README.md::sha256:2ca8b5692848b7d1050a30730f12b5b1ea2e2490f7d8a6302c279bfa4e78477e", "Dockerfile::sha256:36bcdcfcb0253387a6eff89c1017b742280719f4eeb17c3971bc8db416df3472", "docker-compose.yml::sha256:cf441918fa1f2bf1a7a6f13d19858adc63c568276197a2aed5b8a4f360f80458", "deploy/README.md::sha256:1be51d8fcff3951555a9421e2f990006ba05f701b4fc0b7a0c6ec9f79ca61377", "deploy/portable-agent.compose.yml::sha256:1edcb9dcef1d5dda13c24f1f353e8f5a6a2721a92c8bf8312ef40850209c024c", "backend/app.py::sha256:ed789f9e6f70b6ce5493608e0329cdfc94792bdddeea9bf73aee5de4eab64095", "backend/integrations/portable_agent/service.py::sha256:d64119e0a26e53617a686417a13d8fbb5cab3aa200b59c9c81615bb0ebaa0d6c", "docs/data/storage.md::sha256:09eb8831fc27645aed706fea1c5a66ac81c7f13f3f5dcbdbf345822dfb089353", "docs/data/acquisition-protocol.md::sha256:74b6148b3d575859a67ac42da42f2c56027bac2efd2cb125be173828ffb5f1cc", "docs/indicators/cpp-aot-contracts.md::sha256:df25aeae098bcd6b50f42121b4f9052d0bd9f8aa3d9122a21f31dc47aab19732", "docs/research/portable-agent-platform-integration.md::sha256:aa7052eab7818c2bb91f0f4a1183c48026ad6e39e682f5980740d6ea393e32fe", "start_services.sh::sha256:d38ac48a99fcc4ced24e68f87bd111d05eac4355217acf7224dc829b2a4f9f57"]
evidence_kind: "mixed"
aliases: ["部署", "运行", "生产门槛"]
---

# 怎样启动与部署，哪些门槛必须单独验证？

[开发入口](../navigation/developer.md) · [系统分层](developer-architecture.md) · [交付与验收](developer-delivery.md)

## 这页回答什么

开发能打开页面、健康检查可访问、容器配置能解析、源码已发布，各自只证明一部分。本页把基础应用和独立助手的启动/部署门槛分开，避免用离线绿灯宣布生产可用。

## 核心认识

### 第一层：环境和数据要真实存在

项目要求受控Python3.12及已安装后端依赖，前端要求Node20或以上。开发时Vite代理API；构建后FastAPI可以同源托管SPA。`python3`名称存在不等于解释器版本与依赖已就绪。

数据不是普通Git源码。存储入口可能指向受管理的独立数据区，运行前核对身份、挂载和锁；掉盘不能静默回退旧副本或创建同名空目录。迁移整个数据区会涉及行情、研究、配置及本地凭据，须依存储协议准备与恢复。

基础Dockerfile含 `COPY data ./data`，而已审阅的干净checkout没有data目录。运行卷不能补救缺失的构建输入；镜像构建前既要准备获准输入，也要检查构建上下文中是否混入私有数据和凭据。本页不建议复制任何未获准材料。

### 第二层：端口存活不等于计算就绪

`app.py`用存储生命周期包装应用启动，并在`lifespan`内准备主要数值内核、保存计划及指标worker。预热不完整会阻止相应就绪流程；正式请求不能依赖首次调用临时编译。

`/api/health`返回`numba_warmup`及`pit_audit`。需要核对主进程/worker准备结果，PIT诊断也应单独解释；只读到`ok: true`或看到端口监听不证明所有研究数据取得资格。存储中途不可用时，多数请求返回503，存储诊断入口保留。

### 第三层：基础应用生产条件

基础形态是一个FastAPI服务托管API与前端，配持久化数据卷、公开HTTPS入口及健康检查。跨域仅在确有需要时配置明确来源。公开部署的数据刷新/凭据入口需要HTTPS和身份保护，不能将开发默认值视为实际安全配置。重部署或迁移前按原运维流程保护持久数据。

### 第四层：Portable是独立交付

助手覆盖配置增加独立Agent与同源网关。只有网关公开端口，公开路径按白名单代理，内部工具不能从外部绕过；SSE需验证流式转发。业务与Agent各有独立持久卷，不能互相挂载数据或运行实现。

生产需要已审阅固定镜像digest和release文件，绑定源码提交、协议主版本、widget、manifest与能力集合；身份来源、当前权限查询及内部服务凭据另行配置。`release_contract()`在production缺锁或锁不匹配时失败关闭，不能换成demo凭据或浮动latest绕过。

开发时Agent另起进程，平台`start_services.sh`只管理平台。`VITE_PORTABLE_AGENT_TARGET`只是代理地址，不自动建立可信登录。迁移历史业务数据需要独立授权、dry-run、幂等/中断恢复与权限复核，离线夹具通过不能代替真实迁移。

## 当前与边界

平台已有Portable接入源码及带版本的历史离线工程证据；独立框架当前HEAD、最终受审固定制品、生产HTTPS/身份/外部端口/SSE和真实数据迁移在本知识整理中均未确认。C++ AOT只提供显式局部适配，不能据此宣布平台NJIT启动成本已消失。

本页未启动服务、构建镜像、访问生产或执行任何登录。生产切换是独立动作，不能从写文档、安装Obsidian或源码发布推定已获授权或已完成。

## 依据与继续阅读

- [项目运行说明](../../../README.md)、[启动脚本](../../../start_services.sh)、[Dockerfile](../../../Dockerfile)、[基础Compose](../../../docker-compose.yml)
- [部署说明](../../../deploy/README.md)、[存储协议](../../data/storage.md)、[采集协议](../../data/acquisition-protocol.md)
- [应用lifespan与health](../../../backend/app.py)、[release门禁](../../../backend/integrations/portable_agent/service.py)、[助手Compose](../../../deploy/portable-agent.compose.yml)
- [Portable版本、迁移与分版本验收](../../research/portable-agent-platform-integration.md)、[C++适配范围](../../indicators/cpp-aot-contracts.md)

## 复核条件

解释器/依赖、构建输入、存储协议、启动预热、公开网络拓扑、身份权限、release锁或迁移流程变化时复核。只有取得目标环境的制品与实际运行证据，才能新增相应部署结论；历史离线记录不能自动升级。
