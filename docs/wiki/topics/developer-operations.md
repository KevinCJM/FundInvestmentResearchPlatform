---
id: "developer-operations"
type: "topic"
title: "怎样启动与部署，哪些门槛必须单独验证？"
domains: ["developer"]
review_state: "partial"
result: "supported"
scope: "2026-10-04原文概述；2026-10-12仅复核本地助手启动链路及其依赖字节，其余历史证据和生产边界保留。"
reviewed_at: "2026-10-12"
reviewed_by: "AI 原文与实现核对"
source_revision: "24ac20ad1d23bd5f6e7c6408dd306e75003a8c3e"
dependencies: ["README.md::sha256:afa1e8b266853eabf5073134a347f1956c2014d7f3d364a5e27397d61fc91b1c", "Dockerfile::sha256:36bcdcfcb0253387a6eff89c1017b742280719f4eeb17c3971bc8db416df3472", "docker-compose.yml::sha256:cf441918fa1f2bf1a7a6f13d19858adc63c568276197a2aed5b8a4f360f80458", "deploy/README.md::sha256:092426bc202453443f8aafffd2225f2738c421dc2c844166741a5676ee463de7", "deploy/portable-agent.compose.yml::sha256:1edcb9dcef1d5dda13c24f1f353e8f5a6a2721a92c8bf8312ef40850209c024c", "backend/app.py::sha256:ed789f9e6f70b6ce5493608e0329cdfc94792bdddeea9bf73aee5de4eab64095", "backend/integrations/portable_agent/service.py::sha256:d64119e0a26e53617a686417a13d8fbb5cab3aa200b59c9c81615bb0ebaa0d6c", "docs/data/storage.md::sha256:09eb8831fc27645aed706fea1c5a66ac81c7f13f3f5dcbdbf345822dfb089353", "docs/data/acquisition-protocol.md::sha256:74b6148b3d575859a67ac42da42f2c56027bac2efd2cb125be173828ffb5f1cc", "docs/indicators/cpp-aot-contracts.md::sha256:df25aeae098bcd6b50f42121b4f9052d0bd9f8aa3d9122a21f31dc47aab19732", "docs/research/portable-agent-platform-integration.md::sha256:88b5d323dd49365d616395ef531363f520c7d9debd21201fb9a76b0b612615c9", "start_services.sh::sha256:b0bf28bd899c230ad6f43b3346c5459b6d26e1cfdda8a656052816974817cbae"]
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

本地开发使用`./start_services.sh`统一启动平台后端、网页和独立安装的v2 Agent；两个Python环境保持隔离。脚本生成受限本机身份与服务凭证，一次性导入原模型配置，重复启动保留后续用户设置并补启缺失服务；复用后端前核对当前身份、应用签名和服务凭证，不兼容的已归属旧后端会安全重启后重新检查。已有助手数据缺少配对配置时要求恢复备份，不创建新主体。模型地址从全部配置来源逐项校验（含空白/控制字符与错误主机字符），失败不创建目录/密钥且不回显地址；原模型的超时、推理等级和上下文窗口随一次性导入保留。首次安装在密钥命令前保存身份与初始化状态，临时目录中的密钥经验证后原子发布；失败可沿用同一身份和已发布密钥重试，未完成时不能正常启动。独立子进程只继承进程、语言、临时目录与 TLS 证书设置，运行中的 Agent 另获得配对宿主授权，不继承平台或云服务密钥；推进初始化前同步文件和父目录项，同步失败可保留身份和已发布密钥重试。接入检查覆盖登录、上下文授权、v2能力、网页组件、模型选择及同源`/assistant`代理；暂时未就绪会重试，失败会说明原因。公开部署仍走正式身份与固定镜像，不能沿用本机登录。迁移历史业务数据需要独立授权、dry-run、幂等/中断恢复与权限复核，离线夹具通过不能代替真实迁移。

## 当前与边界

平台已有Portable接入源码及带版本的历史离线工程证据；独立框架当前HEAD、最终受审固定制品、生产HTTPS/身份/外部端口/SSE和真实数据迁移在本知识整理中均未确认。C++ AOT只提供显式局部适配，不能据此宣布平台NJIT启动成本已消失。

修复后的候选启动/接入离线回归146项通过，宿主/边界补充回归70项通过；初始启动版本在原本机工作区已验证补启缺失Agent和浏览器助手回复，后续配对保护与旧后端升级修复通过离线反例回归。新候选的后端鉴权检查另对原本机服务实测匹配，没有执行重启；另在临时目录用实际 v2 密钥命令及离线工具目录夹具验证首次/重复初始化配对不变，未连接模型。不代表冷启动全量重启或生产验收。本页未构建镜像或访问生产。生产切换是独立动作，不能从写文档、安装Obsidian或源码发布推定已获授权或已完成。

## 依据与继续阅读

- [项目运行说明](../../../README.md)、[启动脚本](../../../start_services.sh)、[Dockerfile](../../../Dockerfile)、[基础Compose](../../../docker-compose.yml)
- [部署说明](../../../deploy/README.md)、[存储协议](../../data/storage.md)、[采集协议](../../data/acquisition-protocol.md)
- [应用lifespan与health](../../../backend/app.py)、[release门禁](../../../backend/integrations/portable_agent/service.py)、[助手Compose](../../../deploy/portable-agent.compose.yml)
- [Portable版本、迁移与分版本验收](../../research/portable-agent-platform-integration.md)、[C++适配范围](../../indicators/cpp-aot-contracts.md)

## 复核条件

解释器/依赖、构建输入、存储协议、启动预热、公开网络拓扑、身份权限、release锁或迁移流程变化时复核。只有取得目标环境的制品与实际运行证据，才能新增相应部署结论；历史离线记录不能自动升级。

本轮重读 deploy 说明及新增地址/迁移差异：非法地址测试逐项核验配置与数据字节未变，非默认上下文窗口进入 v2 配置请求。仅更新这一条实际变化的说明依赖；没有改写其他历史部署和数值证据。

第四轮重读子进程环境隔离、文件/父目录 fsync 及三项失败重试回归，临时目录真实密钥 CLI 复验通过。仅更新实际变化的 deploy 依赖；未据故障注入宣称实际掉电、硬件或生产验收。
