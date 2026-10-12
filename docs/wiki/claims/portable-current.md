---
id: "portable-current"
type: "claim"
title: "当前助手：Portable宿主链与旧Harness边界"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "当前平台宿主装配、历史文档定位及本次静态边界检查；不证明外部发布或生产迁移；2026-10-12 仅复核本次启动差异，其他原文及历史证据保持各自适用日期"
reviewed_at: "2026-10-12"
reviewed_by: "AI"
source_revision: "158f059b58cd5545aafc79874c4dcce7b98ef77c"
evidence_kind: "mixed"
watch_globs: ["backend/**", "frontend/src/**", "backend/requirements.txt", "frontend/package.json", "frontend/package-lock.json", "Dockerfile", "docker-compose.yml", ".gitmodules", "deploy/**", "config/portable-agent/**", "scripts/check_portable_agent_boundary.py", "start_services.sh", "scripts/service_process.py", "scripts/portable_agent_local.py"]
dependencies: ["docs/research/portable-agent-platform-integration.md::sha256:88b5d323dd49365d616395ef531363f520c7d9debd21201fb9a76b0b612615c9", "docs/research/ai-functions-design.md::sha256:bfe87a5e27b2e0299df673270801b27476864974009edea4dbb23067e78f2fc6", "docs/research/ai-agent-harness-progress-design-2026-09-20.md::sha256:5a706c6e849bdc8a6adbf84d3f0c7b7f882ecf408b604ae3d2f69289af764e53", "backend/app.py::sha256:ed789f9e6f70b6ce5493608e0329cdfc94792bdddeea9bf73aee5de4eab64095", "backend/integrations/portable_agent/service.py::sha256:d64119e0a26e53617a686417a13d8fbb5cab3aa200b59c9c81615bb0ebaa0d6c", "frontend/src/integrations/portable-agent/PortableAgentMount.tsx::sha256:0dd8cf72030a6fb233b269414d7e3b2334c85ed7b3c4992c75b76a05dacbe01f", "scripts/check_portable_agent_boundary.py::sha256:db0f9b600f05d82aaca8e30dcd8a95c6082ca56525f62881f1504a87e9558d82"]
---

# 当前助手：Portable宿主链与旧Harness边界

## 主张

当前平台以Portable外部组件和宿主业务接口为装配入口。旧AI/Harness文档是迁移前契约与历史证据，不能作为当前调用链。平台静态边界通过不证明独立框架固定制品已发布，也不证明生产身份、网关和数据迁移完成。

## 证据

- [Portable整体迁移设计](../../research/portable-agent-platform-integration.md)：第1、4、15和17.1节定义当前平台职责、历史基线及MIG-07剩余条件。
- [AI功能设计](../../research/ai-functions-design.md)及[旧Harness设计](../../research/ai-agent-harness-progress-design-2026-09-20.md)顶部迁移声明明确旧路径属于历史行为基线。
- [app.py](../../../backend/app.py) 第328–368行使用 `install_portable_agent` 和 `research_access.research_pages.runtime_callbacks`安装宿主适配。
- [portable_agent/service.py](../../../backend/integrations/portable_agent/service.py)依赖research_access；`release_contract`第29–44行在production缺发行锁时返回ASSISTANT_RELEASE_REQUIRED 503。
- [PortableAgentMount.tsx](../../../frontend/src/integrations/portable-agent/PortableAgentMount.tsx) 第67–103行执行registerContext→bootstrap→创建portable-agent元素→按外部endpoint和expectedRelease配置。
- 固定版本的 `git ls-files` 未列出backend/agent、frontend/src/components/agent、standalone-agent或默认config/portable-agent/release.json。
- 本次恢复后的新云环境于 **2026-10-04 07:51:44 UTC**，使用 **Python 3.12.14** 从仓库根执行 `python3 scripts/check_portable_agent_boundary.py`：退出码0，`status=passed`、`errors=[]`。[脚本](../../../scripts/check_portable_agent_boundary.py)实际扫描了当前平台旧路径、导入、前端旧API引用及配置依赖。

本次边界命令是新环境实际执行结果；不是把重置前记录或旧验收数当作当次测试。其他金融计算和业务套件本次未运行。

## 四轴

- 批准目标：独立框架持有通用智能体，平台持有业务权威
- 当前实现：宿主适配、research_access业务层及外部组件挂载
- 已验证范围：当前源码静态核对与本次实际边界脚本通过
- 实际部署：独立框架HEAD、固定镜像、真实身份/网关及数据导入未核；MIG-07仍保留未完成门槛

## 限制

默认release.json未跟踪不等于任何部署都未配置外部锁路径。脚本检查已知路径、导入和文本模式，不能证明所有动态/改名情况绝无运行器。历史HTTP、模型和浏览器记录只适用于各自版本，本次没有读取外部框架仓库或认证真实模型效果与费用。

## 复核条件

watch_globs覆盖backend、frontend/src、依赖/部署配置及Portable配置目录；这些范围新增、删除或修改文件均需重跑边界脚本。脚本自身变化也使旧结果失效。单个既有文件hash不变不能维持全树检查的有效性。外部发行锁、制品、身份、网关或数据切换需取得对应独立运行证据，不能由静态passed关闭生产门槛。

2026-10-12 本地生命周期复核：重读迁移契约、本页及当前启动差异，并通过宿主/边界补充回归 70 项。平台新增本地编排独立安装的 v2 Agent，仍使用独立环境和宿主权限；启动不下载源码或内嵌通用运行器。修复后的启动/接入回归另有 169 项通过，不与旧金融验收相加；外部生产制品、正式身份与 MIG-07 门槛仍未关闭。另外核对本轮配对配置缺失保护与旧后端鉴权重启修复：复用前校验当前身份、签名和服务凭证，已有数据缺配对配置要求恢复；新脚本仍只编排独立服务，未改变上述业务职责边界。

本轮 Bot 修复复核：核对所有模型地址的 HTTP(S)/主机/端口及无凭据、查询、片段校验，在运行目录与密钥创建前完成；导入提交保留旧模型上下文窗口，131072 token 非默认值已覆盖。第三轮另复核主机空白/控制字符拒绝以及可恢复初始化：密钥命令前已保存配对身份，临时密钥验证后原子发布；失败重试不换身份或正式主密钥，未完成状态阻止服务启动。新增地址来源、无副作用及中断恢复反例仅验证本机启动适配，不改变独立运行器、业务授权或生产边界。

第四轮 Bot 修复复核：重读最小子进程环境白名单与文件/目录同步顺序。Agent、探针及密钥命令不继承平台、供应商或云服务密钥，仅运行中的 Agent 显式取得配对宿主授权；配对身份、父目录和已发布主密钥在推进阶段前 fsync，同步失败后的重试保留身份与密钥。三项同步故障注入和真实 v2 CLI 临时目录验证通过，未执行实际掉电或生产验收。

第五轮 Bot 修复复核：初始化前按项目根目录规范化解释器绝对路径，保留 venv 可执行文件软链接，缺失文件不创建配置/数据。相对和绝对路径、不同当前目录、正常服务读取与重试的五项回归通过；实际独立 v2 CLI 在临时目录复验相对路径首次/重复初始化，不涉及供应商、原服务或数据。

第六轮 Bot 修复复核：显式模型端口在初始化副作用前限于 1–65535；省略端口保留协议默认值。重读模型地址校验与三来源失败反例、合法边界用例；本轮 159 项启动/接入回归通过，未改变独立运行器或生产范围。

第七轮 Bot 修复复核：本机后端、浏览器与 Agent URL 的显式端口也在读取/生成身份前限制为 1–65535；0 与超范围端口给出明确范围提示。重读三类服务六项无副作用反例和合法边界检查，本轮 169 项启动/接入回归通过，原运行与生产边界保留。
