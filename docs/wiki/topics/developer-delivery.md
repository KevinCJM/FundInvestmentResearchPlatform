---
id: "developer-delivery"
type: "topic"
title: "一次开发怎样交付，什么证据才算完成？"
domains: ["developer"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["AGENTS.md::sha256:c691bc21980b359d49811d69517f864e66d680956b07bd2292a23f62892dee80", "docs/governance/documentation.md::sha256:2de6ec7df1733848cb8d2c891f37afcd1e4604c6d34983f2c7df41a9a484bca6", "docs/governance/numeric-computing.md::sha256:ed5c01fbd46d8dd2048dfdb4922d853835a9a38f76f77ad9b14d817523ffdb7b", "docs/governance/operator-contracts.md::sha256:c33cc8cb3037d908208906b64ad1135efdb323175d2d946cb640b812f5303d94", "docs/governance/branch-submission-rules.md::sha256:571898a2dd34445dce22bf4b34f0102789d0671ecea565b38295f157b54557ad", "docs/governance/submission-workflow.md::sha256:e0602d59daa5fda40ed41d4dfece7f7d3b850c3708d07204eb67ff9542ac6e5b", "docs/governance/branch-protection.md::sha256:367e7ca94d41ffe294d49f06758bc3a35a4903b8a6c187033478c2cc416ee79f", "docs/frontend/README.md::sha256:258fc38478f77af14b79f63bba270da51ea3c1382b874be4d58f28d62d611cb6", "docs/data/acquisition-protocol.md::sha256:74b6148b3d575859a67ac42da42f2c56027bac2efd2cb125be173828ffb5f1cc", "docs/verification/engineering.md::sha256:676b4ce2c3e9e77cbcaa07f541beb39c4688971efa8c0087d3f1eeb5a380f80b"]
evidence_kind: "mixed"
aliases: ["研发约束", "交付", "测试验收"]
---

# 一次开发怎样交付，什么证据才算完成？

[开发入口](../navigation/developer.md) · [设计决定](developer-decisions.md) · [部署门槛](developer-operations.md)

## 这页回答什么

交付不止是改完文件或看到测试绿灯。本页说明如何限定变更、找到权威契约、选择验证并保留尚未完成的范围。具体协议仍以原文件为准。

## 核心认识

### 先找对责任和授权范围

工程任务先读AGENTS，再依次读repo_map、task_routes和pitfalls。任务路由决定先看哪些模块与何时扩展，模块事实提供代码、测试、配置及最小回归；Wiki帮助理解关系，不能改这套读取顺序。

开始时检查工作区差异，明确本任务的文件和共享文件中的改动归属。需求授权决定能改什么；“优化”“统一实现”或文档冲突不自动授权删改既有算法。开发授权也不自动包含commit、push、合并、生产部署或远端规则修改。

### 验收针对实际行为，不针对文件名称

数值改动需证明真实调用进入允许的执行后端，核对数值等价、NaN/Inf、空样本、边界窗口、dtype、时点及确定性。零拷贝需要实际共享内存和生命周期证据；性能结论需要可重复基准，不能只看装饰器或`copy=False`。

算子和模板还要验证可编辑步骤、真实连线、历史定义、单节点预览及公式往返。历史结果不能因重构被悄悄换算法；变更数学口径要明确新版本。

前端改动先读设计准则，首页另读首页要求。单测、类型检查和构建覆盖工程结构；浏览器才验证受影响交互、不同视口、键盘/焦点、失败恢复、对比度和必要降级。静态样稿或截图不能替代当前页面验收。

下载相关任务另按采集协议核对权限、全局限频、超时、隔离smoke和快照激活；普通离线回归使用固定夹具，不污染正式研究数据。

### 四轴分别记账

- **批准目标**：用户要求、现行契约及明确采纳决定
- **当前实现**：指定源码版本的装配、调用链与输出
- **已验证范围**：实际执行的版本、环境、输入、命令和结果；失败、未跑和历史记录分别写
- **实际部署**：目标环境的制品、配置、数据迁移与运行观察

文件存在不是实现接通，测试文件存在不是测试通过，push或合并不是生产部署。某个局部通过只能支持它自己的范围。

### 文档与提交是独立关口

里程碑和收尾按实际差异核对原主题、专业契约、计划和验收记录。改变行为就更新权威说明；无行为变化可以说明为何不改。原repo_map唯一维护文档角色和归属，task_routes负责匹配，pitfalls负责已证实的可复用问题。

文档检查验证目录、链接、索引、计划和候选；Hermes验证路由引用与覆盖。它们不能证明语义、金融资格或生产成功。历史计划中的未决问题必须有对应复验证据才能关闭。

提交与PR遵循原分支流程。Bot审核须对应最新完整HEAD；仅“已完成”、旧提交通过或无反对意见都不算批准。Owner例外须满足原协议，人类普通合并授权不等于允许绕过Bot。源码发布与实际部署继续分开。

## 当前与边界

本页是流程概述，没有执行业务回归、PR检查或远端配置核验。原文保留的历史CI证据不证明当前required checks或Bot服务状态。选择测试时回到匹配模块的原最小回归，避免维护另一份逐渐过期的命令清单。

## 依据与继续阅读

- [AGENTS](../../../AGENTS.md)、[模块事实](../../repo_map.json)、[任务路由](../../task_routes.json)、[已证实坑点](../../pitfalls.json)
- [数值规范](../../governance/numeric-computing.md)、[算子治理](../../governance/operator-contracts.md)、[前端准则](../../frontend/README.md)、[采集协议](../../data/acquisition-protocol.md)
- [文档维护](../../governance/documentation.md)、[提交规范](../../governance/branch-submission-rules.md)、[提交全流程](../../governance/submission-workflow.md)、[分支保护技术边界](../../governance/branch-protection.md)
- [工程历史验收](../../verification/engineering.md)：学习证据的版本与范围，不把旧结果算成本轮重跑

## 复核条件

读取协议、提交授权规则、数值/前端契约或检查器职责改变时复核；依赖变化会使本页需要重读，但不自动说明原规则错误。每次交付仍须依据该任务最终差异和真实执行证据判断完成。
