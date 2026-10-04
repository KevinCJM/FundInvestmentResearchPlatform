---
id: "business-process"
type: "topic"
title: "怎样从一个投资目标完成研究并交接？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/pre-investment/README.md::sha256:2b07c559f9a8760869474181968b278527e1d7a8d713fc8fd91eaa1b960cdc6b", "docs/pre-investment/versioning.md::sha256:71ea867342ee951620957e96cc32b12b80d4ec936735e0726e02673b281568c3", "backend/strategic_allocation/versioning.py::sha256:8cd3212126b95c7d32bebe94da80bfb7a7348a66412a6b1e8a16e9af90e2dc17", "docs/pre-investment/implementation.md::sha256:1c9ba2c90ed50b08f7219bb9908ef71b40cf2de0447b19976f492a05d34b7394", "backend/pre_investment/sources.py::sha256:e190e59f629c655b978143972da60421cd8b349c5aac1f3998df8f72262fc32a", "docs/regimes/README.md::sha256:51ff8ea5f087218661f35ec24ad8cd19c5d11fb34725f2bdadfbe471c8839afd"]
evidence_kind: "mixed"
aliases: ["完整投研流程", "端到端投研流程", "研究交接"]
---

# 怎样从一个投资目标完成研究并交接？

## 这页回答什么

说明每一步消费什么、形成什么，以及研究完成后如何面对真实投资。产品研究是长期横向能力，不应被缩成某个投前页面的附属步骤。

## 核心认识

投前主链分三段：

1. **确定基础**：先有已发布 RiskScale 参考；在 Mandate 中确认目标、资金事实与风险授权，再选择研究范围，生成并确认 LTCMA
2. **形成配置**：SAA 使用目标和精确 LTCMA 版本研究长期大类政策；按需做 TAA；产品配置承接 SAA 或 TAA 的预算
3. **验证定稿**：确认实施映射、产品风险、费用、现金与剩余资金，把同一候选汇成研究包，冻结后验证，再形成研究定稿

研究范围有两条平等路径：Product-first 从已发布产品池构建产品范围和大类；Strategic-first 先定义战略资产与指数／产品研究代理。两条路径在 LTCMA 汇合，不能跳过长期假设直接获得正式 SAA。当前范围新建流程要求绑定已发布目标；LTCMA 计算职责独立于授权，不表示用户流程可以绕过范围绑定。

**TAA 可选。** 纯 SAA 研究直接进入产品与资金，不必伪造零偏离 TAA。研究代理用于估计收益与风险，并不自动形成实施映射。合格的大类 SAA／TAA 可在没有实际产品映射时研究和保存，真实产品应用前再核验映射、可投资域和日期。

成果沿链钉住精确版本与内容 hash。上游改变不回写旧成果；旧版本仍可读，但新下游是否可用要看统一的 ready／stale／blocked 状态。风险标尺新版本本身不使旧绑定失效；战略范围的非配置改版也有明确等价例外，不能把“任何变化全部重做”当统一规则。

## 当前与边界

产品与资金、候选验证、研究定稿已有基础实现，直接 SAA 来源也由实施服务处理。流程第 07 步是汇总视图，不是保存产品候选后进入验证的额外强制页面。能自由打开模块列表，不代表已经满足计算或采纳门禁。

研究定稿之后，目标体系还包括主体／账户／真实组合、外部事实分摊与 Booking、投资账和绩效反馈；这些环节尚未形成完整正式闭环。AI 可辅助解释、构造提案和回填编辑器，不能授予实时或投资资格。修改输入必须重新验证，旧报告不能为新候选放行。

## 依据与继续阅读

- [如何完成一次研究](../../pre-investment/README.md#如何完成一次研究)、[三段九步与交接](../../pre-investment/README.md#投前总览与操作路径)是用户流程权威
- [版本依赖图](../../pre-investment/versioning.md#依赖图)、[可用性规则](../../pre-investment/versioning.md#可用性规则)及[统一状态实现](../../../backend/strategic_allocation/versioning.py)解释旧成果与新引用
- [实施与研究包主链](../../pre-investment/implementation.md#统一主链与最小数据对象)、[SAA／TAA 来源解析](../../../backend/pre_investment/sources.py)支持可选 TAA 交接
- [AI 辅助设计算法](../../regimes/README.md#ai-辅助设计算法)限定辅助权限；[目标配置](business-allocation.md)与[产品实施](business-products.md)展开各段

## 复核条件

步骤准入、来源种类、版本生命周期、产品映射或验证／定稿契约变化时复核。此次未重新跑完整用户流程，不据此声明实盘、历史 PIT 或生产交接通过。
