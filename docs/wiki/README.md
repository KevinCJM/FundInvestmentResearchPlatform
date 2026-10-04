# 项目知识库

理解资产配置投研与组合管理平台的业务逻辑、系统结构和设计来由，并把新的研究变成有出处、可复核、能持续维护的知识。

## 两个入口

### [业务知识](navigation/business.md)

从“服务谁、解决什么问题”出发，理解产品研究、目标与约束、长期假设、资产配置、产品实施、验证和真实投资管理的完整关系。

### [开发知识](navigation/developer.md)

从系统分层、模块职责、数据与计算链出发，理解当前实现、架构决定、前端交互、研发约束和运行交付条件。

## 按你现在的问题进入

- **先了解项目全貌**：[平台定位与边界](topics/business-purpose.md) → [完整投研流程](topics/business-process.md) → [系统怎样分层](topics/developer-architecture.md)
- **开展一次资产配置研究**：[目标与配置对象](topics/business-allocation.md) → [产品研究与实施](topics/business-products.md) → [金融口径和证据资格](topics/business-evidence.md)
- **理解市场状态和风险研究**：[状态、历史事件与情景](topics/business-regimes.md) → [数据生命周期与计算](topics/developer-data-compute.md)
- **判断研究以后还缺什么**：[真实组合、运营与核算](topics/business-investment.md) → [运行与生产门槛](topics/developer-operations.md)
- **回溯为什么这样设计**：[业务决定与当前能力](topics/business-decisions.md) · [技术决定与历史演进](topics/developer-decisions.md)
- **着手开发或修改模块**：[模块与前端交接](topics/developer-modules.md) → [研发、验证与交付](topics/developer-delivery.md)
- **加入研报、论文或竞品研究**：[研究资料怎样变成知识](topics/research-library.md) → [来源与比较模板](templates.md) → [完整操作手册](workflow.md)

## 找全原文、证据和状态

[全项目知识与附档目录](catalog.md)覆盖所有受管文档、Hermes模块及相关图片、原型、JSON和benchmark材料；[原文档索引](../README.md)继续作为原项目入口。目录中的active是文档生命周期，不能读作功能已交付。

主题页是有源解释，原需求、专业契约、代码和验收仍在原处。具体风险、反例与核验卡放在各主题的证据层；历史记录、未采纳草稿和未知项均保留身份，不组成另一套当前设计。

## 检索与维护

AI 任务从 [项目 Wiki 技能](../../skills/obsidian-wiki/SKILL.md) 进入五种按需流程。修改前查相关知识及原证据，修改后判断影响、必要更新并验收；它复用现有脚本/Hermes，不安装另一套 Wiki。

- 在Obsidian全文搜索关键词，从主题回原文；反向链接查看某项证据被哪里使用
- AI问答按[取证协议](workflow.md#ai问答与按需取证)区分全局、模块、历史原因和外部对照，只取必要材料
- 新来源经入库去重→未核草稿→证据审阅→主题整合；更正经反馈→复核→修订，步骤和命令见[运行手册](workflow.md)
- 用check/queue检查过期依赖、冲突、待核及未整合材料；hash未变不证明事实正确，历史测试不代表现在重跑

知识文件随Git共享，机器配置独立；云端和本机都以各自项目根打开vault。完整方案和交付范围见[设计](llm-wiki-design.md)，两端实际验收分别记录。
