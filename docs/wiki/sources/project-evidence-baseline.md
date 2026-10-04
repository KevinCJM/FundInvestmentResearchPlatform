---
id: "project-evidence-baseline"
type: "source"
title: "项目证据基线与历史审计范围"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "仓库内2026-10-04历史文档审计的记录范围与本次六卡静态核验基线；不是业务或生产认证"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
source_kind: "repo_record"
source_uri: "repo:docs/wiki/documentation-audit-2026-10-04.md"
accessed_at: "2026-10-04"
rights: "项目仓库内记录的派生摘要；原文保留原路径，外部引用不复制受限全文"
dependencies: ["docs/wiki/documentation-audit-2026-10-04.md::sha256:5099d59f22f8cbabc6c06bb525d96b807fceb52a0b85d45072c147c66070fba9", "docs/wiki/audit-files-2026-10-04.md::sha256:674295085df953ff6f4f42405f67c545b7099172e429138d146c2a33b596c3c7"]
---

# 项目证据基线与历史审计范围

## 来源事实

原始记录为[2026-10-04文档审计](../documentation-audit-2026-10-04.md)，逐文件范围见[历史覆盖台账](../audit-files-2026-10-04.md)。两者是带时间和版本的仓库记录，不是持续生效的业务测试报告。

原报告记录审核基线 `64e9b5254ad48641bf2645ec23927d4df58506df`，覆盖既有71篇docs Markdown及根入口、部署说明，并把静态核对、独立小算例、实际命令、历史未复验及生产未核分开。它保留RiskScale币种、因果预算、复权经济口径及真实数据/外部框架等未解决项。

本次知识卡固定源码基线为远端可获取的 `8ce7c06cc655d027cec6856caa873a1eafa1e51d`。本地审核时使用 `331e7a4fcaa73573e9c155eccb5666df1cf5aab3`，发布前核实两者的完整 Git tree 同为 `626497ab557754bfb875891c60104b0c896f7184`，仅提交身份不同。记录映射到远端基线，以便新 clone 解析；仅更新受这次元数据变化影响的依赖指纹，不改变审阅状态、结论或历史测试范围。先前已读取AGENTS、Hermes原JSON和相关原文/源码；环境重置后确认HEAD一致，依赖指纹重新绑定恢复环境的当前文件字节。没有把本次文档恢复表述为重新运行历史业务测试。

## 自述与推断

“原报告记载通过”属于仓库作者的历史记录，不能自动升级为本次执行事实。六张卡以各自原文、代码符号和范围回答有界问题：

- [RiskScale币种/FX证据](../claims/risk-scale-fx.md)
- [因果32节点预算](../claims/causal-budget-32.md)
- [复权经济口径与注释冲突](../claims/adjustment-economic-evidence.md)
- [NIW日频与信息量](../claims/niw-frequency.md)
- [C++局部覆盖](../claims/cpp-local-coverage.md)
- [当前助手与旧Harness](../claims/portable-current.md)

其中复权解释层为conflict，经济等价仍未核。其余supported只覆盖各卡明确写出的静态实现范围或缺口；不等于完整金融主张已经证实。

## 限制

新环境本次仅实际重跑Portable静态边界脚本，详情见对应卡。正式data/不存在，未重跑金融实证、历史测试数量、原生wheel/性能、完整前后端业务或生产验收。未查询独立框架当前HEAD、固定制品、真实身份/网关或正式迁移。外部链接仅保留原文出处，本次没有重新访问，不能将引用本身视作原始来源全文核验。

此记录只做派生索引和证据分类，不替代AGENTS、Hermes JSON、专业契约或原verification记录；不赋予业务修改、提交、推送或投资执行权限。

## 复核条件

原审计/台账变化会触发本卡字节依赖复核。具体业务结论仍以各claim自身依赖为准；不能因本卡未变推断全部代码未变。外部来源修订、数据vintage变化或部署变化不由本地hash自动发现，需要人工核对。发现新反例或更强证据时保留旧适用版本，不重写历史为“本次已验”。
