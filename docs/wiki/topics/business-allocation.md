---
id: "business-allocation"
type: "topic"
title: "目标、长期假设与配置决策怎样分工？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/wiki/claims/niw-frequency.md::sha256:bccdb9f8551f7c066724781bb2461d90dfb247b0d0b7c387022469b942134821", "docs/pre-investment/risk-scale.md::sha256:887d8a5c54c2d09fa480d1cebee8085e161b0681e32cb0aac7fd1def62c3eac8", "docs/pre-investment/mandate.md::sha256:04e6ecda893735ca1df21647facf0c2dad531a82a73f88ab0b86fa75affcc071", "docs/pre-investment/ltcma.md::sha256:964e2ce0e570c7cc6d7ebaedec8677d4d812b36041b439adb8347de4100d5d54", "backend/strategic_allocation/cma_selection.py::sha256:8fe54d210ba0d5278939fb790f05c5cfa5786da1aad93120d710e8b4f15a289a", "docs/pre-investment/saa.md::sha256:0ae0a140c8edd18b9b5f25d3dcaa9ff2f9bd06f76e681d6a7a44bb46431ece1d", "docs/pre-investment/taa.md::sha256:c95b3da5fdac99990cedbece6d7547deb8929865b0e8cb66af1528406fefed38", "docs/pre-investment/funding.md::sha256:7f93a3cc24bcac190c822d15e4707b56133e726de677bf2856f89438494971da", "docs/pre-investment/asset-classification.md::sha256:54772f21011fd30214834a9ff266d83ddc58fbd58cb3cdd0139777fffca9ef75", "docs/pre-investment/historical-frontier.md::sha256:22d2d94b4b842abbdc76fdb1c6b223f571d936b0752a454e73adab40f8aea85a"]
evidence_kind: "mixed"
aliases: ["资产配置", "RiskScale Mandate LTCMA SAA TAA", "目标约束"]
---

# 目标、长期假设与配置决策怎样分工？

## 这页回答什么

把“投资者需要多少”“市场模型预计多少”“允许承担多少风险”和“最终采用哪些权重”分开，避免用一个收益率或一条前沿代替全部判断。

## 核心认识

**RiskScale 是参考尺。** 它从参考资产历史数据、共同收益样本和有效前沿生成冻结 C1–C5，不依赖某个投资者的期限，也不是未来收益预测。C1–C5 是项目风险表达，不是行业统一适当性评级。

**Mandate 是目标与授权。** 绝对收益、基准相对、资金目标有不同语义。最低算术年收益、期间复利要求与现金流反推的所需收益不能直接取数值最大者。风险等级默认表达最高允许风险，不强迫组合增险；现金、流动性、期限与支付条件仍独立有效。

**LTCMA 是长期假设。** 它提供一致资产轴上的收益、资产协方差及均值估计不确定性，不决定授权、风险分档或组合权重。历史统计是样本估计，保存成 CMA 不自动增加前瞻证据；Manual、BL 和情景可以承载判断，但判断来源仍需研究。

**SAA 是长期政策选择。** 单模型消费一套 CMA；模式 A 当前做参数平均，原模型结果保留为诊断；模式 B 要找同一组权重，在每个原模型下满足共同要求，不能用平均矩替代。约束取 Mandate 与研究设置的交集，下游不能静默放宽。

**TAA 是有边界的临时偏离。** 它继承冻结 SAA，区分信号可得、决策和执行时钟，比较成本、风险与复核条件。保留 SAA 是有效结果；留出样本只报告，不能失败后自动重选赢家。

资金成功另行判断：路径中所有必要支付完成、期末目标达到；一次支付不足不能被之后投入抹掉。独立验证随机种子降低选优与模拟样本复用的问题，但不是新的真实市场证据。

## 当前与边界

长期情景可按年化矩和既有资格进入 SAA；条件情景研究指定期限的变化，目前可保存比较，不能作为 SAA 或 NIW 先验，服务端独立拦截。E3 原生混合路径、多 CMA 椭球、模式 B 产品联合 QCQP 等不属于现有基础能力。

“找到合格权重”可以支持数值可行；“未找到”“预算耗尽”不能直接写成数学无解。历史前沿不替代前瞻 CMA，资金收益门槛初筛不替代支付成功率。NIW 当前仍有[日频／252边界](../claims/niw-frequency.md)。

## 依据与继续阅读

- [RiskScale 核心业务](../../pre-investment/risk-scale.md#核心业务)与[目标收益和风险要求](../../pre-investment/mandate.md#统一收益与风险要求)定义参考和授权
- [LTCMA 业务边界](../../pre-investment/ltcma.md#业务边界)、[长期／条件情景](../../pre-investment/ltcma.md#自动长期情景与条件情景)及[下游资格代码](../../../backend/strategic_allocation/cma_selection.py)定义假设用途
- [SAA 当前可执行边界](../../pre-investment/saa.md#当前可执行边界)、[可行性与最优性](../../pre-investment/saa.md#可行性与最优性的分开输出)区分三种政策模式
- [TAA 时钟与信号](../../pre-investment/taa.md#04taa信号决策和执行)、[资金成功及独立验证](../../pre-investment/funding.md#概率区间选优与独立验证)说明后续研究；[大类构建](../../pre-investment/asset-classification.md#输入与输出契约)和[历史前沿](../../pre-investment/historical-frontier.md#数学与求解边界)是辅助工具

## 复核条件

新增 CMA 方法、目标口径、优化模式、资金分布、频率或应用门禁时复核。此次只支持有源职责概述与指定静态门禁，不认证全部公式、预测表现或投资资格。
