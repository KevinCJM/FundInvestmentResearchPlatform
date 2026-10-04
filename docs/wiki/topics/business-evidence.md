---
id: "business-evidence"
type: "topic"
title: "怎样读懂金融结果，并判断证据够不够？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/wiki/claims/risk-scale-fx.md::sha256:557b92464f2d307364471e20b0fe85342e8ee9023d1d7fef2406cdf061fe3a8b", "docs/wiki/claims/adjustment-economic-evidence.md::sha256:2b4b2ac086f92da7bc3abe32feed57d8c12081a57905359095481989422db950", "docs/wiki/claims/niw-frequency.md::sha256:bccdb9f8551f7c066724781bb2461d90dfb247b0d0b7c387022469b942134821", "docs/data/pit.md::sha256:3d2e36ef827cbc86b85dba45466af9b146d1c53619c44c41cb33c54d21a87abf", "docs/data/adjusted-price.md::sha256:7d3b549e892d5a76f671ac7fececc29695f770b06a050f699bbcd11f7334a009", "docs/data/README.md::sha256:9b7139b71c37e06050207ae5e401a8f248a3e5aad054966be5f1fcc76b260d22", "docs/data/tushare-download.md::sha256:70fce750c1a06de57a258996fb64efae55f6c1f7dde00478df81f3dd4f23f43d", "docs/pre-investment/ltcma.md::sha256:964e2ce0e570c7cc6d7ebaedec8677d4d812b36041b439adb8347de4100d5d54", "docs/pre-investment/taa-numeric-contract.md::sha256:7afef72655d914a85a82750e1156672722eab740db60b9ed692b10581a902b64", "docs/product/research-parameters.md::sha256:523d3050e425cd915f6be4cac7597f4caf3b44f385164982fe10a94ace1e6ad6", "docs/pre-investment/implementation.md::sha256:1c9ba2c90ed50b08f7219bb9908ef71b40cf2de0447b19976f492a05d34b7394", "docs/indicators/causality.md::sha256:1791f0d6fe96e9764c74ffb26c2e8223d7c8c6bbdc81258bb6191f02caa751db"]
evidence_kind: "mixed"
aliases: ["金融口径", "证据资格", "收益风险"]
---

# 怎样读懂金融结果，并判断证据够不够？

## 这页回答什么

比较结果前先核对其经济含义、时间和证据范围。一个绿色结果、漂亮曲线或成功保存，不能替代这些条件。

## 核心认识

**先核对口径。** 原始价格、复权价格、复权净值不能静默互换；缺失因子不补 1。周期简单收益、年化算术参数、一年复合收益随机变量与资金所需 CAGR 不是同一量。费用还要注明 NAV 已含什么、模型额外扣了什么，避免重复计算。

**再核对风险含义。** 资产收益协方差衡量实现风险；均值估计协方差衡量参数不确定性；模型分歧和模拟抽样误差又是另外两层。不能把后验均值协方差直接加进风险后仍声称是原资产风险，也不能把 Wilson 区间当作未来收益保证。

**同时核对四条时间轴。** event time 是事件属于哪天，available time 是何时可取得，version time 是哪次修订，decision time 是研究站在哪天。今天全样本训练后裁短曲线，不是历史决策回放。PIT 元数据指纹也不等于逐行内容 hash，不能单凭封版证明任意历史 vintage 可还原。

**最后核对实际比较对象。** 资产轴、币种、频率、样本、费用和持仓规则必须相容。LTCMA／RiskScale 的共同相邻单日样本，与 TAA 的完整共同观察区间有不同目的；不能为了统一而把跨日持有收益冒称单日收益，或直接丢掉 TAA 跨日涨跌。

证据至少分四层读：批准的目标与契约、当前实现、已验证范围、实际部署。研究包还独立展示研究 scope、实施条件、历史 PIT 和独立复核。硬规则失败或不可用不能靠勾选知悉放行；不适用需要规则依据。

## 当前与边界

现有系统已有多处时点、版本和验证门禁，但全链物理 vintage、真实费用、经济有效性及认证独立审批并未由通用测试统一证明。固定种子能复现模拟，不能让已反复查看的市场留出样本重新独立。

三个有界核验可作为阅读实例：[RiskScale 的币种／FX边界](../claims/risk-scale-fx.md)说明 CNY 标签不足以证明代理已经换汇；[复权迁移的经济等价证据](../claims/adjustment-economic-evidence.md)区分代码迁移与经济认证；[NIW 频率边界](../claims/niw-frequency.md)限制直接换频解释。它们不代表其他主张都已核验。

## 依据与继续阅读

- [四条时间轴](../../data/pit.md#四条时间轴)、[PIT 诚实边界](../../data/pit.md#数据能力与诚实边界)与[仍需核验的缺口](../../data/pit.md#仍需核验的缺口)
- [复权数据契约](../../data/adjusted-price.md#3-数据契约)、[多源先可比再比较](../../data/README.md#先可比再比较)、[下载状态如何判断](../../data/tushare-download.md#状态如何判断)
- [四种收益概念](../../pre-investment/ltcma.md#四种收益概念不得混用)、[TAA共同区间](../../pre-investment/taa-numeric-contract.md#共同区间数值口径)、[研究参数分离目标设计](../../product/research-parameters.md#1-需求结论与边界)
- [验证结果按事实表达](../../pre-investment/implementation.md#验证结果按事实表达)、[定稿与资格分开](../../pre-investment/implementation.md#定稿状态与资格分开)及[因果审计方法边界](../../indicators/causality.md#方法核验边界2026-10-04)

## 复核条件

收益、复权、币种、频率、样本、费用、PIT 或资格规则改变时复核对应主张及依赖，不只刷新指纹。本页是有源阅读框架，未完成全项目金融复算、真实数据和生产验证。
