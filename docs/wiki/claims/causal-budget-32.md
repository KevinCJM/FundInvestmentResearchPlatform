---
id: "causal-budget-32"
type: "claim"
title: "因果审计：32节点预算不等于全图通过"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "audit_expression节点预算、warning与裁决合并的静态语义；未验证真实保存或发布入口可绕过"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
evidence_kind: "static"
dependencies: ["docs/indicators/causality.md::sha256:1791f0d6fe96e9764c74ffb26c2e8223d7c8c6bbdc81258bb6191f02caa751db", "backend/causality/audit.py::sha256:cc30e9bac71d199c16d9e90db9fb08dba9bc14f52a5819ca7f6afdbe8c70a249", "backend/causality/probes.py::sha256:1ceaf828603588e11aeec12251524ca2ed2b3767828268d5854fea7e96cedb5c", "backend/causality/rolling_scope.py::sha256:6b1029c7eea4ef5b22946ad07a0d97b2fa4c71abdcb554e08b704d8ab9064f8b", "backend/cal_indicators/rolling_scope.py::sha256:a0cd568463d06f478244199b100012b726c1e04c1ed39429f4018fc0ac5507f9"]
---

# 因果审计：32节点预算不等于全图通过

## 主张

`audit_expression` 默认最多审核32个计入预算的节点。达到预算只写“其余节点未审计”warning并退出；warning没有作为 UNKNOWN 加入裁决。因此已审核部分无反例时，汇总仍可能为 CAUSAL，该枚举不能单独证明全图完整审核。

## 证据

- [因果性审计：公式级审计](../../indicators/causality.md#6-公式级审计l2运行时)：第195–212行描述实际探针与覆盖缺口。
- [audit.py](../../../backend/causality/audit.py)：`audit_expression` 第393–497行；第399行默认 max_nodes=32，第435–437行 warning+break，第470–475行仅 findings、WINDOW_CONSUMING 进入合并，空列表则追加 CAUSAL。
- 同文件 `ExpressionReport.blocked` 第313–317行仅对 LEAK 返回 true。这个类字段本身不证明所有调用者如何处理 UNKNOWN 或 warning。
- [probes.py](../../../backend/causality/probes.py)：`_SEVERITY`、`merge_verdicts` 第41–65行只合并传入 verdict 列表。
- [audit.py](../../../backend/causality/audit.py) 的 `_probe_plan` 使用尾部扰动，rolling_apply 分派到 [causality/rolling_scope.py](../../../backend/causality/rolling_scope.py) 的 `probe_scoped_plan`；外层节点由 [cal_indicators/rolling_scope.py](../../../backend/cal_indicators/rolling_scope.py) 的 `outside_nodes` 取得。

证据为当次静态阅读；未构造并运行超过32节点的真实入口反例。当前生产公式审计不能表述为普遍同时运行P1/P2。

## 四轴

- 批准目标：未知不冒充完整通过，有限探针保留局限
- 当前实现：逐节点/滚动子图审核；预算不足作为warning
- 已验证范围：预算、节点选择、合并和报告字段的静态分支
- 实际部署：未观察每个保存/发布入口对覆盖不足的最终处理

## 限制

32是计入审核的节点预算，不是原始AST总节点数；变量及部分无时间轴节点会跳过，滚动作用域另有审核。不能写成“第33个任意节点必未审”，也不能声称已证实生产绕过。有限探针即使覆盖全部选定节点，也不是所有可能输入的形式证明。

## 复核条件

预算、节点枚举、滚动作用域、合并或blocked语义变化需复核。关闭缺口需明确覆盖不足采用UNKNOWN、阻断或显式接受的业务决定，并沿真实入口运行针对性反例；不能以无关测试通过替代。
