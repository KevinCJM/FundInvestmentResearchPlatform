---
id: "adjustment-economic-evidence"
type: "claim"
title: "复权经济口径：恒等式证据与源码注释冲突"
domains: ["business", "developer"]
review_state: "reviewed"
result: "conflict"
scope: "前收盘价推导公式及其解释层；经济等价未核，历史大样本未在本次复现"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
evidence_kind: "mixed"
dependencies: ["docs/data/adjusted-price.md::sha256:7d3b549e892d5a76f671ac7fececc29695f770b06a050f699bbcd11f7334a009", "backend/data_sources/price_adjustment.py::sha256:03911f27c797eb235e673a7a377326afeb0acc4c767f32b7837a176683b22920", "backend/data_sources/price_adjustment_numba.py::sha256:78f230ceda3be6a97ae87156489a8fbf10cc3617805d08fa323899385086926d", "backend/tests/test_price_adjustment.py::sha256:ba841cfe6f4d7bd8332d5e9b6e548e3cc5dacae2b1df33c6b857df25f5eea198"]
---

# 复权经济口径：恒等式证据与源码注释冲突

## 主张

当前推导实现 `F[t]=F[t−1]×close[t−1]/pre_close[t]`。因此相邻复权收益等于 `close[t]/pre_close[t]−1` 是代数结果，不能独立证明真实企业行动、现金分红再投或官方因子等价。现行文档明确保留此限制，但源码注释仍无条件宣称相应经济语义，形成解释层冲突。

## 证据

- [复权价格契约：3.2因子口径、3.3口径选择、4.1新增](../../data/adjusted-price.md)：1,577,335行、4.44e−16、6542行及2.27e−03均明确属于历史记录；同源恒等不构成独立经济证据。
- [price_adjustment_numba.py](../../../backend/data_sources/price_adjustment_numba.py)：`pre_close_factor_kernel` 第70–99行直接实现递推，非法比值之后截断。其docstring仍无条件称 pre_close 是 ex-rights previous close，且收益是 real dividend-inclusive return。
- [price_adjustment.py](../../../backend/data_sources/price_adjustment.py)：`source_factor`、`factor_divergence`、`attach_adjusted_prices`保留官方因子对齐、偏差统计及三种策略；DEFAULT_FACTOR_POLICY前第31–34行注释称两条来源是 same quantity out of a different field，强于现行文档的证据边界。
- [test_price_adjustment.py](../../../backend/tests/test_price_adjustment.py)：`test_factor_is_one_until_an_event_and_reproduces_the_real_return` 第40–55行手工构造“第三天除息0.3”的previous数组，再断言同源恒等。它是合成测试文本，不是独立企业行动核验。

本次静态核对支持公式及解释冲突。未运行业务测试，未重新取得官方因子或企业行动。原文引用的外部资料本次没有重新访问。

## 四轴

- 批准目标：因子来源明确；没有因子时不能用未复权价冒充
- 当前实现：三种策略、归一化、断链留空、偏差统计
- 已验证范围：公式、测试输入来源与文档/注释冲突；历史大样本未重跑
- 实际部署：当前账号覆盖、正式快照和实际分红处理没有核验

## 限制

卡的 conflict 仅针对解释层过强表述；经济等价本身仍是 unverified，不据此否定全部数值实现。正式 data/ 在本云检出不存在。不能以历史极小残差、AI摘要或测试函数名称替代独立证据，也不能擅自替换默认策略或历史制品。

## 复核条件

公式、默认策略、注释、权威契约或相关测试变化即应复核。要关闭经济口径未核，需独立逐事件企业行动、现金流、正式因子和输入vintage证据，并明确分红处理定义。修订过强源码注释也须获得相应授权；仅重复同源残差不足以关闭。
