---
id: "niw-frequency"
type: "claim"
title: "NIW：当前日频252与先验信息量边界"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "当前NIW请求、续更、年化内核与强度输入组件；不认证全部模型数理、校准或投资效果"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
evidence_kind: "static"
dependencies: ["docs/pre-investment/ltcma.md::sha256:964e2ce0e570c7cc6d7ebaedec8677d4d812b36041b439adb8347de4100d5d54", "backend/strategic_allocation/cma_model_contracts.py::sha256:ec6594d9a58e011e08fd97414a635fbd42e656d5c667d74b5c3f107aba696466", "backend/strategic_allocation/cma_statistical_models.py::sha256:ae6df6e208f46fb3245f470f0fa91d61bc620d4c96726af8c30069b0a07e7f81", "backend/strategic_allocation/cma_statistical_kernels.py::sha256:ed990fb805dc6669b5b4315f29cd14bf2f34ac1bfe04221cfc838878c0a82dfd", "frontend/src/components/ltcma/LtcmaNiwStrengths.tsx::sha256:d1d77540d3104b0442f1dbc09741ec6ca52fa0f3a3e46b5ac9881b6d314e0961"]
---

# NIW：当前日频252与先验信息量边界

## 主张

当前NIW请求、样本与续更链限定日频/252。重新建立先验须显式填写均值和风险的日频等效观察数；续更继承同频后验并只接受新增样本。当前没有日/月切换或弱/中/强模板。未来换频不能仅转换矩单位便宣称先验信息量和分布等价。

## 证据

- [LTCMA：先验参数来源](../../pre-investment/ltcma.md#先验参数来源)：第412–451行，尤其第422、438–440行；同文“第一版实现与失败条件”第507–511行明确当前能力。
- [cma_model_contracts.py](../../../backend/strategic_allocation/cma_model_contracts.py)：`StatisticalCmaContext` 第180–196行使用daily、252 Literal；`BayesianCmaRequest.prior_strength` 第204–219行要求recenter提供两种强度，continue不得重复提供。
- [cma_statistical_models.py](../../../backend/strategic_allocation/cma_statistical_models.py)：`statistical_result` 第39–72行检查续更后验的daily/252身份，拒绝与旧截止日重叠。
- [cma_statistical_kernels.py](../../../backend/strategic_allocation/cma_statistical_kernels.py)：`niw_update` 第228–261行及 `recenter_niw_prior` 第265行起显式使用252；资产协方差乘252，均值后验协方差乘252²。
- [LtcmaNiwStrengths.tsx](../../../frontend/src/components/ltcma/LtcmaNiwStrengths.tsx)：两个显式NumberInput；样本参考显示T及比例，没有自动回填强度或强度模板。

上述为本次静态核对，不是本次NIW数值测试或浏览器验收。

## 四轴

- 批准目标：先验与似然同频；均值不确定性和资产风险分开
- 当前实现：日频约束、显式强度和同频新增证据续更
- 已验证范围：请求契约→编排→内核→强度组件的静态链
- 实际部署：未观察实际页面、正式数据或模型运行结果

## 限制

本卡不认证完整NIW矩阵推导、采样、预测、校准和实证效果。日频年化包含模型假设。跨频率设计是未来扩展，不能写成当前能力；“24个等效月”不能直接解释为24条日观察。

## 复核条件

频率契约、年化、先验强度、续更、样本或模板UI变化需复核。支持新频率前需重新定义信息量并提供专项数理、接口和实际执行证据。
