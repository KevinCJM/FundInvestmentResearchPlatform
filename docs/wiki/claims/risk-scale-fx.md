---
id: "risk-scale-fx"
type: "claim"
title: "RiskScale：CNY标签不构成来源币种与FX证据"
domains: ["business", "developer"]
review_state: "reviewed"
result: "supported"
scope: "当前平台参考输入请求、来源身份校验和收益合成链；不覆盖实际标尺结果、投资资格或生产部署"
reviewed_at: "2026-10-04"
reviewed_by: "AI"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
evidence_kind: "static"
dependencies: ["docs/pre-investment/risk-scale.md::sha256:887d8a5c54c2d09fa480d1cebee8085e161b0681e32cb0aac7fd1def62c3eac8", "backend/strategic_allocation/reference_contracts.py::sha256:936968d2122f808aba491d21364dd88994100955d8705a48aaa3d8dd0c4e95b0", "backend/strategic_allocation/reference_sources.py::sha256:aa52b85384edf0f910f2503e0b120c0e64fb09a8a7eda1908e1349bdca76f1a1", "backend/strategic_allocation/reference_inputs.py::sha256:b0e51585628930cebb00529f6b6acc109526bd57110c2de7931085d8f32cb622", "backend/historical_regimes/data.py::sha256:112796843421dd7772f3a6f6ede111beaad0e6b34a191a8f3a7134679de15693"]
---

# RiskScale：CNY标签不构成来源币种与FX证据

## 主张

当前 RiskScale 请求固定声明 CNY 与 same_currency_no_conversion，但加载、校验和冻结参考来源时没有建立逐序列币种认证。原值进入收益合成，因此固定标签不能证明跨币种代理已经构成经验证的 CNY 组合风险。

## 证据

- [风险标尺：代理与再平衡](../../pre-investment/risk-scale.md#代理与再平衡)：代理选择不做币种准入；明确固定标签与来源证据之间的缺口。
- [reference_contracts.py](../../../backend/strategic_allocation/reference_contracts.py)：`ReferenceInputRequest` 第67–92行把 currency 固定为 CNY、fx_basis 固定为 same_currency_no_conversion；`ProxyComponent` 没有币种字段。
- [reference_sources.py](../../../backend/strategic_allocation/reference_sources.py)：`ReferenceSources._capability` 依据类型、字段与可读状态；`load` 第73行起返回 `frame['value']`，冻结 identity 含类型、字段、频率、内容hash和复权checksum，没有逐来源币种认证。
- [reference_inputs.py](../../../backend/strategic_allocation/reference_inputs.py)：`ReferenceInputs._input_calculation` 第171–282行校验 kind、series_id、field、frequency、hash，随后执行 `adjacent_returns`、`proxy_returns` 和 `annual_moments`。
- [historical_regimes/data.py](../../../backend/historical_regimes/data.py)：`_index_bundle` 第213行起把选定列直接赋给 value；`resolve_target` 第380行起按来源分发。这里没有建立指数币种转换证据。

上述为2026-10-04对固定版本的静态核对，恢复时文件指纹绑定当前字节；未运行业务计算。

## 四轴

- 批准目标：广泛代理选择用于研究参考；ETF/基金使用复权数据
- 当前实现：固定请求标签，原序列收益合成，缺少逐来源币种认证
- 已验证范围：请求→来源→身份→收益链的静态证据缺口
- 实际部署：没有实际标尺、生产输入或FX数据观察

## 限制

此卡支持“所查链路没有建立该证据”，不声称每个来源一定币种不同，也不声称某个已保存结果已经错误。币种/FX证据缺失时，跨币种原值混合至多按本币价格变化的合成参考解释。业务解释、币种门禁或FX适配需要另行决定，不能由本卡自动放行投资资格。

## 复核条件

请求、来源加载、冻结身份、收益链或引用契约变化即应复核。若新增逐代理币种认证、FX序列及反例测试，需要核对真实输入、日期对齐和经济含义。仅修改标签不能关闭缺口。
