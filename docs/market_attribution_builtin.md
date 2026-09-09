# 市场归因与指标中心的当前边界

指标中心不再注册市场归因；普通目录、兼容目录及直接读取/计算入口都不提供此前的固定沪深300指标。对应固定指数变量、专用取数分支及算子编译特例已清除。删除前对本地持久化配置进行了引用检索，未发现该固定指标 ID 或变量引用；未修改用户指标、研究记录、指数数据或快照。

市场模型研究属于因子研究中心的业务范围，但本次清理不表示已经在因子研究中心新增或迁移了该模型。当前指标中心使用独立标量指标；相关回撤指标共享内部计算图和区间状态，不再提供业务层多标量指标。

当前契约由 `backend/tests/test_builtin_market_attribution.py` 的移除回归检查，以及 `backend/tests/test_independent_indicator_workflow.py` 验证。正常的组合 `benchmark_returns` 入参和因子研究数据源不受影响。

旧市场归因源码内容已经清空。DevSpace 本次没有文件删除动作，`backend/custom_indicators/market_attribution.py` 的空文件仍待物理删除；这不计为已完成的文件删除。

当前实施与测试结果见 `indicator_operator_primitives_acceptance.md`，物理待清理清单见 `indicator_retirement_manifest.json`。历史源码追溯使用 Git，不在文档中保存旧实现。
