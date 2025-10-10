# Task Checklist — 指定目标策略回测改造

## 任务 1：梳理窗口裁剪逻辑
- [x] 状态：已完成
- **说明**：在 `backend/backtest_engine.py` 的 `_slice_fit_data` 与 `_compute_model_weights` 中移除 `use_all_data` 默认干扰，严格以 `window_mode`+`data_len` 控制样本截取：`all`=全部历史，`rollingN`=最近 N 条。对非法 N 需容错（例如小于 2 时强制设为 2）。
- **验证**：编写单元测试构造 10 日样本，分别对 `window_mode='all'` 与 `rollingN`, `data_len=4` 验证 `_slice_fit_data` 返回的索引列表与期望相符。

## 任务 2：实现“样本不足顺延”机制
- [ ] 状态：待处理
- **说明**：在 `_static_or_rebalanced` 内判断各调仓日是否拥有足够数据：
  - `all` 模式至少 2 条记录。
  - `rollingN` 模式至少 `data_len` 条（若 `data_len` 大于可用样本上限时，需在策略层报错或回退）。
  - 若首批数据不足，将调仓日期顺延到首个满足条件的日期，并以该日期权重作为初始权重。若全程都不足，返回可读错误。
- **验证**：设计测试：给定 20 日 NAV + 月度调仓，设 `rollingN`=15；确认首期标记在首个满足条件的日期，之前序列为空；再平衡时 markers 与 series 首点一致。

## 任务 3：统一前后端权重计算接口
- [ ] 状态：待处理
- **说明**：`backend/services/strategy_routes.py` 的 `compute-weights` 与 `compute-schedule-weights` API 必须复用与回测一致的窗口逻辑，避免硬编码 `window_mode='rollingN'`。可将截取/顺延逻辑提取为共用函数。
- **验证**：通过 API 集成测试：使用 `window_mode='all'` 与 `rollingN` 两种配置调用接口，比较返回权重与直接调用回测中 `_compute_model_weights` 的结果是否一致。

## 任务 4：保持回测输出结构稳定
- [ ] 状态：待处理
- **说明**：顺延首期后需确保 `series` 与 `markers` 同步：首个再平衡点的 markers 中 `date`、`value` 与 series 对应点相符，静态策略仍无 markers。若顺延导致序列头部缺口，应在前端提示。
- **验证**：E2E 回测测试：构建带/不带再平衡的策略组合，比较 `markers` 列表长度及首元素日期与前端展示逻辑。

## 任务 5：文档与沟通
- [ ] 状态：待处理
- **说明**：在 `AGENTS.md` 或新增开发记录中补充窗口模式/样本要求解读，提醒数据不足时的行为与配置建议。
- **验证**：文档自查，确保包含四种场景描述及异常处理说明。

## 任务 6：回归与 CI
- [ ] 状态：待处理
- **说明**：完成代码调整后运行 `pytest`（若已有测试集）及 `npm run build`，补充新的单测到后端测试套件。
- **验证**：CI 通过；本地命令输出通过记录在 PR 描述。
