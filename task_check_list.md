# Task Checklist — 指定目标策略回测改造

## 任务 1：梳理窗口裁剪逻辑
- [x] 状态：已完成
- **说明**：在 `backend/backtest_engine.py` 的 `_slice_fit_data` 与 `_compute_model_weights` 中移除 `use_all_data` 默认干扰，严格以 `window_mode`+`data_len` 控制样本截取：`all`=全部历史，`rollingN`=最近 N 条。对非法 N 需容错（例如小于 2 时强制设为 2）。
- **测试覆盖度**：`backend/tests/test_window_slice.py` 覆盖 (1) `all` 模式全量、(2) `rollingN` 基础, (3) data_len 小于 2 时的下限保护, (4) `window_mode=None` 默认行为, (5) 大小写不敏感, (6) `data_len=None` 时的最少样本约束。

## 任务 2：实现“样本不足顺延”机制
- [ ] 状态：待处理
- **说明**：在 `_static_or_rebalanced` 内判断各调仓日是否拥有足够数据：
  - `all` 模式至少 2 条记录。
  - `rollingN` 模式至少 `data_len` 条（若 `data_len` 大于可用样本上限时，需在策略层报错或回退）。
  - 若首批数据不足，将调仓日期顺延到首个满足条件的日期，并以该日期权重作为初始权重。若全程都不足，返回可读错误。
- **测试覆盖度**：计划新增单测与集成测试，覆盖 `all`/`rollingN` 在数据不足时的顺延、完全不足时的报错、以及首个权重与 markers 对齐。

## 任务 3：统一前后端权重计算接口
- [ ] 状态：待处理
- **说明**：`backend/services/strategy_routes.py` 的 `compute-weights` 与 `compute-schedule-weights` API 必须复用与回测一致的窗口逻辑，避免硬编码 `window_mode='rollingN'`。可将截取/顺延逻辑提取为共用函数。
- **测试覆盖度**：计划使用 FastAPI 测试客户端构造 `all` 与 `rollingN` 案例，对比接口响应与内部函数的权重，确保截取逻辑一致。

## 任务 4：保持回测输出结构稳定
- [ ] 状态：待处理
- **说明**：顺延首期后需确保 `series` 与 `markers` 同步：首个再平衡点的 markers 中 `date`、`value` 与 series 对应点相符，静态策略仍无 markers。若顺延导致序列头部缺口，应在前端提示。
- **测试覆盖度**：计划通过回测函数集成测试校验 `markers`/`series` 对齐、静态策略无 markers，以及 JSON 输出结构稳定。

## 任务 5：文档与沟通
- [ ] 状态：待处理
- **说明**：在 `AGENTS.md` 或新增开发记录中补充窗口模式/样本要求解读，提醒数据不足时的行为与配置建议。
- **测试覆盖度**：文档审阅 checklist（人工检查），确保四种场景、异常处理与配置建议均被覆盖。

## 任务 6：回归与 CI
- [ ] 状态：待处理
- **说明**：完成代码调整后运行 `pytest`（若已有测试集）及 `npm run build`，补充新的单测到后端测试套件。
- **测试覆盖度**：全量 `pytest` + 前端 `npm run build` 作为最低标准，并在 PR 中记录命令输出。
