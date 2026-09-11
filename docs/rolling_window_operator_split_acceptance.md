# 滚动窗口算子拆分验收记录

## 结论

本次改造通过验收。当前作者协议使用 `rolling_window` 表达窗口语义，再连接普通统计归约；`rolling_mean / rolling_std / rolling_min / rolling_max` 不再出现在当前新建资源目录中，只保留历史 2.3 兼容能力。

## 实施结果

- 当前 typed DSL / operator registry：`2.4.0`；编译器：`typed-numba-5`。
- 新增逻辑中间类型 `window<time,window>[T,W]` 与算子 `rolling_window(values, window[, min_periods])`。
- `rolling_window` 不能直接作为指标输出，也不能参与普通加减乘除。
- `mean / std / variance / min_value / max_value` 可消费滚动窗口并输出时间序列。
- 当前公式示例：`std(rolling_window(returns, 20), 1)`；画布真实显示“滚动窗口 -> 全元素标准差”两个节点。
- 生产执行不物化 `T×W` 矩阵；编译器将“窗口 + 归约”直接融合到固定签名 NJIT 滚动内核。
- 当前算子 NJIT 注册覆盖保持 `117/117`，`rolling_window` 属于 `compiler_fused_no_materialization` 逻辑节点；`python_fallback=0`。
- 可变参数所有权迁移到 `rolling_window.window / min_periods`；`std.ddof` 继续属于标准差算子。
- 标量转滚动时序改为生成 `mean(rolling_window(...)) / std(rolling_window(...), ddof)` 等透明组合。
- Excel 继续生成原生窗口范围 + AVERAGE/STDEV/MIN/MAX 等公式，不暴露 DSL 私有函数、不生成窗口矩阵。
- 数学 LaTeX / 完整计算说明按拆分后的真实 DAG 展示。
- 因果审计新增 `rolling_window=causal`；逻辑窗口只依赖当前及过去观察值。

## 历史兼容

- 内置时序指标当前版本为 revision 2 / DSL 2.4。
- 同一内置指标 revision 1 / DSL 2.3 仍可按 `indicator_id + revision` 获取并运行。
- 2.3 的 `rolling_mean / rolling_std / rolling_min / rolling_max` 数值实现及注册表保持冻结。
- 当前 2.4 若接收到旧 `rolling_*` 拼写，会在编译期展开为 `rolling_window + 普通归约`，不会形成第二套当前数值算法。

## 验收证据

1. 后端核心 minimum regression：257 passed。
2. 时序、Excel、运行参数、公式往返、画布专项：122 passed。
3. 前端指标中心 / 画布专项：51 passed。
4. 因果审计专项：56 passed，baseline 无 missing/changed/stale。
5. 前端生产构建：通过。
6. i18n 校验：通过，0 errors。
7. Python `py_compile`：通过。
8. `git diff --check`：通过。
9. Playwright 真实浏览器验收：desktop-1440 通过；mobile-320 通过。浏览器实际画布检查到独立的“滚动窗口”和“全元素算术平均值”节点，同时验证参数默认值、运行时覆盖和 Excel 下载。
10. 当前公共算子目录检查：存在 `rolling_window`；不存在四个旧 `rolling_*`；2.3 历史目录仍存在四个旧算子。
11. NJIT 状态检查：operator coverage `117/117`，warmup ready，`python_fallback=0`。
12. AI Hermes：本需求代码路径 self-evolve 覆盖检查通过，routing-only 检查通过，R70 路由可正常解析。全仓 routing validator 仍被工作区中其他需求已经写入路由、但尚未 Git 跟踪的文件阻塞；本需求没有修改或掩盖这些无关文件。

测试仅出现既有依赖提示：Browserslist / baseline-browser-mapping 数据较旧，以及 FastAPI TestClient 的 Starlette 弃用警告；均未影响本次功能和测试结果。

本次未提交或推送 Git；工作区内其他既有未提交修改未回退、未纳入本需求处理。
