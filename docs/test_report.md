# 全量测试记录

本次提交前按照项目要求执行了以下自检命令，确认核心功能在当前环境下均可构建并通过回归测试：

| 序号 | 命令 | 结果摘要 |
| --- | --- | --- |
| 1 | `npm run build --prefix frontend` | ✅ 构建成功，Vite 报告 `✓ built in ...`，仅提示默认的 bundle 体积告警。 |
| 2 | `pytest backend/tests` | ✅ 19 项测试全部通过（`19 passed`），伴随 NumPy 自带的 `RuntimeWarning` 与 Pydantic 的弃用提示，未出现失败用例。 |

> 备注：执行 `pip install -r backend/requirements.txt` 时仍可能遇到代理导致的 `httpx` 获取失败；当前环境已预装兼容版本，因此上述测试均已顺利完成。若需在新环境复现，请预先解决外网代理或在内网私服补齐依赖。

