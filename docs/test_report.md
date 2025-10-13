# 全量测试记录

本次提交前按照项目要求执行了以下自检命令，确认核心功能在当前环境下均可构建与通过回归测试：

- `npm run build --prefix frontend`
- `pytest backend/tests`

其中后端依赖 `httpx` 由于网络代理限制未能通过 `pip install` 安装，但本地已预装满足测试需求的版本，相关命令的失败信息已记录在执行日志中供追溯。

