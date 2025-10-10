# Repository Guidelines

## 项目结构与模块组织
- `backend/`：FastAPI 服务入口位于 `app.py`，业务逻辑拆分在 `fit.py`、`optimizer.py` 等模块，`run.py` 提供统一 CLI。
- `frontend/`：React + Vite 源码保存在 `src/`，Tailwind 直接写入 JSX，`dist/` 存放构建产物供后端静态托管。
- `data/`：仅保留小型、可复现实验数据和测试夹具，路径通过 `DATA_DIR` 工具函数解析。
- 根目录：公共脚本如 `T01_get_data.py` 与环境配置 `config.py`，文档放在此处，避免散落。

## 构建、测试与开发命令
- `cd backend && pip install -r requirements.txt`：使用推荐虚拟环境 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12` 安装依赖。
- `uvicorn app:app --reload --host 0.0.0.0 --port 8000`：本地热加载 API 服务，自动提供 `/api`。
- `cd frontend && npm install`：安装前端依赖；后续命令默认在同一路径执行。
- `npm run dev` 与 `npm run build`：前者启动代理到后端的开发服务器，后者输出生产包至 `frontend/dist`。
- `python backend/run.py`：打包前端后的一体化演示入口，方便业务验证。

## 代码风格与命名约定
- Python 采用 PEP 8 与四空格缩进；函数、变量使用 `snake_case`，Pydantic 模型使用 `PascalCase`。
- 前端使用函数式组件与 PascalCase 文件名，局部样式与测试与组件同目录保存。
- 避免硬编码路径，优先复用配置和工具模块；必要注释保持简洁并解释设计意图。

## 测试指引
- 后端测试放在 `backend/tests/`，文件命名 `test_*.py`，使用 `pytest` 验证路由状态码与响应结构。
- 前端测试使用 Vitest + React Testing Library，测试文件与组件同级，命名 `*.test.tsx`。
- 所有测试需可重复执行，使用 `data/fixtures/` 提供固定样本，禁止网络调用。

## 提交与合并请求规范
- 提交信息遵循 `feat:`, `fix:`, `docs:` 等前缀，聚焦单一改动并描述影响面。
- PR 需说明目的、关键变化、运行过的命令，并在涉及 UI 时附上截图或视频。
- 关联相关 Issue 或任务编号，确认无敏感信息泄露后再发起合并。

## 安全与配置提醒
- 配置值由环境变量或 `.env` 读取，不要将密钥或令牌写入仓库。
- CORS 当前仅用于开发，部署前需收紧允许的来源并复核日志策略。
- 若发现仓库外部新改动或异常文件，先暂停操作并与团队沟通后再处理。
