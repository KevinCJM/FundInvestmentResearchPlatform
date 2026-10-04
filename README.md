# 资产配置投研与组合管理平台

面向个人、企业和家族办公室等资产所有者，提供 ETF／公募基金研究、SAA/TAA、产品实施研究、回测及情景分析。研究计算已具备核心链路；真实账户、投资运营、核算与完整投后仍以原型或局部能力为主。

**[打开完整文档索引](docs/README.md)**：按任务查找主题入口、专业契约、操作流程和验收证据。

- 了解范围和后续方向：[需求与路线图](docs/product/requirements.md)
- 开始一次研究：[投前研究](docs/pre-investment/README.md)
- 查找专业契约：[文档索引](docs/README.md)；知识归属及核验：[Wiki 入口](docs/wiki/README.md)
- 开发前阅读：[AGENTS.md](AGENTS.md)、[领域语言](docs/product/domain-language.md)

平台不提供基金募集、客户份额登记、法定基金会计或交易下单。研究定稿不等于交易授权或收益保证。

## 项目知识库与 Obsidian

项目 Wiki 支持通过 Obsidian 管理和浏览，入口是 **[项目知识库](docs/wiki/README.md)**，按[业务知识](docs/wiki/navigation/business.md)和[开发知识](docs/wiki/navigation/developer.md)组织。主题页解释项目全貌、模块关系与设计理由，并链接原契约、代码和证据；完整操作见[知识库手册](docs/wiki/workflow.md)。

协作者可在 Obsidian 选择 **Open folder as vault**，打开本仓库根目录，再打开 `docs/wiki/README.md`。不需要移动或复制文档；未安装 Obsidian 时仍可直接阅读这些 Markdown。

辅助自己的 Vibe Coding 时，先按 [AGENTS](AGENTS.md) 和 Hermes 路由确认任务范围，再从主题/检索结果读取原文，给 AI 提供必要段落、来源路径、版本和未验证边界。Obsidian 不会自动把整个知识库注入任意 AI 客户端；也不能把历史测试、草稿或 `current` 指纹状态当作当前实现/生产通过。

```bash
# 在已满足项目 Python 3.12 要求的仓库根执行
python3 scripts/knowledge_base.py search "计算" --domain developer --limit 5
python3 scripts/knowledge_base.py context "配置" --intent overview --domain business
python3 scripts/knowledge_base.py check --summary
```

知识文件由 Git 共享，各机 `.obsidian/` 配置、工作区、缓存和 CLI 注册独立且不提交。官方 Obsidian CLI 需在本机单独启用并验证目标 vault；同步前保护本地改动，凭据、私人原件和受限资料不进入项目 Wiki。

## 结构与运行

React + TypeScript + Vite 前端，FastAPI 模块化后端；数值计算使用启动预热的 Numba NJIT。研究输入以本地 Parquet、版本化 JSON 和不可变运行制品为主。`backend/app.py` 是 API 入口，`frontend/src/app/processRegistry.ts` 管理业务流程与能力状态；生产构建由后端同源托管。

| 目录 | 用途 |
| --- | --- |
| `backend/` | API、领域服务、计算内核与测试 |
| `frontend/` | 页面、共享组件、单元与浏览器测试 |
| `data/` | 本地数据、工作区和不可变快照；正式数据不提交 |
| `docs/` | 主题入口、专业契约、AI 路由与验收纪要 |
| `deploy/` | 部署配置和操作说明 |

## 快速开始

### 环境要求

- Python 3.12
- Node.js 20 或更高版本（当前锁文件中的 React Router 和 Playwright 要求；Docker 构建也使用 Node 20）
- 先激活安装了项目依赖的 Python 3.12 环境，确认 `python3 --version` 为 3.12；下文及路由中的 `python3` 均指该受控解释器，不能用系统默认版本代替。开发者已有本机环境可继续使用其原路径，云环境不依赖该绝对路径。

### 安装依赖

```bash
cd backend
pip install -r requirements.txt

cd ../frontend
npm install
```

### 开发模式

启动后端：

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

首次启动会执行 NJIT 和 worker 预热，完成前服务不会进入就绪状态。健康检查：`http://127.0.0.1:8000/api/health`。

另开终端启动前端：

```bash
cd frontend
npm run dev
```

前端默认通过 Vite 将 `/api` 代理到后端。

AI 助手由独立的 Portable Web Agent 服务提供；平台只保留业务工具和页面接入。开发时还需启动独立服务并配置 `/assistant` 代理与登录身份，步骤见[独立助手接入](deploy/README.md#independent-assistant-candidate)，当前开发验收和生产切换边界见[迁移设计](docs/research/portable-agent-platform-integration.md#182-2026-10-01-dev整合与验收收尾)。

### 一体化运行

```bash
npm run build --prefix frontend
python backend/run.py
```

访问 `http://127.0.0.1:8000`。FastAPI 会同时提供 API 和 `frontend/dist`。

### Docker

部署方式与持久化见 [部署说明](deploy/README.md)。运行配置见 [.env.example](.env.example)，数据入口为“设置 → 数据源与下载”，下载契约见 [docs/data/tushare-download.md](docs/data/tushare-download.md)。Token 只从受控本机配置读取，禁止提交或输出。

## 测试与协作

按任务路由选择受影响回归。后端使用 Python 3.12、固定本地夹具；前端使用 Vitest 和 Playwright。常用入口：

```bash
python -m pytest backend/tests -q
npm run test --prefix frontend -- --run
npm run build --prefix frontend
npm run test:e2e --prefix frontend
```

测试不得依赖外网或污染正式研究数据。服务启动后应检查 `/api/health` 中的 NJIT 和 worker readiness，不能只看端口存活。提交和 PR 流程以 [提交规范](docs/governance/branch-submission-rules.md) 与 [代码提交全流程](docs/governance/submission-workflow.md) 为准。

## 许可

本项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE)，仅允许许可规定的非商业使用。商业使用须获得版权方单独书面授权；第三方依赖和数据源的许可与条款仍分别适用。
