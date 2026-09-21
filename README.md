# 资产配置投研与组合管理平台

面向个人、企业和家族办公室等资产所有者，提供 ETF／公募基金研究、SAA/TAA、产品实施研究、回测及情景分析。研究计算已具备核心链路；真实账户、投资运营、核算与完整投后仍以原型或局部能力为主。

**[打开完整文档索引](docs/README.md)**：按任务查找主题入口、专业契约、操作流程和验收证据。

- 了解范围和后续方向：[需求与路线图](docs/product/requirements.md)
- 开始一次研究：[投前研究](docs/pre-investment/README.md)
- 查找专业契约：[文档索引](docs/README.md)
- 开发前阅读：[AGENTS.md](AGENTS.md)、[领域语言](docs/product/domain-language.md)

平台不提供基金募集、客户份额登记、法定基金会计或交易下单。研究定稿不等于交易授权或收益保证。

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
- Node.js 18 或更高版本
- 本地开发建议使用 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`

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
