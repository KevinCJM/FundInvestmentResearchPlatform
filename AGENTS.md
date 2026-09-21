# Repository Guidelines

## AI 对话规则
- 对话中必须使用直接简要、清晰明了、人类友好的语言答复用户，不回复长篇大论。

## 代码变更注意事项
- 除非得到人类明确的需求或允许，否则不得删除或修改已有的功能、流程、模型、算法及其行为契约。
- 变更必须限定在人类明确授权的范围内；不得以重构、优化、清理冗余或统一实现为由，擅自删减、替换或停用已有能力。超出授权范围时，必须先说明影响并取得人类明确允许；已有明确授权的事项无需重复确认。
- 本文件中的存量代码清理、实现替换与算法治理要求仅适用于已获授权的变更，不能作为擅自修改已有功能、流程或模型的依据。

## 项目结构与模块组织
- `backend/`：FastAPI 服务入口位于 `app.py`，业务逻辑拆分在 `fit.py`、`optimizer.py` 等模块，`run.py` 提供统一 CLI。
- `frontend/`：React + Vite 源码保存在 `src/`，Tailwind 直接写入 JSX，`dist/` 存放构建产物供后端静态托管。
- `data/`：仅保留小型、可复现实验数据和测试夹具，路径通过 `DATA_DIR` 工具函数解析。
- 根目录 Markdown 仅保留 `README.md` 与 `AGENTS.md`。主题知识和专业契约归入 `docs/`，从[文档索引](docs/README.md)进入；部署和技能说明与各自目录同放。

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

## 前端设计执行协议
- 修改前端视觉、交互或图片资产前，必须完整阅读 [前端设计准则](docs/frontend/README.md)；涉及首页时另读 [首页需求与设计](docs/frontend/homepage.md)。视觉细则统一维护在设计准则中，不在本文件复制另一套标准。
- 先确认首页与工作台的适用范围，复用既有设计令牌和共享组件；参考图片、外部风格说明中的估算值不能直接成为全站规则。差异与例外必须在设计准则对应章节明确说明。
- 页面或资产修改的验收须覆盖受影响的真实交互状态、响应式布局、文字对比度及必要的降级效果；静态检查通过不能替代浏览器验收。仅修改规范时检查文档一致性、链接和路由，不据此宣称页面已经实现或通过视觉验收。

## 代码版本与存量代码治理
- Git 是本项目唯一的代码版本管理机制。源码目录只保留一份代表当前版本、实际参与调用、构建和发布的实现；历史版本统一通过 Git commit、tag 或 branch 追溯，不在当前代码中并行保存。
- 禁止仅为留档、对比或回滚而保留旧实现，包括但不限于 `old`、`legacy`、`backup`、`copy` 等副本、注释掉的大段旧代码、永远不会进入当前调用链的分支，以及同一功能的新旧双实现。
- 替换或重构功能时，必须在同一变更中删除被取代的代码，并同步清理失效的导入、导出、路由注册、配置、测试、文档和依赖；不得以“以后可能用到”为理由保留未使用代码。
- 回滚必须使用 `git revert`、切换 commit/tag/branch 等 Git 能力，不得通过恢复旧文件、旧函数或旧路由到当前源码来预埋回滚路径。
- 只有仍被当前系统真实调用、属于明确兼容契约且有测试覆盖的适配代码可以保留；兼容需求结束后必须立即删除。代码评审与验收需确认实际调用链只指向唯一当前实现，并检查无未引用代码、重复入口和失效配置。
- API/数据格式版本、数据库迁移、数据快照与模型产物版本可因外部协议或审计要求保留，但不得借此在源码中维持同一功能的多套历史实现。

## 算子与组合算法
- 新增或修改算子、计算图、模板或其契约前，必须完整阅读[算子治理](docs/governance/operator-contracts.md)。遵守最小独立计算语义、类型/时点边界、不可变历史契约及等价验收，禁止以黑盒组合替代可编辑步骤。

## 数值计算与性能
- 修改数值计算、回测窗口、内存布局或预热前，必须完整阅读[计算规范](docs/governance/numeric-computing.md)。项目数值路径默认固定签名 NJIT，启动预热完整后才就绪，禁止请求期编译及静默 Python 回退。
- 数组边界、零拷贝、只读所有权、第三方模型豁免和数值验收须遵守该规范；禁止用 Pandas 承载生产数值计算，不能仅凭装饰器或 `copy=False` 声称满足要求。

## 测试指引
- 后端测试放在 `backend/tests/`，文件命名 `test_*.py`，使用 `pytest` 验证路由状态码与响应结构。
- 前端测试使用 Vitest + React Testing Library，测试文件与组件同级，命名 `*.test.tsx`。
- 所有测试需可重复执行，使用 `data/fixtures/` 提供固定样本，禁止网络调用。

## 提交与合并请求规范
- AI 在创建开发分支、准备 commit/push、创建或更新 PR、执行 PR 审核、合并、发布、回滚及维护提交规则前，必须完整阅读 [提交规范](docs/governance/branch-submission-rules.md) 和 [代码提交全流程](docs/governance/submission-workflow.md)，不能仅凭记忆或本节摘要操作。
- 分支流向、提交范围、Bot 审核、测试证据、合并条件及异常处理由提交规范统一规定；全流程文档解释执行顺序，不另立规则。
- Bot 未通过最新提交的审核时不得正常合并。AI 代 Owner 绕过 Bot 审核，必须先给出具体意见、代码/测试依据及拟回复，取得人类针对该 PR、完整 HEAD SHA 和争议项的明确允许，再逐条回复 Bot 说明理由，最后才可合并。普通“帮我合并”不等于绕过授权；新提交或新增争议不能沿用旧许可。此要求由 AI 自觉执行，不做 AI/人类身份的技术区分；细节以提交规范的 Owner 例外流程为准。
- 按用户已经授权的任务范围执行，不重复询问已授权动作；写文档或修改代码本身不表示已获准提交、推送、合并或修改远端仓库设置。

## 文档维护与任务收尾
- 关键里程碑完成、结束含代码/测试/配置/文档变更的任务、或执行已获授权的提交前，必须按[文档维护协议](docs/governance/documentation.md)完成影响核对。
- 依据本任务实际差异及 AI Hermes 路由定位主题、专业契约、计划和操作说明。行为变化同步更新权威文档；核对后无需修改时说明具体理由。不要机械改日期或每轮新增报告。
- 计划按实际实现和验收证据结项；部分完成、未验证、延期、取消分别记录。历史证据保留适用日期与版本，未解决问题须有对应复验证据才能关闭。
- 确认的可复用坑点更新 `docs/pitfalls.json`，文档角色与模块归属更新 `docs/repo_map.json`，任务匹配更新 `docs/task_routes.json`。不得为消除文档冲突而擅改未获授权的业务行为。
- 运行 `python3 scripts/check_documentation.py`，根据任务差异核对输出的受影响文档；路由变更另跑 Hermes validate/evolve。共享脏工作区必须明确本任务范围；提交前检查暂存候选，不能用未暂存内容掩盖缺漏。
- 收尾报告已更新文档、相关验证及剩余事项。脚本与 CI 检查不等于语义正确、投资资格或真实部署通过；本协议不会自动赋予提交、推送、合并或修改远端设置权限。

## 安全与配置提醒
- 配置值由环境变量或 `.env` 读取，不要将密钥或令牌写入仓库。
- CORS 当前仅用于开发，部署前需收紧允许的来源并复核日志策略。
- 若发现仓库外部新改动或异常文件，先暂停操作并与团队沟通后再处理。

## Tushare 下载与文档同步
- 涉及下载、刷新、新增接口、数据源或下载脚本时，必须先使用 `$tushare-fetcher` 并完整阅读[采集协议](docs/data/acquisition-protocol.md)及[下载契约](docs/data/tushare-download.md)。账户权限、全局限频、超时、重试、完整性、隔离 smoke 和快照激活要求全部适用。
- API、动作、文件、schema、时点或下载/激活逻辑变化时，同步更新 `docs/data/tushare-download.md` 并运行其文档契约检查；历史数据快照不得表述为当前状态。

<!-- AI-HERMES-ROUTING-PROTOCOL:BEGIN -->
# AI Hermes Routing Protocol

## Purpose

Machine-first routing protocol for downstream agents operating from the current working directory.

## Scope Boundary

- Treat `.` as the writable project boundary unless higher-priority instructions say otherwise.
- External folders may be read for task understanding, comparison, or integration analysis; do not route edits outside the target project.
- Treat external services, DB schema, and invisible callers/callees as `out_of_scope` unless directly observed from readable files.
- Keep routing facts in JSON files under `docs/`; keep `AGENTS.md` protocol-only.

## Required Read Order

1. `AGENTS.md`
2. `docs/repo_map.json`
3. `docs/task_routes.json`
4. `docs/pitfalls.json`
5. `docs/governance/branch-submission-rules.md` and `docs/governance/submission-workflow.md` before branch creation, commit/push, PR creation or review, merge, release, rollback, or submission-policy maintenance
6. `docs/frontend/README.md` before frontend visual, interaction, or asset changes; also `docs/frontend/homepage.md` when the homepage is involved
7. Domain terminology before business-scope changes: `docs/product/domain-language.md`; numerical and operator contracts before related computation changes
8. `docs/governance/documentation.md` before documentation maintenance and at task closeout
9. Routed code, tests, and configs

## Routing Ownership

- `docs/task_routes.json` owns task matching, module expansion, and operational-list merge policy.
- `docs/repo_map.json` owns module facts, operational file lists, tests, configs, and regression commands.
- `docs/pitfalls.json` owns hidden contracts, recurring pitfalls, affected modules, and safe checks.
- `AGENTS.md` owns protocol, required read order, scope rules, and tool workflow only.
- Do not duplicate module-level file, test, config, or regression lists in `docs/task_routes.json`.

## Default Operating Sequence

1. Match the task in `docs/task_routes.json`.
2. Load `first_read_modules` from the selected route.
3. Expand into `expand_to_modules` only when route rule codes trigger.
4. Resolve `first_read_files`, `then_check_files`, `related_tests`, `related_configs`, and `minimum_regression` from `docs/repo_map.json` using `docs/task_routes.json` merge policy.
5. Load linked pitfalls from `docs/pitfalls.json`.
6. Verify claims from code, tests, configs, or command output before promoting them to routing memory.

## AI Routing Validation

- Use `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` after editing `AGENTS.md`, `docs/repo_map.json`, `docs/task_routes.json`, `docs/pitfalls.json`, or matching service routing files.
- The validator checks route/module/pitfall references, routed path existence, git-tracked reproducibility for stable references, minimum regression command targets, and `grounding.fact_status` values.

# AI Routing Self-Evolution

- Treat `docs/ai_routing_evolution_policy.json` as governance only; routing facts belong in `docs/task_routes.json`, `docs/repo_map.json`, and `docs/pitfalls.json`.
- Update `AGENTS.md` only when protocol, required read order, scope rules, or tool workflow changes.
- Promote verified hidden contracts and recurring pitfalls to the correct JSON owner.
- Use `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py` after code, test, config, tool, or routing changes to check coverage.
- For routing-only work, run `skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --routing-only` with explicit changed paths.
- Re-run `skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py` after routing file changes.

## Output Discipline

- Keep routing facts in JSON only.
- Keep `AGENTS.md` protocol-only.
- Stop exploration once routing is sufficient for first-pass narrowing.
<!-- AI-HERMES-ROUTING-PROTOCOL:END -->
