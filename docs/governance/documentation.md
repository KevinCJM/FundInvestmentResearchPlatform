# 文档维护协议

本文规定文档归属、计划结项、坑点沉淀和任务收尾。适用于代码、测试、配置、工具和 Markdown 变更。入口为[文档索引](../README.md)，执行约束见[AGENTS](../../AGENTS.md)，提交权限与合并规则仍以[提交规范](branch-submission-rules.md)为准。

## 文档结构与权威来源

- 根目录仅有 README（项目入口）和 AGENTS（智能体协议）；完整索引在 `docs/README.md`。部署和技能说明与各自代码同放。
- `product/` 定义业务语言、范围及交付顺序；`pre-investment/`、`data/`、`indicators/`、`product-research/`、`regimes/`、`frontend/` 各有一个 README 主题入口，必要专业契约独立保存。因子研究当前由单页承接。
- `governance/` 保存开发、计算、文档及提交规范；`research/` 保存方法依据和明确标为草稿的研究；`verification/` 按主题保留关键证据、决定及未解决问题。
- 当前功能以源码、测试及实际运行证据核实；文档描述不能代替执行事实。业务要求与授权也不能因实现偏离而被自动改写。
- 文档角色、模块归属和读取场景只登记在 `docs/repo_map.json` 的 `documentation.documents`。模块代码/测试路径仍由同一文件的 modules 维护，任务匹配在 task_routes，坑点事实在 pitfalls。不要另建一套文件映射。
- 索引表由目录元数据生成，修改目录后运行 `python3 scripts/check_documentation.py --print-index`，仅替换索引的标记区；其余导航说明手工维护。不要在索引复制算法、回归命令或实施状态。

## 何时核对和更新

关键里程碑完成、任务收尾及已获授权的提交前集中执行。逐次编辑可以做轻量检查，无须每次保存都改文档。只读问答、无需落盘的调查不强制制造文档修改。

| 变更 | 核对与更新 |
| --- | --- |
| 功能交付、支持范围变化 | 主题当前说明、对应专业契约、活动计划及必要的上层路线图 |
| 参数、API、算法、数据或时点变化 | 对应契约、示例、适用检查与边界，保留兼容性和未验证范围 |
| 启动、运维、配置与恢复流程变化 | 实际入口、默认值、前置条件、成功判断和失败恢复 |
| 修复或发现隐藏问题 | 对应 pitfall、可复现测试、未解决问题的状态及关闭依据 |
| 文档新增、移动、删除 | 目录角色、索引、所有入链、路由、源码引用和文档测试路径；额外检索退役路径及文件名的纯文本/注释引用，不能只依赖 Markdown 链接检查 |
| 纯排版、拼写或等价内部调整 | 核对后可以无需修改专业文档，说明哪些契约没有变化 |

处理顺序：确认任务范围 → 获取实际差异 → 根据路由和检查报告定位文档 → 回到源码与证据判断 → 更新权威内容 → 检查链接、活动计划、路由和相关契约 → 报告结果。

共享工作区须保留开始时的差异/哈希基线，只处理本任务授权范围。`--changed-file` 是明确文件范围，不会自动识别共享文件的 hunk 归属；混合内容要人工核对或在隔离候选中检查。禁止把其他任务已有修改当成自己的交付。

## 计划与证据生命周期

每个长期主题只维护一份当前待办。上层路线图记录阶段与依赖，链接具体主题；不复制多个细粒度任务清单。复杂跨轮任务可使用临时计划，结束后提炼决策、有效证据和剩余问题，过程细节由 Git 追溯。

活动计划使用下表格式，ID 在所属文档内稳定且唯一。检查器只识别这一显式表头，不凭“完成”“后续”等关键词猜测状态：

| ID | 状态 | 工作项 | 完成判据 | 证据/剩余事项 |
| --- | --- | --- | --- | --- |
| DOC-01 | verified | 文档归位、目录与收尾检查 | 链接、候选读取和路由检查通过，现有专业规则完整保留 | [2026-09-20 本地验收](../verification/engineering.md#文档工程升级2026-09-20)；远端发布不在本任务内 |
| DOC-02 | verified | PR 文档检查配置 | 本地候选检查通过，并取得实际 PR 工作流运行证据 | [PR #45 实际运行证据](../verification/engineering.md#pr-远端检查验证2026-09-20)；仅验证工作流运行，不包含 required check 配置 |

状态为 `planned`（待开发）、`in_progress`（进行中）、`implemented_unverified`（已实现待验证）、`verified`（约定范围已验证）、`deferred`（延期）或 `cancelled`（取消）。verified 必须链接具备适用范围的证据；延期、取消需说明原因。验证字段完整不等于证据真实，仍须审核。

部分完成必须拆开已完成与剩余范围。测试、用户交互、部署、数据资格、真实费用、PIT 和独立审批分别记录，不能互相代替。不能仅以“存在函数”标成功能完成，或用其他测试通过关闭特定反例。

历史验收保留原日期和版本；关闭旧问题时补充新版本下对应复验依据，不重写旧事实。新计划不得默认为已获开发授权；核对到文档缺口不会自动扩大本次业务范围。

## 坑点与设计决定

确认可复用价值后更新 `docs/pitfalls.json` 的原 ID 或新增 ID，记录触发条件、影响、正确做法及代码/测试依据。一次性环境错误和未验证推测不能升级为项目规则。修复 bug 后，防复发检查仍可能有效。

长期设计取舍归所属主题或现有研究纪要；普通代码变更无需再记一份流水。历史建议失效时标明适用版本及替代依据；不要同时维护互相冲突的当前规则。

## 可执行检查

安装仅用于文档检查的依赖：`python3 -m pip install -r scripts/requirements-docs.txt`。推荐使用项目 Python 3.12 环境。检查器不写项目文件、不运行 Markdown 中的命令、不发网络请求或下载数据。

```bash
# 工作区结构、链接、索引、活动计划及变更影响
python3 scripts/check_documentation.py
# 限定本任务涉及的文件；结构仍检查完整当前文档库
python3 scripts/check_documentation.py --changed-file backend/pre_investment/service.py --json
# 精确读取暂存版本，未暂存修复不能掩盖暂存缺陷
python3 scripts/check_documentation.py --staged --json
# 使用 merge-base 比较，读取 head 的提交内容
python3 scripts/check_documentation.py --base-ref origin/Dev --head-ref HEAD --json
# 检查工具自身的离线回归
python3 -m pytest scripts/tests/test_documentation.py -q
```

报告包含当前候选 fingerprint、受影响文档和触发路径。删除或迁移的受管文档即使已从目录移除，也保留旧路径作为 `role=deleted` 的审核项；须核对替代契约、路由和源码引用，不能靠删除目录项跳过审核。该值只用于影响报告，不是目录角色。每个审核项需人工或智能体审核并给出 `updated / reviewed_no_change / needs_review`。不要仅改日期或随意改字满足检查。

可将审阅结论放在仓库外的临时 JSON，通过 `--review-file <文件> --require-review` 在任务收尾严格核对。格式为 `{"fingerprint":"报告中的值","documents":[{"path":"文档相对路径","status":"reviewed_no_change","reason":"具体不受影响的契约与依据"}]}`。updated 必须属于本次变更范围；候选变化使回执失效。回执是审核声明，不是自动证明，也不需要永久新增一份 Markdown。

本工具验证：目录覆盖、文档角色/归属、根目录边界、相对链接及锚点、索引一致性、活动计划字段、暂存/PR 候选内容和可选回执（Git 候选中的符号链接不作为文档内容读取）。计划表中的 `\|` 按 GFM 转义规则保留在单元格内；行内代码中的竖线也须转义。语义矛盾、完成判据是否真的满足、证据真实性仍须源码和审核确认。扫描到相关文件变化只表示需复核，不代表文档必然错误。

候选隔离同时覆盖所复用的 Hermes 辅助代码：暂存检查从 index 加载，指定提交检查从该提交加载，工作区修复不能代替候选中的版本。辅助代码也纳入审核回执指纹，版本变化使旧回执失效；候选缺少辅助代码或加载失败时直接报错。

标题锚点在同一文档内全局去重，包括标题本身含编号的情况：`Foo`、`Foo-1`、`Foo` 对应 `foo`、`foo-1`、`foo-2`。不能只统计同名标题，否则有效的编号锚点链接会被误报。

路由变化继续运行 Hermes validate/evolve；Tushare 相关文档运行 `backend/tests/test_tushare_data_script.py -k document`。专业契约测试按实际变更选择，不为纯路径移动启动全量业务回归或正式下载。

## 触发与部署边界

- AGENTS 已规定里程碑和收尾必须执行；它是智能体执行协议，不是原生自动事件。
- `.github/workflows/documentation.yml` 在 PR 中调用同一检查器并保留影响报告，执行结构检查及检查器回归；它不会代替语义审核，也不是 Bot 审核替代品。
- 工作流文件进入远端后才会触发。设置 required check 是另一个远端配置动作；本次不修改远端设置。
- 本项目未安装客户端原生 Stop hook，也未改 Git hooksPath。未来需要接入时先核实实际客户端支持，复用此入口，设置去重、超时和有限重试，尊重用户中断。
- 无需定时重写所有文档；后续定期维护如需启用应单独配置。

## 方法依据

采用 [OpenAI 的轻量入口与仓库知识实践](https://openai.com/index/harness-engineering/)、[持续维护执行计划](https://developers.openai.com/cookbook/articles/codex_exec_plans)、[Docs as Code](https://www.writethedocs.org/guide/docs-as-code/) 和 [Diátaxis 的文档职责区分](https://diataxis.fr/start-here/)。客户端 hook 的机制可参考 [Claude Code 文档](https://code.claude.com/docs/en/hooks)，但其配置不直接适用于其他客户端。
