# 上游来源与本项目改编

- 来源：[Ar9av/obsidian-wiki](https://github.com/Ar9av/obsidian-wiki)
- 固定 commit：[113454c7598ada6b1e4cbc38dd04f64abd3036c6](https://github.com/Ar9av/obsidian-wiki/commit/113454c7598ada6b1e4cbc38dd04f64abd3036c6)，读取日期 2026-10-04
- 许可：MIT，Copyright (c) 2026 Ar9av。完整原文保存在 [LICENSE.txt](LICENSE.txt)，对应[上游许可](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/LICENSE)
- 获取方式：只读 GitHub commit/tree 与固定 commit 的 raw 技能源/许可；未执行上游 setup、runtime、hooks 或安装器
- 采用方式：改编流程指导，不复制或运行上游 CLI。项目技能正本是本目录 SKILL.md，发现入口是仓库内普通 Markdown 文件，不用上游 symlink 安装机制

## 技能源映射

| 固定版本技能源 | 本项目引用 |
| --- | --- |
| [`.skills/wiki-ingest/SKILL.md`](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/.skills/wiki-ingest/SKILL.md) | references/ingest.md |
| [`.skills/wiki-query/SKILL.md`](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/.skills/wiki-query/SKILL.md) | references/query.md |
| [`.skills/wiki-update/SKILL.md`](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/.skills/wiki-update/SKILL.md) | references/update.md |
| [`.skills/wiki-lint/SKILL.md`](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/.skills/wiki-lint/SKILL.md) | references/lint.md |
| [`.skills/wiki-context-pack/SKILL.md`](https://github.com/Ar9av/obsidian-wiki/blob/113454c7598ada6b1e4cbc38dd04f64abd3036c6/.skills/wiki-context-pack/SKILL.md) | references/context-pack.md |

## 保留与重写

- ingest：保留先读来源、区分出处/推断/歧义、去重、关联现有主题的原则；改用已有 intake → 未核 source/claim → 证据复核 → integrate。去掉聊天历史采集、全局配置解析、自动下载、raw/staging/category 目录与 manifest/log/hot/QMD 维护
- query：保留由轻量候选逐步进入必要正文、引用来源、说明已读范围和缺口；上游虽标 READ-ONLY 仍有 Step 6 写 log 的例外，本项目完全删除该写入，以 search/context 只读查询实现。不依上游 index-only 摘要直接推断已核事实
- update：保留增量差异、架构原因和取舍、检查旧关系的思路；以实际任务 Git diff、Hermes 和现有证据依赖替代 last_commit_synced/manifest，不读取私人 agent memory，不自动发布
- lint：保留结构/来源/冲突/历史有效性的分离；映射到 check/coverage/queue/check_documentation/Hermes。删除按年龄/置信度自动晋级、整理日志、自动快照 commit/reset/clean 等流程；修复必须是获准的语义复核
- context-pack：保留只读取证包与不可信摘录边界；上游需要自己的 CLI 和 token budget，本项目改为 context --intent overview/module/history/compare 与候选 limit，不承诺 token 上限。包不证明原文已读

## 本项目唯一所有权

知识和工作流状态继续由 docs/wiki 与 scripts/knowledge_base.py 管理；原权威文档保持原位置，目录/路由/风险分别由 repo_map/task_routes/pitfalls 的现有 Hermes 所有者管理。此改编不是第二套知识平台，也不要求上游的其他技能、守护程序、索引服务或全局配置。

上游后续更新须先只读比较固定版本和许可，再按本项目实际需求评估；不得拉取后直接运行 setup 或覆盖本项目技能。

## 所读文本 SHA-256

固定 commit 的原文件 UTF-8 文本，未做换行转换；仅用于取得版本的复核，不作为项目知识事实批准。

| 上游路径 | SHA-256 |
| --- | --- |
| `.skills/wiki-context-pack/SKILL.md` | `23c7c372f2f39be30c80dc6345c870af72ed182d6f18b874599a57594c3dc832` |
| `.skills/wiki-ingest/SKILL.md` | `91bd73e32cbe58c2a4293cc9460c39356c6d3aeb584bcf32b1b06ef42f336bdb` |
| `.skills/wiki-lint/SKILL.md` | `12ab5c0df73cc7c90a2655f50775a862fc0186914df482fbb56b2c3e1f19756e` |
| `.skills/wiki-query/SKILL.md` | `5088930473e10a92b6bd69961d77ac3e0edd056b21e24e6f9447417581960e91` |
| `.skills/wiki-update/SKILL.md` | `f447133ab79dd1b57f6e8bc49a4ba9b6bedd9f284bb1e123321d0e525098d65e` |
| `LICENSE` | `70c79a07317545794fe309811949c1e637fb4cf8fb7bfcddc3658c25e758b019` |
