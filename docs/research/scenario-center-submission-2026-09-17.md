# 情景中心独立提交记录

日期：2026-09-17。开发分支：`ISSUE2609/ScenarioCenter`，目标：`Dev`。

## 拆分依据和范围

起点为 `fa42a5eda9a869d4008c49e13c5056fea8e65556`。以9月16日第二轮审核结束时的文件SHA-256、风险等级中心开始前基线和投资目标任务基线交叉比对，在独立worktree组装提交。

- 此前情景中心及缺陷修复共148个文件，含历史参考、实时识别、可靠性、前向验证、来源续接、压力情景、TAA消费及交互修复、测试和研究文档。
- 146个文件可直接复用当前内容；共享的 `backend/app.py` 和 `docs/repo_map.json` 拆除后续风险等级中心/投资目标改动后，与旧审核哈希一致。
- 其他需求文件111个不进入本PR，其中109个是后续新增或修改的文件，另2个是早期已存在的风险等级/双路径设计文档；这2个文档按需求归属排除，不能算成本次情景变更。
- 后续改动中的 `backend/qp_numba.py`、`backend/frontier_moments.py`、`backend/sensitivity/repository.py` 属于风险等级中心依赖；不是本次情景修复的一部分。
- 新增本提交记录，补齐情景模块路由文件/测试覆盖，规范两份Markdown文档硬换行；这些仅为提交整理，不改变业务行为。
- 所有 `.tmp_*`、测试日志、缓存、构建和正式数据不进入提交。原工作区、当前分支和未提交的新功能保持原状。

## 当前结论和文档

历史审核记录按日期保留。较早审核中的“暂不通过”是修复前结论；逐项解决情况以[第二轮审核与修复](scenario-audit-followup-2026-09-16.md)及本次独立分支验证为准。

- 需求：[情景研究与中心优化](scenario-regime-research-best-practices-and-center-optimization-2026-09-15.md)。
- 技术：[情景识别实施设计](regime-completion-design-2026-09-15.md)、[前向验证API](regime-prospective-api-2026-09-15.md)。
- 修复：[首轮整改](scenario-audit-remediation-2026-09-16.md)、[第二轮复核](scenario-audit-followup-2026-09-16.md)。

## 本次独立分支验证

本次代码来自拆分后的独立工作区，测试使用隔离存储；不使用后续风险等级中心与投资目标实现补齐依赖。

| 检查 | 实际结果 |
| --- | --- |
| 后端51个测试文件 | 首次974 passed、1 failed；唯一失败为测试编排从stdin调用pytest导致multiprocessing找不到`<stdin>`。随后用标准`python -m pytest`复验该完整应用启动用例，1 passed；没有修改业务代码或断言。975个独立用例均取得通过结果，不把首次整组命令写成成功。 |
| 前端完整Vitest | 147 files / 1147 passed |
| 真实隔离API浏览器 | 21 passed，覆盖320/768/1440 |
| 发布情景 / TAA协议浏览器 | 3 passed / 10 passed |
| TypeScript / Vite / i18n / design | 全部命令退出0；i18n无错误，设计无新增回归，保留既有大包提示。 |
| 路由 | 完整validator通过，含Git已暂存文件可复现性。evolve全149路径有1个工具误报：其路径识别明确排除含空格文件名；`Market Regime Detection Framework.md`已在backend_regime_research.then_check_files精确登记并人工核对。其余148路径的自动覆盖检查退出0。未扩大目录覆盖或修改检查器掩盖问题。 |
| 提交范围与原工作区 | 所有业务源码与9月16日审核最终哈希一致；原工作区463个既有文件逐一核对无变化，未重启其服务。`git diff --cached --check`通过。 |

测试命令与结果保存在本次提交的外部证据目录；历史报告的测试数字仍仅代表各自运行，不与本次累加。

不能把协议夹具、合成数值测试或历史市场研究记录视为生产投资资格；已保存研究版本不回写。独立LTCMA估计和正式历史数据时点认证不属于本PR完成范围。
