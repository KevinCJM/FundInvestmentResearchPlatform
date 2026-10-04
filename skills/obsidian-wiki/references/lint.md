# Lint：结构、状态与真实候选

检查默认只读；只报告问题，不自动修复、晋级、刷新 hash、写查询日志或提交快照。

```bash
python3 scripts/knowledge_base.py check --summary
python3 scripts/knowledge_base.py coverage
python3 scripts/knowledge_base.py queue
python3 scripts/check_documentation.py
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```

- `check`：结构、依赖与链式失效。退出 0 仍可有 conflict/needs_review；不能将它们当通过，也不能为绿色报告抹除
- `coverage`：现有 repo_map 中受管文档、模块、附档和支持技能的可达/目录一致，不证明逐篇语义正确
- `queue`：需复核、未使用来源、未整合 claim、开放反馈与歧义工作流记录；不是自动审批或催办
- 文档检查器与 Hermes：链接、目录、候选和路由覆盖；语义仍须回到来源、代码与真实测试

退出 1 表示检测到问题；退出 2 表示调用/环境错误。保留原始报告和失败，不因部分步骤成功宣称全部通过。

只有获准变更且已核实内容时，按 `docs/governance/documentation.md` 更新 repo_map，执行 `catalog --write` 并把 `check_documentation.py --print-index` 输出替换进原索引标记区。路由改动跑 Hermes evolve/validate；其使用方法以现有 Hermes 技能为准。不为新入口另建 Harness。

提交候选检查须针对真实 index/PR HEAD，不让工作区修补遮住候选问题。未获准 staging 时，不修改真实 index；可以用仓库外临时索引核验本次精确候选并明确这只是本地验证。Documentation CI 的真实库 check/coverage与回归互补，不代替 Bot 审核、required-check 配置或语义复核。
