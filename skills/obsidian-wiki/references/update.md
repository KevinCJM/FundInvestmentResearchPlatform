# Update：实际变更、反馈、复核、重新整合

用于获准修改后的知识影响收尾和获准纠错。以本次实际 Git diff（包含新增/删除/改名和其他任务边界）为范围，先查主题/证据和原契约，复用 Hermes；不扫描聊天历史、不建立 last_commit_synced 状态。

1. 运行 `check --summary` 与 `queue`，读取受影响正文和变化来源；区分内容实质改变与元数据/目录改变。依赖变化仅触发复核，不能批量刷新 fingerprint 或清空冲突。没有知识影响时说明具体未变的主张及原因。
2. 已确认行为变化先在获准范围更新原权威说明，再维护有源主题、claim/source 与原路由目录。只记录复用价值高的决定、原因与边界；不用流水日志或多份文件清单代替现有目录。
3. 更正流程保留审计：用 `hash` 读取目标当前字节，再执行 `feedback`；核实反证后以新的当前 hash 调用 `resolve-feedback`，decision 只表示该反馈的 resolved/deferred，不能自动恢复证据 current：

```bash
python3 scripts/knowledge_base.py hash docs/wiki/claims/claim-slug.md
python3 scripts/knowledge_base.py feedback docs/wiki/claims/claim-slug.md --message "具体反证或范围变化" --by "实际提出者" --expected-sha256 "当前完整hash"
python3 scripts/knowledge_base.py queue
python3 scripts/knowledge_base.py resolve-feedback docs/wiki/claims/claim-slug.md --feedback-id "真实反馈ID" --decision resolved --message "核实结果与证据" --by "实际审阅者" --expected-sha256 "重新读取的完整hash"
```

4. 真正重新阅读证据、更新有界结论与逐项依赖，记录复核范围及版本，再复核下游 topic。保留其他未解决反馈、金融 conflict 和原四轴限制。`source_revision` 是真实完整 Git 基线；工作区证据另以精确依赖 hash 与审阅说明绑定，不能声称未提交树已经发布。
5. 重新整合：旧命令回执绑定旧 claim 快照，`integrate` 不会覆盖它。同 claim 证据改变后，不要重复命令强行替换或伪造标记；可在获准的语义修订中明确审阅旧主题段落/旧回执，或使用新的版本化 claim。旧结论需要保留为历史并明确当前适用性；若旧 claim 仍作为依赖，它也必须经过真实复核，否则下游继续失效。待 claim/topic 检查满足条件，再按运行手册 `integrate`。不变重试用上次返回的 after_sha256，目标变化则停止重读。
6. 按 [lint](lint.md) 检查最终候选并报告实际改动、理由、测试与尚未核实内容。发布或本机同步是另一个授权范围，不在更新知识时自动执行。
