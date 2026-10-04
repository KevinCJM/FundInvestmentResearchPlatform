# 知识库操作手册与验收

[主页](README.md) · [业务知识](navigation/business.md) · [开发知识](navigation/developer.md) · [完整设计](llm-wiki-design.md)

## 直接开始

在Obsidian以项目checkout根目录打开vault，再打开 `docs/wiki/README.md`。两个入口组织8个业务、6个开发和1个研究主题，先理解高层关系，再回原契约、代码和证据。具体核验卡属于底层材料，不代表知识库的范围。

- 全局：平台定位→完整投研流程→系统分层
- 研究：目标与配置→产品实施→证据资格
- 开发：模块/前端→数据计算→研发交付
- 理由：业务/技术决定→原研究和历史验收，未知不编造
- 新资料：研究主题→模板→入库、审阅、整合和反馈

[生成目录](catalog.md)显示全部受管文档、模块、附档/benchmark及原技能说明。active只是文档生命周期，不表示功能完成；图片/原型/历史基准不等于本轮运行证明。

## AI 技能与闭环入口

正本为 [skills/obsidian-wiki/SKILL.md](../../skills/obsidian-wiki/SKILL.md)，普通文件 `.agents/skills/obsidian-wiki/SKILL.md` 只作发现转向；没有 symlink 或全局安装。AGENTS 要求修改前显式读取，不能依赖所有客户端都会自动发现。查询、入库、更新、lint 和 context-pack 只是同一工具的五种操作指引，不建立第二套状态。

闭环是：修改前按 Hermes 缩小范围 → 查主题及必要原文 → 在授权范围修改 → 判断知识影响 → 必要的来源/主张证据复核与主题整合 → 对最终候选运行文档/知识/覆盖/路由检查。无影响时说明具体理由；只读问答不写日志。子任务获得相关入口、范围和可写路径，不加载整库。上游固定版本与 MIT 许可见 [UPSTREAM](../../skills/obsidian-wiki/UPSTREAM.md)。

## 环境与入口

从项目根使用Python3.12和原文档依赖 `scripts/requirements-docs.txt`。检索、来源和核验操作使用标准库；整合、反馈处理、queue及catalog/coverage复用已声明的Markdown解析器，需要上述文档依赖，不新增服务或包。

```bash
python3 -m pip install -r scripts/requirements-docs.txt
python3 scripts/knowledge_base.py --help
python3 scripts/knowledge_base.py check --summary
python3 scripts/knowledge_base.py coverage
python3 scripts/knowledge_base.py queue
```

check退出0仅表示无结构错误/已检测失效，不消除conflict或未核。退出1表示结构/失效或coverage失败，退出2表示调用/环境错误。current仅表示登记依赖字节未变；stale需复核，untracked是无证据基线，invalid不可用，not_assessed表示原文未整篇核验。

## AI问答与按需取证

工程任务先读AGENTS/Hermes。context返回取证包，实际回答仍需读取原文必要段落：

```bash
python3 scripts/knowledge_base.py context "平台" --intent overview --domain business
python3 scripts/knowledge_base.py context "计算" --intent module --domain developer
python3 scripts/knowledge_base.py context "决定" --intent history
python3 scripts/knowledge_base.py context "竞品" --intent compare
python3 scripts/knowledge_base.py search "LTCMA" --domain business --status active --limit 6
python3 scripts/knowledge_base.py search "Harness" --status historical --limit 6
```

- overview：主题→原需求/架构，给目标、对象关系、当前覆盖和限制
- module：路由/契约→源码/测试，分清批准、静态实现、本轮执行和部署
- history：当前与历史决定并看，保留时间/版本/替代关系，无原因证据就说未知
- compare：事实、厂商自述、推断、独立验证分开，逐项对照项目，采纳由负责人决定

search是字面检索，空格分词AND；主题aliases帮助定位，不承诺语义搜索。长句无命中时缩短词或从主题进入，不编造。领域来自原目录/记录，历史过滤不证明当前有效。scope、result、review覆盖、freshness和needs_review分别判断；未核/冲突/否定/替代来源继续向下游传播。

## 来源入库与去重

先确认允许在本Git仓库记录的摘要与许可，私人/付费原件、账户信息、token和签名URL不入库；工具不下载网页，也无法识别任意正文秘密。

```bash
python3 scripts/knowledge_base.py intake source-slug --title "具体论文或方案标题" --domain business --source-kind paper --uri "https://example.org/paper" --version "明确版本/日期" --summary "获准的必要摘要" --rights "可记录范围和许可依据"
```

替换示例参数。source-kind支持repo_record、official_source、paper、vendor_claim、user_summary、ai_summary。讨论/私有出处只能用获准、不含个人信息的opaque ID，如discussion:approved-summary-01，不能用私人路径。

- 同URI/版本/摘要返回原path，不重复创建
- 同URI/版本但文字不同停止，要求反馈/修订或明确新版本
- 异源同摘要报告possible_duplicate_content；短摘要相同不证明同源，确认不同才用--distinct-source
- 新版明确替代旧版：用新slug/版本，加--supersedes指向旧source；保旧文件并触发下游复核
- 不声明替代时保留并列版本与关联，不自动假定新版本使所有旧用途无效

默认pending/unverified。敏感URI校验覆盖intake、source draft和手工记录检查，拒绝定义的query/fragment凭据、签名与非web authority。写锁与排他创建保护脚本并发，不替代人工协调。

## 草稿与证据核验

```bash
python3 scripts/knowledge_base.py draft claim claim-slug --title "可证伪的有界主张" --domain business
python3 scripts/knowledge_base.py fingerprint docs/pre-investment/ltcma.md backend/strategic_allocation/cma_model_contracts.py
```

填写主张、精确证据、四轴、限制和复核条件；source区分原作者自述、本项目推断、独立验证/反证。详见[模板](templates.md)。真实reviewer才填写姓名/身份、日期、scope、完整Git commit、依赖和结论。

fingerprint仅允许已跟踪证据，不自动写卡或批准。新资料可保未核草稿；不能为绕过检查在未获授权时stage正式仓库。hash读取当前笔记字节供并发保护，不等同证据批准。金融冲突/采纳需负责人判断，没有自动approve或全部刷新hash。

## 将知识整合进主题

claim和目标topic必须current且needs_review=false。先读目标hash：

```bash
python3 scripts/knowledge_base.py hash docs/wiki/topics/目标主题.md
python3 scripts/knowledge_base.py integrate docs/wiki/claims/已核主张.md --topic docs/wiki/topics/目标主题.md --summary "有界整合摘要" --reviewer "实际审阅者" --reason "与本主题的关系" --expected-sha256 "当前完整hash"
```

只写派生topic，不改原需求/算法。整合保存claim链接、scope、版本、reviewer、理由、目标修改前hash，返回修改后hash并绑定依赖。相同整合内容可幂等重试：摘要、审阅人、理由及绑定的claim证据须精确一致，且仍须传当前目标hash；首次成功后用返回的after_sha256重试，目标发生变化则停止重读，不能复用整合前hash。原日期和写前hash仅作为历史回执保留。首次写入前也拒绝保留的整合标记及多解字段边界；依赖过期或既有块不同则停止。人工主题引用仍可用，不要求全部改成命令块；改变原权威另需授权和diff审阅。

## 反馈修订与复核队列

```bash
python3 scripts/knowledge_base.py hash docs/wiki/claims/待复核主张.md
python3 scripts/knowledge_base.py feedback docs/wiki/claims/待复核主张.md --message "具体更正、反证或范围变化" --by "提出者" --expected-sha256 "当前hash"
python3 scripts/knowledge_base.py queue
python3 scripts/knowledge_base.py resolve-feedback docs/wiki/claims/待复核主张.md --feedback-id "返回的ID" --decision resolved --message "处理结果、新证据或不采纳理由" --by "审阅者" --expected-sha256 "重新读取的hash"
```

feedback可指source/claim/topic，保旧result并置stale；同作者/内容不重复，但幂等返回也须核当前hash，不能用写入前的旧hash。处理可resolved/deferred，但不自动清除stale。reviewer须更新正文、证据和受影响下游，不能批量刷新hash冒充复核。

旧整合回执绑定旧 claim 快照。证据改变后直接重试不会覆盖旧回执；需在授权范围内审阅修订旧主题/回执，或新增版本化 claim 并明确旧结论历史适用性。保留为依赖的旧 claim 也需真正复核，不能移除依赖来掩盖失效；重新审阅 claim 与 topic 后才可再次整合。

queue展示需复核、无claim使用的source、尚无合法命令整合回执的已核claim、开放反馈。只有顶层完整、有界、身份与证据字段匹配的记录可进入流程；引用/围栏/内联示例、导入材料中的标记和不完整旧记录不得冒充事件，歧义内容提示人工复核。来源摘要、反馈文本等输入拒绝保留的KB工作流标记；正常资料请保留必要摘要或明确可读转述。新反馈含END边界，旧无END记录需核对后人工补全，不自动猜测或执行。它是信息清单，不是自动催办或审批。外部撤回、数据vintage或生产变化要明确新证据，不能靠本地hash发现。

## 全项目目录与增量维护

唯一目录归属仍在repo_map；knowledge_domains为业务/开发双分类。新增资料登记后重建派生目录和原索引：

```bash
python3 scripts/knowledge_base.py catalog --write
python3 scripts/knowledge_base.py coverage
python3 scripts/check_documentation.py --print-index
python3 scripts/check_documentation.py
python3 scripts/knowledge_base.py check --summary
```

print-index仅输出生成区，由维护者替换原docs/README对应标记。coverage核受管文档、模块、附档、原技能及精确 `.agents/skills/obsidian-wiki/SKILL.md` 入口可达与目录一致；不通配扫描其他隐藏/私密目录。原技能不计managed数量，仍保留独立入口。主题也绑定原文/代码依赖；全树watch覆盖新增/删除/修改和被忽略源码文件名，不读取秘密/缓存。目录可发现不等于逐图/逐benchmark复验。

## 官方Obsidian CLI

在运行应用的同一桌面终端使用。隔离shell看不到socket不等于应用不可用，也不是降低沙箱的理由。

```bash
python3 scripts/knowledge_base.py obsidian read docs/wiki/README.md --vault FundInvestmentResearchPlatform
python3 scripts/knowledge_base.py obsidian read docs/wiki/topics/business-purpose.md --vault FundInvestmentResearchPlatform
python3 scripts/knowledge_base.py obsidian search 研究 --vault FundInvestmentResearchPlatform
python3 scripts/knowledge_base.py obsidian property docs/wiki/topics/business-purpose.md --property review_state --vault FundInvestmentResearchPlatform
python3 scripts/knowledge_base.py obsidian backlinks docs/wiki/topics/business-purpose.md --vault FundInvestmentResearchPlatform
```

命令未注册可用--executable指定该机器官方obsidian-cli。每次核真实vault根/精确path，read比对磁盘。官方1.13.7错目标可能exit0，适配器检查stdout/stderr错误并返回失败。只允许4只读操作，无eval/command/删除/插件/同步。

## 本机与云端共同使用

用户已自行安装并打开本机官方Obsidian。本机选择同一项目checkout根，打开相同主页；本机CLI要独立启用/验收，不能由云端推断。

Git共享知识和相对链接；.obsidian、workspace、缓存、CLI注册各机独立，不提交。知识库须经开发分支提交、PR审核并合入Dev后，由各机获取相应Git版本；云端提交不会自动更新本机。同步前检查git status并保护改动，不reset/stash覆盖；冲突按证据语义合并，再检查和本机验收。不要双重启用Git与Obsidian Sync。

## 完整验收矩阵

| 范围 | 交付与验收目标 |
| --- | --- |
| KB-01 | 8业务＋6开发＋1研究主题；高层关系/边界/原因，底层卡不主导主页 |
| KB-02 | 全部受管文档、36模块、相关附档/benchmark与原技能，coverage无缺口 |
| KB-03 | 来源身份/内容去重、敏感URI、明确换版与并发负例 |
| KB-04 | 默认未核、真实commit、主题与链式失效、循环和上游信任 |
| KB-05 | 仅派生整合、明确审阅、hash/幂等、原权威不变 |
| KB-06 | 反馈、处理记录、仍需复核、队列使用缺口 |
| KB-07 | 四种intent与15个真实高层问题检索，不造无源答案 |
| KB-08 | 根vault、主页/主题/原文、搜索/反链/属性和CLI实际验收 |
| KB-09 | 离线回归、原文档/索引/Hermes和精确候选审阅，无业务源码改动 |
| KB-10 | 相对路径、机器状态隔离、Git同步边界、耐久源码改动包 |

### 重复验收命令

```bash
python3 -m pytest scripts/tests/test_knowledge_base.py scripts/tests/test_documentation.py -q
python3 scripts/knowledge_base.py check --summary
python3 scripts/knowledge_base.py coverage
python3 scripts/check_documentation.py
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
python3 skills/ai-hermes-self-evolve/scripts/route_task.py --route-id R04 --mode context
```

新稳定文件尚未进Git时，普通Hermes会如实报未跟踪；可用外部临时索引验证精确候选，不改真实暂存区、不commit、不用artifact白名单掩盖。结构/工具通过不替代金融资格或生产验证。

### PR #68 交付验收（历史）

本地与独立审核已重跑248项离线测试（171知识工作流、76文档检查器、1 Portable）。105受管文档/1371本地链接、36模块、92附档、4技能覆盖保持通过；22条知识记录无invalid/stale/untracked，保留原2项needs_review。队列没有未解决的工作流记录格式告警。

来源、反馈、整合与队列使用一致的记录边界；引用/围栏/导入文字不是流程事件。完整候选写入前验证新旧事件仍可识别，来源核验快照变化则停止。知识库测试已纳入现有Documentation工作流的同一pytest步骤，无新包或权限。此为 PR #68 历史记录，后续技能闭环的真实库门禁变更见下节。实际远端CI、Bot审核及合并状态以[PR #68](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/68)对应HEAD为准；本地通过不代替这些结果。

### 恢复后首次验收记录

2026-10-04环境曾发生整体快照替换，旧环境的167项测试没有沿用为新机通过。完整内容已按原设计、保留的主题原文、证据和已验收工具契约恢复；恢复代码不声称与丢失版本逐字相同，已重新验收：

- 177项离线测试通过：101知识工作流＋76原文档检查器；含15个真实高层首命中、完整main命令闭环和全部已发现安全负例
- 105份受管文档、36模块、92附档/benchmark、4份原技能完整可发现；coverage无缺口，105文档/1371本地链接检查通过
- 22条记录的invalid/stale/untracked均为0；保留2项needs_review，即复权解释conflict及引用它的金融证据主题
- 新环境Obsidian主页→业务主题→原研究章节、主页→技术主题→原契约实际通过；CLI主页/两主题逐字读取、属性/反链通过，wrapper正确目标exit0，错误目标exit2
- 本次38文件只涉及README、知识库/治理/目录和工具/测试，业务源码未改；临时Git候选的文档/Hermes检查通过，实现验收时真实暂存区未变、未commit/push/merge；后续发布状态以远端PR为准
- 完整改动包另保存在本次交付的耐久附件，含基线、全部38文件、逐文件hash与tracked patch，排除.git、凭据、数据及.obsidian；恢复时先保护现有工作，不盲目覆盖

独立review已定点复验敏感URI各入口、目录解析/写入的symlink保护、literal pathspec、真实commit、来源信任传播及安全写入。云端通过不替代本机CLI/知识同步，二者仍未执行；外部论文/竞品、金融实验、生产与逐图/benchmark内容也未因本次知识工程重新验证。恢复过程不操作生产、不读取私人会话、不进行AWS登录。

### 发布前可复现性核对

2026-10-04获准提交后，核实本地审计提交与远端审计提交的完整tree相同，将22条记录的source_revision映射到远端已有的8ce7c06完整SHA，并复核5处受元数据变化影响的依赖指纹。审阅状态、金融冲突和证据范围不变。发布前重跑178项离线测试（上述177项及1项Portable边界测试）与Portable边界脚本通过；远端PR、CI、Bot审核和合并须分别核实，不能从本条本地记录推断。

### PR #68 首轮审核修复

2026-10-04核实Codex Bot的三项建议均为真实问题，已作最小修复：watch_globs保留改名/复制两端路径；来源版本在比较和保存前统一首尾空白，并兼容旧记录；intake使用同一次原始字节读取生成解析内容与预期hash，拒绝覆盖读取后发生的编辑。并发保护仍是乐观hash核对，不承诺对不遵守锁的外部编辑器提供原子事务。

新增12项定点回归，前三问题在修复前已有失败反例；修复后190项离线测试通过（113知识库、76文档、1 Portable），含真实R100暂存/已提交改名、C100及特殊文件名、旧版空白来源身份、外部编辑插入和CRLF字节。Portable静态边界检查通过，22条记录保留原2项needs_review，invalid/stale/untracked仍为0。最新远端审核结论与合并状态仍以PR为准。

后续审核核实整合幂等的子串判断会误认不同摘要或审阅记录，现改为唯一BEGIN/END边界内的整块精确比较，拒绝不完整、重复或审阅字段边界多解的块，保持旧块格式与原历史日期/hash。新增11项回归在修复前10失败、1通过，修复后201项通过（124知识库、76文档、1 Portable）；不因此宣称金融冲突或生产状态已通过。

再次核对整合写入边界后，幂等成功也执行当前目标hash门禁与返回前复查；旧整合前hash不再作为重试凭据，未变目标可使用上次返回的after_sha256。首次写入与重试共用回执校验，摘要、审阅人、理由和claim标题/范围中的保留整合标记及多解字段会在写入前被拒绝。新增16项回归先复现失败再修复，最终217项通过（140知识库、76文档、1 Portable）；相关安全重试及手工历史块兼容性均重新核对。

工作流状态边界专项核对补充31项回归：解析器复用原markdown-it-py，只接受末尾复核章节之后的顶层完整记录；反馈与处理记录具有明确边界，引用/围栏/旧无END等内容列入workflow_records_needing_review，不作为待处理事件。来源摘要/文本不能通过标记操纵队列，修改后claim快照不会与旧审阅结果拼接。248项完整回归通过；反馈作者使用非空单行文本，处理原因保留在有界回执中。


### 项目 Wiki 技能闭环验收（2026-10-04）

本轮基线为已合并 Dev 的 `0c9dba1`；原本地 `2051590` 与其 tree 相同。新增项目内五流程技能正本、普通文件发现入口与固定上游 MIT 来源记录，复用原知识脚本和 Hermes；AGENTS 显式要求修改前查知识/原证据、修改后判断影响和验收。Documentation 配置在 PR HEAD 上增加真实库 check/coverage，并分别保留 JSON 报告，未修改远端 required checks。

- Python 3.12.14 下273项离线测试通过：171既有知识工作流、25新增技能/闭环契约、76文档、1 Portable；两个技能入口另通过 skill-creator 基础校验
- 新隔离夹具覆盖 intake/去重、未核阻断、证据审阅、整合、四种 context、来源链失效、反馈、处理反馈后仍 stale、重新审阅与整合；原权威除明确模拟来源变更外保持原样
- 真正只读的跨进程快照覆盖 search/context/check/queue/hash/fingerprint/catalog/coverage，含无命中边界；消除了原 catalog/coverage 首次导入在仓库内写 bytecode 的副作用
- 独立行为试用实际回答复权经济资格问题，回读原设计/实现/合成测试，保留 reviewed/current 与 conflict/needs_review 并存的真实含义；隔离供应商反证试用保留旧回执、传播失效，拒绝过期 hash 与未复核整合，未执行来源中的命令或发布指令
- 真实库22条记录，invalid/stale/untracked均0，原复权冲突及其业务证据主题两项needs_review保留；105文档、36模块、92附档和12份支持说明/技能可达，catalog一致
- 四份受本次AGENTS/文档协议/路由差异影响的主题逐项语义复核并留下注明范围的说明，只更新相应依赖，不刷新其他卡或清空冲突
- 文档结构/链接、Portable边界、Hermes validate/evolve与R04路由通过；新文件的Git可复现性通过仓库外临时索引核对，真实暂存区保持不变

本轮只交付本地实现和测试；尚无这次候选的远端CI/Bot审核、提交推送、PR或合并结果，也未执行用户本机同步/CLI、业务数值回归或生产验证。工具和技能结构通过不代表所有客户端均会自动加载，也不代表待核金融问题已解决。
