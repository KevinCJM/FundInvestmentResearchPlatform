# 情景识别：数据续接收口与最终验收

日期：2026-09-15。分支：`ISSUE2609/BetterSaaTaa`。

本文是本轮最终证据，补充并优先于 regime-completion-final-audit-2026-09-15.md 中“固定指数快照不能续接”的旧状态。没有提交、推送、合并或改写生产数据；没有下载数据。前轮已有模板、历史质量、稳定性、bootstrap及前瞻代码继续保留，仅沿同一执行链补齐实际数据续接。

## 1. 完成内容

- 历史状态定义：开放计算图/公式/向导、已有及新增事后模板、区间质量和参数/边界敏感性、不可变参考与质量报告。原 PS/沪深300模板数学未变。
- 实时识别：绑定精确参考、状态/区间/转折评价，独立校准与留出评分，参数/窗口/适用种子稳定性、配对移动块统计区间，未知/不足时明确不估计。
- 前瞻闭环：冻结登记、当时捕获、后续成熟参考、固定窗口资格核验、显式采用。历史诊断报告不会被改写为部署通过。
- 本轮新增单一指数固定快照的安全续接：检查新数据版本 → 确认续接 → 使用新数据记录当前判断。参考的仅数据修订通过原版本服务创建，但仍须单独运行和确认发布，不自动发布标签。
- 采用前瞻合格校准时，同时采用已批准数据绑定和qualification_id，形成模型新修订。TAA继续核验资格、实际源绑定、时点和状态轴；过期、缺失或模型变化时不放行。

## 2. 版本与时点保护

source_versions.py仅承担I/O、哈希、版本编排，没有新数值实现。来源、PIT、prepare、图执行、temporal audit及数值内核沿用原服务。预览在有界TTL内存中；确认只接受预览hash、在服务端重验并签名追加。客户端不能提交路径、日期、数据或通过状态。

新源必须有新增已知观察，原日期/数值/可得时间前缀完全相同；原快照文件仍校验。删除历史行、修订历史值/发布时间、换指数/字段/算法/状态、陈旧预览均拒绝。原定义、报告、参考结果和文件字节不变。

参考仅接受原定义或该协议明确批准的仅数据修订。捕获前检查所有批准参考修订的既有发布，禁止知道标签后补录预测。重读旧资格仅使用该资格之前的续接记录，不让未来版本改写历史检验。

## 3. 自审核发现与修复

1. 原本绑定数据根目录的定义在新子目录快照激活后无法读取：保留根目录的明确generation解释，既有身份/路径/checksum检查不取消。
2. 新数据续接后验证通过，但采用时仍可能读取旧源：显式采用时携带批准的四项绑定，后端资格验证要求当前批准的模型hash。无资格仍执行原精确hash门禁。
3. 原始捕获末点effective_date可能为null：这是尚未成为可交易信号，不是数据异常。共享TAA消费者现在返回calibration_signal_time_missing而不是TypeError；非法日期也拒绝，不能用as_of补造生效时点。
4. source_version_id与上一捕获checkpoint容易混淆：记录分为实际source_version_id、input_checkpoint_id及execution_model_binding_hash。
5. 旧协议因执行器变更失效不应阻塞全部研究启动：预热报告列出blocked protocol，旧捕获/资格仍拒绝；真正数值预热失败和journal完整性失败没有被吞掉。
6. 前端采用回调保持旧两参数兼容，只有实际续接时传第三个source bindings；错误文本仅允许明确的安全业务文案，任意含中文的异常也不会直接暴露。

## 4. 最终测试

所有测试使用临时目录/合成行情。后端实际调用源解析、Parquet、PIT、图执行、发布、签名journal及共享TAA消费者，不用前端mock证明数学结果。

| 范围 | 结果 | 日志 |
| --- | --- | --- |
| 相关后端最终完整复跑 | **285 passed**，143.32秒 | `/tmp/regime-source-final-backend2.log` |
| 前端全量 | **145文件，1108 passed**，79.67秒 | `/tmp/regime-source-final-ui.log` |
| TypeScript | 通过 | `/tmp/regime-source-final-tsc.log` |
| design:check / i18n | 均通过，预算未放宽 | `/tmp/regime-source-final-design.log`、`/tmp/regime-source-final-i18n.log` |
| 隔离生产构建 | 通过，6.53秒 | `/tmp/regime-source-final-build.log` |
| 前瞻/源续接浏览器，320/768/1440 | **7 passed** | `/tmp/regime-source-browser-forward.log` |
| 历史质量/实时诊断浏览器，三个宽度 | **18 passed** | `/tmp/regime-source-browser-diagnostics.log` |
| 真实隔离前后端质量/可靠性API联动 | **2 passed** | `/tmp/regime-source-browser-integration.log` |
| 原工作台、人工事件和时点库浏览器 | **22 passed，2 skipped** | `/tmp/regime-source-browser-existing.log` |

浏览器合计 **49项通过，2项原有桌面限定用例在非桌面跳过**。覆盖键盘、实际DOM/图表或既有降级、文字对比度、无页面横向溢出、预览不写入、确认、重载、引用变更失效与迟到响应。浏览器续接使用离线接口夹具；续接真实计算全流程另由test_regime_source_continuation.py覆盖，不混淆两类证据。

本轮早期前端一次失败来自新增可选回调参数影响旧断言；后端一次失败发现null生效日，已按第3节修复，之后上述最终回归通过。没有修改测试门槛制造通过。

保留既有React act提示、Starlette/httpx弃用和构建大chunk提示。数值仍为既有固定签名NJIT，无Python fallback。源码变更后git diff --check通过。

## 5. 数据续接专项反例

10项专项测试包含完整受控前向样本、60个模拟捕获、成熟参考评分/资格/TAA、进程重启预热、两个快照续接、原字节保护、数值/可得时间/缺行修订拒绝、旧预览/客户端伪造拒绝、旧参考算法编辑保护、已知标签禁止补录、真实HTTP字段门禁，以及旧失效协议不阻塞其他研究启动。

这些模拟时钟只能证明软件契约，不意味着真实市场已有任何模型取得资格。

## 6. 明确边界

- 本轮数据续接只支持**单一日频指数来源**。宏观、多源、上传文件、跨频率的通用续接没有新增；它们仍可进行已有历史研究及诊断，不能以删除校验和强行续接。
- 不自动采集、定时记录或发布参考。现有系统完成数据更新后，由用户显式确认续接。
- 历史参考不是客观真值；bootstrap区间是条件于当前模型/参考/校准器的历史统计量区间，不是当天“真实牛市概率”的区间，也不覆盖全部模型选择误差。
- 真实前瞻资格必须等待真实后续数据及成熟参考。检验采用预注册门槛，不宣称统计显著性或获利保证。
- 本地签名日志不能替代独立外部审计，也不防止掌握本机密钥的管理员重新签名。
- CMA/SAA主体、新Student-t HMM、BOCPD/PELT等独立算法扩展未在本轮开发。

## 7. 复现命令

```sh
TEST_ROOT=$(mktemp -d /tmp/regime-source-review.XXXXXX)
export CUSTOM_INDICATOR_DATA_DIR="$TEST_ROOT" HISTORICAL_REGIME_DATA_DIR="$TEST_ROOT"
export PYTHONPATH=.:backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m pytest backend/tests/test_regime_source_continuation.py backend/tests/test_regime_prospective.py backend/tests/test_regime_prospective_integration.py backend/tests/test_regime_reliability_consumer.py -q
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm run design:check --prefix frontend
node scripts/check_i18n.mjs
npm run build --prefix frontend -- --outDir /tmp/bettersaataa-regime-source-final-build
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-source-final-build npm run test:e2e --prefix frontend -- --config=e2e/regime-prospective.config.ts
REGIME_OFFLINE_BUILD=/tmp/bettersaataa-regime-source-final-build npm run test:e2e --prefix frontend -- --config=e2e/regime-offline.config.ts --workers=2
REGIME_INTEGRATION_BUILD=/tmp/bettersaataa-regime-source-final-build npm run test:e2e --prefix frontend -- --config=playwright.regime-integration.config.ts
npm run test:e2e --prefix frontend -- historical-regime-workbench.spec.ts regime-manual-events.spec.ts regime-temporal-event-library.spec.ts --workers=2
```

实际使用项目推荐Python3.12解释器；生产构建保存在临时目录，没有覆盖共享dist。所有启动的测试服务由Playwright管理并已退出。

## 8. 提交与路由

所有改动仍在本地开发分支，未暂存/commit/push。新增稳定源文件和测试等待正式提交范围确认后登记路由，不把真实代码伪装为临时产物来绕过git-tracked规则。

最终显式19个本轮路径的覆盖检查：10已覆盖、9未登记、missing_required_files为空，exit 1；日志/tmp/regime-source-routing-coverage.json。未登记包括新的续接测试、prospective前端服务/测试/e2e和相关设计验收文档。源码主体路径已有目录覆盖，但路径覆盖不代表已完善全部业务语义索引。未将检查器阈值调低或将真实源文件加入临时白名单。

独立validate_ai_routing.py通过，git diff --check通过；两者不替代新增文件覆盖登记。该剩余项属于提交前的路由文档治理，不能描述为全部仓库检查均已通过。最终确认测试后端8769端口无监听进程。
