# 新版投研首页设计验收

## 范围与参照

- 需求及技术设计：`docs/homepage-design.md`；实现入口：`/`，唯一组件 `ProcessHome`。
- 用户参考图：`codex-clipboard-63b576dc-f491-459c-91f3-51142f95ab2a.png`，原始像素 2994 × 4198。
- 参考图按 2 倍密度推断为 1497 × 2099 CSS px；该密度是对齐假设，并非原浏览器元数据。
- 实现通过 Chrome 在 1497 × 2099 CSS px、DPR 1 截取，整页 1497 × 2105 像素。比较时参考图按宽度缩至一半，不拉伸高度，6px 内容高度差单独记录。
- 初次对比将参考图与整页实现图放在同一次图像工具调用中共同检查；随后分别复核 Hero、核心能力、地球区域、320px 中英文和弹窗。
- 完整日志临时目录：`/private/tmp/fund-homepage-qa.QvgcIH`。持久化截图：[桌面](docs/homepage/screenshots/desktop.png)、[手机](docs/homepage/screenshots/mobile.png)、[搜索](docs/homepage/screenshots/search.png)。

## 视觉核对

- 浅色导航、左侧标题与双按钮、小牛背景、五阶段卡片、四类工具、引用、地球面板、快捷入口及页脚顺序一致。
- 复用原有 WebP 素材而非重新生成近似图；小牛头部、电脑、背景图表与地球位置保持完整。中文桌面版的主要区块高度与参考图一致，未发生遮挡或拉伸。
- 文本为可选择的 HTML，图标来自现有 Heroicons；不是整页图片。字体优先使用系统 Avenir/PingFang，与参考图的中文粗细和层级接近。
- 320px 手机采用真实重排：两列阶段/能力卡、三列快捷入口，Hero 图片在文案之后。英文结构说明采用两列，避免长单词挤出视口。
- 轻微差异（P3）：图标库的描边、引用气泡和品牌 SVG 底色与附件不完全相同；全文高度差约 6px，参考图与实际内容水平对齐约有 8px 差异。不影响层级、内容可读性或操作。
- 有意差异：实际平台首页不保留“未连接业务服务”的演示声明，改为明确研究示例和不下单边界。详情见设计文档。

## 交互与代码自审核

- 搜索、流程、更多菜单、移动菜单和工具入口只使用现有路由；首页不调用业务写接口。
- 导览前后步骤和结束、三个示例说明、帮助、关于与本机工作区说明均可打开和关闭。
- 搜索立即聚焦；Tab/Shift+Tab 限定在弹窗；Escape 一次关闭非空搜索框并恢复触发点；卸载释放监听、恢复 body 滚动。应用内浏览器已复核输入“指标”及 Escape 恢复焦点。
- 最近访问只保存最多 8 个白名单模块 ID，不保存具体产品或参数；空状态不伪造记录。清除只影响独立的首页记录键，存储错误不阻断导航。
- 首页和工作台复用同一语言状态；工作台原导航、阶段下拉和 PIT 未知/严格模式契约保留。旧首页实现与无用文案删除，无隐藏旧路由或双版本开关。
- 无新增生产计算、下载或数据存储操作。导航回归的会计数值调用使用独立无数据服务，执行已有固定签名 NJIT 内核；其他接口用受控响应隔离。

## 修正记录

1. React 18 的 `fetchPriority` 提示：改用同义原生小写属性，保持图片加载优先级。
2. 搜索输入在原生 dialog 打开前自动聚焦无效：改为 `showModal` 后显式聚焦；补充首尾 Tab 和 Escape 行为。
3. 英文手机页的品牌与结构说明超宽：允许品牌换行、结构说明改为两列，并检查全页无横向溢出。
4. 原流程 E2E 依赖本机端口 8000：改用临时端口、真实数值 NJIT 测试服务，避免正式数据依赖。
5. 旧 PIT 回归从首页查找研究口径控件：改在产品研究工作台验证同一失败/重试语义。
6. 多套重负载测试同时运行导致三个未修改的历史情景单元测试超时：不放宽断言，限制 Vitest 为两个 worker 重新完整执行。
7. 补查断点发现英文导航在 1251px 横向溢出：仅在 1251–1400px 收紧英文导航间距，补充 1251px/1366px 浏览器回归，保持中文和参考尺寸不变。
8. 完整复查发现指标画布测试关闭页面时仍有代理请求，触发 `route.fetch: Test ended` 并影响同 worker 后续用例：在 `afterEach` 等待 `unrouteAll({ behavior: 'wait' })`，不吞掉网络错误或放宽业务断言。

## 最终验证

- 前端全量单元测试：122 个文件、898 项通过（限制两个 worker，91.89 秒）。
- 相关后端回归：业务数值 NJIT、本地化与语言矩阵 51 项通过。
- TypeScript、Vite 生产构建、i18n 契约、AI Hermes 路由覆盖与引用校验通过；系统文案 625 项、业务文案 406 项。
- 首轮完整 Playwright：271 通过、4 失败、7 跳过。失败是旧 PIT 控件定位（三种视口）和非空搜索框 Escape 行为；修复后，三个受影响测试文件在全部视口复测：43 通过、2 跳过。
- 第二轮完整 Playwright：272 通过、3 失败、7 跳过（9.7 分钟）；三项失败均为指标画布测试退出时未等待代理请求结束。补充清理后，画布全部用例在三个视口重复两轮：30 项全部通过（2.9 分钟）。该全量执行本身不是单轮全绿，不将其失败结果隐去。
- 首页最终补测：10 通过、2 跳过，包含 1251px/1366px 英文导航断点。
- 7 个预期跳过为：三个显式关闭的真实工作区因子研究用例、两次无需重复执行的历史情景桌面专用交互、两次桌面专用参考截图。
- 实际浏览器检查：中文桌面、320px 中英文、搜索空状态、焦点恢复与导航正常；未遗留本次改动导致的 P0/P1/P2 问题。
- 非阻断提示：应用已有的大 bundle 构建提示、浏览器兼容数据库更新提示、部分存量测试的 React act 提示。不代表生产构建失败。
- 测试使用受控模拟响应或独立临时服务；未执行显式要求正式工作区写入的验收，未操作共享 Tushare 数据，也不据此宣称生产部署或投资策略有效性。
- 验收结论基于完整回归及受影响集合的修复复测：已执行的业务/交互用例均有通过证据，未遗留未解决的失败；最终代码通过自审核。未配置远端 CI，本记录是本地验证，不是 CI 通过证明。

### 可重复命令

在仓库根目录执行，`INDICATOR_TEST_PYTHON` 指向已安装项目依赖的 Python 3.12 环境：

```sh
npm run test --prefix frontend -- --run --maxWorkers=2 --minWorkers=2
npx --prefix frontend tsc --noEmit --project frontend/tsconfig.json
npm run build --prefix frontend
node scripts/check_i18n.mjs
PYTHONPATH=.:backend "$INDICATOR_TEST_PYTHON" -m pytest backend/tests/test_business_numeric_numba.py backend/tests/test_localization.py backend/tests/test_localization_matrix.py -q
npm run test:e2e --prefix frontend -- --workers=3
npm run test:e2e --prefix frontend -- indicator-canvas.spec.ts --workers=3 --repeat-each=2
python skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py
python skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
git diff --check
```

final result: passed
