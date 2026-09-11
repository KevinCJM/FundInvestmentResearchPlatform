import { expect, test } from '@playwright/test'
import { taaBaseline, taaCatalog, taaPreflight } from '../src/test/tacticalAllocationFixtures'

test('流程首页在当前视口完整展示且不存在水平溢出', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: '公募基金量化投研流程' })).toBeVisible()
  await expect(page.getByLabel('反馈与迭代回流至产品研究')).toBeVisible()
  await expect(page.getByRole('heading', { name: '基金会计与管理人账务' })).toBeVisible()
  await expect(page.getByText(/个节点/)).toHaveCount(0)
  await expect(page.getByRole('heading', { name: '组合中心 · 真实组合库' })).toBeVisible()
  const portfolioSolutionsBox = await page.getByRole('link', { name: /组合方案展示中心/ }).boundingBox()
  const settingsBox = await page.getByRole('link', { name: /设置 · 公共能力/ }).boundingBox()
  expect(portfolioSolutionsBox).not.toBeNull()
  expect(settingsBox).not.toBeNull()
  expect(settingsBox!.y).toBeGreaterThan(portfolioSolutionsBox!.y + portfolioSolutionsBox!.height)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('静态 Booking 页统一业务录入与复式记账', async ({ page }) => {
  await page.goto('/investment-execution/execution')

  await expect(page.getByText('静态功能演示｜未接入真实数据与后端服务')).toBeVisible()
  await expect(page).toHaveURL(/\/fund-accounting\/booking$/)
  await expect(page.getByRole('heading', { name: '组合 Booking 与复式记账' })).toBeVisible()
  await expect(page.getByLabel('选择 Booking 数据文件')).toBeVisible()
  await expect(page.getByText('actual_portfolio_id')).toBeVisible()
  await expect(page.getByLabel('Booking 真实组合')).toBeVisible()
  await expect(page.getByLabel('Booking 来源账户')).toBeVisible()
  await expect(page.getByLabel('Booking 基金经理任职关系')).toBeVisible()
  await expect(page.getByLabel('组合或基金账规则预判')).toHaveAttribute('readonly', '')
  await expect(page.getByLabel('管理人账规则预判')).toHaveAttribute('readonly', '')
  await expect(page.getByLabel('组合绩效现金流规则预判')).toHaveAttribute('readonly', '')
  await page.getByRole('button', { name: '复式凭证' }).click()
  await expect(page.getByText(/ETF 买入｜交易日确认/)).toBeVisible()
  await expect(page.getByText(/借贷差额：¥0.00/).first()).toBeVisible()
  await expect(page.getByRole('button', { name: /下单|委托提交|撤单/ })).toHaveCount(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('TAA 节点提供可运行工作台而不是静态职责说明', async ({ page }) => {
  const writes: string[] = []
  const preflights: string[] = []
  await page.route('**/api/**', route => {
    const request = route.request(), path = new URL(request.url()).pathname
    // Opening the workbench may check inputs, but must not calculate or persist a decision.
    if (request.method() === 'POST' && path === '/api/tactical-allocation/preflight') {
      preflights.push(path)
      return route.fulfill({ json: taaPreflight })
    }
    if (request.method() !== 'GET') writes.push(path)
    if (path === '/api/tactical-allocation/catalog') return route.fulfill({ json: taaCatalog })
    if (path === '/api/tactical-allocation/baselines/SAA-1') return route.fulfill({ json: taaBaseline })
    if (path === '/api/historical-regimes/runs') return route.fulfill({ json: { items: [] } })
    return route.fulfill({ status: 404, json: { detail: 'Offline process navigation fixture' } })
  })
  await page.goto('/pre-investment/taa?baseline=SAA-1')

  await expect(page.getByRole('heading', { name: '本次准备怎么配？' })).toBeVisible()
  for (const name of ['观点与规则', '回测与选优', '情景模拟', '版本与审计']) {
    await expect(page.getByRole('tab', { name, exact: true })).toBeVisible()
  }
  await expect(page.getByRole('radio', { name: /研究员观点/ })).toBeVisible()
  await expect(page.getByRole('button', { name: '计算并比较方案' })).toBeEnabled()
  expect(preflights.length).toBeGreaterThan(0)
  expect(writes).toEqual([])
  await expect(page.getByTestId('non-interactive-blueprint')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /下单|委托提交|撤单/ })).toHaveCount(0)
})

test('研究口径与参数中心位于设置并兼容旧地址', async ({ page }) => {
  await page.goto('/product-research')
  await expect(page.getByRole('link', { name: /研究框架与数据基础/ })).toHaveCount(0)

  await page.goto('/settings')
  await expect(page.getByRole('link', { name: /进入节点/ }).filter({ hasText: '研究口径与参数中心' })).toBeVisible()

  await page.goto('/product-research/framework-data?template=RP-R4')
  await expect(page).toHaveURL(/\/settings\/research-parameters\?template=RP-R4$/)
  await expect(page.getByRole('heading', { name: '研究口径与参数中心' })).toBeVisible()
  await expect(page.getByText(/业务页面只选择并引用模板/)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('会计报表页分别展示组合基金报表和管理人公司报表', async ({ page }) => {
  await page.goto('/fund-accounting/financial-statements')

  await expect(page.getByRole('heading', { name: '双主体财务报表' })).toBeVisible()
  await expect(page.getByRole('button', { name: '资产负债表' })).toBeVisible()
  await expect(page.getByRole('button', { name: '利润表' })).toBeVisible()
  await expect(page.getByRole('button', { name: '净资产变动表' })).toBeVisible()
  await page.getByRole('button', { name: '附注与勾稽' }).click()
  await expect(page.getByText(/资产 12,648 = 负债 118 \+ 净资产 12,530/)).toBeVisible()
  await page.getByRole('button', { name: /管理人公司账/ }).click()
  await expect(page.getByRole('button', { name: '现金流量表' })).toBeVisible()
  await expect(page.getByRole('button', { name: '所有者权益变动表' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('组合方案展示区分回测与实盘，并展示算法和情景入口', async ({ page }) => {
  await page.goto('/portfolio-solutions/profile')

  await expect(page.getByRole('heading', { name: '组合画像与适用范围' })).toBeVisible()
  await expect(page.getByText(/虚线左侧为历史回测，右侧为实盘跟踪示例/)).toBeVisible()
  const trigger = page.getByRole('button', { name: '方案展示阶段导航' })
  await expect(trigger).toHaveAttribute('aria-expanded', 'false')
  await expect(page.getByLabel('方案展示子页面导航')).toHaveCount(0)
  if ((page.viewportSize()?.width ?? 0) >= 1024) {
    const workspaceBox = await page.getByLabel('方案展示工作区').boundingBox()
    expect(workspaceBox).not.toBeNull()
    expect(workspaceBox!.width).toBeGreaterThan((page.viewportSize()?.width ?? 0) * 0.8)
  }

  await trigger.click()
  const navigation = page.getByLabel('方案展示子页面导航')
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')
  await expect(navigation.getByRole('link', { name: /周期与情景模拟/ })).toBeVisible()
  await expect(navigation.getByRole('link', { name: /配置与算法说明/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('组合中心承接研究方案并为投中提供真实组合主数据', async ({ page }) => {
  await page.goto('/portfolio-center/accounts')

  await expect(page.getByRole('heading', { name: '账户主档与组合关系' })).toBeVisible()
  await expect(page.getByText('LEDGER-PF-DEMO-01')).toBeVisible()
  await expect(page.getByText('EXEC-FM-DEMO-SSE').first()).toBeVisible()
  await expect(page.getByText('稳健多资产一号、固收增强三号')).toBeVisible()

  await page.goto('/investment-execution/onboarding')
  await expect(page.getByRole('heading', { name: '组合落地与启用' })).toBeVisible()
  await expect(page.getByLabel('组合落地边界说明（非交互）')).toContainText('不完成基金法律设立')
  await expect(page.getByRole('button', { name: /下单|委托提交|撤单/ })).toHaveCount(0)
})

test('共享交易通道先做公平分配，再与基金专用资金账户核对', async ({ page }) => {
  await page.goto('/investment-execution/trade-allocation')
  await expect(page.getByRole('heading', { name: '汇总订单与公平交易分配' })).toBeVisible()
  await expect(page.getByText(/尾差 0/)).toBeVisible()
  await expect(page.getByText('CASH-PF01-CNY')).toBeVisible()
  await expect(page.getByText('CASH-PF03-CNY')).toBeVisible()

  await page.goto('/fund-accounting/account-statements')
  await expect(page.getByRole('heading', { name: '外部账户流水、拆分与结算匹配' })).toBeVisible()
  await expect(page.getByText('分配尾差 ¥0.00')).toBeVisible()
  await expect(page.getByText(/共享的是执行通道，不是基金财产/)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})
