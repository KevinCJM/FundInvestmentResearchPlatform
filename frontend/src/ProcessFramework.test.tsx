import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import App from './App'

describe('投研流程框架', () => {
  afterEach(() => window.history.replaceState({}, '', '/'))

  it('主页展示五阶段闭环和独立设置入口，不显示节点数量标签', () => {
    window.history.replaceState({}, '', '/')
    render(<App />)

    expect(screen.getByRole('heading', { name: '公募基金量化投研流程' })).toBeInTheDocument()
    expect(screen.getByLabelText('反馈与迭代回流至产品研究')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '设置 · 公共能力' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '组合方案展示中心' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '组合中心 · 真实组合库' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '基金会计与管理人账务' })).toBeInTheDocument()
    expect(screen.getByTestId('extended-capabilities-stack')).toBeInTheDocument()
    expect(screen.getByText(/公共能力：数据与 PIT/)).toBeInTheDocument()
    expect(screen.queryByText(/个节点/)).not.toBeInTheDocument()
  })

  it('阶段导航默认收起，可触发展开并用 Escape 关闭', async () => {
    const user = userEvent.setup()
    window.history.replaceState({}, '', '/pre-investment')
    render(<App />)

    const trigger = screen.getByRole('button', { name: '投前决策阶段导航' })
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('投前决策子页面导航')).not.toBeInTheDocument()

    await act(async () => { await user.click(trigger) })

    const navigation = screen.getByLabelText('投前决策子页面导航')
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(within(navigation).getByRole('link', { name: /战略资产配置（SAA）/ })).toBeInTheDocument()

    await act(async () => { await user.keyboard('{Escape}') })
    expect(screen.queryByLabelText('投前决策子页面导航')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it('组合方案展示中心提供独立的表现、回测、情景和披露入口', () => {
    window.history.replaceState({}, '', '/portfolio-solutions')
    render(<App />)

    expect(screen.getAllByRole('link', { name: /收益与风险表现/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /历史回测模拟/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /周期与情景模拟/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /披露与展示版本/ }).length).toBeGreaterThan(0)
  })

  it('投中交易数据入口并入 Booking 与复式记账', () => {
    window.history.replaceState({}, '', '/fund-accounting/booking')
    render(<App />)

    expect(screen.getByRole('heading', { name: '组合 Booking 与复式记账' })).toBeInTheDocument()
    expect(screen.getByText('静态功能演示｜未接入真实数据与后端服务')).toBeInTheDocument()
    expect(screen.getByLabelText('Booking 真实组合')).toBeInTheDocument()
    expect(screen.getByLabelText('组合或基金账规则预判')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('管理人账规则预判')).toHaveAttribute('readonly')
    expect(screen.queryByRole('button', { name: /下单|委托提交|撤单/ })).not.toBeInTheDocument()
  })

  it('会计核算目录包含复式 Booking、两类账簿、双账勾稽、报表和绩效数据', () => {
    window.history.replaceState({}, '', '/fund-accounting')
    render(<App />)

    expect(screen.getAllByRole('link', { name: /组合 Booking 与复式记账/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /外部账户流水、拆分与结算匹配/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /组合\/基金总账与明细账/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /管理人公司账映射/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /双账勾稽、差错与关账/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /双主体财务报表/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /绩效核算数据集/ }).length).toBeGreaterThan(0)
  })

  it('投中先完成真实组合落地，组合中心统一维护账户与账簿关系', () => {
    window.history.replaceState({}, '', '/investment-execution')
    const first = render(<App />)
    expect(screen.getAllByRole('link', { name: /组合落地与启用/ }).length).toBeGreaterThan(0)
    expect(screen.getByLabelText('当前真实组合')).toBeInTheDocument()
    first.unmount()

    window.history.replaceState({}, '', '/portfolio-center')
    render(<App />)
    expect(screen.getAllByRole('link', { name: /真实组合登记簿/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /账户主档与组合关系/ }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('link', { name: /基金经理与投资单元/ }).length).toBeGreaterThan(0)
  })

  it('投中提供多组合公平交易分配工作区', () => {
    window.history.replaceState({}, '', '/investment-execution/trade-allocation')
    render(<App />)

    expect(screen.getByRole('heading', { name: '汇总订单与公平交易分配' })).toBeInTheDocument()
    expect(screen.getByText(/跨组合账户与分配工作区/)).toBeInTheDocument()
    expect(screen.getByText('CASH-PF01-CNY')).toBeInTheDocument()
    expect(screen.getByText('CASH-PF03-CNY')).toBeInTheDocument()
  })

  it('阶段目录使用产品配置与择时和投后研究结论命名', () => {
    window.history.replaceState({}, '', '/pre-investment')
    const first = render(<App />)
    expect(screen.getAllByRole('link', { name: /产品配置与择时/ }).length).toBeGreaterThan(0)
    first.unmount()

    window.history.replaceState({}, '', '/post-investment')
    render(<App />)
    expect(screen.getAllByRole('link', { name: /投后研究结论/ }).length).toBeGreaterThan(0)
  })

  it('研究口径与参数中心属于设置，产品研究不再重复维护', () => {
    window.history.replaceState({}, '', '/product-research')
    const first = render(<App />)
    expect(screen.queryByRole('link', { name: /研究框架与数据基础/ })).not.toBeInTheDocument()
    first.unmount()

    window.history.replaceState({}, '', '/settings')
    render(<App />)
    expect(screen.getAllByRole('link', { name: /研究口径与参数中心/ }).length).toBeGreaterThan(0)
  })

  it('旧研究框架地址重定向到研究口径与参数中心并保留查询参数', async () => {
    window.history.replaceState({}, '', '/product-research/framework-data?template=RP-R4')
    render(<App />)

    expect(await screen.findByRole('heading', { name: '研究口径与参数中心' })).toBeInTheDocument()
    expect(window.location.pathname).toBe('/settings/research-parameters')
    expect(window.location.search).toBe('?template=RP-R4')
  })
})
