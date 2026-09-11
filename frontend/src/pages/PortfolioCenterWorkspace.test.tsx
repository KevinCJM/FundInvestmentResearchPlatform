import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ActualPortfolioProvider } from '../app/ActualPortfolioContext'
import PortfolioCenterWorkspace, { type PortfolioCenterView } from './PortfolioCenterWorkspace'

const renderWorkspace = (view: PortfolioCenterView) => render(<ActualPortfolioProvider><PortfolioCenterWorkspace view={view} /></ActualPortfolioProvider>)

describe('PortfolioCenterWorkspace', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('登记簿可筛选真实组合且不请求后端', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const user = userEvent.setup()
    renderWorkspace('register')

    expect(screen.getByRole('heading', { name: '真实组合登记簿' })).toBeInTheDocument()
    expect(screen.getByText('PF-DEMO-01')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('筛选组合状态'), '暂停')
    expect(screen.getByText('均衡配置二号')).toBeInTheDocument()
    expect(screen.queryByText('稳健多资产一号')).not.toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each<[PortfolioCenterView, string]>([
    ['master', '组合主数据'],
    ['relationships', '账户主档与组合关系'],
    ['responsibilities', '基金经理与投资单元'],
    ['versions', '研究方案与目标版本'],
    ['lifecycle', '组合生命周期与状态'],
  ])('%s 视图展示对应工作区', (view, heading) => {
    renderWorkspace(view)
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument()
    expect(screen.getByLabelText('选择真实组合')).toBeInTheDocument()
    expect(screen.getByLabelText('真实组合边界说明（非交互）')).toBeInTheDocument()
  })

  it('账户关系随真实组合切换', async () => {
    const user = userEvent.setup()
    renderWorkspace('relationships')

    await user.selectOptions(screen.getByLabelText('选择真实组合'), 'PF-DEMO-02')
    expect(screen.getByText('LEDGER-PF-DEMO-02')).toBeInTheDocument()
    expect(screen.getAllByText('EXEC-FM-DEMO-02-SZSE')).not.toHaveLength(0)
    expect(screen.queryByText('LEDGER-PF-DEMO-01')).not.toBeInTheDocument()
  })

  it('共享执行账户可关联多个组合但资金账户保持核算主体专用', () => {
    renderWorkspace('relationships')

    expect(screen.getByText('稳健多资产一号、固收增强三号')).toBeInTheDocument()
    expect(screen.getAllByText('允许多组合共享执行').length).toBeGreaterThan(0)
    expect(screen.getAllByText('单一核算主体专用').length).toBeGreaterThan(0)
  })
})
