import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import PortfolioSolutionCatalog from './PortfolioSolutionCatalog'
import PortfolioSolutionShowcase from './PortfolioSolutionShowcase'

describe('组合方案中心静态页面', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('方案目录可按风险等级筛选且不请求后端', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const user = userEvent.setup()
    render(<MemoryRouter><PortfolioSolutionCatalog /></MemoryRouter>)

    await user.selectOptions(screen.getByLabelText('风险等级'), '中风险')
    expect(screen.getByRole('heading', { name: '均衡增长方案' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '稳健多资产方案' })).not.toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('组合画像明确区分历史回测和实盘跟踪', () => {
    render(<MemoryRouter><PortfolioSolutionShowcase view="profile" /></MemoryRouter>)

    expect(screen.getByRole('heading', { name: '组合画像与适用范围' })).toBeInTheDocument()
    expect(screen.getByText(/虚线左侧为历史回测，右侧为实盘跟踪示例/)).toBeInTheDocument()
    expect(screen.getByText(/不构成投资建议或收益承诺/)).toBeInTheDocument()
    expect(screen.getByLabelText('组合适用性说明（非交互）')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '适用目标' })).not.toBeInTheDocument()
  })

  it('展示版本只生成当前会话预览，不发布或保存', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter><PortfolioSolutionShowcase view="publishing" /></MemoryRouter>)

    await user.click(screen.getByRole('button', { name: '生成静态预览' }))
    expect(screen.getByText('已生成当前会话内的静态预览；未发布、未保存。')).toBeInTheDocument()
  })
})
