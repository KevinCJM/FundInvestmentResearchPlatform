import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import PortfolioOnboardingWorkspace from './PortfolioOnboardingWorkspace'

describe('PortfolioOnboardingWorkspace', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('研究方案落地只生成当前会话主数据预览', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const user = userEvent.setup()
    render(<PortfolioOnboardingWorkspace />)

    expect(screen.getByRole('heading', { name: '组合落地与启用' })).toBeInTheDocument()
    expect(screen.getByLabelText('组合落地边界说明（非交互）')).toHaveTextContent('不完成基金法律设立')
    await user.type(screen.getByLabelText('真实组合名称'), '稳健配置测试组合')
    await user.type(screen.getByLabelText('外部产品或委托代码'), 'EXT-DEMO-100')
    await user.type(screen.getByLabelText('基金专用资金账户'), 'CASH-EXT-DEMO-100')
    await user.click(screen.getByRole('button', { name: '生成落地登记预览（演示）' }))

    expect(screen.getByText('稳健配置测试组合 · PF-DEMO-NEW-001')).toBeInTheDocument()
    expect(screen.getByText(/未保存、未依法设立、未开户/)).toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: /下单|委托提交|撤单/ })).not.toBeInTheDocument()
  })

  it('存量组合调整沿用原 portfolio_id', async () => {
    const user = userEvent.setup()
    render(<PortfolioOnboardingWorkspace />)

    await user.click(screen.getByRole('button', { name: /存量组合调整/ }))
    await user.selectOptions(screen.getByLabelText('存量真实组合'), 'PF-DEMO-02')
    await user.clear(screen.getByLabelText('新目标组合版本'))
    await user.type(screen.getByLabelText('新目标组合版本'), 'TARGET-R12.0')
    await user.type(screen.getByLabelText('调整原因'), '记录平台外已批准的目标版本调整')
    await user.click(screen.getByRole('button', { name: '生成存量组合调整预览（演示）' }))

    expect(screen.getByText(/沿用原 portfolio_id PF-DEMO-02/)).toBeInTheDocument()
    expect(screen.getByText(/未修改真实持仓、账套或交易/)).toBeInTheDocument()
    expect(screen.getByLabelText('存量组合不可变身份')).toHaveTextContent('LEDGER-PF-DEMO-02')
  })
})
