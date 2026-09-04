import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ActualPortfolioProvider } from '../app/ActualPortfolioContext'
import FinancialStatementsWorkspace from './FinancialStatementsWorkspace'

describe('FinancialStatementsWorkspace', () => {
  it('分别展示组合基金报表和管理人公司报表', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const user = userEvent.setup()
    render(<ActualPortfolioProvider><FinancialStatementsWorkspace /></ActualPortfolioProvider>)

    expect(screen.getByRole('button', { name: '资产负债表' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '利润表' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '净资产变动表' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '附注与勾稽' }))
    expect(screen.getByText(/资产 12,648 = 负债 118 \+ 净资产 12,530/)).toBeInTheDocument()
    expect(screen.getByText(/每个基金或组合分别出表/)).toBeInTheDocument()
    expect(screen.getByLabelText('报表范围说明（非交互）')).toBeInTheDocument()
    expect(screen.getByLabelText('报表生成链路（非交互）')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Booking 事件' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /管理人公司账/ }))
    expect(screen.getByRole('button', { name: '现金流量表' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '所有者权益变动表' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '附注与勾稽' }))
    expect(screen.getByText(/受托管理的基金财产不作为管理人固有资产/)).toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})
