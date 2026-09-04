import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ActualPortfolioProvider } from '../app/ActualPortfolioContext'
import AccountingBookingWorkspace from './AccountingBookingWorkspace'
import { evaluateLedgerSummary } from '../services/businessNumeric'

vi.mock('../services/businessNumeric', () => ({ evaluateLedgerSummary: vi.fn() }))

const renderWorkspace = () => render(<ActualPortfolioProvider><AccountingBookingWorkspace /></ActualPortfolioProvider>)

describe('AccountingBookingWorkspace', () => {
  beforeEach(() => {
    vi.mocked(evaluateLedgerSummary).mockImplementation(async (input) => {
      const trialBalanced = input.trial_credit === 100_000
      return {
        metrics: { source_event_count: input.pending_flags.length, voucher_count: 4, balanced_count: 4, pending_count: 1 },
        vouchers: [
          { id: 'JV-FUND-0001', debit: 490_008, credit: 490_008, difference: 0, balanced: true },
          { id: 'JV-FUND-0002', debit: 490_008, credit: 490_008, difference: 0, balanced: true },
          { id: 'JV-FUND-0003', debit: 86_420, credit: 86_420, difference: 0, balanced: true },
          { id: 'JV-MGR-0001', debit: 86_420, credit: 86_420, difference: 0, balanced: true },
        ],
        entities: [
          { entity: '组合/基金账', voucher_count: 3, debit: 1_066_436, credit: 1_066_436, difference: 0 },
          { entity: '管理人公司账', voucher_count: 1, debit: 86_420, credit: 86_420, difference: 0 },
        ],
        trial: { debit: 100_000, credit: input.trial_credit, difference: trialBalanced ? 0 : 200, balanced: trialBalanced },
        execution: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0 },
      }
    })
  })

  it('把业务事实、双主体分类和凭证明细字段放在同一 Booking 功能中', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    expect(screen.getByRole('heading', { name: '组合 Booking 与复式记账' })).toBeInTheDocument()
    expect(screen.getByLabelText('Booking 真实组合')).toBeInTheDocument()
    expect(screen.getByLabelText('Booking 来源账户')).toBeInTheDocument()
    expect(screen.getByLabelText('Booking 基金经理任职关系')).toBeInTheDocument()
    expect(screen.getByLabelText('Booking 资金结算账户')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('管理人主体与主账簿')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('组合或基金账规则预判')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('管理人账规则预判')).toHaveAttribute('readonly')
    expect(screen.getByLabelText('组合绩效现金流规则预判')).toHaveAttribute('readonly')
    expect(screen.getByText('portfolio_book_treatment')).toBeInTheDocument()
    expect(screen.getByText('manager_book_treatment')).toBeInTheDocument()
    expect(screen.getByText('performance_cash_flow_type')).toBeInTheDocument()
    expect(screen.getByText('source_account_id')).toBeInTheDocument()
    expect(screen.getByText('portfolio_manager_assignment_id')).toBeInTheDocument()
    expect(screen.getByText('allocation_id')).toBeInTheDocument()
    expect(screen.getByText('account_code')).toBeInTheDocument()
    expect(screen.getByText('debit_amount')).toBeInTheDocument()
    expect(screen.getByText('credit_amount')).toBeInTheDocument()

    const file = new File(['source_record_id,portfolio_id'], 'booking-demo.xlsx', { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
    await user.upload(screen.getByLabelText('选择 Booking 数据文件'), file)
    expect(screen.getByText('已选择：booking-demo.xlsx（未解析、未上传）')).toBeInTheDocument()
    expect(evaluateLedgerSummary).toHaveBeenCalledWith(expect.objectContaining({ entities: ['组合/基金账', '管理人公司账'] }), expect.any(AbortSignal))
  })

  it('手工 Booking 只保存在当前组件会话', async () => {
    const user = userEvent.setup()
    const first = renderWorkspace()
    await user.type(screen.getByLabelText('产品或事项代码'), '159999.SZ')
    await user.type(screen.getByLabelText('本币金额'), '123400')
    await user.click(screen.getByRole('button', { name: '登记业务事实（演示）' }))
    expect(screen.getByText('159999.SZ')).toBeInTheDocument()
    expect(screen.getByText(/未生成凭证、未复核、未过账/)).toBeInTheDocument()

    first.unmount()
    renderWorkspace()
    expect(screen.queryByText('159999.SZ')).not.toBeInTheDocument()
  })

  it('展示逐主体借贷平衡凭证、试算平衡和两套资产负债等式', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    await user.click(screen.getByRole('button', { name: '复式凭证' }))
    expect(screen.getByText('凭证与分录预览')).toBeInTheDocument()
    await screen.findAllByText(/借贷差额：¥0.00/)
    expect(screen.getByText(/ETF 买入｜交易日确认/)).toBeInTheDocument()
    expect(screen.getByText(/ETF 买入｜交收日付款/)).toBeInTheDocument()
    expect(screen.getAllByText('借方合计 ¥490,008.00')).toHaveLength(2)
    expect(screen.getAllByText('贷方合计 ¥490,008.00')).toHaveLength(2)
    expect(screen.getAllByText(/借贷差额：¥0.00/)).toHaveLength(4)

    await user.click(screen.getByRole('button', { name: '试算平衡' }))
    expect(screen.getByText('资产 = 负债 + 净资产 · 等式差额 0.00')).toBeInTheDocument()
    expect(screen.getByText('资产 = 负债 + 所有者权益 · 等式差额 0.00')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '双账勾稽' }))
    expect(screen.getByText('基金承担管理费')).toBeInTheDocument()
    expect(screen.getByText('管理人确认管理费收入')).toBeInTheDocument()
    expect(screen.getAllByText('本主体借贷平衡')).toHaveLength(2)
  })

  it('借贷不平时硬阻断，修正后只允许进入复核', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    await user.click(screen.getByRole('button', { name: '校验异常' }))
    expect(await screen.findByText(/差额 ¥200.00 · 凭证不平/)).toBeInTheDocument()
    const credit = screen.getByLabelText('校验贷方合计')
    await user.clear(credit)
    await user.type(credit, '100000')
    expect(await screen.findByText(/差额 ¥0.00 · 凭证平衡，可进入会计复核/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /过账|提交凭证/ })).not.toBeInTheDocument()
  })

  it('会计口径和阻断规则明确标记为非交互说明', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    await user.click(screen.getByRole('button', { name: '会计规则' }))
    expect(screen.getByLabelText('会计规则说明（非交互）')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Booking 与凭证分层' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '校验异常' }))
    expect(screen.getByLabelText('阻断规则说明（非交互）')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '凭证硬阻断' })).not.toBeInTheDocument()
  })
})
