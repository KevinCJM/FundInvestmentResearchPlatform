import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AccountStatementAllocationWorkspace from './AccountStatementAllocationWorkspace'
import { evaluateNumericControls } from '../services/businessNumeric'

vi.mock('../services/businessNumeric', () => ({ evaluateNumericControls: vi.fn() }))

describe('AccountStatementAllocationWorkspace', () => {
  beforeEach(() => {
    vi.mocked(evaluateNumericControls).mockResolvedValue({
      items: [{ key: 'statement-allocation', total: 6_720_000, difference: 0, within_tolerance: true, positive: true, normalized_shares: [0.6, 0.4] }],
      execution: {
        execution_backend: 'numba_njit_fixed_signature',
        nopython: true,
        object_mode: 0,
        python_fallback: 0,
        request_time_compilation: 0,
        kernel_signatures: { numeric_control_kernel: ['fixed'] },
      },
    })
  })

  it('区分共享执行回单与基金专用资金流水', async () => {
    const user = userEvent.setup()
    render(<AccountStatementAllocationWorkspace />)

    expect(screen.getByRole('heading', { name: '外部账户流水、拆分与结算匹配' })).toBeInTheDocument()
    expect(screen.getByText('EXEC-FM-DEMO-CIBM · 银行间成交结算通知')).toBeInTheDocument()
    expect(screen.getAllByText('CASH-PF01-CNY')).not.toHaveLength(0)
    expect(screen.getAllByText('CASH-PF03-CNY')).not.toHaveLength(0)
    expect(screen.getByText(/周衡 · 联席基金经理/)).toBeInTheDocument()
    expect(screen.getByText(/林岚 · 主基金经理/)).toBeInTheDocument()
    expect(await screen.findByText('分配尾差 ¥0.00')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('筛选账户流水状态'), '待认领')
    expect(screen.getByText('银行手续费待取得回单')).toBeInTheDocument()
    expect(screen.queryByText('BANK-PF03-20260902-041')).not.toBeInTheDocument()
    expect(evaluateNumericControls).toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ key: 'statement-allocation', values: [4_032_000, 2_688_000] })]), expect.any(AbortSignal))
  })
})
