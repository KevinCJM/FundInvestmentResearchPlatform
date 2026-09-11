import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TradeAllocationWorkspace from './TradeAllocationWorkspace'
import { evaluateTradeAllocation } from '../services/businessNumeric'

vi.mock('../services/businessNumeric', () => ({ evaluateTradeAllocation: vi.fn() }))

describe('TradeAllocationWorkspace', () => {
  beforeEach(() => {
    vi.mocked(evaluateTradeAllocation).mockImplementation(async (input) => {
      const balanced = input.allocations[0]?.quantity === 600_000
      return {
        source_quantity: 1_000_000,
        unit_price: 4.2,
        source_amount: 4_200_000,
        allocated_total: balanced ? 1_000_000 : 900_000,
        residual: balanced ? 0 : 100_000,
        balanced,
        allocations: [
          { key: 'PF-DEMO-01', quantity: input.allocations[0]?.quantity ?? 0, amount: balanced ? 2_520_000 : 2_100_000 },
          { key: 'PF-DEMO-03', quantity: 400_000, amount: 1_680_000 },
        ],
        execution: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0 },
      }
    })
  })

  it('把共享执行通道成交拆到组合、经理任职和专用结算账户', async () => {
    const user = userEvent.setup()
    render(<TradeAllocationWorkspace />)

    expect(screen.getByRole('heading', { name: '汇总订单与公平交易分配' })).toBeInTheDocument()
    expect(screen.getByText('EXEC-FM-DEMO-SSE')).toBeInTheDocument()
    expect(screen.getByText('CASH-PF01-CNY')).toBeInTheDocument()
    expect(screen.getByText('CASH-PF03-CNY')).toBeInTheDocument()
    expect(screen.getByText(/周衡 · 联席基金经理/)).toBeInTheDocument()
    expect(screen.getByText(/林岚 · 主基金经理/)).toBeInTheDocument()
    expect(await screen.findByText(/尾差 0/)).toBeInTheDocument()

    const firstAllocation = screen.getByLabelText('稳健多资产一号成交分配数量')
    await user.clear(firstAllocation)
    await user.type(firstAllocation, '500000')
    expect(await screen.findByText(/硬阻断：不得形成组合 Booking/)).toBeInTheDocument()
    expect(evaluateTradeAllocation).toHaveBeenCalledWith(expect.objectContaining({ source_quantity: 1_000_000, unit_price: 4.2 }), expect.any(AbortSignal))
  })
})
