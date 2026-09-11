import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

export interface BusinessNumericExecution extends FixedNjitExecutionAudit {
  execution_backend: 'numba_njit_fixed_signature'
  nopython: true
  object_mode: 0
  python_fallback: 0
  request_time_compilation: 0
  [key: string]: unknown
}

export interface NumericControlInput {
  key: string
  values: number[]
  target?: number
  tolerance?: number
}

export interface NumericControlResult {
  key: string
  total: number
  difference: number
  within_tolerance: boolean
  positive: boolean
  normalized_shares: number[]
}

export interface NumericControlsResponse {
  items: NumericControlResult[]
  execution: BusinessNumericExecution
}

export interface TradeAllocationResponse {
  source_quantity: number
  unit_price: number
  source_amount: number
  allocated_total: number
  residual: number
  balanced: boolean
  allocations: Array<{ key: string; quantity: number; amount: number }>
  execution: BusinessNumericExecution
}

export interface LedgerSummaryResponse {
  metrics: {
    source_event_count: number
    voucher_count: number
    balanced_count: number
    pending_count: number
  }
  vouchers: Array<{ id: string; debit: number; credit: number; difference: number; balanced: boolean }>
  entities: Array<{ entity: string; voucher_count: number; debit: number; credit: number; difference: number }>
  trial: { debit: number; credit: number; difference: number; balanced: boolean }
  execution: BusinessNumericExecution
}

async function requestBusinessNumeric<T extends { execution: BusinessNumericExecution }>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/business-numeric/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload?.detail || `业务数值内核返回错误 ${response.status}`)
  }
  assertFixedNjitExecution(payload?.execution, '业务数值内核')
  return payload as T
}

export const evaluateNumericControls = (groups: NumericControlInput[], signal?: AbortSignal) =>
  requestBusinessNumeric<NumericControlsResponse>('controls', { groups }, signal)

export const evaluateTradeAllocation = (input: {
  source_quantity: number
  unit_price: number
  allocations: Array<{ key: string; quantity: number }>
  tolerance?: number
}, signal?: AbortSignal) => requestBusinessNumeric<TradeAllocationResponse>('trade-allocation', input, signal)

export const evaluateLedgerSummary = (input: {
  entities: string[]
  vouchers: Array<{ id: string; entity: string; lines: Array<{ debit: number; credit: number }> }>
  pending_flags: boolean[]
  trial_debit: number
  trial_credit: number
  tolerance?: number
}, signal?: AbortSignal) => requestBusinessNumeric<LedgerSummaryResponse>('ledger-summary', input, signal)
