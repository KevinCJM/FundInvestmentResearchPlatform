import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { HistoricalRegimeRun } from '../services/historicalRegimes'
import {
  eligibleFormalBacktestPublications,
  HistoricalRegimeBacktestSelector,
  RegimeConditioningPanel,
} from './HistoricalRegimeBacktest'

function run(overrides: Partial<HistoricalRegimeRun> = {}): HistoricalRegimeRun {
  const runHash = 'a'.repeat(64)
  return {
    id: 'run-formal', schema_version: '2.0', definition_id: 'definition-1', definition_revision: 3,
    definition_snapshot_hash: 'b'.repeat(64),
    name: '牛熊状态', mode: 'realtime', created_at: '2026-09-04', immutable: true, content_hash: runHash,
    states: [], series: [], segments: [], conditional_stats: [], transition: { states: [], counts: [], probabilities: [] },
    causality: { classification: 'causal', is_causal: true, uses_future_data: false, repaints: false, realtime_eligible: true, publish_eligible_usages: ['formal_backtest'], blockers: [], warnings: [] },
    stability: {}, walk_forward: {}, diagnostics: [], calculation_audits: [],
    governance: { formal_gate_passed: true, publish_eligible_usages: ['formal_backtest'] },
    publications: [{ id: 'publication-formal', usage: 'formal_backtest', published_at: '2026-09-04T10:00:00Z', definition_revision: 3, run_id: 'run-formal', run_content_hash: runHash, gate: 'comprehensive_formal_gate_passed' }],
    ...overrides,
  }
}

describe('HistoricalRegimeBacktest', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('只暴露严格匹配的 realtime、因果、formal_backtest v2 发布', () => {
    const items = eligibleFormalBacktestPublications([
      run(),
      run({ id: 'retrospective', mode: 'retrospective' }),
      run({ causality: { ...run().causality, is_causal: false } }),
      run({ causality: { ...run().causality, uses_future_data: true } }),
      run({ causality: { ...run().causality, repaints: true } }),
      run({ schema_version: '1.0', publications: [{ ...run().publications[0], gate: 'causality_passed' }] }),
      run({ publications: [{ ...run().publications[0], gate: 'causality_passed' }] }),
      run({ definition_snapshot_hash: undefined }),
      run({ id: 'wrong-usage', publications: [{ ...run().publications[0], run_id: 'wrong-usage', usage: 'taa' }] }),
      run({ id: 'wrong-lineage', publications: [{ ...run().publications[0], run_id: 'wrong-lineage', run_content_hash: 'c'.repeat(64) }] }),
    ])
    expect(items.map((item) => item.run.id)).toEqual(['run-formal'])
  })

  it('选择时返回精确运行和发布 ID', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ items: [run()] }) }))
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<HistoricalRegimeBacktestSelector value={null} onChange={onChange} />)

    const select = await screen.findByLabelText('历史情景条件化（可选）')
    await user.selectOptions(select, 'run-formal|publication-formal')
    expect(onChange).toHaveBeenCalledWith({ run_id: 'run-formal', publication_id: 'publication-formal' })
  })

  it('展示锁定血缘、覆盖率、条件表现，并默认折叠逐期明细', async () => {
    const user = userEvent.setup()
    render(<RegimeConditioningPanel result={{
      binding: { run_id: 'run-formal', publication_id: 'publication-formal', definition_id: 'definition-1', definition_revision: 3 },
      coverage: { periods: 100, classified_periods: 80, classified_ratio: .8 },
      conditional_performance: { 平衡组合: [{ state_id: 'bull', state_label: '牛市', observations: 42, return_observations: 41, annualized_return: .12, volatility: .18, max_drawdown: -.08, sharpe: .8, positive_rate: .59 }] },
      period_states: [{ date: '2026-09-04', period_start: '2026-09-03', state_id: 'bull', state_label: '牛市', regime_effective_date: '2026-09-02', confidence: .9 }],
    }} />)

    expect(screen.getByText('run-formal')).toBeInTheDocument()
    expect(screen.getByText('80.00% · 80/100 期')).toBeInTheDocument()
    expect(screen.getByText('12.00%')).toBeInTheDocument()
    const details = screen.getByText('逐期状态明细（1 期）').closest('details')
    expect(details).not.toHaveAttribute('open')
    await user.click(screen.getByText('逐期状态明细（1 期）'))
    await waitFor(() => expect(details).toHaveAttribute('open'))
    expect(screen.getByText('2026-09-02')).toBeInTheDocument()
  })
})
