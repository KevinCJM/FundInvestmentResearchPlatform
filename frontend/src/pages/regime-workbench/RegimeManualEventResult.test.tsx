import { render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { adaptRegimeResult } from './regimeResultAdapter'
import { resultFixture } from './regimeResultFixtures'
import RegimeManualEventResult from './RegimeManualEventResult'

vi.mock('echarts-for-react', () => ({ default: ({ 'aria-label': label }: { 'aria-label'?: string }) => <div data-testid="mock-chart">{label || 'chart'}</div> }))

it('人工事件结果按多标签语义展示事件轨道、重叠统计和事件明细', () => {
  const { overview, rows } = resultFixture('manual-result', 10)
  overview.result_kind = 'manual_events'
  overview.manual_events = [
    { id: 'subprime', label: '次贷危机', start_date: rows[2].observation_date, end_date: rows[7].observation_date, color: '#dc2626', description: '金融危机阶段', covered_observations: 6, first_observation_index: 2, last_observation_index: 7, first_observation_date: rows[2].observation_date, last_observation_date: rows[7].observation_date },
    { id: 'liquidity', label: '流动性冲击', start_date: rows[4].observation_date, end_date: rows[5].observation_date, color: '#7c3aed', covered_observations: 2, first_observation_index: 4, last_observation_index: 5, first_observation_date: rows[4].observation_date, last_observation_date: rows[5].observation_date },
  ]
  overview.manual_event_summary = { event_count: 2, covered_observations: 6, overlap_observations: 2, max_concurrent_events: 2 }
  const result = adaptRegimeResult(overview, rows)
  render(<RegimeManualEventResult result={result} />)

  expect(screen.getByRole('heading', { name: '人工历史事件结果' })).toBeVisible()
  expect(screen.getByText('重叠观测').parentElement).toHaveTextContent('2')
  expect(screen.getByText('最大同时事件').parentElement).toHaveTextContent('2')
  expect(screen.getByText('次贷危机')).toBeVisible()
  expect(screen.getByText('流动性冲击')).toBeVisible()
  expect(screen.getByRole('table', { name: '人工历史事件明细' })).toContainElement(screen.getByText('金融危机阶段'))
  expect(screen.getByText('允许重叠')).toBeVisible()
  expect(screen.queryByText('状态切换')).not.toBeInTheDocument()
})
