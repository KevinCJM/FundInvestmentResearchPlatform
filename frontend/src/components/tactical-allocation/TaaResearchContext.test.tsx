import { act, render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import { i18n } from '../../i18n/runtime'
import { taaBaseline, taaRequest, taaPreflight, taaPreview } from '../../test/tacticalAllocationFixtures'
import type { TaaAlignment } from '../../services/tacticalAllocation'
import TaaResearchContext from './TaaResearchContext'

const alignment: TaaAlignment = {
  method: 'common_observation_intervals/1.0.0', common_observations: 298, return_periods: 297,
  non_common_dates: 3, multi_observation_periods: 2, max_calendar_days: 5, day_count: 'ACT/365.25', calendar_verified: false,
  sources: [{ series_id: 'index:SPX', name: 'SPX 用户原名', observations: 298, not_observed_dates: 3,
    start_date: '2025-01-01', end_date: '2026-01-01', date_examples: ['2025-07-04'] }],
}
const props = { baseline: taaBaseline, request: taaRequest, preview: null, preflight: { ...taaPreflight, alignment },
  checking: false, error: '', onChange: vi.fn(), onEdit: vi.fn(), onRetry: vi.fn() }
afterEach(async () => { await act(() => i18n.changeLanguage('zh-CN')) })

it('explains retained interval returns, exposes source dates and switches language without renaming sources', async () => {
  render(<MemoryRouter><TaaResearchContext {...props} /></MemoryRouter>)
  expect(screen.getByText(/298 个价格观察日、297 个收益区间/)).toBeVisible()
  fireEvent.click(screen.getByText('查看日期对齐明细', { selector: 'summary' }))
  expect(screen.getByText('SPX 用户原名')).toBeVisible()
  expect(screen.getByText('2025-07-04')).toBeVisible()
  expect(screen.getByText(/未用各市场完整日历核验原因/)).toBeVisible()
  await act(() => i18n.changeLanguage('en-US'))
  expect(screen.getByText(/298 common price dates/)).toBeVisible()
  expect(screen.getByText(/Annual risk uses actual interval lengths/)).toBeVisible()
  expect(screen.getByText('SPX 用户原名')).toBeVisible()
})

it('does not assign a new convention to saved old results or show stale alignment during rechecks', () => {
  const view = render(<MemoryRouter><TaaResearchContext {...props} preview={taaPreview} /></MemoryRouter>)
  expect(screen.queryByText(/已按共同日期对齐/)).not.toBeInTheDocument()
  view.rerender(<MemoryRouter><TaaResearchContext {...props} checking /></MemoryRouter>)
  expect(screen.queryByText(/已按共同日期对齐/)).not.toBeInTheDocument()
  view.rerender(<MemoryRouter><TaaResearchContext {...props} error="读取失败" /></MemoryRouter>)
  expect(screen.queryByText(/已按共同日期对齐/)).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('重试条件检查'))
  expect(props.onRetry).toHaveBeenCalledOnce()
})
