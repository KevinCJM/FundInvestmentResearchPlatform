import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ResearchContextProvider, useResearchContext, useResearchDay } from './ResearchContext'
import PitBadge from '../components/PitBadge'
import PitProvenance from '../components/PitProvenance'
import PitDecisionNotice from '../components/PitDecisionNotice'
import { pageReload, setPitOverride } from '../services/pitOverride'
import type { PitSettingsPayload } from '../services/pit'

const strictSettings: PitSettingsPayload = {
  settings: { active_release_id: null, as_of: '2026-09-01', run_mode: 'STRICT_PIT', updated_at: null, note: '' },
  effective: { as_of: '2026-09-01', as_of_source: 'explicit', run_mode: 'STRICT_PIT', run_mode_label: '严格 PIT', data_release_id: null, no_pit: false, label: '站在 2026-09-01 · 严格 PIT' },
  release: null, release_error: null, available_releases: [], can_apply: true,
}

function Probe() {
  const context = useResearchContext()
  const day = useResearchDay()
  return <><output data-testid="context">{JSON.stringify({ noPit: context.noPit, runMode: context.runMode, day: day === undefined ? 'unknown' : day, label: context.label })}</output><button onClick={context.refresh}>刷新系统设置</button><PitBadge /></>
}

function renderContext() {
  return render(<MemoryRouter><ResearchContextProvider><Probe />
    <PitProvenance lineage={{ as_of: '2026-08-31', as_of_applied: true, run_mode: 'STRICT_PIT', availability_field: 'ann_date', rows_before_cut: 10, rows_after_cut: 8, rows_dropped_by_as_of: 2, rows_without_announcement: 0, announcement_fallback: false, warnings: [] }} />
    <PitDecisionNotice lineage={{ alloc_name: '测试配置', as_of: '2026-08-31', run_mode: 'STRICT_PIT', series_as_of: '2026-08-31', series_variants: [], hindsight_series: false, rows_dropped_by_as_of: 2, availability_available: true, warnings: [] }} />
  </ResearchContextProvider></MemoryRouter>)
}

const readContext = () => JSON.parse(screen.getByTestId('context').textContent!)

describe('ResearchContext read failures', () => {
  beforeEach(() => {
    setPitOverride(null)
    vi.spyOn(pageReload, 'run').mockImplementation(() => {})
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => strictSettings }))
  })
  afterEach(() => { setPitOverride(null); vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('首次读取失败为未知，并可重试恢复服务端严格 PIT；结果脚注不受影响', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('连接中断'))
    renderContext()
    const badge = await screen.findByTestId('pit-badge')
    expect(badge).toHaveTextContent('PIT 口径未知')
    expect(readContext()).toMatchObject({ noPit: null, runMode: null, day: 'unknown' })
    expect(screen.getByTestId('pit-provenance')).toHaveTextContent('严格 PIT')
    expect(screen.getByTestId('pit-decision-notice')).toHaveTextContent('研究日 2026-08-31')
    await userEvent.click(badge)
    expect(screen.getByRole('alert')).toHaveTextContent('连接中断')
    expect(screen.getByTestId('pit-switcher')).toHaveTextContent('系统默认：未知（读取失败）')
    await userEvent.click(screen.getByRole('button', { name: '重试读取 PIT 口径' }))
    await waitFor(() => expect(badge).toHaveTextContent('PIT 打开：2026-09-01'))
    expect(readContext()).toMatchObject({ noPit: false, runMode: 'STRICT_PIT', day: '2026-09-01' })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('已经确认严格 PIT 后再次读取失败也不能冒充关闭或继续冒充当前严格模式', async () => {
    renderContext()
    await waitFor(() => expect(readContext().runMode).toBe('STRICT_PIT'))
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({ detail: '系统设置暂不可读' }) } as Response)
    await userEvent.click(screen.getByRole('button', { name: '刷新系统设置' }))
    await waitFor(() => expect(screen.getByTestId('pit-badge')).toHaveTextContent('PIT 口径未知'))
    expect(readContext()).toMatchObject({ noPit: null, runMode: null, day: 'unknown' })
    expect(screen.getByTestId('pit-badge')).not.toHaveTextContent('PIT 关闭')
    expect(screen.getByTestId('pit-provenance')).toHaveTextContent('严格 PIT')
  })

  it('成功状态码但响应缺少口径时仍为未知并提供重试', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
    renderContext()
    const badge = await screen.findByTestId('pit-badge')
    expect(badge).toHaveTextContent('PIT 口径未知')
    await act(async () => { await userEvent.click(badge) })
    expect(screen.getByRole('alert')).toHaveTextContent('响应不完整')
    expect(screen.getByRole('button', { name: '重试读取 PIT 口径' })).toBeEnabled()
  })

  it('明确的本页 off 与未知系统设置分开展示，失败不清除用户选择', async () => {
    setPitOverride({ off: true, releaseId: null, runMode: 'RESEARCH' })
    vi.mocked(fetch).mockRejectedValue(new Error('服务不可用'))
    renderContext()
    await userEvent.click(await screen.findByTestId('pit-badge'))
    await screen.findByRole('alert')
    expect(screen.getByTestId('pit-badge')).toHaveTextContent('PIT 关闭')
    expect(screen.getByTestId('pit-switcher')).toHaveTextContent('系统默认：未知（读取失败）')
    expect(readContext()).toMatchObject({ noPit: true, runMode: 'RESEARCH', day: null })
    expect(readContext().label).toContain('临时口径')
  })

  it('没有运行模式的裸日期继承已确认的系统严格模式', async () => {
    setPitOverride({ off: false, releaseId: null, asOf: '2026-08-01', runMode: null })
    renderContext()
    await waitFor(() => expect(readContext().runMode).toBe('STRICT_PIT'))
    expect(readContext()).toMatchObject({ noPit: false, day: '2026-08-01' })
  })

  it('裸日期的继承模式未读取到时不编造 RESEARCH', async () => {
    setPitOverride({ off: false, releaseId: null, asOf: '2026-08-01', runMode: null })
    vi.mocked(fetch).mockRejectedValue(new Error('连接中断'))
    renderContext()
    await act(async () => {})
    expect(readContext()).toMatchObject({ noPit: false, runMode: null, day: 'unknown' })
    expect(screen.getByTestId('pit-badge')).toHaveTextContent('PIT 口径未知')
  })

  it('服务端明确关闭 PIT 时可正常显示关闭及无截止日', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => ({ ...strictSettings, effective: { ...strictSettings.effective, no_pit: true, as_of: null, run_mode: 'RESEARCH', label: '无 PIT 口径 · 使用全部磁盘数据' } }) } as Response)
    renderContext()
    expect(await screen.findByTestId('pit-badge')).toHaveTextContent('PIT 关闭')
    expect(readContext()).toMatchObject({ noPit: true, runMode: 'RESEARCH', day: null })
  })
})
