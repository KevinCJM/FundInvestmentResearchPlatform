import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PitSnapshots from './PitSnapshots'
import PitBadge from '../components/PitBadge'
import PitProvenance from '../components/PitProvenance'
import { ResearchContextProvider } from '../app/ResearchContext'
import { getPitOverride, pageReload, setPitOverride } from '../services/pitOverride'

const NAV_DATASET = {
  dataset_id: 'etf_nav',
  label: 'ETF / 场内基金净值',
  file: 'etf_daily_df.parquet',
  event_field: 'nav_date',
  availability_field: 'ann_date',
  declared_lag_days: 1,
  revisable: false,
  note: 'Tushare 提供公告日。',
  present: true,
  rows: 1547292,
  availability_coverage: 0.9998,
  event_range: { start: '2015-01-05', end: '2026-09-03' },
  availability_range: { start: '2015-01-06', end: '2026-09-04' },
  lag: {
    p50: 1,
    p95: 2,
    max: 25,
    negative_rows: 0,
    histogram: [
      { bucket: '0', rows: 31451 },
      { bucket: '1', rows: 1438179 },
      { bucket: '2', rows: 47391 },
      { bucket: '3-5', rows: 8361 },
      { bucket: '6-10', rows: 0 },
      { bucket: '11-20', rows: 2058 },
      { bucket: '21+', rows: 12026 },
    ],
  },
  grade: 'A' as const,
  grade_label: 'A · 严格 PIT',
  fingerprint: 'abc',
  available_through: '2026-09-04',
}

const INFO_DATASET = {
  ...NAV_DATASET,
  dataset_id: 'etf_info',
  label: 'ETF 合同与分类信息',
  file: 'etf_info_df.parquet',
  event_field: null,
  availability_field: null,
  revisable: true,
  rows: 1760,
  availability_coverage: null,
  lag: { p50: null, p95: null, max: null, negative_rows: 0, histogram: [] },
  grade: 'C' as const,
  grade_label: 'C · 无 PIT',
  available_through: null,
  note: '维表按最新状态整体覆盖写入。',
}

const SUMMARY = {
  declared: 2,
  present: 2,
  missing: 0,
  grade_a: 1,
  grade_b: 0,
  grade_c: 1,
  total_rows: 1549052,
  available_through: '2026-09-04',
  latest_dataset_end: '2026-09-04',
  available_through_basis: 'grade_a_b',
}

const RELEASE_SUMMARY = {
  id: 'release-abc123def456',
  name: '2026Q3 基线',
  note: '',
  created_at: '2026-09-07T02:00:00.000000+00:00',
  sequence: 1,
  available_through: '2026-09-04',
  grade_a: 1,
  grade_b: 0,
  grade_c: 1,
  table_count: 2,
  total_rows: 1549052,
  release_fingerprint: 'f'.repeat(64),
}

const NO_PIT_SETTINGS = {
  settings: { active_release_id: null, as_of: null, run_mode: 'RESEARCH' as const, updated_at: null, note: '' },
  effective: {
    as_of: null,
    as_of_source: null,
    run_mode: 'RESEARCH' as const,
    run_mode_label: '研究模式',
    data_release_id: null,
    no_pit: true,
    label: '无 PIT 口径 · 使用全部磁盘数据',
  },
  release: null,
  release_error: null,
  available_releases: [] as (typeof RELEASE_SUMMARY)[],
  can_apply: false,
}

let releases: unknown[] = []
let settings: any = NO_PIT_SETTINGS
let sealedBody: any = null
let appliedBody: any = null

function appliedSettings(runMode: 'RESEARCH' | 'STRICT_PIT') {
  return {
    settings: { active_release_id: RELEASE_SUMMARY.id, as_of: null, run_mode: runMode, updated_at: '2026-09-07T03:00:00+00:00', note: '' },
    effective: {
      as_of: '2026-09-04',
      as_of_source: 'release' as const,
      run_mode: runMode,
      run_mode_label: runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式',
      data_release_id: RELEASE_SUMMARY.id,
      no_pit: false,
      label: `站在 2026-09-04 · 2026Q3 基线 · ${runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}`,
    },
    release: RELEASE_SUMMARY,
    release_error: null,
    available_releases: [RELEASE_SUMMARY],
    can_apply: true,
  }
}

function stubFetch() {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? 'GET'
    if (url.startsWith('/api/pit/universe')) {
      const asOf = new URL(url, 'http://x').searchParams.get('as_of')
      return new Response(
        JSON.stringify({
          kind: 'fund',
          kind_label: '基金产品域',
          as_of: asOf,
          run_mode: 'RESEARCH',
          coverage: asOf ? 'INTERVAL' : 'LATEST_ONLY',
          replayable: Boolean(asOf),
          history_begins_at: null,
          member_count: asOf ? 224 : 1792,
          latest_member_count: 1792,
          excluded_by_replay: asOf ? 1568 : 0,
          sample: [{ code: '510010.SH', name: '交银上证180公司治理ETF' }],
          warnings: [],
        }),
        { status: 200 },
      )
    }
    if (url.startsWith('/api/pit/audit')) {
      return new Response(
        JSON.stringify({ datasets: [NAV_DATASET, INFO_DATASET], summary: SUMMARY, latest_release: null }),
        { status: 200 },
      )
    }
    if (url === '/api/pit/releases' && method === 'GET') {
      return new Response(JSON.stringify({ releases }), { status: 200 })
    }
    if (url === '/api/pit/releases' && method === 'POST') {
      sealedBody = JSON.parse(String(init?.body))
      const release = {
        ...RELEASE_SUMMARY,
        name: sealedBody.name,
        note: sealedBody.note,
        parent_release_id: null,
        immutable: true,
        tables: [{ dataset_id: 'etf_nav' }, { dataset_id: 'etf_info' }],
        summary: SUMMARY,
      }
      releases = [release]
      settings = { ...settings, available_releases: [RELEASE_SUMMARY], can_apply: true }
      return new Response(JSON.stringify(release), { status: 200 })
    }
    if (url === '/api/pit/settings' && method === 'GET') {
      return new Response(JSON.stringify(settings), { status: 200 })
    }
    if (url === '/api/pit/settings' && method === 'PUT') {
      appliedBody = JSON.parse(String(init?.body))
      if (!appliedBody.activeReleaseId && appliedBody.runMode === 'STRICT_PIT') {
        return new Response(
          JSON.stringify({ detail: '未应用数据版本时无法启用严格 PIT，请先封版并选择一个版本。' }),
          { status: 400 },
        )
      }
      settings = appliedBody.activeReleaseId
        ? appliedSettings(appliedBody.runMode)
        : { ...NO_PIT_SETTINGS, available_releases: [RELEASE_SUMMARY], can_apply: true }
      return new Response(JSON.stringify(settings), { status: 200 })
    }
    return new Response(JSON.stringify({ detail: `unexpected ${method} ${url}` }), { status: 500 })
  })
}

function renderPage(node = <PitSnapshots />) {
  return render(
    <MemoryRouter>
      <ResearchContextProvider>{node}</ResearchContextProvider>
    </MemoryRouter>,
  )
}

// The provider reloads on purpose so a page can never mix two口径 at once; jsdom
// has no navigation, so the seam is replaced instead.
const reload = vi.fn()
const nativeReload = pageReload.run

beforeEach(() => {
  releases = []
  settings = NO_PIT_SETTINGS
  sealedBody = null
  appliedBody = null
  reload.mockClear()
  setPitOverride(null)
  vi.stubGlobal('fetch', stubFetch())
  pageReload.run = reload
})

afterEach(() => {
  setPitOverride(null)
  pageReload.run = nativeReload
  vi.unstubAllGlobals()
})

describe('PIT 能力体检', () => {
  it('renders the measured capability sheet rather than a prototype placeholder', async () => {
    renderPage()
    expect(await screen.findByText('ETF / 场内基金净值')).toBeInTheDocument()

    const row = screen.getByText('ETF / 场内基金净值').closest('tr') as HTMLElement
    expect(within(row).getByText('ann_date')).toBeInTheDocument()
    expect(within(row).getByText('99.98%')).toBeInTheDocument()
    expect(within(row).getByText('1,547,292')).toBeInTheDocument()

    // A dataset with no availability column has to say so, not show a blank.
    const infoRow = screen.getByText('ETF 合同与分类信息').closest('tr') as HTMLElement
    expect(within(infoRow).getByText('无')).toBeInTheDocument()

    expect(screen.getByText('A/B 数据可得至').parentElement).toHaveTextContent('2026-09-04')
  })

  it('shows the lag distribution for the selected dataset and swaps on click', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('ETF / 场内基金净值')

    expect(await screen.findByText(/公告滞后分布/)).toBeInTheDocument()
    expect(screen.getByText('1,438,179')).toBeInTheDocument()

    await user.click(screen.getByText('ETF 合同与分类信息'))
    expect(await screen.findByText(/该数据集没有可得时间列，滞后分布无从计算/)).toBeInTheDocument()
  })
})

describe('系统级 PIT 口径', () => {
  it('lets a research day be set on its own, with nothing sealed', async () => {
    const user = userEvent.setup()
    renderPage()
    const panel = await screen.findByTestId('pit-apply')
    expect(panel).toHaveTextContent('无 PIT 口径 · 使用全部磁盘数据')
    // The whole point of the redesign: "stand on 2009-12-31" needs no release.
    // The old page offered only a release picker, so this was unreachable.
    expect(screen.getByLabelText(/站在哪一天看/)).not.toBeDisabled()
    expect(screen.getByRole('button', { name: /严格 PIT/ })).toBeDisabled()

    await user.type(screen.getByLabelText(/站在哪一天看/), '2009-12-31')
    await waitFor(() => expect(screen.getByRole('button', { name: /严格 PIT/ })).not.toBeDisabled())
    await user.click(screen.getByRole('button', { name: '应用到全平台' }))

    await waitFor(() => expect(appliedBody).not.toBeNull())
    expect(appliedBody).toEqual({
      activeReleaseId: null,
      asOf: '2009-12-31',
      runMode: 'RESEARCH',
      note: '',
    })
  })

  it('shows what standing on that day costs before it is applied', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByTestId('pit-apply')

    await user.type(screen.getByLabelText(/站在哪一天看/), '2009-12-31')

    const panel = screen.getByTestId('pit-apply')
    await waitFor(() => expect(panel).toHaveTextContent('站在 2009-12-31'))
    expect(panel).toHaveTextContent('届时可选基金')
  })

  it('applies a release platform-wide and reports the new 口径', async () => {
    const user = userEvent.setup()
    settings = { ...NO_PIT_SETTINGS, available_releases: [RELEASE_SUMMARY], can_apply: true }
    releases = [{ ...RELEASE_SUMMARY, parent_release_id: null, immutable: true, tables: [], summary: SUMMARY }]
    renderPage()

    const select = await screen.findByLabelText(/用哪一批数据/)
    await waitFor(() => expect(select).not.toBeDisabled())
    await user.selectOptions(select, RELEASE_SUMMARY.id)
    await user.type(screen.getByLabelText(/站在哪一天看/), '2026-09-04')
    await user.click(screen.getByRole('button', { name: /严格 PIT/ }))
    await user.click(screen.getByRole('button', { name: '应用到全平台' }))

    await waitFor(() => expect(appliedBody).not.toBeNull())
    expect(appliedBody).toEqual({
      activeReleaseId: RELEASE_SUMMARY.id,
      asOf: '2026-09-04',
      runMode: 'STRICT_PIT',
      note: '',
    })
    expect(await screen.findByText(/已应用 站在 2026-09-04 · 2026Q3 基线 · 严格 PIT，全平台按此口径展示/)).toBeInTheDocument()
  })

  it('can be cleared back to no PIT', async () => {
    const user = userEvent.setup()
    settings = appliedSettings('RESEARCH')
    renderPage()

    const select = await screen.findByLabelText(/用哪一批数据/)
    await waitFor(() => expect(select).toHaveValue(RELEASE_SUMMARY.id))
    await user.selectOptions(select, '')
    await user.click(screen.getByRole('button', { name: '应用到全平台' }))

    await waitFor(() => expect(appliedBody).not.toBeNull())
    // Clearing both knobs must also drop strict mode, which needs a research day.
    expect(appliedBody).toEqual({ activeReleaseId: null, asOf: null, runMode: 'RESEARCH', note: '' })
    expect(await screen.findByText(/已切换为无 PIT 口径/)).toBeInTheDocument()
  })

  it('seals a release without silently switching the platform 口径', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('ETF / 场内基金净值')

    expect(screen.getByText(/尚未封版/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '封版' }))
    expect(await screen.findByText('请先填写数据版本名称')).toBeInTheDocument()
    expect(sealedBody).toBeNull()

    await user.type(screen.getByLabelText('数据版本名称'), '2026Q3 基线')
    await user.click(screen.getByRole('button', { name: '封版' }))

    await waitFor(() => expect(sealedBody).not.toBeNull())
    expect(sealedBody.name).toBe('2026Q3 基线')
    expect(await screen.findByText(/如需全平台按此口径展示，请在上方「应用口径」中点击应用/)).toBeInTheDocument()
    // Sealing is not applying: the platform 口径 is untouched until asked.
    expect(appliedBody).toBeNull()
    expect(screen.getByTestId('pit-apply')).toHaveTextContent('无 PIT 口径')
  })

  it('surfaces a release that has gone missing instead of keeping a stale as_of', async () => {
    settings = {
      ...NO_PIT_SETTINGS,
      settings: { active_release_id: 'release-gone', run_mode: 'STRICT_PIT' as const, updated_at: '2026-09-01T00:00:00+00:00', note: '' },
      release_error: '未找到数据版本 release-gone。',
    }
    renderPage()
    expect(await screen.findByText(/未找到数据版本 release-gone。 已自动退回无 PIT 口径。/)).toBeInTheDocument()
  })
})

describe('顶栏 PIT 标签', () => {
  function renderBadge() {
    return render(
      <MemoryRouter>
        <ResearchContextProvider>
          <PitBadge />
        </ResearchContextProvider>
      </MemoryRouter>,
    )
  }

  it('reads the system setting rather than a per-browser preference', async () => {
    settings = appliedSettings('STRICT_PIT')
    renderBadge()
    const badge = await screen.findByTestId('pit-badge')
    expect(badge).toHaveTextContent('2026Q3 基线')
    expect(badge).toHaveTextContent('2026-09-04')
    expect(badge).toHaveTextContent('严格')
    expect(badge).not.toHaveTextContent('临时')
  })

  it('says 无口径 when nothing is applied', async () => {
    renderBadge()
    expect(await screen.findByTestId('pit-badge')).toHaveTextContent('无口径')
  })

  it('系统口径是默认而不是强制：本标签页可以临时换版本', async () => {
    settings = appliedSettings('RESEARCH')
    renderBadge()
    await userEvent.click(await screen.findByTestId('pit-badge'))

    const switcher = screen.getByTestId('pit-switcher')
    expect(switcher).toHaveTextContent('系统默认：站在 2026-09-04 · 2026Q3 基线 · 研究模式')
    await userEvent.click(within(switcher).getByRole('button', { name: '严格 PIT' }))

    // 只改这一个标签页，系统级设置不受影响。
    expect(getPitOverride()).toEqual({
      off: false,
      releaseId: RELEASE_SUMMARY.id,
      asOf: null,
      runMode: 'STRICT_PIT',
    })
    expect(appliedBody).toBeNull()
    expect(reload).toHaveBeenCalled()
  })

  it('本标签页也可以只站在某一天看，不需要任何封版', async () => {
    settings = NO_PIT_SETTINGS
    renderBadge()
    await userEvent.click(await screen.findByTestId('pit-badge'))

    const switcher = screen.getByTestId('pit-switcher')
    await userEvent.type(within(switcher).getByLabelText('只看某一天为止'), '2009-12-31')
    await userEvent.click(within(switcher).getByRole('button', { name: '应用到本页' }))

    expect(getPitOverride()).toEqual({
      off: false,
      releaseId: null,
      asOf: '2009-12-31',
      runMode: 'RESEARCH',
    })
    expect(appliedBody).toBeNull()
  })

  it('也可以整个关掉 PIT 只看磁盘上的全部数据', async () => {
    settings = appliedSettings('STRICT_PIT')
    renderBadge()
    await userEvent.click(await screen.findByTestId('pit-badge'))
    await userEvent.click(screen.getByTestId('pit-turn-off'))
    expect(getPitOverride()).toEqual({ off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' })
    expect(appliedBody).toBeNull()
  })

  it('临时口径会被标记出来，并且可以一键跟随系统默认', async () => {
    settings = appliedSettings('RESEARCH')
    setPitOverride({ off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' })
    renderBadge()
    const badge = await screen.findByTestId('pit-badge')
    expect(badge).toHaveTextContent('临时')
    expect(badge).toHaveTextContent('无口径')

    await userEvent.click(badge)
    await userEvent.click(screen.getByTestId('pit-follow-system'))
    expect(getPitOverride()).toBeNull()
  })

  it('被删掉的版本不会把这个标签页锁死在失效口径上', async () => {
    settings = appliedSettings('RESEARCH')
    setPitOverride({ off: false, releaseId: 'release-deleted', runMode: 'RESEARCH' })
    renderBadge()
    // 否则它的请求头会让每个数据请求都 400，包括能改回来的那个。
    await waitFor(() => expect(getPitOverride()).toBeNull())
  })
})

describe('结果口径脚注', () => {
  const lineage = {
    as_of: '2026-09-04',
    as_of_applied: true,
    run_mode: 'STRICT_PIT' as const,
    availability_field: 'ann_date',
    rows_before_cut: 6780,
    rows_after_cut: 5714,
    rows_dropped_by_as_of: 1066,
    rows_without_announcement: 0,
    announcement_fallback: false,
    warnings: [],
  }

  it('prints the cut that actually happened', () => {
    render(<MemoryRouter><PitProvenance lineage={lineage} /></MemoryRouter>)
    const note = screen.getByTestId('pit-provenance')
    expect(note).toHaveTextContent('研究日 2026-09-04')
    expect(note).toHaveTextContent('严格 PIT')
    expect(note).toHaveTextContent('剔除 1,066 行当时尚未公告的净值')
    expect(note).toHaveTextContent('可用 5,714 行')
  })

  it('labels a run that had no PIT 口径 so it cannot be mistaken for one', () => {
    render(
      <MemoryRouter>
        <PitProvenance lineage={{ ...lineage, as_of: null, as_of_applied: false, run_mode: 'RESEARCH' }} />
      </MemoryRouter>,
    )
    const note = screen.getByTestId('pit-provenance')
    expect(note).toHaveTextContent('无 PIT 口径')
    expect(note).toHaveTextContent('不具备时点可复现性')
    expect(note.querySelector('a')).toHaveAttribute('href', '/settings/pit-snapshots')
  })

  it('flags rows that fell back to the value date', () => {
    render(
      <MemoryRouter>
        <PitProvenance lineage={{ ...lineage, announcement_fallback: true, rows_without_announcement: 42 }} />
      </MemoryRouter>,
    )
    expect(screen.getByTestId('pit-provenance')).toHaveTextContent('42 行缺少公告日')
  })

  it('renders nothing when a result carries no lineage', () => {
    const { container } = render(<MemoryRouter><PitProvenance lineage={null} /></MemoryRouter>)
    expect(container.querySelector('[data-testid="pit-provenance"]')).toBeNull()
  })
  it('把研究日的可选产品域和今天的表摆在一起', async () => {
    renderPage()

    const panel = await screen.findByTestId('pit-universe-panel')
    // Without a research day the panel shows today's table and says so; the
    // contrast only becomes a number once a day is picked.
    expect(within(panel).getByText(/域仅最新态/)).toBeInTheDocument()

    await userEvent.clear(within(panel).getByLabelText('研究日'))
    await userEvent.type(within(panel).getByLabelText('研究日'), '2019-06-28')

    await waitFor(() => expect(within(panel).getByText('224')).toBeInTheDocument())
    expect(within(panel).getByText('1,568')).toBeInTheDocument()
    expect(within(panel).getByText(/当时不可选的产品/)).toBeInTheDocument()
  })
})
