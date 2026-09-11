import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PitSnapshots from './PitSnapshots'
import PitBadge from '../components/PitBadge'
import PitProvenance from '../components/PitProvenance'
import PitDecisionNotice from '../components/PitDecisionNotice'
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
  as_of: '2026-09-04',
  run_mode: 'RESEARCH' as const,
  available_through: '2026-09-04',
  grade_a: 1,
  grade_b: 0,
  grade_c: 1,
  table_count: 2,
  total_rows: 1549052,
  release_fingerprint: 'f'.repeat(64),
}

const NO_PIT_SETTINGS = {
  settings: { active_release_id: null, as_of: null, run_mode: null, updated_at: null, note: '' },
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
let editedBody: any = null
let deletedId = ''

function appliedSettings(runMode: 'RESEARCH' | 'STRICT_PIT') {
  return {
    settings: { active_release_id: RELEASE_SUMMARY.id, as_of: null, run_mode: null, updated_at: '2026-09-07T03:00:00+00:00', note: '' },
    effective: {
      as_of: '2026-09-04',
      as_of_source: 'release' as const,
      run_mode: runMode,
      run_mode_label: runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式',
      data_release_id: RELEASE_SUMMARY.id,
      no_pit: false,
      label: `站在 2026-09-04 · 2026Q3 基线 · ${runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}`,
    },
    release: { ...RELEASE_SUMMARY, run_mode: runMode },
    release_error: null,
    available_releases: [{ ...RELEASE_SUMMARY, run_mode: runMode }],
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
        as_of: sealedBody.asOf,
        run_mode: sealedBody.runMode,
        parent_release_id: null,
        immutable: true,
        tables: [{ dataset_id: 'etf_nav' }, { dataset_id: 'etf_info' }],
        summary: SUMMARY,
      }
      releases = [release]
      settings = { ...settings, available_releases: [RELEASE_SUMMARY], can_apply: true }
      return new Response(JSON.stringify(release), { status: 200 })
    }
    if (url.startsWith('/api/pit/releases/') && method === 'PUT') {
      editedBody = JSON.parse(String(init?.body))
      const current = releases[0] as any
      const updated = {
        ...current,
        name: editedBody.name,
        note: editedBody.note,
        as_of: editedBody.asOf,
        run_mode: editedBody.runMode,
        updated_at: '2026-09-09T04:00:00+00:00',
      }
      releases = [updated]
      return new Response(JSON.stringify(updated), { status: 200 })
    }
    if (url.startsWith('/api/pit/releases/') && method === 'DELETE') {
      deletedId = url.split('/').pop() ?? ''
      if (settings.settings?.active_release_id === deletedId) {
        return new Response(
          JSON.stringify({ detail: '版本「2026Q3 基线」正在被全平台使用；请先切到别的版本或「不用 PIT」并应用，然后再删除。' }),
          { status: 409 },
        )
      }
      releases = []
      settings = { ...NO_PIT_SETTINGS, available_releases: [], can_apply: false }
      return new Response(JSON.stringify({ deleted_id: deletedId }), { status: 200 })
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
      // A version answers for its own mode; the request no longer states one.
      const chosen = (settings.available_releases ?? []).find(
        (item: any) => item.id === appliedBody.activeReleaseId,
      )
      settings = appliedBody.activeReleaseId
        ? appliedSettings(appliedBody.runMode ?? chosen?.run_mode ?? 'RESEARCH')
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
  editedBody = null
  deletedId = ''
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
  it('用 PIT 的唯一入口是版本：没有版本就只有"不用 PIT"', async () => {
    const user = userEvent.setup()
    renderPage()
    const panel = await screen.findByTestId('pit-apply')
    expect(panel).toHaveTextContent('无 PIT 口径 · 使用全部磁盘数据')
    // One concept, one control: there is no second platform-level date box to
    // keep in sync with the version — the day lives inside the version.
    expect(screen.queryByLabelText(/站在哪一天看/)).toBeNull()
    expect(panel).toHaveTextContent('要用 PIT，先建一个版本')

    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理（已有 0 个）/ }))
    expect(screen.getByLabelText(/站在哪一天看/)).not.toBeDisabled()
  })

  it('shows what standing on that day costs while the version is being defined', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByTestId('pit-apply')
    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理/ }))

    await user.type(screen.getByLabelText(/站在哪一天看/), '2009-12-31')

    const panel = screen.getByTestId('pit-apply')
    // A count is what makes "stand on 2009-12-31" concrete, and it belongs
    // beside the day being chosen rather than in the commit bar.
    await waitFor(() => expect(panel).toHaveTextContent('站在 2009-12-31'))
    expect(panel).toHaveTextContent('可选基金')
  })

  it('applies a version platform-wide,口径 and all', async () => {
    const user = userEvent.setup()
    const strict = { ...RELEASE_SUMMARY, run_mode: 'STRICT_PIT' as const }
    settings = { ...NO_PIT_SETTINGS, available_releases: [strict], can_apply: true }
    releases = [{ ...strict, parent_release_id: null, immutable: true, tables: [], summary: SUMMARY }]
    renderPage()

    const select = await screen.findByLabelText(/用哪个 PIT 版本/)
    await waitFor(() => expect(select).not.toBeDisabled())
    // The option states the whole口径, so there is nothing else to set.
    expect(within(select).getByRole('option', { name: /站在 2026-09-04 · 严格/ })).toBeInTheDocument()
    // Applying is done where the version is: in its own row, next to 删除.
    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理/ }))
    await user.click(screen.getByRole('button', { name: '应用到全平台' }))

    await waitFor(() => expect(appliedBody).not.toBeNull())
    // Null day and null mode: "whatever the version says", so the platform
    // setting can never drift from the version it names.
    expect(appliedBody).toEqual({
      activeReleaseId: RELEASE_SUMMARY.id,
      asOf: null,
      runMode: null,
      note: '',
    })
    expect(await screen.findByText(/已应用 站在 2026-09-04 · 2026Q3 基线 · 严格 PIT，全平台按此口径展示/)).toBeInTheDocument()
  })

  it('can be cleared back to no PIT', async () => {
    const user = userEvent.setup()
    settings = appliedSettings('RESEARCH')
    renderPage()

    const select = await screen.findByLabelText(/用哪个 PIT 版本/)
    await waitFor(() => expect(select).toHaveValue(RELEASE_SUMMARY.id))
    // The picker is the setting, not a draft: choosing applies.
    await user.selectOptions(select, '')

    await waitFor(() => expect(appliedBody).not.toBeNull())
    expect(appliedBody).toEqual({ activeReleaseId: null, asOf: null, runMode: null, note: '' })
    expect(await screen.findByText(/已切换为无 PIT 口径/)).toBeInTheDocument()
  })

  it('a new version is defined with the day it stands on, and is not applied for you', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('ETF / 场内基金净值')

    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理（已有 0 个）/ }))
    expect(screen.getByText(/还没有版本/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '创建版本' }))
    expect(await screen.findByText('请先填写数据版本名称')).toBeInTheDocument()
    expect(sealedBody).toBeNull()

    await user.type(screen.getByLabelText(/站在哪一天看/), '2014-12-31')
    await user.type(screen.getByLabelText('数据版本名称'), '站在 2014 年末的投前基线')
    await user.click(screen.getByRole('button', { name: /严格 PIT/ }))
    await user.click(screen.getByRole('button', { name: '创建版本' }))

    await waitFor(() => expect(sealedBody).not.toBeNull())
    // The day and the mode are part of the version, not of a separate setting.
    expect(sealedBody).toEqual({
      name: '站在 2014 年末的投前基线',
      note: '',
      asOf: '2014-12-31',
      runMode: 'STRICT_PIT',
    })
    expect(await screen.findByText(/已建好版本「站在 2014 年末的投前基线」，站在 2014-12-31/)).toBeInTheDocument()
    // Creating is not applying: the platform 口径 is untouched until asked.
    expect(appliedBody).toBeNull()
    expect(screen.getByTestId('pit-apply')).toHaveTextContent('无 PIT 口径')
  })

  it('一个版本的口径可以改，钉住的数据不跟着改', async () => {
    const user = userEvent.setup()
    settings = { ...NO_PIT_SETTINGS, available_releases: [RELEASE_SUMMARY], can_apply: true }
    releases = [{ ...RELEASE_SUMMARY, parent_release_id: null, immutable: true, tables: [], summary: SUMMARY }]
    renderPage()
    await screen.findByTestId('pit-apply')
    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理/ }))

    // 编辑把这一行读进同一个表单，而不是弹一个第二处维护同样四个字段的对话框。
    await user.click(screen.getByRole('button', { name: '编辑' }))
    expect(screen.getByLabelText('数据版本名称')).toHaveValue('2026Q3 基线')
    expect(screen.getByLabelText(/站在哪一天看/)).toHaveValue('2026-09-04')

    await user.clear(screen.getByLabelText(/站在哪一天看/))
    await user.type(screen.getByLabelText(/站在哪一天看/), '2014-12-31')
    await user.click(screen.getByRole('button', { name: '保存修改' }))

    await waitFor(() => expect(editedBody).not.toBeNull())
    expect(editedBody).toEqual({
      name: '2026Q3 基线',
      note: '',
      asOf: '2014-12-31',
      runMode: 'RESEARCH',
    })
    expect(await screen.findByText(/已改好版本「2026Q3 基线」，现在站在 2014-12-31/)).toBeInTheDocument()
    // 改口径不等于应用它。
    expect(appliedBody).toBeNull()
  })

  it('正在被全平台使用的版本不给删除入口，而不是点了再被拒', async () => {
    const user = userEvent.setup()
    settings = appliedSettings('RESEARCH')
    releases = [{ ...RELEASE_SUMMARY, parent_release_id: null, immutable: true, tables: [], summary: SUMMARY }]
    renderPage()
    await screen.findByTestId('pit-apply')
    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理/ }))

    // The backend refuses this delete with a 409 by design. A live button that
    // can only ever fail is how that refusal gets reported as 「删除失败」.
    const remove = screen.getByRole('button', { name: '删除' })
    expect(remove).toBeDisabled()
    // A disabled button never shows its title, so the reason has to be text.
    expect(screen.getByText('全平台使用中')).toBeInTheDocument()
    await user.click(remove)
    expect(deletedId).toBe('')
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
    // 一眼只回答「开着吗、用哪个口径」；日期与严格与否留给悬停和面板。
    expect(badge).toHaveTextContent('PIT 打开：2026Q3 基线')
    expect(badge).not.toHaveTextContent('!')
  })

  it('says PIT 关闭 when nothing is applied', async () => {
    renderBadge()
    expect(await screen.findByTestId('pit-badge')).toHaveTextContent('PIT 关闭')
  })

  it('系统口径是默认而不是强制：本标签页可以临时换版本', async () => {
    settings = appliedSettings('RESEARCH')
    renderBadge()
    await userEvent.click(await screen.findByTestId('pit-badge'))

    const switcher = screen.getByTestId('pit-switcher')
    expect(switcher).toHaveTextContent('系统默认：站在 2026-09-04 · 2026Q3 基线 · 研究模式')
    // One button per version: it carries its own day and mode, so there is no
    // 研究模式/严格 pair to pick from here.
    await userEvent.click(within(switcher).getByRole('button', { name: /2026Q3 基线/ }))

    // 只改这一个标签页，系统级设置不受影响。
    expect(getPitOverride()).toEqual({
      off: false,
      releaseId: RELEASE_SUMMARY.id,
      asOf: null,
      runMode: null,
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

  it('本页选的版本就是系统正在用的那个时，不算临时口径', async () => {
    settings = appliedSettings('RESEARCH')
    setPitOverride({ off: false, releaseId: RELEASE_SUMMARY.id, asOf: null, runMode: null })
    renderBadge()
    const badge = await screen.findByTestId('pit-badge')
    // 口径一模一样却报「和系统默认不一致」，警告就变成了噪音。
    expect(badge).toHaveTextContent('PIT 打开：2026Q3 基线')
    expect(within(badge).queryByText(/与系统默认（.+）不一致/)).toBeNull()
  })

  it('临时口径会被标记出来，并且可以一键跟随系统默认', async () => {
    settings = appliedSettings('RESEARCH')
    setPitOverride({ off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' })
    renderBadge()
    const badge = await screen.findByTestId('pit-badge')
    // 和系统默认不一致时只挂一个感叹号，理由留在悬停提示里。
    expect(within(badge).getByText(/与系统默认（.+）不一致/)).toBeInTheDocument()
    expect(badge).toHaveTextContent('PIT 关闭')

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

  it('prints the universe finding beside the口径 that carries it', () => {
    render(
      <MemoryRouter>
        <PitProvenance
          lineage={{
            ...lineage,
            universe: {
              source: 'investable_universe_snapshot',
              replayable: false,
              established_at: '2026-09-04',
              clean: false,
              findings: [
                {
                  code: 'UNIVERSE_LOOKAHEAD',
                  label: '可投资域「测试域」',
                  message: '可投资域「测试域」是用截至 2026-09-04 的数据筛出来的，却被用于 2021-09-01 的决策——该决策带入了未来信息。',
                },
              ],
            },
          }}
        />
      </MemoryRouter>,
    )
    expect(screen.getByTestId('pit-universe-finding')).toHaveTextContent('带入了未来信息')
  })

  const allocationLineage = {
    alloc_name: '股债分类',
    as_of: '2024-02-20',
    run_mode: 'RESEARCH' as const,
    series_as_of: '2023-12-01',
    series_variants: ['2023-12-01'],
    hindsight_series: false,
    rows_dropped_by_as_of: 0,
    availability_available: true,
    warnings: [],
  }

  it('回测口径要印出搭这套配置的产品池，以及它是哪天筛的', () => {
    render(
      <MemoryRouter>
        <PitDecisionNotice
          lineage={{
            ...allocationLineage,
            universe: {
              source: 'asset_nv',
              replayable: false,
              established_at: '2023-12-01',
              snapshot_id: 'universe-1',
              snapshot_established_at: '2026-01-01',
              clean: false,
              findings: [
                {
                  code: 'UNIVERSE_LOOKAHEAD',
                  label: '可投资域「2026筛出来的池子」',
                  message: '可投资域「2026筛出来的池子」是用截至 2026-01-01 的数据筛出来的，却被用于 2024-02-20 的决策——该决策带入了未来信息。',
                },
              ],
            },
          }}
        />
      </MemoryRouter>,
    )
    const note = screen.getByTestId('pit-decision-notice')
    expect(note).toHaveTextContent('产品域研究日 2026-01-01')
    expect(screen.getByTestId('pit-universe-finding')).toHaveTextContent('带入了未来信息')
  })

  it('没记录产品池的配置要说"不知道"，不能看着跟干净的一样', () => {
    render(<MemoryRouter><PitDecisionNotice lineage={allocationLineage} /></MemoryRouter>)
    expect(screen.getByTestId('pit-decision-notice')).toHaveTextContent('未记录产品域')
  })

  it('renders nothing when a result carries no lineage', () => {
    const { container } = render(<MemoryRouter><PitProvenance lineage={null} /></MemoryRouter>)
    expect(container.querySelector('[data-testid="pit-provenance"]')).toBeNull()
  })
  it('产品域明细跟着"当前在用的那一天"走，全页只有一个日期框', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByTestId('pit-apply')

    expect(screen.queryByTestId('pit-universe-panel')).toBeNull()
    await user.click(screen.getByRole('button', { name: '查看产品域明细' }))

    const panel = await screen.findByTestId('pit-universe-panel')
    expect(within(panel).getByText(/域仅最新态/)).toBeInTheDocument()
    // The panel has no date input of its own to drift from the version's.
    expect(within(panel).queryByLabelText('研究日')).toBeNull()

    await user.click(screen.getByRole('button', { name: /新建版本 \/ 管理/ }))
    await user.type(screen.getByLabelText(/站在哪一天看/), '2019-06-28')

    await waitFor(() => expect(within(panel).getByText('224')).toBeInTheDocument())
    expect(within(panel).getByText('1,568')).toBeInTheDocument()
    expect(within(panel).getByText(/当时不可选的产品/)).toBeInTheDocument()
  })
})
