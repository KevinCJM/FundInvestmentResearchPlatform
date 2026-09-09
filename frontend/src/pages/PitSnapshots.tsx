import { useEffect, useMemo, useState } from 'react'
import { useResearchContext } from '../app/ResearchContext'
import {
  GRADE_TONE,
  applyPitSettings,
  createDataRelease,
  fetchDataReleases,
  fetchPitAudit,
  fetchPitSettings,
  formatCoverage,
  formatLagDays,
  type DataRelease,
  type PitAudit,
  type PitDatasetAudit,
  type PitSettingsPayload,
  type PitUniverseView,
  type RunMode,
  fetchPitUniverse,
  UNIVERSE_COVERAGE_LABELS,
  UNIVERSE_COVERAGE_TONE,
} from '../services/pit'

const NUM = 'tabular-nums'

function Kpi({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex min-w-0 flex-col justify-center border-r border-slate-200 px-4 py-2 last:border-r-0">
      <span className="truncate text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</span>
      <span className={`${NUM} truncate text-base font-semibold ${tone ?? 'text-slate-900'}`}>{value}</span>
    </div>
  )
}

function GradeChip({ grade, label }: { grade: 'A' | 'B' | 'C'; label: string }) {
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[11px] font-semibold ${GRADE_TONE[grade]}`}>
      {label}
    </span>
  )
}


const UNIVERSE_KINDS = [
  { id: 'fund', label: '基金产品' },
  { id: 'index', label: '指数' },
  { id: 'stock', label: '股票' },
]

/**
 * The universe question the three dataset clocks cannot answer.
 *
 * A per-row cut says which prints were visible; it says nothing about which
 * products a screen could have picked. The contrast between today's table and
 * the research day's is the survivorship bias every historical backtest on this
 * platform carries, and it is a number, not an opinion.
 */
function UniversePanel({ defaultAsOf }: { defaultAsOf: string | null }) {
  const [kind, setKind] = useState('fund')
  const [asOf, setAsOf] = useState(defaultAsOf ?? '')
  const [view, setView] = useState<PitUniverseView | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setBusy(true)
    setError('')
    fetchPitUniverse({ kind, asOf: asOf || null, signal: controller.signal })
      .then(setView)
      .catch((exc: Error) => {
        if (controller.signal.aborted) return
        setView(null)
        setError(exc.message)
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false)
      })
    return () => controller.abort()
  }, [kind, asOf])

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3" data-testid="pit-universe-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">站在某日的可选产品域</h3>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select
            className="rounded border border-slate-300 px-2 py-1"
            value={kind}
            onChange={(event) => setKind(event.target.value)}
            aria-label="产品域"
          >
            {UNIVERSE_KINDS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          <input
            type="date"
            className="rounded border border-slate-300 px-2 py-1"
            value={asOf}
            onChange={(event) => setAsOf(event.target.value)}
            aria-label="研究日"
          />
        </div>
      </div>
      {error && <p className="mt-2 text-xs text-rose-700">{error}</p>}
      {busy && !view && <p className="mt-2 text-xs text-slate-500">读取中…</p>}
      {view && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span
              className={`rounded border px-1.5 py-0.5 font-semibold ${UNIVERSE_COVERAGE_TONE[view.coverage]}`}
            >
              {UNIVERSE_COVERAGE_LABELS[view.coverage]}
            </span>
            {view.history_begins_at && (
              <span className="text-slate-500">维表历史起于 {view.history_begins_at}</span>
            )}
          </div>
          <div className="grid grid-cols-3 divide-x divide-slate-200 rounded border border-slate-200">
            <Kpi label={view.as_of ? `${view.as_of} 可选` : '当前可选'} value={view.member_count.toLocaleString()} />
            <Kpi label="今日在表" value={view.latest_member_count.toLocaleString()} />
            <Kpi
              label="回放剔除"
              value={view.excluded_by_replay.toLocaleString()}
              tone={view.excluded_by_replay > 0 ? 'text-rose-700' : undefined}
            />
          </div>
          {view.as_of && view.excluded_by_replay > 0 && (
            <p className="text-[11px] leading-5 text-slate-600">
              直接拿今天的 {view.latest_member_count.toLocaleString()} 只做 {view.as_of} 的筛选，会多出{' '}
              {view.excluded_by_replay.toLocaleString()} 只当时不可选的产品——这部分就是产品池层面的未来信息。
            </p>
          )}
          {view.sample.length > 0 && (
            <p className={`${NUM} text-[11px] text-slate-500`}>
              示例：{view.sample.map((item) => `${item.code} ${item.name}`).join('、')}
            </p>
          )}
          {view.warnings.map((text) => (
            <p key={text} className="text-[11px] text-rose-700">
              {text}
            </p>
          ))}
        </div>
      )}
    </section>
  )
}

function LagHistogram({ dataset }: { dataset: PitDatasetAudit }) {
  const buckets = dataset.lag.histogram
  const total = buckets.reduce((sum, item) => sum + item.rows, 0)
  if (!total) {
    return (
      <p className="text-xs text-slate-500">
        {dataset.availability_field
          ? '该数据集没有可比对的事件时间，无法给出公告滞后分布。'
          : '该数据集没有可得时间列，滞后分布无从计算——这本身就是它拿不到 A 级的原因。'}
      </p>
    )
  }
  const peak = Math.max(...buckets.map((item) => item.rows), 1)
  return (
    <div className="space-y-1">
      {buckets.map((item) => (
        <div key={item.bucket} className="flex items-center gap-2 text-xs">
          <span className={`${NUM} w-12 shrink-0 text-right text-slate-500`}>{item.bucket} 天</span>
          <div className="h-3.5 min-w-0 flex-1 rounded-sm bg-slate-100">
            <div
              className="h-full rounded-sm bg-blue-500"
              style={{ width: `${Math.max(item.rows > 0 ? 1.5 : 0, (item.rows / peak) * 100)}%` }}
            />
          </div>
          <span className={`${NUM} w-24 shrink-0 text-right text-slate-700`}>{item.rows.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

export default function PitSnapshots() {
  const { refresh: refreshSystemContext, override, label: viewLabel, applyOverride } = useResearchContext()
  const [pitSettings, setPitSettings] = useState<PitSettingsPayload | null>(null)
  const [applying, setApplying] = useState(false)
  const [applyMessage, setApplyMessage] = useState('')
  const [draftRelease, setDraftRelease] = useState('')
  const [draftAsOf, setDraftAsOf] = useState('')
  const [draftMode, setDraftMode] = useState<RunMode>('RESEARCH')
  const [audit, setAudit] = useState<PitAudit | null>(null)
  const [releases, setReleases] = useState<DataRelease[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [releaseName, setReleaseName] = useState('')
  const [releaseNote, setReleaseNote] = useState('')
  const [sealing, setSealing] = useState(false)
  const [message, setMessage] = useState('')

  const adoptSettings = (payload: PitSettingsPayload) => {
    setPitSettings(payload)
    setDraftRelease(payload.settings.active_release_id ?? '')
    setDraftAsOf(payload.settings.as_of ?? '')
    setDraftMode(payload.settings.run_mode)
  }

  const load = (refresh = false) => {
    setLoading(true)
    setError('')
    Promise.all([fetchPitAudit({ refresh }), fetchDataReleases(), fetchPitSettings()])
      .then(([auditPayload, releasePayload, settingsPayload]) => {
        setAudit(auditPayload)
        setReleases(releasePayload.releases)
        adoptSettings(settingsPayload)
        setSelectedId((current) => current || auditPayload.datasets.find((item) => item.present)?.dataset_id || '')
        setLoading(false)
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason))
        setLoading(false)
      })
  }

  useEffect(() => {
    load(false)
    // Loaded once per visit: the audit is a file scan, not a live feed.
  }, [])

  const selected = useMemo(
    () => audit?.datasets.find((item) => item.dataset_id === selectedId) ?? null,
    [audit, selectedId],
  )

  const asOf = pitSettings?.effective.as_of ?? null
  const runMode = pitSettings?.effective.run_mode ?? 'RESEARCH'
  const dataReleaseId = pitSettings?.effective.data_release_id ?? null

  const visibility = useMemo(() => {
    if (!selected || !asOf) return null
    const end = selected.available_through
    if (!end) return { visible: false, reason: '该数据集没有可得时间，无法判断研究日当天能看到什么' }
    if (end < asOf) return { visible: true, reason: `最新可得 ${end}，比研究日早 —— 研究日附近的数据尚未入库` }
    return { visible: true, reason: `可得至 ${end}，覆盖研究日 ${asOf}` }
  }, [asOf, selected])

  // The research day cannot run past the vintage that answers for it, so the
  // ceiling comes from the selected release rather than from today.
  const maxAsOf = useMemo(() => {
    if (!draftRelease) return audit?.summary.available_through ?? null
    return (
      (pitSettings?.available_releases ?? []).find((item) => item.id === draftRelease)?.available_through ?? null
    )
  }, [audit, draftRelease, pitSettings])

  const draftLabel = useMemo(() => {
    const vintage =
      (pitSettings?.available_releases ?? []).find((item) => item.id === draftRelease)?.name ?? '最新数据（未封版）'
    const mode = draftAsOf && draftMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'
    if (!draftAsOf && !draftRelease) return '无 PIT 口径 · 使用全部磁盘数据'
    if (!draftAsOf) return `${vintage} · ${mode}`
    return `站在 ${draftAsOf} · ${vintage} · ${mode}`
  }, [draftAsOf, draftMode, draftRelease, pitSettings])

  // Prose cannot make "stand on 2009-12-31" concrete; a count can. This is the
  // same replay the universe panel below runs, shown before the user commits.
  const [universePreview, setUniversePreview] = useState<PitUniverseView | null>(null)
  useEffect(() => {
    if (!draftAsOf) {
      setUniversePreview(null)
      return
    }
    const controller = new AbortController()
    fetchPitUniverse({ kind: 'fund', asOf: draftAsOf, signal: controller.signal })
      .then(setUniversePreview)
      .catch(() => {
        if (!controller.signal.aborted) setUniversePreview(null)
      })
    return () => controller.abort()
  }, [draftAsOf])

  const onApply = (releaseId: string | null, asOfValue: string | null, mode: RunMode) => {
    setApplying(true)
    setApplyMessage('')
    applyPitSettings({ activeReleaseId: releaseId, asOf: asOfValue, runMode: mode })
      .then((payload) => {
        adoptSettings(payload)
        // The badge and every page read the server setting, so tell them to re-read.
        refreshSystemContext()
        setApplyMessage(
          payload.effective.no_pit
            ? '已切换为无 PIT 口径：全平台使用全部磁盘数据'
            : `已应用 ${payload.effective.label}，全平台按此口径展示`,
        )
        setApplying(false)
      })
      .catch((reason: unknown) => {
        setApplyMessage(reason instanceof Error ? reason.message : String(reason))
        setApplying(false)
      })
  }

  const onSeal = () => {
    const name = releaseName.trim()
    if (!name) {
      setMessage('请先填写数据版本名称')
      return
    }
    setSealing(true)
    setMessage('')
    createDataRelease(name, releaseNote.trim())
      .then((release) => {
        setReleases((current) => [release, ...current])
        // Sealing records a vintage; applying it platform-wide is a separate,
        // deliberate act — so preselect it but do not switch口径 behind the user.
        setDraftRelease(release.id)
        setReleaseName('')
        setReleaseNote('')
        setMessage(`已封版 ${release.name}（${release.id}）。如需全平台按此口径展示，请在上方「应用口径」中点击应用。`)
        setSealing(false)
      })
      .catch((reason: unknown) => {
        setMessage(reason instanceof Error ? reason.message : String(reason))
        setSealing(false)
      })
  }

  const summary = audit?.summary

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">PIT 时点快照</h2>
          <p className="mt-0.5 text-xs text-slate-600">
            在这里决定全平台<strong>站在哪一天、用哪一批数据、有多严格</strong>。三者互相独立：研究日管"看到哪天为止"，数据版本管"读的是哪一次的历史"。设定后全平台默认按此展示；各功能页可在顶栏临时切换，只影响自己的标签页。
          </p>
        </div>
        <button
          type="button"
          className="rounded border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          onClick={() => load(true)}
          disabled={loading}
        >
          {loading ? '扫描中…' : '重新扫描'}
        </button>
      </div>

      {error && <p className="rounded border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800">{error}</p>}

      {/* Otherwise this page shows the system口径 while the rest of the tab shows
          something else, and the numbers look inexplicably inconsistent. */}
      {override && (
        <p
          className="flex flex-wrap items-center gap-2 rounded border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-900"
          data-testid="pit-override-notice"
        >
          <span>本标签页正在用临时口径查看数据：<strong>{viewLabel}</strong>。系统级设置不受影响。</span>
          <button type="button" className="underline hover:no-underline" onClick={() => applyOverride(null)}>
            恢复跟随系统默认
          </button>
        </p>
      )}

      {/* Three questions in the order a person actually decides them: which day
          am I standing on, which copy of the data answers, how strict. The old
          layout offered only the middle one, so a user who wanted to research
          "as of 2009-12-31" had nowhere to say it and typed it into a note. */}
      <section
        className={`rounded-lg border ${
          pitSettings?.effective.no_pit ? 'border-amber-300 bg-amber-50' : 'border-emerald-300 bg-emerald-50'
        }`}
        data-testid="pit-apply"
      >
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-black/5 px-3 py-2.5">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">研究口径（系统级 · 全平台生效）</h3>
            <p className={`mt-0.5 ${NUM} text-xs text-slate-700`}>
              当前：<strong>{pitSettings?.effective.label ?? '读取中…'}</strong>
              {pitSettings?.settings.updated_at && (
                <span className="ml-2 text-slate-500">更新于 {pitSettings.settings.updated_at}</span>
              )}
            </p>
            {pitSettings?.release_error && (
              <p className="mt-1 text-xs font-medium text-rose-700">
                {pitSettings.release_error} 已自动退回无 PIT 口径。
              </p>
            )}
          </div>
        </div>

        <div className="space-y-3 px-3 py-3">
          <div>
            <label className="text-xs font-semibold text-slate-800" htmlFor="pit-as-of">
              ① 站在哪一天看？
            </label>
            <p className="mt-0.5 text-[11px] leading-5 text-slate-600">
              只使用该日<strong>当时已经公开</strong>的数据。之后才公布的净值、之后才上市的产品，一律不可见。
              留空表示不设研究日，使用磁盘上的全部数据。
            </p>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <input
                id="pit-as-of"
                type="date"
                className={`${NUM} w-44 rounded border border-slate-300 px-2 py-1 text-xs`}
                value={draftAsOf}
                max={maxAsOf ?? undefined}
                onChange={(event) => setDraftAsOf(event.target.value)}
              />
              <button
                type="button"
                className="rounded border border-slate-300 bg-white px-2 py-1 text-[11px] text-slate-700 hover:bg-slate-50"
                onClick={() => setDraftAsOf(maxAsOf ?? '')}
              >
                数据最新一天
              </button>
              <button
                type="button"
                className="rounded border border-slate-300 bg-white px-2 py-1 text-[11px] text-slate-700 hover:bg-slate-50"
                onClick={() => setDraftAsOf('')}
              >
                不设研究日
              </button>
              {maxAsOf && draftAsOf > maxAsOf && (
                <span className="text-[11px] text-rose-700">晚于所选数据版本的可得截止日 {maxAsOf}</span>
              )}
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-800" htmlFor="pit-release">
              ② 用哪一批数据？
            </label>
            <p className="mt-0.5 text-[11px] leading-5 text-slate-600">
              和研究日无关：研究日决定<strong>看到哪一天为止</strong>，数据版本决定<strong>读的是哪一次的历史</strong>——
              供应商回溯修订过的净值，会让同一个研究日在上周和本周给出两个答案。封版把文件本身钉住。
            </p>
            <select
              id="pit-release"
              className="mt-1.5 w-80 rounded border border-slate-300 px-2 py-1 text-xs disabled:bg-slate-100 disabled:text-slate-400"
              value={draftRelease}
              disabled={applying}
              onChange={(event) => setDraftRelease(event.target.value)}
            >
              <option value="">最新数据（未封版）</option>
              {(pitSettings?.available_releases ?? []).map((release) => (
                <option key={release.id} value={release.id}>
                  {release.name} · 可得至 {release.available_through ?? '—'}
                </option>
              ))}
            </select>
            {!pitSettings?.can_apply && (
              <p className="mt-1 text-[11px] text-slate-600">
                还没有封过版。<strong>研究日单独就能用</strong>；封版是为了让结论在数据刷新后仍然可复现。
              </p>
            )}
          </div>

          <div>
            <span className="text-xs font-semibold text-slate-800">③ 严不严格？</span>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {(['RESEARCH', 'STRICT_PIT'] as RunMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`rounded border px-2.5 py-1.5 text-left text-[11px] leading-4 ${
                    draftMode === mode
                      ? 'border-slate-900 bg-white font-semibold text-slate-900'
                      : 'border-slate-300 bg-white/60 text-slate-600 hover:bg-white'
                  } disabled:opacity-40`}
                  disabled={mode === 'STRICT_PIT' && !draftAsOf}
                  onClick={() => setDraftMode(mode)}
                >
                  <span className="block">{mode === 'RESEARCH' ? '研究模式' : '严格 PIT'}</span>
                  <span className="mt-0.5 block font-normal text-slate-500">
                    {mode === 'RESEARCH'
                      ? '缺公告日的数据按净值日近似，并在结果上标注'
                      : '拿不出时点证明的数据直接拒绝使用'}
                  </span>
                </button>
              ))}
            </div>
            {!draftAsOf && (
              <p className="mt-1 text-[11px] text-slate-600">严格 PIT 需要先填 ① 研究日——没有截止日就没什么可执行。</p>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded border border-black/10 bg-white/70 px-2.5 py-2">
            <div className="min-w-0 text-[11px] leading-5 text-slate-700">
              <span className="font-semibold">将要生效：</span>
              <span className={NUM}>{draftLabel}</span>
              {universePreview && draftAsOf && (
                <span className="ml-2 text-slate-600">
                  届时可选基金 <strong className={NUM}>{universePreview.member_count.toLocaleString()}</strong> 只
                  {universePreview.excluded_by_replay > 0 && (
                    <>
                      ，比今天的表少 <strong className={NUM}>{universePreview.excluded_by_replay.toLocaleString()}</strong> 只
                    </>
                  )}
                </span>
              )}
            </div>
            <button
              type="button"
              className="shrink-0 rounded bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-700 disabled:opacity-50"
              onClick={() => onApply(draftRelease || null, draftAsOf || null, draftAsOf ? draftMode : 'RESEARCH')}
              disabled={applying}
            >
              {applying ? '应用中…' : '应用到全平台'}
            </button>
          </div>
        </div>
        {applyMessage && <p className="border-t border-black/5 px-3 py-2 text-xs text-slate-800">{applyMessage}</p>}
      </section>

      <div className="grid grid-cols-2 divide-slate-200 overflow-hidden rounded-lg border border-slate-200 bg-white sm:grid-cols-3 lg:grid-cols-6">
        <Kpi label="声明数据集" value={summary ? String(summary.declared) : '—'} />
        <Kpi label="A 级 严格" value={summary ? String(summary.grade_a) : '—'} tone="text-emerald-700" />
        <Kpi label="B 级 近似" value={summary ? String(summary.grade_b) : '—'} tone="text-amber-700" />
        <Kpi label="C 级 无 PIT" value={summary ? String(summary.grade_c) : '—'} tone="text-rose-700" />
        <Kpi label="A/B 数据可得至" value={summary?.available_through ?? '—'} />
        <Kpi label="已封版本" value={String(releases.length)} />
      </div>

      <section className="overflow-hidden rounded-lg border border-slate-200 bg-white">
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
          <h3 className="text-sm font-semibold text-slate-900">数据集 PIT 能力</h3>
          <span className="text-[11px] text-slate-500">点击一行查看滞后分布</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] border-collapse text-xs">
            <thead className="bg-slate-50 text-left text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2 font-semibold">数据集</th>
                <th className="px-3 py-2 font-semibold">事件时间</th>
                <th className="px-3 py-2 font-semibold">可得时间</th>
                <th className="px-3 py-2 text-right font-semibold">行数</th>
                <th className="px-3 py-2 text-right font-semibold">覆盖率</th>
                <th className="px-3 py-2 text-right font-semibold">P50</th>
                <th className="px-3 py-2 text-right font-semibold">P95</th>
                <th className="px-3 py-2 font-semibold">可得至</th>
                <th className="px-3 py-2 font-semibold">等级</th>
              </tr>
            </thead>
            <tbody>
              {(audit?.datasets ?? []).map((item) => (
                <tr
                  key={item.dataset_id}
                  className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                    item.dataset_id === selectedId ? 'bg-blue-50/60' : ''
                  } ${item.present ? '' : 'text-slate-400'}`}
                  onClick={() => setSelectedId(item.dataset_id)}
                >
                  <td className="px-3 py-1.5">
                    <span className="font-medium text-slate-900">{item.label}</span>
                    {!item.present && <span className="ml-2 text-[11px] text-slate-400">文件缺失</span>}
                    <div className="text-[11px] text-slate-500">{item.file}</div>
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-600`}>{item.event_field ?? '—'}</td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-600`}>
                    {item.availability_field ?? <span className="text-rose-600">无</span>}
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>{item.rows.toLocaleString()}</td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>
                    {formatCoverage(item.availability_coverage)}
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>{formatLagDays(item.lag.p50)}</td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>{formatLagDays(item.lag.p95)}</td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-600`}>{item.available_through ?? '—'}</td>
                  <td className="px-3 py-1.5">
                    <GradeChip grade={item.grade} label={item.grade} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <UniversePanel defaultAsOf={asOf} />

      {selected && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-900">{selected.label} · 公告滞后分布</h3>
              <GradeChip grade={selected.grade} label={selected.grade_label} />
            </div>
            <div className="mt-3">
              <LagHistogram dataset={selected} />
            </div>
            {selected.lag.negative_rows > 0 && (
              <p className="mt-2 rounded border border-rose-200 bg-rose-50 px-2 py-1.5 text-[11px] text-rose-800">
                {selected.lag.negative_rows.toLocaleString()} 行的公告日早于事件日——这是数据质量问题，不是时点特性。
              </p>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-3">
            <h3 className="text-sm font-semibold text-slate-900">在当前研究上下文下</h3>
            <dl className="mt-3 space-y-1.5 text-xs">
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">研究日</dt>
                <dd className={`${NUM} text-slate-900`}>{asOf ?? '未指定'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">运行模式</dt>
                <dd className="text-slate-900">{runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">数据版本</dt>
                <dd className={`${NUM} text-slate-900`}>{dataReleaseId ?? '未指定'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">事件时间范围</dt>
                <dd className={`${NUM} text-slate-900`}>
                  {selected.event_range.start ?? '—'} ~ {selected.event_range.end ?? '—'}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">可得时间范围</dt>
                <dd className={`${NUM} text-slate-900`}>
                  {selected.availability_range.start ?? '—'} ~ {selected.availability_range.end ?? '—'}
                </dd>
              </div>
              {runMode === 'STRICT_PIT' && selected.grade === 'C' && (
                <p className="rounded border border-rose-200 bg-rose-50 px-2 py-1.5 text-[11px] text-rose-800">
                  严格 PIT 模式下该数据集被禁用，引用它的计算会直接报错而不是静默降级。
                </p>
              )}
              {visibility && <p className="text-[11px] text-slate-600">{visibility.reason}</p>}
            </dl>
            <p className="mt-3 border-t border-slate-100 pt-2 text-[11px] leading-5 text-slate-600">{selected.note}</p>
          </section>
        </div>
      )}

      <section className="rounded-lg border border-slate-200 bg-white">
        <div className="border-b border-slate-200 px-3 py-2">
          <h3 className="text-sm font-semibold text-slate-900">数据版本（封版）</h3>
          <p className="mt-0.5 text-[11px] text-slate-600">
            <strong>封版不是用来选日期的</strong>——要"只看某天为止"，请用上面的 ① 研究日。
            封版解决的是另一件事：同一个研究日，上周与本周的底层数据可能已被供应商回溯修订，两次会跑出两个答案。封版把文件本身钉住。
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-2 border-b border-slate-200 px-3 py-2.5">
          <label className="text-xs">
            <span className="mb-1 block font-medium text-slate-600">版本名称</span>
            <input
              className="w-56 rounded border border-slate-300 px-2 py-1 text-xs"
              value={releaseName}
              onChange={(event) => setReleaseName(event.target.value)}
              placeholder="例如 2026Q3 投前研究基线"
              aria-label="数据版本名称"
            />
          </label>
          <label className="min-w-0 flex-1 text-xs">
            <span className="mb-1 block font-medium text-slate-600">备注</span>
            <input
              className="w-full rounded border border-slate-300 px-2 py-1 text-xs"
              value={releaseNote}
              onChange={(event) => setReleaseNote(event.target.value)}
              placeholder="选填：本次封版的用途或数据变更说明"
              aria-label="数据版本备注"
            />
          </label>
          <button
            type="button"
            className="rounded bg-blue-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-600 disabled:opacity-50"
            onClick={onSeal}
            disabled={sealing}
          >
            {sealing ? '封版中…' : '封版'}
          </button>
        </div>
        {message && <p className="px-3 py-2 text-xs text-slate-700">{message}</p>}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-xs">
            <thead className="bg-slate-50 text-left text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2 font-semibold">版本</th>
                <th className="px-3 py-2 font-semibold">封版时间</th>
                <th className="px-3 py-2 text-right font-semibold">表数</th>
                <th className="px-3 py-2 text-right font-semibold">总行数</th>
                <th className="px-3 py-2 font-semibold">指纹</th>
                <th className="px-3 py-2 font-semibold">父版本</th>
              </tr>
            </thead>
            <tbody>
              {releases.length === 0 && (
                <tr>
                  <td className="px-3 py-3 text-slate-500" colSpan={6}>
                    尚未封版。先封一个基线版本，之后每次数据刷新再封一版，就能回答“这个结论用的是哪批数据”。
                  </td>
                </tr>
              )}
              {releases.map((release) => (
                <tr
                  key={release.id}
                  className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                    release.id === dataReleaseId ? 'bg-blue-50/60' : ''
                  }`}
                  onClick={() => setDraftRelease(release.id)}
                  title="选中以便在上方「应用口径」中应用"
                >
                  <td className="px-3 py-1.5">
                    <span className="font-medium text-slate-900">{release.name}</span>
                    <div className={`${NUM} text-[11px] text-slate-500`}>{release.id}</div>
                    {release.note && <div className="text-[11px] text-slate-600">{release.note}</div>}
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-700`}>{release.created_at}</td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>{release.tables.length}</td>
                  <td className={`${NUM} px-3 py-1.5 text-right text-slate-800`}>
                    {release.summary.total_rows.toLocaleString()}
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-500`} title={release.release_fingerprint}>
                    {release.release_fingerprint.slice(0, 12)}
                  </td>
                  <td className={`${NUM} px-3 py-1.5 text-slate-500`}>
                    {release.parent_release_id ? release.parent_release_id.slice(-6) : '基线'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
