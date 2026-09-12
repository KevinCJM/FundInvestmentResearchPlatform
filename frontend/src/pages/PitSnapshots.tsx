import { useEffect, useMemo, useState } from 'react'
import { useResearchContext } from '../app/ResearchContext'
import {
  GRADE_TONE,
  applyPitSettings,
  createDataRelease,
  deleteDataRelease,
  updateDataRelease,
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
      <span className="truncate text-xs font-medium uppercase tracking-wide text-slate-600">{label}</span>
      <span className={`${NUM} truncate text-base font-semibold ${tone ?? 'text-slate-900'}`}>{value}</span>
    </div>
  )
}

function GradeChip({ grade, label }: { grade: 'A' | 'B' | 'C' | null; label: string }) {
  // A pending dataset has no grade yet. Painting it C would read as a measured
  // verdict on a file nobody has opened.
  const tone = grade ? GRADE_TONE[grade] : 'border-slate-200 bg-slate-50 text-slate-600'
  return (
    <span className={`inline-block rounded-lg border px-1.5 py-0.5 text-xs font-semibold ${tone}`}>
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
/**
 * What "standing on that day" costs, in products.
 *
 * It lives *inside* step ① and reads that step's date rather than carrying one
 * of its own: two date boxes for the same concept on one page is the confusion
 * this page exists to remove, and the earlier layout had exactly that.
 */
function UniversePanel({ asOf }: { asOf: string }) {
  const [kind, setKind] = useState('fund')
  const [view, setView] = useState<PitUniverseView | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setError('')
    fetchPitUniverse({ kind, asOf: asOf || null, signal: controller.signal })
      .then(setView)
      .catch((exc: Error) => {
        if (controller.signal.aborted) return
        setView(null)
        setError(exc.message)
      })
    return () => controller.abort()
  }, [kind, asOf])

  return (
    <div className="mt-1.5 rounded-xl border border-slate-200 bg-white p-2.5" data-testid="pit-universe-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-700">
          {asOf ? `${asOf} 当天的可选产品域` : '当前可选产品域（未设研究日）'}
        </span>
        <select
          className="rounded-lg border border-slate-300 px-2 py-1 text-xs"
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
      </div>
      {error && <p className="mt-2 text-xs text-rose-700">{error}</p>}
      {!view && !error && <p className="mt-2 text-xs text-slate-600">读取中…</p>}
      {view && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span
              className={`rounded-lg border px-1.5 py-0.5 font-semibold ${UNIVERSE_COVERAGE_TONE[view.coverage]}`}
            >
              {UNIVERSE_COVERAGE_LABELS[view.coverage]}
            </span>
            {view.history_begins_at && (
              <span className="text-slate-600">维表历史起于 {view.history_begins_at}</span>
            )}
          </div>
          <div className="grid grid-cols-3 divide-x divide-slate-200 rounded-lg border border-slate-200">
            <Kpi label={view.as_of ? `${view.as_of} 可选` : '当前可选'} value={view.member_count.toLocaleString()} />
            <Kpi label="今日在表" value={view.latest_member_count.toLocaleString()} />
            <Kpi
              label="回放剔除"
              value={view.excluded_by_replay.toLocaleString()}
              tone={view.excluded_by_replay > 0 ? 'text-rose-700' : undefined}
            />
          </div>
          {view.as_of && view.excluded_by_replay > 0 && (
            <p className="text-xs leading-5 text-slate-600">
              直接拿今天的 {view.latest_member_count.toLocaleString()} 只做 {view.as_of} 的筛选，会多出{' '}
              {view.excluded_by_replay.toLocaleString()} 只当时不可选的产品——这部分就是产品池层面的未来信息。
            </p>
          )}
          {view.sample.length > 0 && (
            <p className={`${NUM} text-xs text-slate-600`}>
              示例：{view.sample.map((item) => `${item.code} ${item.name}`).join('、')}
            </p>
          )}
          {view.warnings.map((text) => (
            <p key={text} className="text-xs text-rose-700">
              {text}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}


function LagHistogram({ dataset }: { dataset: PitDatasetAudit }) {
  const buckets = dataset.lag.histogram
  const total = buckets.reduce((sum, item) => sum + item.rows, 0)
  if (!total) {
    return (
      <p className="text-xs text-slate-600">
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
          <span className={`${NUM} w-12 shrink-0 text-right text-slate-600`}>{item.bucket} 天</span>
          <div className="h-3.5 min-w-0 flex-1 rounded-lg bg-slate-100">
            <div
              className="h-full rounded-lg bg-accent-500"
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
  const { refresh: refreshSystemContext, temporary, label: viewLabel, applyOverride } = useResearchContext()
  const [pitSettings, setPitSettings] = useState<PitSettingsPayload | null>(null)
  const [applying, setApplying] = useState(false)
  const [applyMessage, setApplyMessage] = useState('')
  // Both disclosures sit inside the step whose subject they serve. Collapsed by
  // default because the common path is "pick a version and apply".
  const [showUniverse, setShowUniverse] = useState(false)
  const [showReleases, setShowReleases] = useState(false)
  // The new-version form. The research day is defined *here*, on the version,
  // not as a second platform setting — one PIT concept, not two.
  const [newAsOf, setNewAsOf] = useState('')
  const [newMode, setNewMode] = useState<RunMode>('RESEARCH')
  // One form for both jobs. A separate edit dialog would be a second place to
  // keep the same four fields in step.
  const [editingId, setEditingId] = useState('')
  const [audit, setAudit] = useState<PitAudit | null>(null)
  const [releases, setReleases] = useState<DataRelease[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [releaseName, setReleaseName] = useState('')
  const [releaseNote, setReleaseNote] = useState('')
  const [sealing, setSealing] = useState(false)
  // Refusals arrive here too — a 409 「正在被全平台使用」 or a 405 from a backend
  // that predates the route — so the banner has to be able to look like a
  // failure. Rendered in the same grey as a success it reads as "nothing
  // happened", which is precisely how a rejected delete gets reported as a bug.
  const [message, setMessage] = useState<{ text: string; bad?: boolean } | null>(null)
  const failure = (reason: unknown) => ({
    text: reason instanceof Error ? reason.message : String(reason),
    bad: true,
  })

  const adoptSettings = (payload: PitSettingsPayload) => {
    setPitSettings(payload)
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

  // The scan runs behind the request, so the page has to come back for the
  // result. Polling only while something is pending keeps it off the idle path.
  const scanning = audit?.scan?.state === 'running'
  useEffect(() => {
    if (!scanning) return undefined
    const timer = window.setInterval(() => {
      fetchPitAudit({}).then(setAudit).catch(() => undefined)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [scanning])

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

  const activeId = pitSettings?.settings.active_release_id ?? ''

  // A new version pins today's files; an edited one still answers for the
  // vintage it sealed, and that vintage did not grow because the disk did.
  const maxAsOf = editingId
    ? (releases.find((item) => item.id === editingId)?.summary.available_through ?? null)
    : (audit?.summary.available_through ?? null)

  // Whichever day is currently in play: the one being defined, else the one the
  // selected version stands on. Prose cannot make "stand on 2009-12-31"
  // concrete; a count can.
  const previewAsOf = showReleases ? newAsOf : (asOf ?? '')
  const [universePreview, setUniversePreview] = useState<PitUniverseView | null>(null)
  useEffect(() => {
    if (!previewAsOf) {
      setUniversePreview(null)
      return
    }
    const controller = new AbortController()
    fetchPitUniverse({ kind: 'fund', asOf: previewAsOf, signal: controller.signal })
      .then(setUniversePreview)
      .catch(() => {
        if (!controller.signal.aborted) setUniversePreview(null)
      })
    return () => controller.abort()
  }, [previewAsOf])

  const onApply = (releaseId: string | null, asOfValue: string | null, mode: RunMode | null) => {
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

  const resetForm = () => {
    setEditingId('')
    setReleaseName('')
    setReleaseNote('')
    setNewAsOf('')
    setNewMode('RESEARCH')
  }

  const startEdit = (release: DataRelease) => {
    setEditingId(release.id)
    setReleaseName(release.name)
    setReleaseNote(release.note ?? '')
    setNewAsOf(release.as_of ?? '')
    setNewMode(release.run_mode === 'STRICT_PIT' ? 'STRICT_PIT' : 'RESEARCH')
    setShowReleases(true)
    setMessage(null)
  }

  const onSeal = () => {
    const name = releaseName.trim()
    if (!name) {
      setMessage({ text: '请先填写数据版本名称', bad: true })
      return
    }
    setSealing(true)
    setMessage(null)
    const request = editingId
      ? updateDataRelease(editingId, { name, note: releaseNote.trim(), asOf: newAsOf || null, runMode: newMode })
      : createDataRelease({ name, note: releaseNote.trim(), asOf: newAsOf || null, runMode: newMode })
    const wasEditing = Boolean(editingId)
    request
      .then((release) => {
        setReleases((current) =>
          wasEditing
            ? current.map((item) => (item.id === release.id ? release : item))
            : [release, ...current],
        )
        // Creating or editing a version and switching the platform onto it are
        // separate, deliberate acts — so leave the platform where it is.
        resetForm()
        // Refetch so the picker shows the version with its own口径.
        fetchPitSettings().then(setPitSettings).catch(() => undefined)
        setMessage({
          text: wasEditing
            ? `已改好版本「${release.name}」，现在站在 ${release.as_of ?? '数据最后一天'}。在下面的列表里点「应用到全平台」才会生效。`
            : `已建好版本「${release.name}」，站在 ${release.as_of ?? '数据最后一天'}。在下面的列表里点「应用到全平台」才会生效。`,
        })
        setSealing(false)
      })
      .catch((reason: unknown) => {
        setMessage(failure(reason))
        setSealing(false)
      })
  }

  const onDelete = (release: DataRelease) => {
    if (!window.confirm(`删除版本「${release.name}」？它钉住的数据指纹会一起消失，引用过它的结论将无法复现。`)) return
    setMessage(null)
    deleteDataRelease(release.id)
      .then(() => {
        setReleases((current) => current.filter((item) => item.id !== release.id))
        if (editingId === release.id) resetForm()
        fetchPitSettings().then(setPitSettings).catch(() => undefined)
        setMessage({ text: `已删除版本「${release.name}」。` })
      })
      .catch((reason: unknown) => setMessage(failure(reason)))
  }

  const summary = audit?.summary

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">PIT 时点快照</h2>
          <p className="mt-0.5 text-xs text-slate-600">
            一个 PIT 版本 = <strong>站在哪一天 + 用哪一批数据 + 有多严格</strong>。选一个版本应用到全平台，所有页面都按它展示；各功能页可在顶栏临时换版本，只影响自己的标签页。
          </p>
        </div>
        <button
          type="button"
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          onClick={() => load(true)}
          disabled={loading}
        >
          {loading ? '扫描中…' : '重新扫描'}
        </button>
      </div>

      {error && <p className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800">{error}</p>}

      {/* Otherwise this page shows the system口径 while the rest of the tab shows
          something else, and the numbers look inexplicably inconsistent. */}
      {temporary && (
        <p
          className="flex flex-wrap items-center gap-2 rounded-lg border border-accent-200 bg-accent-50 px-3 py-2 text-xs text-accent-900"
          data-testid="pit-override-notice"
        >
          <span>本标签页正在用临时口径查看数据：<strong>{viewLabel}</strong>。系统级设置不受影响。</span>
          <button type="button" className="underline hover:no-underline" onClick={() => applyOverride(null)}>
            恢复跟随系统默认
          </button>
        </p>
      )}

      {/* One concept: a version. It carries the day it stands on, the vintage
          that answers, and how strict — so choosing a version sets the whole
          口径 in one act. It used to be three steps, which meant a user could
          name a version "2015前数据" and still be standing on today. */}
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
                <span className="ml-2 text-slate-600">更新于 {pitSettings.settings.updated_at}</span>
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
            <label className="text-xs font-semibold text-slate-800" htmlFor="pit-release">
              用哪个 PIT 版本？
            </label>
            <p className="mt-0.5 text-xs leading-5 text-slate-600">
              一个版本就是<strong>一整套口径</strong>：站在哪一天、用哪一批数据、有多严格。选定后全平台按它展示；
              各功能页可在顶栏临时换版本，只影响自己的标签页。
            </p>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <select
                id="pit-release"
                className="w-96 rounded-lg border border-slate-300 px-2 py-1 text-xs disabled:bg-slate-100 disabled:text-slate-600"
                value={activeId}
                disabled={applying}
                onChange={(event) => onApply(event.target.value || null, null, null)}
              >
                <option value="">不用 PIT · 使用全部磁盘数据</option>
                {(pitSettings?.available_releases ?? []).map((release) => (
                  <option key={release.id} value={release.id}>
                    {release.name} · 站在 {release.as_of ?? '—'}
                    {release.run_mode === 'STRICT_PIT' ? ' · 严格' : ''}
                  </option>
                ))}
              </select>
              {/* Choosing a version and creating one are the same subject, so the
                  form lives inside this step rather than in a section of its own. */}
              <button
                type="button"
                className="text-xs text-slate-600 underline hover:no-underline"
                aria-expanded={showReleases}
                onClick={() => setShowReleases((value) => !value)}
              >
                {showReleases ? '收起版本管理' : `新建版本 / 管理（已有 ${releases.length} 个）`}
              </button>
            </div>
            {!pitSettings?.can_apply && !showReleases && (
              <p className="mt-1 text-xs text-slate-600">
                还没有任何版本。<strong>要用 PIT，先建一个版本</strong>，在里面写明站在哪一天。
              </p>
            )}

            {/* The consequence of the day in play, printed right under it. */}
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs leading-5">
              {previewAsOf && universePreview ? (
                <span className="text-slate-700">
                  站在 <strong className={NUM}>{previewAsOf}</strong>，可选基金{' '}
                  <strong className={NUM}>{universePreview.member_count.toLocaleString()}</strong> 只
                  {universePreview.excluded_by_replay > 0 && (
                    <>
                      ，比今天的表少{' '}
                      <strong className={NUM}>{universePreview.excluded_by_replay.toLocaleString()}</strong> 只
                    </>
                  )}
                </span>
              ) : (
                <span className="text-slate-600">选一个版本（或填一个研究日）后，这里会显示当天有多少产品可选。</span>
              )}
              <button
                type="button"
                className="text-slate-600 underline hover:no-underline"
                aria-expanded={showUniverse}
                onClick={() => setShowUniverse((value) => !value)}
              >
                {showUniverse ? '收起产品域明细' : '查看产品域明细'}
              </button>
            </div>
            {showUniverse && <UniversePanel asOf={previewAsOf} />}

            {showReleases && (
              <div className="mt-2 overflow-hidden rounded-xl border border-slate-200 bg-white">
                <div className="space-y-2.5 border-b border-slate-200 px-2.5 py-2.5">
                  <p className="text-xs leading-5 text-slate-600">
                    {editingId ? (
                      <>
                        正在<strong>修改</strong>下表中选中的版本：改的是它的口径（哪一天、多严格、名字），
                        它钉住的数据指纹不变。
                      </>
                    ) : (
                      <>
                        新建一个版本：<strong>钉住当前磁盘上的数据</strong>，并写明站在哪一天、多严格。
                      </>
                    )}
                  </p>
                  <div>
                    <label className="text-xs font-semibold text-slate-800" htmlFor="pit-new-as-of">
                      站在哪一天看？
                    </label>
                    <p className="mt-0.5 text-xs leading-5 text-slate-600">
                      这个版本只使用该日<strong>当时已经公开</strong>的数据：之后才公布的净值、之后才上市的产品，一律不可见。
                      留空表示站在这批数据的最后一天。
                    </p>
                    <div className="mt-1.5 flex flex-wrap items-center gap-2">
                      <input
                        id="pit-new-as-of"
                        type="date"
                        className={`${NUM} w-44 rounded-lg border border-slate-300 px-2 py-1 text-xs`}
                        value={newAsOf}
                        max={maxAsOf ?? undefined}
                        onChange={(event) => setNewAsOf(event.target.value)}
                      />
                      <button
                        type="button"
                        className="rounded-xl border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 hover:bg-slate-50"
                        onClick={() => setNewAsOf(maxAsOf ?? '')}
                      >
                        数据最新一天
                      </button>
                      {maxAsOf && newAsOf > maxAsOf && (
                        <span className="text-xs text-rose-700">晚于当前数据的可得截止日 {maxAsOf}</span>
                      )}
                    </div>
                  </div>

                  <div>
                    <span className="text-xs font-semibold text-slate-800">有多严格？</span>
                    <div className="mt-1.5 flex flex-wrap gap-2">
                      {(['RESEARCH', 'STRICT_PIT'] as RunMode[]).map((mode) => (
                        <button
                          key={mode}
                          type="button"
                          className={`rounded-lg border px-2.5 py-1.5 text-left text-xs leading-4 ${
                            newMode === mode
                              ? 'border-slate-900 bg-white font-semibold text-slate-900'
                              : 'border-slate-300 bg-white/60 text-slate-600 hover:bg-white'
                          } disabled:opacity-40`}
                          disabled={mode === 'STRICT_PIT' && !newAsOf}
                          onClick={() => setNewMode(mode)}
                        >
                          <span className="block">{mode === 'RESEARCH' ? '研究模式' : '严格 PIT'}</span>
                          <span className="mt-0.5 block font-normal text-slate-600">
                            {mode === 'RESEARCH'
                              ? '缺公告日的数据按净值日近似，并在结果上标注'
                              : '拿不出时点证明的数据直接拒绝使用'}
                          </span>
                        </button>
                      ))}
                    </div>
                    {!newAsOf && (
                      <p className="mt-1 text-xs text-slate-600">严格 PIT 需要一个研究日——没有截止日就没什么可执行。</p>
                    )}
                  </div>

                  <div className="flex flex-wrap items-end gap-2">
                    <label className="text-xs">
                      <span className="mb-1 block font-medium text-slate-600">版本名称</span>
                      <input
                        className="w-52 rounded-lg border border-slate-300 px-2 py-1 text-xs"
                        value={releaseName}
                        onChange={(event) => setReleaseName(event.target.value)}
                        placeholder="例如 站在 2014 年末的投前基线"
                        aria-label="数据版本名称"
                      />
                    </label>
                    <label className="min-w-0 flex-1 text-xs">
                      <span className="mb-1 block font-medium text-slate-600">备注</span>
                      <input
                        className="w-full rounded-lg border border-slate-300 px-2 py-1 text-xs"
                        value={releaseNote}
                        onChange={(event) => setReleaseNote(event.target.value)}
                        placeholder="选填：这个版本用来回答什么问题"
                        aria-label="数据版本备注"
                      />
                    </label>
                    <button
                      type="button"
                      className="rounded-lg bg-accent-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-accent-600 disabled:opacity-50"
                      onClick={onSeal}
                      disabled={sealing}
                    >
                      {sealing ? '保存中…' : editingId ? '保存修改' : '创建版本'}
                    </button>
                    {editingId && (
                      <button
                        type="button"
                        className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
                        onClick={resetForm}
                      >
                        取消
                      </button>
                    )}
                  </div>
                </div>
                {message && (
                  <p
                    className={`px-2.5 py-2 text-xs ${message.bad ? 'font-medium text-rose-700' : 'text-slate-700'}`}
                    role={message.bad ? 'alert' : undefined}
                  >
                    {message.text}
                  </p>
                )}
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[820px] border-collapse text-xs">
                    <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-600">
                      <tr>
                        <th scope="col" className="px-2.5 py-1.5 font-semibold">版本</th>
                        <th scope="col" className="px-2.5 py-1.5 font-semibold">站在哪一天</th>
                        <th scope="col" className="px-2.5 py-1.5 font-semibold">模式</th>
                        <th scope="col" className="px-2.5 py-1.5 font-semibold">创建时间</th>
                        <th scope="col" className="px-2.5 py-1.5 text-right font-semibold">表数</th>
                        <th scope="col" className="px-2.5 py-1.5 text-right font-semibold">总行数</th>
                        <th scope="col" className="px-2.5 py-1.5 font-semibold">指纹</th>
                        <th scope="col" className="px-2.5 py-1.5 text-right font-semibold">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {releases.length === 0 && (
                        <tr>
                          <td className="px-2.5 py-2.5 text-slate-600" colSpan={8}>
                            还没有版本。建一个写明研究日的版本，之后每次数据刷新再建一个，就能回答“这个结论站在哪一天、用的是哪批数据”。
                          </td>
                        </tr>
                      )}
                      {releases.map((release) => {
                        // Deleting the applied version is refused by design (409):
                        // it would drop every page back to no-PIT. Offering the
                        // button anyway is how that refusal gets reported as
                        // 删除失败.
                        const inUse = pitSettings?.settings.active_release_id === release.id
                        return (
                        <tr
                          key={release.id}
                          className={`border-t border-slate-100 hover:bg-slate-50 ${
                            release.id === editingId ? 'bg-amber-50' : inUse ? 'bg-accent-50/60' : ''
                          }`}
                        >
                          <td className="px-2.5 py-1.5">
                            <span className="font-medium text-slate-900">{release.name}</span>
                            <div className={`${NUM} text-xs text-slate-600`}>{release.id}</div>
                            {release.note && <div className="text-xs text-slate-600">{release.note}</div>}
                            {release.updated_at && (
                              <div className={`${NUM} text-xs text-amber-700`}>口径已改于 {release.updated_at}</div>
                            )}
                          </td>
                          <td className={`${NUM} px-2.5 py-1.5 font-medium text-slate-900`}>
                            {release.as_of ?? release.summary.available_through ?? '—'}
                            {!release.as_of && <span className="ml-1 text-xs font-normal text-slate-600">数据最后一天</span>}
                          </td>
                          <td className="px-2.5 py-1.5 text-slate-700">
                            {release.run_mode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}
                          </td>
                          <td className={`${NUM} px-2.5 py-1.5 text-slate-700`}>{release.created_at}</td>
                          <td className={`${NUM} px-2.5 py-1.5 text-right text-slate-800`}>
                            {release.tables.length}
                          </td>
                          <td className={`${NUM} px-2.5 py-1.5 text-right text-slate-800`}>
                            {release.summary.total_rows.toLocaleString()}
                          </td>
                          <td className={`${NUM} px-2.5 py-1.5 text-slate-600`} title={release.release_fingerprint}>
                            {release.release_fingerprint.slice(0, 12)}
                          </td>
                          <td className="px-2.5 py-1.5 text-right whitespace-nowrap">
                            {inUse ? (
                              // Says why 删除 next to it is dead: a disabled button's
                              // title never renders, so the reason has to be visible.
                              <span className="rounded-lg bg-accent-100 px-2 py-0.5 text-xs font-medium text-accent-800">
                                全平台使用中
                              </span>
                            ) : (
                              <button
                                type="button"
                                className="rounded-lg bg-slate-900 px-2 py-0.5 text-xs font-semibold text-white hover:bg-slate-700 disabled:opacity-50"
                                disabled={applying}
                                onClick={() => onApply(release.id, null, null)}
                              >
                                {applying ? '应用中…' : '应用到全平台'}
                              </button>
                            )}
                            <button
                              type="button"
                              className="ml-1.5 rounded-lg border border-slate-200 px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-100"
                              onClick={() => startEdit(release)}
                            >
                              编辑
                            </button>
                            <button
                              type="button"
                              disabled={inUse}
                              className="ml-1.5 rounded-lg border border-rose-200 px-2 py-0.5 text-xs text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-600 disabled:hover:bg-transparent"
                              onClick={() => onDelete(release)}
                            >
                              删除
                            </button>
                          </td>
                        </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
        {applyMessage && <p className="border-t border-black/5 px-3 py-2 text-xs text-slate-800">{applyMessage}</p>}
      </section>

      {/* Second block: everything you read rather than set. Naming it is what
          makes the page two things instead of five loose sections. */}
      <div className="flex items-baseline gap-2 pt-1">
        <h3 className="text-sm font-semibold text-slate-900">数据能力体检</h3>
        <span className="text-xs text-slate-600">上面的口径能不能成立，取决于下面这些表能证明什么</span>
      </div>

      <div className="grid grid-cols-2 divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white sm:grid-cols-3 lg:grid-cols-5">
        <Kpi label="声明数据集" value={summary ? String(summary.declared) : '—'} />
        <Kpi label="A 级 严格" value={summary ? String(summary.grade_a) : '—'} tone="text-emerald-700" />
        <Kpi label="B 级 近似" value={summary ? String(summary.grade_b) : '—'} tone="text-amber-700" />
        <Kpi label="C 级 无 PIT" value={summary ? String(summary.grade_c) : '—'} tone="text-rose-700" />
        <Kpi label="A/B 数据可得至" value={summary?.available_through ?? '—'} />
      </div>

      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white">
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
          <h3 className="text-sm font-semibold text-slate-900">数据集 PIT 能力</h3>
          <span className="text-xs text-slate-600">
            {scanning
              ? `正在后台扫描 ${audit?.summary.pending ?? 0} 个数据集…扫完自动刷新`
              : audit?.scan?.state === 'failed'
                ? `扫描失败：${audit.scan.error ?? '未知原因'}`
                : '点击一行查看滞后分布'}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] border-collapse text-xs">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-600">
              <tr>
                <th scope="col" className="px-3 py-2 font-semibold">数据集</th>
                <th scope="col" className="px-3 py-2 font-semibold">事件时间</th>
                <th scope="col" className="px-3 py-2 font-semibold">可得时间</th>
                <th scope="col" className="px-3 py-2 text-right font-semibold">行数</th>
                <th scope="col" className="px-3 py-2 text-right font-semibold">覆盖率</th>
                <th scope="col" className="px-3 py-2 text-right font-semibold">P50</th>
                <th scope="col" className="px-3 py-2 text-right font-semibold">P95</th>
                <th scope="col" className="px-3 py-2 font-semibold">可得至</th>
                <th scope="col" className="px-3 py-2 font-semibold">等级</th>
              </tr>
            </thead>
            <tbody>
              {(audit?.datasets ?? []).map((item) => (
                <tr
                  key={item.dataset_id}
                  className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                    item.dataset_id === selectedId ? 'bg-accent-50/60' : ''
                  } ${item.present ? '' : 'text-slate-600'}`}
                  onClick={() => setSelectedId(item.dataset_id)}
                >
                  <td className="px-3 py-1.5">
                    <span className="font-medium text-slate-900">{item.label}</span>
                    {!item.present && <span className="ml-2 text-xs text-slate-600">文件缺失</span>}
                    <div className="text-xs text-slate-600">{item.file}</div>
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
                    <GradeChip grade={item.grade} label={item.grade ?? '扫描中'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {selected && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="rounded-xl border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-900">{selected.label} · 公告滞后分布</h3>
              <GradeChip grade={selected.grade} label={selected.grade_label} />
            </div>
            <div className="mt-3">
              <LagHistogram dataset={selected} />
            </div>
            {selected.lag.negative_rows > 0 && (
              <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-2 py-1.5 text-xs text-rose-800">
                {selected.lag.negative_rows.toLocaleString()} 行的公告日早于事件日——这是数据质量问题，不是时点特性。
              </p>
            )}
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-3">
            <h3 className="text-sm font-semibold text-slate-900">在当前研究上下文下</h3>
            <dl className="mt-3 space-y-1.5 text-xs">
              <div className="flex justify-between gap-3">
                <dt className="text-slate-600">研究日</dt>
                <dd className={`${NUM} text-slate-900`}>{asOf ?? '未指定'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-600">运行模式</dt>
                <dd className="text-slate-900">{runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-600">数据版本</dt>
                <dd className={`${NUM} text-slate-900`}>{dataReleaseId ?? '未指定'}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-600">事件时间范围</dt>
                <dd className={`${NUM} text-slate-900`}>
                  {selected.event_range.start ?? '—'} ~ {selected.event_range.end ?? '—'}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-600">可得时间范围</dt>
                <dd className={`${NUM} text-slate-900`}>
                  {selected.availability_range.start ?? '—'} ~ {selected.availability_range.end ?? '—'}
                </dd>
              </div>
              {runMode === 'STRICT_PIT' && selected.grade === 'C' && (
                <p className="rounded-lg border border-rose-200 bg-rose-50 px-2 py-1.5 text-xs text-rose-800">
                  严格 PIT 模式下该数据集被禁用，引用它的计算会直接报错而不是静默降级。
                </p>
              )}
              {visibility && <p className="text-xs text-slate-600">{visibility.reason}</p>}
            </dl>
            <p className="mt-3 border-t border-slate-100 pt-2 text-xs leading-5 text-slate-600">{selected.note}</p>
          </section>
        </div>
      )}
    </div>
  )
}
