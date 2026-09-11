import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useResearchContext } from '../app/ResearchContext'

/**
 *口径 indicator and per-tab口径 switch, in the existing header row.
 *
 * The system-level setting on `/settings/pit-snapshots` stays the default every
 * page displays against; nothing here writes it. What this does offer is the
 * temporary view — another vintage, or PIT off — because a default that cannot
 * be looked past stops being a default and becomes a wall. It borrows the
 * header's row rather than adding a band to every page.
 */
export default function PitBadge() {
  const { settings, override, temporary, overrideRelease, noPit, label, asOf, runMode, loading, error, refresh, applyOverride } =
    useResearchContext()
  const [open, setOpen] = useState(false)
  const [draftDay, setDraftDay] = useState(override?.asOf ?? '')
  // Retain the initial loading placeholder; a failed read must remain visible.
  if (loading && !settings && !error && !override) return null
  const unknown = noPit === null || runMode === null
  const systemLabel = loading ? '正在读取' : error ? '未知（读取失败）' : settings?.effective.label ?? '未知'

  // The badge answers one question — on or off, and if on, which口径. The day,
  // the mode and the system default are one click away in the popover.
  const name = override && !override.off
    ? (overrideRelease?.name ?? asOf ?? '最新数据')
    : (settings?.release?.name ?? asOf ?? '最新数据')
  const strict = runMode === 'STRICT_PIT'
  const tone = unknown
    ? 'border-amber-400/60 bg-amber-400/10 text-amber-200'
    : temporary
    ? 'border-sky-400/60 bg-sky-400/10 text-sky-200'
    : noPit
      ? 'border-slate-600 text-slate-300'
      : strict
        ? 'border-amber-400/60 bg-amber-400/10 text-amber-200'
        : 'border-emerald-400/60 bg-emerald-400/10 text-emerald-200'

  const releases = settings?.available_releases ?? []
  // A version carries its own day and mode, so viewing one needs nothing else.
  const view = (releaseId: string) =>
    applyOverride({ off: false, releaseId, asOf: null, runMode: null })

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        className={`flex shrink-0 items-center gap-1.5 rounded border px-2 py-1 text-xs font-medium tabular-nums ${tone}`}
        title={`${label}${strict ? '（严格 PIT）' : ''}；点击切换本页查看口径`}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        data-testid="pit-badge"
      >
        <span>{unknown ? 'PIT 口径未知' : noPit ? 'PIT 关闭' : `PIT 打开：${name}`}</span>
        {temporary && (
          <span className="group relative flex items-center" title="">
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-red-600 text-[10px] font-bold leading-none text-white">
              !
            </span>
            <span className="pointer-events-none absolute right-0 top-full z-50 mt-1.5 hidden w-64 rounded border border-slate-200 bg-white p-2 text-left text-[11px] font-normal leading-4 text-slate-700 shadow-lg group-hover:block">
              {loading || error ? '本页已指定临时口径；系统默认尚未确认，不能判断两者是否一致。' : `本页正在临时查看的口径与系统默认（${systemLabel}）不一致，只影响这个标签页；点开可跟随系统默认。`}
            </span>
          </span>
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-40 cursor-default"
            aria-label="关闭口径切换"
            onClick={() => setOpen(false)}
          />
          <div
            className="absolute left-0 z-50 mt-2 w-80 max-w-[calc(100vw-2rem)] rounded border border-slate-200 bg-white p-3 text-left text-xs text-slate-700 shadow-lg xl:left-auto xl:right-0"
            data-testid="pit-switcher"
          >
            <p className="font-semibold text-slate-900">本页查看口径</p>
            <p className="mt-1 text-slate-500">
              系统默认：{systemLabel}。此处的切换只影响你自己的这个标签页，不改变平台设置。
            </p>
            {error && <div role="alert" className="mt-2 rounded border border-amber-200 bg-amber-50 p-2 text-amber-900">
              <p>PIT 设置读取失败：{error}。未确认系统口径，请重试；实际结果以服务端返回的口径为准。</p>
              <button type="button" onClick={refresh} disabled={loading} className="mt-2 rounded border border-amber-300 px-2 py-1 font-semibold disabled:opacity-50">{loading ? '正在重试…' : '重试读取 PIT 口径'}</button>
            </div>}

            <div className="mt-2 space-y-1">
              <button
                type="button"
                className={`w-full rounded border px-2 py-1.5 text-left ${
                  temporary ? 'border-slate-200 hover:bg-slate-50' : 'border-sky-300 bg-sky-50 font-medium text-sky-900'
                }`}
                onClick={() => applyOverride(null)}
                data-testid="pit-follow-system"
              >
                跟随系统默认
              </button>

              {/* The commonest thing a reader wants is another day, not another
                  vintage — and until now the popover only offered vintages. */}
              <div className="rounded border border-slate-200 px-2 py-1.5">
                <label className="font-medium text-slate-900" htmlFor="pit-badge-as-of">
                  只看某一天为止
                </label>
                <div className="mt-1 flex items-center gap-2">
                  <input
                    id="pit-badge-as-of"
                    type="date"
                    className="w-36 rounded border border-slate-300 px-1.5 py-0.5 tabular-nums"
                    value={draftDay}
                    onChange={(event) => setDraftDay(event.target.value)}
                  />
                  <button
                    type="button"
                    className="rounded border border-slate-200 px-2 py-0.5 hover:bg-slate-50 disabled:opacity-40"
                    disabled={!draftDay}
                    onClick={() =>
                      applyOverride({ off: false, releaseId: null, asOf: draftDay, runMode: 'RESEARCH' })
                    }
                  >
                    应用到本页
                  </button>
                </div>
              </div>

              {releases.map((release) => (
                <button
                  key={release.id}
                  type="button"
                  className={`w-full rounded border px-2 py-1.5 text-left ${
                    override && !override.off && override.releaseId === release.id
                      ? 'border-sky-300 bg-sky-50 text-sky-900'
                      : 'border-slate-200 hover:bg-slate-50'
                  } disabled:opacity-40`}
                  disabled={!release.available_through}
                  onClick={() => view(release.id)}
                >
                  <span className="flex items-baseline justify-between gap-2">
                    <span className="font-medium text-slate-900">{release.name}</span>
                    <span className="tabular-nums text-slate-500">站在 {release.as_of ?? '—'}</span>
                  </span>
                  <span className="mt-0.5 block font-normal text-slate-500">
                    {release.run_mode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'}
                  </span>
                </button>
              ))}

              <button
                type="button"
                className={`w-full rounded border px-2 py-1.5 text-left ${
                  override?.off ? 'border-sky-300 bg-sky-50 font-medium text-sky-900' : 'border-slate-200 hover:bg-slate-50'
                }`}
                onClick={() => applyOverride({ off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' })}
                data-testid="pit-turn-off"
              >
                关闭 PIT · 查看全部磁盘数据
                <span className="mt-0.5 block font-normal text-slate-500">结果不具备时点可复现性，仅用于查看。</span>
              </button>
            </div>

            <Link
              to="/settings/pit-snapshots"
              className="mt-2 inline-block text-sky-700 underline hover:no-underline"
              onClick={() => setOpen(false)}
            >
              管理数据版本与系统默认口径
            </Link>
          </div>
        </>
      )}
    </div>
  )
}
