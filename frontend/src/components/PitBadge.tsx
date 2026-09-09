import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useResearchContext } from '../app/ResearchContext'
import type { RunMode } from '../services/pit'

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
  const { settings, override, overrideRelease, noPit, label, asOf, runMode, loading, applyOverride } =
    useResearchContext()
  const [open, setOpen] = useState(false)
  const [draftDay, setDraftDay] = useState(override?.asOf ?? '')
  if (loading && !settings) return null

  const temporary = Boolean(override)
  const name = temporary
    ? override?.off
      ? '无口径'
      : (overrideRelease?.name ?? '最新数据')
    : (settings?.release?.name ?? '最新数据')
  const strict = runMode === 'STRICT_PIT'
  const tone = temporary
    ? 'border-sky-400/60 bg-sky-400/10 text-sky-200'
    : noPit
      ? 'border-slate-600 text-slate-300'
      : strict
        ? 'border-amber-400/60 bg-amber-400/10 text-amber-200'
        : 'border-emerald-400/60 bg-emerald-400/10 text-emerald-200'

  const releases = settings?.available_releases ?? []
  const view = (releaseId: string, mode: RunMode) =>
    applyOverride({ off: false, releaseId, asOf: null, runMode: mode })

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        className={`flex shrink-0 items-center gap-1.5 rounded border px-2 py-1 text-xs font-medium tabular-nums ${tone}`}
        title={`${label}；点击切换本页查看口径`}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        data-testid="pit-badge"
      >
        <span className="opacity-70">PIT</span>
        {temporary && <span className="opacity-70">临时</span>}
        {noPit ? <span>无口径</span> : <><span>{name}</span><span className="opacity-70">{asOf}</span>{strict && <span>严格</span>}</>}
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
            className="absolute right-0 z-50 mt-2 w-80 rounded border border-slate-200 bg-white p-3 text-left text-xs text-slate-700 shadow-lg"
            data-testid="pit-switcher"
          >
            <p className="font-semibold text-slate-900">本页查看口径</p>
            <p className="mt-1 text-slate-500">
              系统默认：{settings?.effective.label ?? '无 PIT 口径'}。此处的切换只影响你自己的这个标签页，不改变平台设置。
            </p>

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
                <div key={release.id} className="rounded border border-slate-200 px-2 py-1.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-medium text-slate-900">{release.name}</span>
                    <span className="tabular-nums text-slate-500">{release.available_through ?? '无可得截止日'}</span>
                  </div>
                  <div className="mt-1 flex gap-2">
                    <button
                      type="button"
                      className={`rounded border px-2 py-0.5 ${
                        override && !override.off && override.releaseId === release.id && override.runMode === 'RESEARCH'
                          ? 'border-sky-300 bg-sky-50 text-sky-900'
                          : 'border-slate-200 hover:bg-slate-50'
                      }`}
                      disabled={!release.available_through}
                      onClick={() => view(release.id, 'RESEARCH')}
                    >
                      研究模式
                    </button>
                    <button
                      type="button"
                      className={`rounded border px-2 py-0.5 ${
                        override && !override.off && override.releaseId === release.id && override.runMode === 'STRICT_PIT'
                          ? 'border-sky-300 bg-sky-50 text-sky-900'
                          : 'border-slate-200 hover:bg-slate-50'
                      }`}
                      disabled={!release.available_through}
                      onClick={() => view(release.id, 'STRICT_PIT')}
                    >
                      严格 PIT
                    </button>
                  </div>
                </div>
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
