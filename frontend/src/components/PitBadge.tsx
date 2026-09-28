import { researchMessage } from '../i18n/researchMessages'
import { systemText, useI18n } from '../i18n/runtime'
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
  useI18n()
  const { settings, override, temporary, overrideRelease, noPit, label, asOf, runMode, loading, error, refresh, applyOverride } =
    useResearchContext()
  const [open, setOpen] = useState(false)
  const [draftDay, setDraftDay] = useState(override?.asOf ?? '')
  // Retain the initial loading placeholder; a failed read must remain visible.
  if (loading && !settings && !error && !override) return null
  const unknown = noPit === null || runMode === null
  const systemLabel = loading ? systemText('preInvestment.pitBadge.loading') : error ? systemText('preInvestment.pitBadge.unknownLoadFailed') : settings?.effective.label ?? systemText('preInvestment.pitBadge.unknown')

  // The badge answers one question — on or off, and if on, which口径. The day,
  // the mode and the system default are one click away in the popover.
  const name = override && !override.off
    ? (overrideRelease?.name ?? asOf ?? systemText('preInvestment.pitBadge.latestData'))
    : (settings?.release?.name ?? asOf ?? systemText('preInvestment.pitBadge.latestData'))
  const strict = runMode === 'STRICT_PIT'
  const tone = unknown
    ? 'border-amber-400/60 bg-amber-400/10 text-amber-200'
    : temporary
    ? 'border-accent-400/60 bg-accent-400/10 text-accent-200'
    : noPit
      ? 'border-slate-600 text-slate-200'
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
        className={`flex shrink-0 items-center gap-1.5 rounded-lg border px-2 py-1 text-xs font-medium tabular-nums ${tone}`}
        title={systemText('preInvestment.pitBadge.clickToChangeThisTabSData', { p0: researchMessage(label), p1: strict ? systemText('preInvestment.pitBadge.strictPit') : '' })}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        data-testid="pit-badge"
      >
        <span>{unknown ? systemText('preInvestment.pitBadge.pitContextUnknown') : noPit ? systemText('preInvestment.pitBadge.pitOff') : systemText('preInvestment.pitBadge.pitOn', { p0: name })}</span>
        {temporary && (
          <span className="group relative flex items-center" title="">
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-rose-700 text-xs font-bold leading-none text-white">
              !
            </span>
            <span className="pointer-events-none absolute right-0 top-full z-50 mt-1.5 hidden w-64 rounded-xl border border-slate-200 bg-white p-2 text-left text-xs font-normal leading-4 text-slate-700 shadow-lg group-hover:block">
              {loading || error ? systemText('preInvestment.pitBadge.aTemporaryContextIsSetForThis') : systemText('preInvestment.pitBadge.thisTabSTemporaryContextDiffersFrom', { p0: systemLabel })}
            </span>
          </span>
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-40 cursor-default"
            aria-label={systemText('preInvestment.pitBadge.closeDataContextSwitcher')}
            onClick={() => setOpen(false)}
          />
          <div
            className="absolute left-0 z-50 mt-2 w-80 max-w-[calc(100vw-2rem)] rounded-xl border border-slate-200 bg-white p-3 text-left text-xs text-slate-700 shadow-lg xl:left-auto xl:right-0"
            data-testid="pit-switcher"
          >
            <p className="font-semibold text-slate-900">{systemText('preInvestment.pitBadge.thisTabSDataContext')}</p>
            <p className="mt-1 text-slate-600">
              {systemText('preInvestment.pitBadge.systemDefault')}{systemLabel}{systemText('preInvestment.pitBadge.changesHereAffectOnlyYourCurrentTab')}</p>
            {error && <div role="alert" className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-2 text-amber-900">
              <p>{systemText('preInvestment.pitBadge.unableToLoadPitSettings')}{error}{systemText('preInvestment.pitBadge.theSystemContextIsUnconfirmedRetryActual')}</p>
              <button type="button" onClick={refresh} disabled={loading} className="mt-2 rounded-lg border border-amber-300 px-2 py-1 font-semibold disabled:opacity-50">{loading ? systemText('preInvestment.pitBadge.retrying') : systemText('preInvestment.pitBadge.retryLoadingPitContext')}</button>
            </div>}

            <div className="mt-2 space-y-1">
              <button
                type="button"
                className={`w-full rounded-lg border px-2 py-1.5 text-left ${
                  temporary ? 'border-slate-200 hover:bg-slate-50' : 'border-accent-300 bg-accent-50 font-medium text-accent-900'
                }`}
                onClick={() => applyOverride(null)}
                data-testid="pit-follow-system"
              >
                {systemText('preInvestment.pitBadge.followSystemDefault')}</button>

              {/* The commonest thing a reader wants is another day, not another
                  vintage — and until now the popover only offered vintages. */}
              <div className="rounded-lg border border-slate-200 px-2 py-1.5">
                <label className="font-medium text-slate-900" htmlFor="pit-badge-as-of">
                  {systemText('preInvestment.pitBadge.viewDataThroughADate')}</label>
                <div className="mt-1 flex items-center gap-2">
                  <input
                    id="pit-badge-as-of"
                    type="date"
                    className="w-36 rounded-lg border border-slate-300 px-1.5 py-0.5 tabular-nums"
                    value={draftDay}
                    onChange={(event) => setDraftDay(event.target.value)}
                  />
                  <button
                    type="button"
                    className="rounded-lg border border-slate-200 px-2 py-0.5 hover:bg-slate-50 disabled:opacity-40"
                    disabled={!draftDay}
                    onClick={() =>
                      applyOverride({ off: false, releaseId: null, asOf: draftDay, runMode: 'RESEARCH' })
                    }
                  >
                    {systemText('preInvestment.pitBadge.applyToThisTab')}</button>
                </div>
              </div>

              {releases.map((release) => (
                <button
                  key={release.id}
                  type="button"
                  className={`w-full rounded-lg border px-2 py-1.5 text-left ${
                    override && !override.off && override.releaseId === release.id
                      ? 'border-accent-300 bg-accent-50 text-accent-900'
                      : 'border-slate-200 hover:bg-slate-50'
                  } disabled:opacity-40`}
                  disabled={!release.available_through}
                  onClick={() => view(release.id)}
                >
                  <span className="flex items-baseline justify-between gap-2">
                    <span className="font-medium text-slate-900">{release.name}</span>
                    <span className="tabular-nums text-slate-600">{systemText('preInvestment.pitBadge.asOf') + " "}{release.as_of ?? '—'}</span>
                  </span>
                  <span className="mt-0.5 block font-normal text-slate-600">
                    {release.run_mode === 'STRICT_PIT' ? systemText('preInvestment.pitBadge.strictPit2') : systemText('preInvestment.pitBadge.researchMode')}
                  </span>
                </button>
              ))}

              <button
                type="button"
                className={`w-full rounded-lg border px-2 py-1.5 text-left ${
                  override?.off ? 'border-accent-300 bg-accent-50 font-medium text-accent-900' : 'border-slate-200 hover:bg-slate-50'
                }`}
                onClick={() => applyOverride({ off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' })}
                data-testid="pit-turn-off"
              >
                {systemText('preInvestment.pitBadge.turnOffPitViewAllOnDisk')}<span className="mt-0.5 block font-normal text-slate-600">{systemText('preInvestment.pitBadge.resultsAreForViewingOnlyAndAre')}</span>
              </button>
            </div>

            <Link
              to="/settings/pit-snapshots"
              className="mt-2 inline-block text-accent-700 underline hover:no-underline"
              onClick={() => setOpen(false)}
            >
              {systemText('preInvestment.pitBadge.manageDataVersionsAndSystemDefaults')}</Link>
          </div>
        </>
      )}
    </div>
  )
}
