import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { fetchPitSettings, type PitReleaseSummary, type PitSettingsPayload, type RunMode } from '../services/pit'
import { getPitOverride, pageReload, setPitOverride, type PitViewOverride } from '../services/pitOverride'

interface ResearchContextValue {
  /** Server-held system setting; null until the first load resolves. */
  settings: PitSettingsPayload | null
  /** This tab's temporary viewing choice, or null while it follows the system. */
  override: PitViewOverride | null
  /** True only when this tab's口径 actually differs from the system default. */
  temporary: boolean
  /** The release this tab is viewing when it overrides the system one. */
  overrideRelease: PitReleaseSummary | null
  /** True while no release is in effect — every row on disk is in play. */
  noPit: boolean
  /** One short string a result footnote can print verbatim. */
  label: string
  asOf: string | null
  runMode: RunMode
  loading: boolean
  error: string
  /** Change this tab's viewing口径; null restores the system default. */
  applyOverride: (next: PitViewOverride | null) => void
  /** Re-read after the PIT settings page writes. */
  refresh: () => void
}

const ResearchContextContext = createContext<ResearchContextValue | null>(null)

const NO_PIT_LABEL = '无 PIT 口径 · 使用全部磁盘数据'

/**
 * The PIT口径 is a system-level setting, so it is read from the server rather
 * than remembered per browser: keeping it in `localStorage` let two analysts
 * open the same page, see different numbers, and never notice.
 *
 * `/settings/pit-snapshots` remains the one place that changes that setting,
 * which is what makes "all pages display against the applied version" true
 * instead of aspirational. On top of it sits a per-tab viewing override — the
 * default is a default, not a lock, and a reader must be able to look at another
 * vintage without republishing the platform口径 for everyone else.
 */
export function ResearchContextProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<PitSettingsPayload | null>(null)
  const [override, setOverride] = useState<PitViewOverride | null>(() => getPitOverride())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    fetchPitSettings(controller.signal)
      .then((payload) => {
        setSettings(payload)
        setError('')
        setLoading(false)
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        // Losing the badge is a nuisance; blocking the page would be worse, and
        // the backend already falls back to no-PIT on its own.
        setSettings(null)
        setError(reason instanceof Error ? reason.message : String(reason))
        setLoading(false)
      })
    return () => controller.abort()
  }, [reloadToken])

  const refresh = useCallback(() => setReloadToken((token) => token + 1), [])

  const applyOverride = useCallback((next: PitViewOverride | null) => {
    setOverride(setPitOverride(next))
    pageReload.run()
  }, [])

  const overrideRelease = useMemo(() => {
    if (!override || override.off || !settings) return null
    return settings.available_releases.find((item) => item.id === override.releaseId) ?? null
  }, [override, settings])

  useEffect(() => {
    // The viewed release was deleted out from under this tab. Left in place its
    // header would make every data request fail, including the one that would
    // let the user recover, so drop it back to the system default.
    if (!settings || !override || override.off) return
    // A day-only override pins no release, so there is nothing to go missing.
    if (!override.releaseId) return
    if (settings.available_releases.some((item) => item.id === override.releaseId)) return
    setOverride(setPitOverride(null))
  }, [override, settings])

  const value = useMemo<ResearchContextValue>(() => {
    const system = settings?.effective
    let noPit = system?.no_pit ?? true
    let label = system?.label ?? NO_PIT_LABEL
    let asOf = system?.as_of ?? null
    let runMode: RunMode = system?.run_mode ?? 'RESEARCH'

    if (override?.off) {
      noPit = true
      label = NO_PIT_LABEL
      asOf = null
      runMode = 'RESEARCH'
    } else if (override && (overrideRelease || override.asOf)) {
      noPit = false
      // A version carries its own day and mode; a bare day is research mode.
      runMode = override.runMode ?? overrideRelease?.run_mode ?? 'RESEARCH'
      asOf = override.asOf ?? overrideRelease?.as_of ?? overrideRelease?.available_through ?? null
      const mode = runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'
      const vintage = overrideRelease?.name ?? '最新数据（未封版）'
      label = `站在 ${asOf} · ${vintage} · ${mode}`
    }

    // An override that resolves to the same口径 as the system default is not a
    // deviation: picking the version already applied must not raise a warning.
    const temporary =
      Boolean(override) &&
      (noPit !== (system?.no_pit ?? true) ||
        asOf !== (system?.as_of ?? null) ||
        runMode !== (system?.run_mode ?? 'RESEARCH') ||
        (override?.off ? null : (overrideRelease?.id ?? null)) !==
          (settings?.settings.active_release_id ?? null))
    if (temporary) label = `临时口径 · ${label}`

    return {
      settings,
      override,
      temporary,
      overrideRelease,
      noPit,
      label,
      asOf,
      runMode,
      loading,
      error,
      applyOverride,
      refresh,
    }
  }, [applyOverride, error, loading, override, overrideRelease, refresh, settings])

  return <ResearchContextContext.Provider value={value}>{children}</ResearchContextContext.Provider>
}

export function useResearchContext(): ResearchContextValue {
  const context = useContext(ResearchContextContext)
  if (!context) throw new Error('useResearchContext must be used inside ResearchContextProvider')
  return context
}

/**
 * The研究日 alone, for pages that only label it.
 *
 * Unlike `useResearchContext` this tolerates no provider: a page that merely
 * annotates its inputs should not fail to render outside the app shell.
 */
export function useResearchDay(): string | null {
  return useContext(ResearchContextContext)?.asOf ?? null
}
