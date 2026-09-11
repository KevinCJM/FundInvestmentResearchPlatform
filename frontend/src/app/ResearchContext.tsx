import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { fetchPitSettings, type PitReleaseSummary, type PitSettingsPayload, type RunMode } from '../services/pit'
import { getPitOverride, pageReload, setPitOverride, type PitViewOverride } from '../services/pitOverride'

interface ResearchContextValue {
  /** Server-held system setting; null until the first load resolves. */
  settings: PitSettingsPayload | null
  /** This tab's temporary viewing choice, or null while it follows the system. */
  override: PitViewOverride | null
  /** An explicit tab override, unless confirmed identical to the system. */
  temporary: boolean
  /** The release this tab is viewing when it overrides the system one. */
  overrideRelease: PitReleaseSummary | null
  /** Null means the current口径 is unknown, not that PIT is off. */
  noPit: boolean | null
  /** One short string a result footnote can print verbatim. */
  label: string
  asOf: string | null
  runMode: RunMode | null
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
        // A failed settings read says nothing about the server's active口径.
        // Do not present a previous response as the current setting.
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
    const system = !loading && !error ? settings?.effective : undefined
    let noPit: boolean | null = system?.no_pit ?? null
    let label = system?.label ?? 'PIT 口径未知'
    let asOf = system?.as_of ?? null
    let runMode: RunMode | null = system?.run_mode ?? null

    if (override?.off) {
      noPit = true
      label = NO_PIT_LABEL
      asOf = null
      runMode = 'RESEARCH'
    } else if (override && ((!loading && !error && overrideRelease) || (!override.releaseId && override.asOf))) {
      noPit = false
      // Match the server: omitted mode inherits the version, then the system.
      runMode = override.runMode ?? overrideRelease?.run_mode ?? system?.run_mode ?? null
      asOf = override.asOf ?? overrideRelease?.as_of ?? overrideRelease?.available_through ?? null
      const mode = runMode === null ? '运行模式未知' : runMode === 'STRICT_PIT' ? '严格 PIT' : '研究模式'
      const vintage = overrideRelease?.name ?? '最新数据（未封版）'
      label = `站在 ${asOf} · ${vintage} · ${mode}`
    }

    // An override that resolves to the same口径 as the system default is not a
    // deviation: picking the version already applied must not raise a warning.
    const temporary =
      Boolean(override) &&
      (!system || noPit !== system.no_pit ||
        asOf !== system.as_of ||
        runMode !== system.run_mode ||
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
export function useResearchDay(): string | null | undefined {
  const context = useContext(ResearchContextContext)
  // Undefined means unknown; null means a confirmed view without a cutoff.
  if (!context || context.noPit === null || context.runMode === null) return undefined
  return context.asOf
}
