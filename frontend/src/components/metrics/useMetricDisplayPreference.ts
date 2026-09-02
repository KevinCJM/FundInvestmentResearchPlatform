import { useEffect, useMemo, useState } from 'react'
import type { IndicatorContextDomain } from '../../services/customIndicators'

export interface MetricDisplayPreference {
  indicatorIds: string[]
  periodsByIndicator: Record<string, string>
}

const storageKey = (version: number, page: string, contextKind: IndicatorContextDomain) =>
  `indicator-display:v${version}:${page}:${contextKind}`

const periodMapFor = (
  indicatorIds: string[],
  periodsByIndicator: Record<string, string>,
  fallbackPeriod: string,
) => Object.fromEntries(indicatorIds.map((id) => [id, periodsByIndicator[id] ?? fallbackPeriod]))

export const metricPeriodFor = (
  preference: MetricDisplayPreference,
  indicatorId: string,
  fallbackPeriod: string,
) => preference.periodsByIndicator[indicatorId] ?? fallbackPeriod

export const withSelectedIndicators = (
  preference: MetricDisplayPreference,
  indicatorIds: string[],
  fallbackPeriod: string,
): MetricDisplayPreference => ({
  indicatorIds,
  periodsByIndicator: periodMapFor(indicatorIds, preference.periodsByIndicator, fallbackPeriod),
})

export const normalizeMetricPeriods = (
  preference: MetricDisplayPreference,
  availablePeriods: string[],
  fallbackPeriod: string,
): MetricDisplayPreference => {
  const allowed = new Set(availablePeriods)
  const validFallback = allowed.has(fallbackPeriod) ? fallbackPeriod : availablePeriods[0] ?? fallbackPeriod
  const periodsByIndicator = Object.fromEntries(preference.indicatorIds.map((id) => {
    const current = preference.periodsByIndicator[id]
    return [id, current && allowed.has(current) ? current : validFallback]
  }))
  const unchanged = preference.indicatorIds.every((id) => periodsByIndicator[id] === preference.periodsByIndicator[id])
    && Object.keys(preference.periodsByIndicator).length === preference.indicatorIds.length
  return unchanged ? preference : { ...preference, periodsByIndicator }
}

export const groupIndicatorsByPeriod = (
  preference: MetricDisplayPreference,
  fallbackPeriod: string,
) => {
  const groups = new Map<string, string[]>()
  preference.indicatorIds.forEach((indicatorId) => {
    const period = metricPeriodFor(preference, indicatorId, fallbackPeriod)
    groups.set(period, [...(groups.get(period) ?? []), indicatorId])
  })
  return [...groups.entries()].map(([period, indicatorIds]) => ({ period, indicatorIds }))
}

export function useMetricDisplayPreference(
  page: string,
  contextKind: IndicatorContextDomain,
  defaultIndicatorIds: string[],
  defaultPeriod: string,
  availableIndicatorIds: string[],
) {
  const key = useMemo(() => storageKey(2, page, contextKind), [contextKind, page])
  const legacyKey = useMemo(() => storageKey(1, page, contextKind), [contextKind, page])
  const [preference, setPreference] = useState<MetricDisplayPreference>({
    indicatorIds: defaultIndicatorIds,
    periodsByIndicator: periodMapFor(defaultIndicatorIds, {}, defaultPeriod),
  })

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(key) ?? window.localStorage.getItem(legacyKey)
      if (!raw) return
      const parsed = JSON.parse(raw) as Partial<MetricDisplayPreference> & { period?: unknown }
      const indicatorIds = Array.isArray(parsed.indicatorIds)
        ? parsed.indicatorIds.filter((id): id is string => typeof id === 'string')
        : defaultIndicatorIds
      const savedPeriodMap = parsed.periodsByIndicator && typeof parsed.periodsByIndicator === 'object'
        ? Object.fromEntries(Object.entries(parsed.periodsByIndicator).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
        : {}
      const legacyPeriod = typeof parsed.period === 'string' ? parsed.period : defaultPeriod
      setPreference({
        indicatorIds,
        periodsByIndicator: periodMapFor(indicatorIds, savedPeriodMap, legacyPeriod),
      })
    } catch {
      setPreference({
        indicatorIds: defaultIndicatorIds,
        periodsByIndicator: periodMapFor(defaultIndicatorIds, {}, defaultPeriod),
      })
    }
  }, [defaultIndicatorIds.join('|'), defaultPeriod, key, legacyKey])

  useEffect(() => {
    if (availableIndicatorIds.length === 0) return
    setPreference((current) => {
      const allowed = new Set(availableIndicatorIds)
      const validIds = current.indicatorIds.filter((id) => allowed.has(id))
      const defaults = defaultIndicatorIds.filter((id) => allowed.has(id))
      let nextIds: string[]
      if (current.indicatorIds.length === 0) {
        nextIds = []
      } else if (validIds.length > 0) {
        nextIds = validIds
      } else if (defaults.length > 0) {
        nextIds = defaults
      } else {
        nextIds = defaultIndicatorIds.length > 0 ? [availableIndicatorIds[0]] : []
      }
      const migrationFallback = current.indicatorIds.length > 0 && validIds.length === 0
        ? Object.values(current.periodsByIndicator)[0] ?? defaultPeriod
        : defaultPeriod
      const nextPeriods = periodMapFor(nextIds, current.periodsByIndicator, migrationFallback)
      const idsUnchanged = nextIds.join('|') === current.indicatorIds.join('|')
      const periodsUnchanged = idsUnchanged
        && nextIds.every((id) => nextPeriods[id] === current.periodsByIndicator[id])
        && Object.keys(current.periodsByIndicator).length === nextIds.length
      if (idsUnchanged && periodsUnchanged) return current
      return { indicatorIds: nextIds, periodsByIndicator: nextPeriods }
    })
  }, [availableIndicatorIds.join('|'), defaultIndicatorIds.join('|'), defaultPeriod, preference.indicatorIds.join('|')])

  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(preference))
  }, [key, preference])

  return [preference, setPreference] as const
}
