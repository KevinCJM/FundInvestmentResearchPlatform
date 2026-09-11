import { useCallback, useMemo, useState, useSyncExternalStore, type Dispatch, type SetStateAction } from 'react'

/** Browser drafts contain inputs and saved-object references, never calculated evidence. */
export interface AllocationJourney {
  name?: string
  researchDate?: string
  universeId?: string
  poolVersionIds?: string[]
  allocationName?: string
  baselineId?: string
  taaRunId?: string
}

const JOURNEY_KEY = 'allocation-journey:v1'
const CHANGE_EVENT = 'allocation-journey-change'
const DRAFT_PREFIX = 'allocation-draft:v1:'
let unavailableSessionJourney: string | null = null

export function readAllocationDraft<T>(scope: string): T | null {
  return decode<T | null>(stored(`${DRAFT_PREFIX}${scope}`), null)
}

export function writeAllocationDraft<T>(scope: string, value: T): void {
  try { localStorage.setItem(`${DRAFT_PREFIX}${scope}`, JSON.stringify(value)) } catch { /* Preserve in-memory editing. */ }
}

function stored(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}

function decode<T>(raw: string | null, fallback: T): T {
  try {
    if (!raw) return fallback
    const value: unknown = JSON.parse(raw)
    const sameShape = (candidate: unknown, expected: unknown) => {
      if (expected === undefined || expected === null) return true
      if (Array.isArray(expected)) return Array.isArray(candidate)
      if (typeof expected === 'object') return candidate !== null && typeof candidate === 'object' && !Array.isArray(candidate)
      return typeof candidate === typeof expected
    }
    if (!sameShape(value, fallback)) return fallback
    if (fallback && typeof fallback === 'object' && !Array.isArray(fallback)) {
      if (!Object.entries(fallback).every(([key, expected]) => sameShape((value as Record<string, unknown>)[key], expected))) return fallback
    }
    return value as T
  } catch { return fallback }
}

function journeyFrom(raw: string | null): AllocationJourney {
  const value = decode<Record<string, unknown>>(raw, {})
  const journey: AllocationJourney = {}
  for (const key of ['name', 'researchDate', 'universeId', 'allocationName', 'baselineId', 'taaRunId'] as const) {
    if (typeof value[key] === 'string' && value[key].trim()) journey[key] = value[key]
  }
  if (Array.isArray(value.poolVersionIds) && value.poolVersionIds.every(id => typeof id === 'string' && id.trim())) journey.poolVersionIds = value.poolVersionIds
  if (journey.researchDate && !/^\d{4}-\d{2}-\d{2}$/.test(journey.researchDate)) delete journey.researchDate
  return journey
}

export function readAllocationJourney(): AllocationJourney {
  return journeyFrom(activeJourneyRaw())
}

/** Each tab adopts the last research once; another tab cannot replace its active journey. */
function activeJourneyRaw(): string {
  try {
    const active = sessionStorage.getItem(JOURNEY_KEY)
    if (active !== null) return active
    const initial = stored(JOURNEY_KEY) ?? '{}'
    sessionStorage.setItem(JOURNEY_KEY, initial)
    return initial
  } catch {
    return unavailableSessionJourney ??= stored(JOURNEY_KEY) ?? '{}'
  }
}

export function updateAllocationJourney(patch: Partial<AllocationJourney>): AllocationJourney {
  const current = readAllocationJourney()
  const next = { ...current }
  if ('universeId' in patch && patch.universeId !== current.universeId) {
    delete next.name; delete next.researchDate; delete next.poolVersionIds
    delete next.allocationName; delete next.baselineId; delete next.taaRunId
  }
  if ('allocationName' in patch && patch.allocationName !== current.allocationName) {
    delete next.baselineId; delete next.taaRunId
  }
  if ('baselineId' in patch && patch.baselineId !== current.baselineId) delete next.taaRunId
  Object.assign(next, patch)
  const raw = JSON.stringify(next)
  try { sessionStorage.setItem(JOURNEY_KEY, raw) } catch { unavailableSessionJourney = raw }
  try { localStorage.setItem(JOURNEY_KEY, raw) } catch { /* The editor still works if storage is full. */ }
  window.dispatchEvent(new Event(CHANGE_EVENT))
  return next
}

function subscribe(listener: () => void) {
  window.addEventListener(CHANGE_EVENT, listener)
  return () => window.removeEventListener(CHANGE_EVENT, listener)
}

export function useAllocationJourney() {
  const raw = useSyncExternalStore(subscribe, activeJourneyRaw, () => null)
  const journey = useMemo(() => journeyFrom(raw), [raw])
  return [journey, updateAllocationJourney] as const
}

export type AllocationJourneyStep = 'pool' | 'classes' | 'saa' | 'taa' | 'products'
export function allocationJourneyPath(step: AllocationJourneyStep, journey = readAllocationJourney()): string {
  if (step === 'products' && journey.taaRunId && !readAllocationDraft(`products:${journey.universeId || 'local'}:${journey.taaRunId}`)) return allocationJourneyPath('taa', journey)
  const paths = {
    pool: '/pre-investment/product-pool', classes: '/pre-investment/saa/asset-classes',
    saa: '/pre-investment/saa/allocation-lab', taa: '/pre-investment/taa',
    products: '/pre-investment/product-allocation-timing/construction',
  }
  const query = new URLSearchParams()
  if (step === 'pool' && !journey.universeId && journey.poolVersionIds?.length === 1) query.set('version', journey.poolVersionIds[0])
  if (step === 'saa' && journey.allocationName) query.set('alloc', journey.allocationName)
  if (['pool', 'classes', 'saa', 'products'].includes(step) && journey.universeId) query.set('universe', journey.universeId)
  if (step === 'products' && journey.taaRunId) query.set('decision', journey.taaRunId)
  if (step === 'taa') {
    if (journey.taaRunId) query.set('decision', journey.taaRunId)
    else if (journey.baselineId) query.set('baseline', journey.baselineId)
  }
  return `${paths[step]}${query.size ? `?${query}` : ''}`
}

/** Scope must include the explicit universe / allocation / baseline identity. */
export function useAllocationDraft<T>(scope: string, initial: T | (() => T)): [T, Dispatch<SetStateAction<T>>, () => void] {
  const key = `${DRAFT_PREFIX}${scope}`
  const fresh = () => typeof initial === 'function' ? (initial as () => T)() : initial
  const [entry, setEntry] = useState(() => ({ key, value: decode(stored(key), fresh()) }))
  // A changed URL identity selects a separate draft before any old value can be saved there.
  const value = entry.key === key ? entry.value : decode(stored(key), fresh())
  if (entry.key !== key) setEntry({ key, value })
  const setValue: Dispatch<SetStateAction<T>> = useCallback((action) => {
    setEntry((previous) => {
      const current = previous.key === key ? previous.value : decode(stored(key), fresh())
      const next = typeof action === 'function' ? (action as (current: T) => T)(current) : action
      // Synchronous write survives an immediate navigation or PIT-triggered reload.
      try { localStorage.setItem(key, JSON.stringify(next)) } catch { /* Preserve in-memory editing. */ }
      return { key, value: next }
    })
  }, [key])
  const clear = useCallback(() => {
    try { localStorage.removeItem(key) } catch { /* Storage may be disabled. */ }
    setEntry({ key, value: fresh() })
  }, [key])
  return [value, setValue, clear]
}
