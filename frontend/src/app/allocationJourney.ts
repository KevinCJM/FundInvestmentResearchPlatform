import { useCallback, useMemo, useState, useSyncExternalStore, type Dispatch, type SetStateAction } from 'react'

/** Browser drafts contain inputs and saved-object references, never calculated evidence. */
export interface AllocationJourney {
  name?: string
  mandateId?: string
  strategicUniverseId?: string
  implementationMappingId?: string
  researchDate?: string
  universeId?: string
  /** 本次研究选定的主 LTCMA 版本；SAA 的假设从这里续接，顶部流程条据此点亮 03。 */
  ltcmaId?: string
  /** Exact selected CMA set; the primary ID alone cannot represent a multi-CMA study. */
  ltcmaIds?: string[]
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
  for (const key of ['name', 'researchDate', 'universeId', 'allocationName', 'baselineId', 'taaRunId', 'mandateId', 'strategicUniverseId', 'implementationMappingId', 'ltcmaId'] as const) {
    if (typeof value[key] === 'string' && value[key].trim()) journey[key] = value[key]
  }
  if (Array.isArray(value.poolVersionIds) && value.poolVersionIds.every(id => typeof id === 'string' && id.trim())) journey.poolVersionIds = value.poolVersionIds
  if (Array.isArray(value.ltcmaIds) && value.ltcmaIds.length <= 20 && value.ltcmaIds.every(id => typeof id === 'string' && id.trim())) journey.ltcmaIds = [...new Set(value.ltcmaIds)]
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

const CMA_SCOPE_KEYS = ['mandateId', 'universeId', 'allocationName', 'strategicUniverseId', 'implementationMappingId'] as const

export function updateAllocationJourney(patch: Partial<AllocationJourney>): AllocationJourney {
  const current = readAllocationJourney()
  const next = { ...current }
  if ('universeId' in patch && patch.universeId !== current.universeId) {
    delete next.name; delete next.researchDate; delete next.poolVersionIds
    delete next.allocationName
    if (!current.strategicUniverseId || current.implementationMappingId) { delete next.baselineId; delete next.taaRunId }
    delete next.implementationMappingId
  }
  if ('allocationName' in patch && patch.allocationName !== current.allocationName) {
    delete next.baselineId; delete next.taaRunId
  }
  if ('mandateId' in patch && patch.mandateId !== current.mandateId) { delete next.baselineId; delete next.taaRunId }
  if ('strategicUniverseId' in patch && patch.strategicUniverseId !== current.strategicUniverseId) {
    delete next.implementationMappingId; delete next.baselineId; delete next.taaRunId
  }
  if ('implementationMappingId' in patch && patch.implementationMappingId !== current.implementationMappingId) {
    delete next.baselineId; delete next.taaRunId
  }
  // LTCMA 绑定目标的币种与研究日区间，也绑定范围身份（后端 preview_policy 的 SAA_MANDATE_CMA_BASIS / SAA_CMA_SOURCE_CHANGED）；上游一改，原来选的假设就不再可用。
  if (CMA_SCOPE_KEYS.some(key => key in patch && patch[key] !== current[key])) { delete next.ltcmaId; delete next.ltcmaIds }
  if ('ltcmaId' in patch && !('ltcmaIds' in patch)) delete next.ltcmaIds
  if ('ltcmaIds' in patch && JSON.stringify(patch.ltcmaIds) !== JSON.stringify(current.ltcmaIds)) { delete next.baselineId; delete next.taaRunId }
  if ('ltcmaId' in patch && patch.ltcmaId !== current.ltcmaId) { delete next.baselineId; delete next.taaRunId }
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

export type AllocationJourneyStep = 'objectives' | 'pool' | 'classes' | 'ltcma' | 'saa' | 'taa' | 'products'
/** 06 要等 TAA 交接出草稿才真的有东西可看；没有草稿时点开只会被送回 TAA。 */
export function productsHandoffReady(journey = readAllocationJourney()): boolean {
  return Boolean(journey.taaRunId && readAllocationDraft(`products:${journey.universeId || 'local'}:${journey.taaRunId}`))
}

export function allocationJourneyPath(step: AllocationJourneyStep, journey = readAllocationJourney()): string {
  if (step === 'products' && journey.taaRunId && !productsHandoffReady(journey)) return allocationJourneyPath('taa', journey)
  const paths = {
    // 已选定目标时直接回到那一版详情，而不是回列表页重找。
    objectives: journey.mandateId ? '/pre-investment/objectives/new' : '/pre-investment/objectives',
    // 已选定范围时直接回到编辑页那一版，而不是回列表页重找。
    pool: (journey.universeId || journey.strategicUniverseId) ? '/pre-investment/product-pool/new' : '/pre-investment/product-pool',
    classes: '/pre-investment/saa/asset-classes',
    // 已选定假设时直接回到那一版，而不是回中心页重找。
    ltcma: journey.ltcmaId ? `/pre-investment/ltcma/${encodeURIComponent(journey.ltcmaId)}` : '/pre-investment/ltcma',
    saa: '/pre-investment/saa/policy', taa: '/pre-investment/taa',
    products: '/pre-investment/product-allocation-timing/construction',
  }
  const query = new URLSearchParams()
  if (step === 'objectives' && journey.mandateId) query.set('view', journey.mandateId)
  if (step === 'pool' && !journey.universeId && journey.poolVersionIds?.length === 1) query.set('version', journey.poolVersionIds[0])
  if (step === 'saa' && journey.baselineId) query.set('baseline', journey.baselineId)
  if (step === 'saa' && !journey.strategicUniverseId && journey.allocationName) query.set('alloc', journey.allocationName)
  if (['pool', 'classes', 'saa', 'products'].includes(step) && journey.universeId) query.set('universe', journey.universeId)
  if (['pool', 'saa'].includes(step) && journey.mandateId) query.set('mandate', journey.mandateId)
  if (step === 'saa') (journey.ltcmaIds?.length ? journey.ltcmaIds : journey.ltcmaId ? [journey.ltcmaId] : []).forEach(id => query.append('cma', id))
  if (['pool', 'saa'].includes(step) && journey.strategicUniverseId) {
    query.set('strategic_universe', journey.strategicUniverseId)
    if (step === 'pool') query.set('scope', 'strategic')
    if (journey.implementationMappingId) query.set('mapping', journey.implementationMappingId)
  }
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
