import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { allocationJourneyPath, readAllocationDraft, readAllocationJourney, updateAllocationJourney, useAllocationDraft, useAllocationJourney } from './allocationJourney'

beforeEach(() => { localStorage.clear(); sessionStorage.clear() })

describe('allocation research continuity', () => {
  it('resumes the complete CMA set and invalidates it when the scope changes', () => {
    updateAllocationJourney({ strategicUniverseId: 'scope-1', mandateId: 'm1', ltcmaId: 'cma-1', ltcmaIds: ['cma-1', 'cma-2'], baselineId: 'baseline', taaRunId: 'taa' })
    expect(new URLSearchParams(allocationJourneyPath('saa').split('?')[1]).getAll('cma')).toEqual(['cma-1', 'cma-2'])
    updateAllocationJourney({ ltcmaId: 'cma-1', ltcmaIds: ['cma-1', 'cma-3'] })
    expect(readAllocationJourney().baselineId).toBeUndefined()
    expect(readAllocationJourney().taaRunId).toBeUndefined()
    updateAllocationJourney({ mandateId: 'm2' })
    expect(readAllocationJourney().ltcmaIds).toBeUndefined()
    expect(readAllocationJourney().ltcmaId).toBeUndefined()
    updateAllocationJourney({ ltcmaId: 'cma-4', ltcmaIds: ['cma-4', 'cma-5'] })
    updateAllocationJourney({ ltcmaId: 'cma-4' })
    expect(new URLSearchParams(allocationJourneyPath('saa').split('?')[1]).getAll('cma')).toEqual(['cma-4'])
  })
  it('restores inputs after remount while keeping explicit universes separate', () => {
    const first = renderHook(({ scope }) => useAllocationDraft(scope, { products: [] as string[], start: '2020-01-01' }), { initialProps: { scope: 'classes:one' } })
    act(() => first.result.current[1]({ products: ['510300.SH'], start: '2021-01-01' }))
    first.rerender({ scope: 'classes:two' })
    expect(first.result.current[0].products).toEqual([])
    act(() => first.result.current[1]({ products: ['511010.SH'], start: '2022-01-01' }))
    first.unmount()
    const restored = renderHook(() => useAllocationDraft('classes:one', { products: [] as string[], start: '' }))
    expect(restored.result.current[0]).toEqual({ products: ['510300.SH'], start: '2021-01-01' })
    expect(readAllocationDraft('classes:two')).toEqual({ products: ['511010.SH'], start: '2022-01-01' })
  })

  it('replaces downstream references when switching product range without leaking an old TAA decision', () => {
    updateAllocationJourney({ name: '股债研究', researchDate: '2026-09-01', poolVersionIds: ['version-a'], universeId: 'one', allocationName: '长期60/40', baselineId: 'baseline-a', taaRunId: 'run-a' })
    const context = renderHook(() => useAllocationJourney())
    act(() => updateAllocationJourney({ name: '新范围研究', universeId: 'two' }))
    expect(context.result.current[0]).toEqual({ name: '新范围研究', universeId: 'two' })
    expect(readAllocationJourney().taaRunId).toBeUndefined()
    expect(allocationJourneyPath('saa', { allocationName: '股债 60/40', universeId: 'one' })).toBe('/pre-investment/saa/policy?alloc=%E8%82%A1%E5%80%BA+60%2F40&universe=one')
    expect(allocationJourneyPath('pool', { universeId: 'two', poolVersionIds: ['version-b'] })).toBe('/pre-investment/product-pool/new?universe=two')
  })

  it('preserves the adopted SAA while clearing the old decision after a new baseline is selected', () => {
    updateAllocationJourney({ universeId: 'one', allocationName: '长期60/40', baselineId: 'a', taaRunId: 'old' })
    updateAllocationJourney({ baselineId: 'b' })
    expect(readAllocationJourney()).toEqual({ universeId: 'one', allocationName: '长期60/40', baselineId: 'b' })
  })

  it('opens safely with corrupt browser state and discards fields of the wrong type', () => {
    localStorage.setItem('allocation-journey:v1', 'null')
    const empty = renderHook(() => useAllocationJourney())
    expect(empty.result.current[0]).toEqual({})
    empty.unmount()
    localStorage.setItem('allocation-journey:v1', JSON.stringify({ name: '研究甲', universeId: 123, poolVersionIds: [null], unknown: true }))
    sessionStorage.clear()
    expect(readAllocationJourney()).toEqual({ name: '研究甲' })
    localStorage.setItem('allocation-draft:v1:classes:bad', JSON.stringify({ products: 7 }))
    const draft = renderHook(() => useAllocationDraft('classes:bad', { products: [] as string[], start: '2020-01-01' }))
    expect(draft.result.current[0]).toEqual({ products: [], start: '2020-01-01' })
  })

  it('keeps the active tab stable after another tab changes the most recent research', () => {
    localStorage.setItem('allocation-journey:v1', JSON.stringify({ name: '研究甲', universeId: 'one' }))
    const active = renderHook(() => useAllocationJourney())
    expect(active.result.current[0].universeId).toBe('one')
    act(() => {
      localStorage.setItem('allocation-journey:v1', JSON.stringify({ name: '研究乙', universeId: 'two' }))
      window.dispatchEvent(new StorageEvent('storage', { key: 'allocation-journey:v1', newValue: localStorage.getItem('allocation-journey:v1') }))
    })
    active.rerender()
    expect(active.result.current[0]).toEqual({ name: '研究甲', universeId: 'one' })
    active.unmount()
    const refreshed = renderHook(() => useAllocationJourney())
    expect(refreshed.result.current[0].universeId).toBe('one')
    act(() => updateAllocationJourney({ baselineId: 'same-tab-baseline' }))
    expect(refreshed.result.current[0].baselineId).toBe('same-tab-baseline')
    refreshed.unmount()
    // A fresh tab/session adopts the latest record once, then owns its context.
    localStorage.setItem('allocation-journey:v1', JSON.stringify({ name: '研究乙', universeId: 'two' }))
    sessionStorage.clear()
    expect(readAllocationJourney()).toEqual({ name: '研究乙', universeId: 'two' })
  })

  it('clears old range metadata on an explicit scope change and accepts authoritative metadata in that patch', () => {
    updateAllocationJourney({ universeId: 'one', name: '旧研究', researchDate: '2026-09-01', poolVersionIds: ['old-version'] })
    updateAllocationJourney({ universeId: 'two', allocationName: '新 SAA', baselineId: 'new-baseline' })
    expect(readAllocationJourney()).toEqual({ universeId: 'two', allocationName: '新 SAA', baselineId: 'new-baseline' })
    updateAllocationJourney({ universeId: 'three', name: '真实研究名', researchDate: '2026-09-03', poolVersionIds: ['new-version'] })
    updateAllocationJourney({ baselineId: 'next-baseline' })
    expect(readAllocationJourney()).toEqual({ universeId: 'three', name: '真实研究名', researchDate: '2026-09-03', poolVersionIds: ['new-version'], baselineId: 'next-baseline' })
  })

  it('改了目标或范围，选定的 LTCMA 连同其后的基线一起作废', () => {
    updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-7', baselineId: 'b1', taaRunId: 't1' })
    expect(readAllocationJourney().ltcmaId).toBe('cma-7')
    // LTCMA 的可用性绑定目标的币种与研究日区间，换目标就不能再沿用。
    updateAllocationJourney({ mandateId: 'm2' })
    expect(readAllocationJourney()).toEqual({ mandateId: 'm2', universeId: 'u1', allocationName: '60/40' })
  })

  it('换一版 LTCMA 会作废基于旧假设的基线和战术版本', () => {
    updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-7', baselineId: 'b1', taaRunId: 't1' })
    updateAllocationJourney({ ltcmaId: 'cma-8' })
    expect(readAllocationJourney()).toEqual({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-8' })
  })

  it('01 落点带回已选目标的详情，没有目标时才停在列表页', () => {
    expect(allocationJourneyPath('objectives', {})).toBe('/pre-investment/objectives')
    expect(allocationJourneyPath('objectives', { mandateId: 'm1' })).toBe('/pre-investment/objectives/new?view=m1')
  })

  it('SAA 落点带回已选假设，LTCMA 落点直接回到那一版', () => {
    updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-7' })
    expect(allocationJourneyPath('saa')).toBe('/pre-investment/saa/policy?alloc=60%2F40&universe=u1&mandate=m1&cma=cma-7')
    expect(allocationJourneyPath('ltcma')).toBe('/pre-investment/ltcma/cma-7')
  })
})
