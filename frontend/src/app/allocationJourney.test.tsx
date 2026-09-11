import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { allocationJourneyPath, readAllocationDraft, readAllocationJourney, updateAllocationJourney, useAllocationDraft, useAllocationJourney } from './allocationJourney'

beforeEach(() => { localStorage.clear(); sessionStorage.clear() })

describe('allocation research continuity', () => {
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
    expect(allocationJourneyPath('saa', { allocationName: '股债 60/40', universeId: 'one' })).toBe('/pre-investment/saa/allocation-lab?alloc=%E8%82%A1%E5%80%BA+60%2F40&universe=one')
    expect(allocationJourneyPath('pool', { universeId: 'two', poolVersionIds: ['version-b'] })).toBe('/pre-investment/product-pool?universe=two')
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
})
