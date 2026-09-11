import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearRecentVisits, readRecentVisits, recordRecentVisit, RECENT_VISITS_KEY } from './history'
import { homeModules } from './catalog'

describe('homepage recent module visits', () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => { vi.restoreAllMocks() })
  it('starts empty and records the most specific real module once', () => {
    expect(readRecentVisits()).toEqual([])
    recordRecentVisit('/settings/scenario-algorithms/workbench')
    recordRecentVisit('/product-research/products/private-product')
    recordRecentVisit('/settings/scenario-algorithms/workbench')
    expect(readRecentVisits()).toEqual(['scenarios', 'products'])
    expect(localStorage.getItem(RECENT_VISITS_KEY)).not.toContain('private-product')
  })
  it('never records home, unrelated paths, or lookalike prefixes', () => {
    for (const path of ['/', '/unknown', '/settings-evil', 'https://evil.test/settings']) recordRecentVisit(path)
    expect(readRecentVisits()).toEqual([])
  })
  it('bounds history and rejects malformed, huge, duplicate or untrusted entries', () => {
    homeModules.forEach(item => recordRecentVisit(item.path))
    expect(readRecentVisits()).toHaveLength(8)
    for (const bad of ['{', '{}', 'null', JSON.stringify(['settings', 'settings', 4, '<script>', '/settings', 'metrics']), ' '.repeat(4097)]) {
      localStorage.setItem(RECENT_VISITS_KEY, bad)
      expect(readRecentVisits()).toEqual(bad.includes('metrics') ? ['settings', 'metrics'] : [])
    }
  })
  it('clears only homepage history, never research or language settings', () => {
    localStorage.setItem('research-definition', 'keep')
    recordRecentVisit('/settings')
    expect(clearRecentVisits()).toBe(true)
    expect(readRecentVisits()).toEqual([])
    expect(localStorage.getItem('research-definition')).toBe('keep')
  })
  it('storage denial cannot block navigation or imply successful clearing', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied') })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new Error('denied') })
    expect(readRecentVisits()).toEqual([])
    expect(() => recordRecentVisit('/settings')).not.toThrow()
    expect(clearRecentVisits()).toBe(false)
  })
})
