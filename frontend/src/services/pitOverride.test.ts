import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getPitOverride, installPitOverrideFetch, pitOverrideHeaders, setPitOverride } from './pitOverride'

describe('本页临时 PIT 口径', () => {
  beforeEach(() => {
    sessionStorage.clear()
    setPitOverride(null)
  })

  afterEach(() => {
    setPitOverride(null)
    vi.unstubAllGlobals()
  })

  it('跟随系统默认时不发送任何口径请求头', () => {
    expect(getPitOverride()).toBeNull()
    expect(pitOverrideHeaders()).toEqual({})
  })

  it('关闭 PIT 与"什么都没说"必须可区分', () => {
    setPitOverride({ off: true, releaseId: null, runMode: 'RESEARCH' })
    expect(pitOverrideHeaders()).toEqual({ 'X-Pit-Off': '1' })
  })

  it('查看指定版本时带上版本与运行模式', () => {
    setPitOverride({ off: false, releaseId: 'release-1', runMode: 'STRICT_PIT' })
    expect(pitOverrideHeaders()).toEqual({ 'X-Pit-Release': 'release-1', 'X-Pit-Run-Mode': 'STRICT_PIT' })
  })

  it('没有版本又没关闭 PIT 的选择等于没有选择', () => {
    setPitOverride({ off: false, releaseId: null, runMode: 'RESEARCH' })
    expect(getPitOverride()).toBeNull()
  })

  it('选择存活在 sessionStorage 里，刷新后仍然有效', () => {
    setPitOverride({ off: false, releaseId: 'release-1', runMode: 'RESEARCH' })
    expect(JSON.parse(sessionStorage.getItem('pit.view.override') as string)).toMatchObject({
      off: false,
      releaseId: 'release-1',
    })
  })

  it('数据请求自动带上口径请求头，PIT 设置接口本身除外', async () => {
    const native = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', native)
    installPitOverrideFetch()
    setPitOverride({ off: true, releaseId: null, runMode: 'RESEARCH' })

    await fetch('/api/fit-classes', { method: 'POST', body: '{}' })
    // PIT 设置接口必须豁免：否则一个失效的临时口径会连"改回来"的入口一起打死。
    await fetch('/api/pit/settings')

    const [dataInit, settingsInit] = native.mock.calls.map((call: any[]) => call[1])
    expect(new Headers(dataInit.headers).get('X-Pit-Off')).toBe('1')
    expect(settingsInit === undefined || new Headers(settingsInit.headers).get('X-Pit-Off') === null).toBe(true)
  })

  it('不覆盖调用方自己设置的其它请求头', async () => {
    const native = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', native)
    installPitOverrideFetch()
    setPitOverride({ off: false, releaseId: 'release-1', runMode: 'RESEARCH' })

    await fetch('/api/auto-class', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })

    const headers = new Headers((native.mock.calls[0] as any[])[1].headers)
    expect(headers.get('Content-Type')).toBe('application/json')
    expect(headers.get('X-Pit-Release')).toBe('release-1')
  })
})
