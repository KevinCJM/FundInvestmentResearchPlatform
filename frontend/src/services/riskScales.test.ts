import { afterEach, describe, expect, it, vi } from 'vitest'
import { riskScales, RiskScaleError } from './riskScales'
import { riskDefinition } from '../test/riskScaleFixtures'
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })
describe('risk scales API boundary', () => {
  it('posts decimal values and preserves exact preview identity', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: 'saved' })))
    vi.stubGlobal('fetch', fetcher)
    await riskScales.confirm({ request: { definition: riskDefinition }, confirm: true, preview_hash: 'exact-hash', idempotency_key: 'test-key' })
    const body = JSON.parse(fetcher.mock.calls[0][1].body); expect(body.preview_hash).toBe('exact-hash'); expect(body.request.definition).toEqual(riskDefinition); expect(fetcher.mock.calls[0][0]).toBe('/api/strategic-allocation/risk-scales/confirm')
  })
  it('returns structured conflict fields and never exposes a raw response body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: { code: 'PREVIEW_STALE', message: 'Changed source', field: 'reference_input_ref' } }), { status: 409 })))
    await expect(riskScales.preview({ definition: riskDefinition })).rejects.toMatchObject({ code: 'PREVIEW_STALE', field: 'reference_input_ref', status: 409 })
  })
  it('handles non-JSON failures safely', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>private traceback</html>', { status: 500 })))
    await expect(riskScales.capabilities()).rejects.toEqual(expect.objectContaining({ code: 'INVALID_RESPONSE', message: '' }))
  })
  it('distinguishes network errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('internal transport details')))
    await expect(riskScales.defaults()).rejects.toBeInstanceOf(RiskScaleError)
  })
  it.each(['timeout', 'navigation'])('aborts the batch label lookup on %s', async cause => {
    vi.useFakeTimers()
    let requestSignal!: AbortSignal
    const fetcher = vi.fn().mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      requestSignal = options.signal
      requestSignal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
    }))
    vi.stubGlobal('fetch', fetcher)
    const controller = new AbortController()
    const result = riskScales.sourceLabels(['index:index_daily:000300.SH'], controller.signal).catch(error => error)
    if (cause === 'timeout') await vi.advanceTimersByTimeAsync(10_000)
    else controller.abort()
    expect(await result).toMatchObject({ name: 'AbortError' })
    expect(requestSignal.aborted).toBe(true)
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ series_ids: ['index:index_daily:000300.SH'] })
    expect(vi.getTimerCount()).toBe(0)
  })
})
