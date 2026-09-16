import { afterEach, expect, it, vi } from 'vitest'
import { confirmRegimeReliability, getRegimeRecognitionEvidence, getRegimeReliability, listHistoricalReferences, listRegimeReliability, previewRegimeReliability } from './regimeGraph'
import { reliabilityPreviewFixture } from '../pages/regime-workbench/regimeReliabilityFixtures'
afterEach(() => { vi.unstubAllGlobals() })
it('按后端精确 URL/payload 调用，确认不上传客户端报告', async () => {
  const mock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ items: [] }) })
  vi.stubGlobal('fetch', mock)
  const preview = reliabilityPreviewFixture()
  const controller = new AbortController()
  await previewRegimeReliability(preview.request, 'compile-token', controller.signal)
  expect(mock.mock.calls[0][0]).toBe('/api/historical-regimes/reliability/preview')
  expect(JSON.parse(mock.mock.calls[0][1].body)).toEqual({ ...preview.request, compile_token: 'compile-token' })
  expect(mock.mock.calls[0][1].signal).toBe(controller.signal)
  await confirmRegimeReliability(preview, controller.signal)
  expect(mock.mock.calls[1][0]).toBe('/api/historical-regimes/reliability/confirm')
  expect(JSON.parse(mock.mock.calls[1][1].body)).toEqual({ request: preview.request, preview_hash: preview.preview_hash })
  await listHistoricalReferences()
  await listRegimeReliability()
  await getRegimeReliability('report/id')
  await getRegimeRecognitionEvidence('report/id')
  expect(mock.mock.calls.slice(2).map(call => call[0])).toEqual(['/api/historical-regimes/references', '/api/historical-regimes/reliability/catalog', '/api/historical-regimes/reliability/reports/report%2Fid', '/api/historical-regimes/reliability/reports/report%2Fid/recognition-evidence'])
})
it('引用过期与错误状态保持失败，不制造报告', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 409, json: async () => ({ detail: { code: 'RELIABILITY_PREVIEW_EXPIRED', message: '预览已过期，请重新验证。' } }) }))
  await expect(confirmRegimeReliability(reliabilityPreviewFixture())).rejects.toThrow('预览已过期，请重新验证。')
})

it('质量 API 只确认请求和 hash，不上传客户端统计或发布参考', async () => {
  const { previewRegimeQuality, confirmRegimeQuality, getRegimeQuality, listRegimeQuality } = await import('./regimeGraph')
  const { qualityPreviewFixture } = await import('../pages/regime-workbench/regimeQualityFixtures')
  const mock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ items: [] }) })
  vi.stubGlobal('fetch', mock)
  const preview = qualityPreviewFixture(), controller = new AbortController()
  await previewRegimeQuality(preview.request, 'token', controller.signal)
  await confirmRegimeQuality(preview, controller.signal)
  await getRegimeQuality('quality/id', controller.signal)
  await listRegimeQuality(controller.signal)
  expect(mock.mock.calls.map(call => call[0])).toEqual(['/api/historical-regimes/reference-quality/preview', '/api/historical-regimes/reference-quality/confirm', '/api/historical-regimes/reference-quality/reports/quality%2Fid', '/api/historical-regimes/reference-quality/catalog'])
  expect(JSON.parse(mock.mock.calls[0][1].body)).toEqual({ ...preview.request, compile_token: 'token' })
  expect(JSON.parse(mock.mock.calls[1][1].body)).toEqual({ request: preview.request, preview_hash: preview.preview_hash })
  expect(mock.mock.calls.every(call => call[1].signal === controller.signal)).toBe(true)
})

it('旧新不可变报告按原响应返回，附加证据与服务器 hash 不被重写', async () => {
  const preview = reliabilityPreviewFixture()
  const oldSaved = { ...preview, id: 'old-report', calibration_id: null, created_at: '2026-09-14', immutable: true, content_hash: 'old-hash' }
  const newSaved = { ...oldSaved, id: 'new-report', content_hash: 'new-hash', report: { ...preview.report, confidence_interval: { status: 'unavailable', reason: 'insufficient_full_blocks', metrics: {} }, stability: { status: 'causal_probes_executed', parameter_sensitivity: { status: 'not_applicable', variants: [] } } } }
  const mock = vi.fn().mockResolvedValueOnce({ ok: true, json: async () => oldSaved }).mockResolvedValueOnce({ ok: true, json: async () => newSaved })
  vi.stubGlobal('fetch', mock)
  expect(await getRegimeReliability('old-report')).toBe(oldSaved)
  expect(await getRegimeReliability('new-report')).toBe(newSaved)
  expect(newSaved.content_hash).toBe('new-hash')
})
