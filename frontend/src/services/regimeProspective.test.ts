import { afterEach, expect, it, vi } from 'vitest'
import * as api from './regimeProspective'
import { prospectiveSaved } from '../pages/regime-workbench/regimeProspectiveFixtures'
import { RegimeGraphApiError } from './regimeGraph'
afterEach(() => { vi.unstubAllGlobals() })
it('writes only exact server contracts and shares request headers, abort and detailed errors', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ items: [] }) })
  vi.stubGlobal('fetch', fetch)
  const signal = new AbortController().signal
  await api.registerRegimeProspective({ calibration_id: 'cal-1' }, signal)
  await api.captureRegimeProspective('p/1', {}, signal)
  await api.assessRegimeProspective('p/1', { reference: prospectiveSaved.request.reference }, signal)
  await api.listRegimeProspective(signal)
  await api.getRegimeProspectiveProgress('p/1', signal)
  await api.getRegimeProspectiveQualification('q/1', signal)
  await api.previewRegimeSourceVersion('p/1', signal)
  await api.confirmRegimeSourceVersion('p/1', 'a'.repeat(64), signal)
  expect(fetch.mock.calls.map(([url]) => url)).toEqual(['register', 'p%2F1/capture', 'p%2F1/assess', 'catalog', 'protocols/p%2F1/progress', 'qualifications/q%2F1', 'p%2F1/sources/preview', 'p%2F1/sources/confirm'].map(url => `/api/historical-regimes/prospective/${url}`))
  expect(fetch.mock.calls.slice(6).map(([, init]) => JSON.parse(init.body))).toEqual([{}, { preview_hash: 'a'.repeat(64) }])
  expect(fetch.mock.calls.slice(0, 3).map(([, init]) => JSON.parse(init.body))).toEqual([{ calibration_id: 'cal-1' }, {}, { reference: prospectiveSaved.request.reference }])
  expect(fetch.mock.calls.every(([, init]) => init.signal === signal && init.headers['Content-Type'] === 'application/json')).toBe(true)
  fetch.mockResolvedValue({ ok: false, status: 409, json: async () => ({ detail: { message: '前瞻来源已变更', diagnostics: [{ code: 'CHANGED', message: '请重新保存候选' }] } }) })
  await expect(api.getRegimeProspectiveProgress('p')).rejects.toBeInstanceOf(RegimeGraphApiError)
})
it.each(['timestamp', 'date', 'as_of', 'probabilities', 'confidence', 'eligible', 'deployment_eligible'])('rejects client %s without sending it', key => {
  const fetch = vi.fn(); vi.stubGlobal('fetch', fetch)
  const forbidden = { [key]: true }
  expect(() => api.registerRegimeProspective({ calibration_id: 'cal', ...forbidden })).toThrow('字段不合法')
  expect(() => api.captureRegimeProspective('p', forbidden as never)).toThrow('字段不合法')
  expect(() => api.assessRegimeProspective('p', { reference: prospectiveSaved.request.reference, ...forbidden })).toThrow('字段不合法')
  expect(() => api.assessRegimeProspective('p', { reference: { ...prospectiveSaved.request.reference, ...forbidden } })).toThrow('字段不合法')
  expect(fetch).not.toHaveBeenCalled()
})
it.each([null, { items: null }])('null catalog is safely empty', async value => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => value }))
  expect(await api.listRegimeProspective()).toEqual([])
})
