import { afterEach, expect, it, vi } from 'vitest'
import { AgentApiError, agentRequest } from './agentClient'

afterEach(() => { vi.unstubAllGlobals() })
it('共用请求保留显式请求头并拒绝无效成功响应', async () => {
  const fetcher = vi.fn().mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ ok: true }) })
    .mockResolvedValueOnce({ ok: true, status: 200, json: async () => { throw new SyntaxError() } })
    .mockResolvedValueOnce({ ok: false, status: 409, json: async () => ({ detail: { code: 'REVISION_CONFLICT', message: '版本已变化', run_id: 'run-1' } }) })
  vi.stubGlobal('fetch', fetcher)
  await agentRequest('/api/agent/meta', { headers: new Headers({ 'x-pit-off': '1' }) })
  expect(fetcher.mock.calls[0][1].headers.get('x-pit-off')).toBe('1')
  expect(fetcher.mock.calls[0][1].headers.get('content-type')).toBe('application/json')
  await expect(agentRequest('/api/agent/meta')).rejects.toMatchObject({ code: 'AGENT_RESPONSE_INVALID' })
  await expect(agentRequest('/api/agent/meta')).rejects.toMatchObject(new AgentApiError('版本已变化','REVISION_CONFLICT',409,'run-1'))
})
