import { afterEach, expect, it, vi } from 'vitest'
import { eventDraft, listLibraryEvents, saveLibraryEvent } from './eventLibrary'

afterEach(() => { vi.unstubAllGlobals() })
it('omits blank optional dates but preserves false and zero query values', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ items: [], total: 0, offset: 0, limit: 50 }) })
  vi.stubGlobal('fetch', fetch)
  await listLibraryEvents({ start: '', end: '', q: '', offset: 0, archived: false, limit: 50 })
  const url = new URL(fetch.mock.calls[0][0], 'http://localhost')
  expect(url.searchParams.has('start')).toBe(false)
  expect(url.searchParams.has('end')).toBe(false)
  expect(url.searchParams.get('offset')).toBe('0')
  expect(url.searchParams.get('archived')).toBe('false')
})
it.each([
  [{ detail: [{ loc: ['body', 'event', 'windows', 0, 'end_date'], msg: 'Invalid date' }] }, 'body.event.windows.0.end_date：Invalid date'],
  [{ detail: { diagnostics: [{ field: 'event.name', message: '名称不能为空' }] } }, 'event.name：名称不能为空'],
  [{ detail: { message: '版本已变更' } }, '版本已变更'],
])('retains field-specific write validation errors', async (response, message) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => response }))
  const draft = eventDraft(); draft.name = '保留草稿'
  await expect(saveLibraryEvent(draft)).rejects.toThrow(message)
  expect(draft.name).toBe('保留草稿')
})

it('reports a readable connection error for a non-JSON gateway response', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Bad gateway</html>', { status: 502 })))
  await expect(listLibraryEvents({})).rejects.toThrow('HTTP 502')
})
