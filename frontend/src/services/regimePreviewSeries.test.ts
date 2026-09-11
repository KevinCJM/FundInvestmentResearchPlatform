import { afterEach, expect, it, vi } from 'vitest'
import { getAllRegimePreviewSeries } from './regimeGraph'

afterEach(() => { vi.unstubAllGlobals() })
const page = { id: 'run', node_id: 'source', port: 'value', value_type: 'series<float64>', offset: 0, total: 2, limit: 1, items: [{ observation_date: '2020-01-01', value: 100 }], upstream_outputs: [] }
const ok = (body: unknown) => ({ ok: true, json: async () => body })

it('合并冻结节点所有分页，保留日期和源数据索引', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(ok(page)).mockResolvedValueOnce(ok({ ...page, offset: 1, items: [{ observation_date: '2020-01-03', value: 120 }] }))
  vi.stubGlobal('fetch', fetch)
  const result = await getAllRegimePreviewSeries('run', 'source', 'value')
  expect(result.offset).toBe(0)
  expect(result.items.map(item => item.date)).toEqual(['2020-01-01', '2020-01-03'])
  expect(fetch.mock.calls[1][0]).toContain('offset=1')
})

it.each([
  { ...page, id: 'another-run' },
  { ...page, node_id: 'other-node' },
  { ...page, offset: 5 },
  { ...page, items: [] },
  { ...page, total: 0 },
])('拒绝错误来源、错页和截断结果 %#', async body => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(ok(body)))
  await expect(getAllRegimePreviewSeries('run', 'source', 'value')).rejects.toThrow('节点结果分页不完整')
})

it('取消后不继续读取分页', async () => {
  const controller = new AbortController()
  const fetch = vi.fn(async () => { controller.abort(); return ok(page) }); vi.stubGlobal('fetch', fetch)
  await expect(getAllRegimePreviewSeries('run', 'source', 'value', controller.signal)).rejects.toThrow('Aborted')
  expect(fetch).toHaveBeenCalledTimes(1)
})
