import { expect, it } from 'vitest'
import type { HistoricalRegimeRun } from './historicalRegimes'
import { researchVersionChoices } from './regimeResearchVersions'

const run = (id: string, revision: number, published: string) => ({ id, immutable: true, definition_id: 'clock', definition_revision: revision,
  mode: 'retrospective', publications: [{ id: `pub-${id}`, run_id: id, usage: 'product_research', published_at: published }],
}) as HistoricalRegimeRun

it('每个算法版本展示最新结果，当前选中的旧结果保持固定', () => {
  const runs = [run('old', 1, '2026-01-01'), run('new', 1, '2026-02-01'), run('v2', 2, '2026-03-01')]
  expect(researchVersionChoices(runs).map(item => item.id)).toEqual(['v2', 'new'])
  expect(researchVersionChoices(runs, 'old').map(item => item.id)).toEqual(['v2', 'old'])
})
it('未启用产品研究、可变结果和错误的发布关联均不可选', () => {
  expect(researchVersionChoices([{ ...run('a', 1, ''), publications: [] }, { ...run('b', 1, ''), immutable: false },
    { ...run('c', 1, ''), publications: run('wrong', 1, '').publications }])).toEqual([])
})
