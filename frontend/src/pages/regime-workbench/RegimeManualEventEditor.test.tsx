import { useState } from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it } from 'vitest'
import type { ManualHistoricalEvent } from '../../services/regimeGraph'
import RegimeManualEventEditor from './RegimeManualEventEditor'

function Host() {
  const [events, setEvents] = useState<ManualHistoricalEvent[]>([])
  return <><RegimeManualEventEditor value={events} onChange={setEvents} /><output aria-label="事件JSON">{JSON.stringify(events)}</output></>
}

it('可直接添加并编辑相互重叠的历史事件，不要求优先级去重', async () => {
  const user = userEvent.setup()
  render(<Host />)
  await user.click(screen.getByRole('button', { name: '添加事件' }))
  await user.clear(screen.getByLabelText('事件1名称'))
  await user.type(screen.getByLabelText('事件1名称'), '次贷危机')
  await user.type(screen.getByLabelText('事件1开始日期'), '2007-11-01')
  await user.type(screen.getByLabelText('事件1结束日期'), '2009-03-31')
  await user.click(screen.getByRole('button', { name: '添加事件' }))
  await user.clear(screen.getByLabelText('事件2名称'))
  await user.type(screen.getByLabelText('事件2名称'), '流动性冲击')
  await user.type(screen.getByLabelText('事件2开始日期'), '2008-09-01')
  await user.type(screen.getByLabelText('事件2结束日期'), '2008-12-31')

  expect(screen.getAllByLabelText(/历史事件 \d/)).toHaveLength(2)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  const saved = JSON.parse(screen.getByLabelText('事件JSON').textContent || '[]')
  expect(saved.map((event: ManualHistoricalEvent) => [event.label, event.start_date, event.end_date])).toEqual([
    ['次贷危机', '2007-11-01', '2009-03-31'],
    ['流动性冲击', '2008-09-01', '2008-12-31'],
  ])
})

it('开始日期晚于结束日期时直接提示，删除只删除当前事件', async () => {
  const user = userEvent.setup()
  render(<Host />)
  await user.click(screen.getByRole('button', { name: '添加事件' }))
  await user.type(screen.getByLabelText('事件1开始日期'), '2020-04-01')
  await user.type(screen.getByLabelText('事件1结束日期'), '2020-03-01')
  expect(screen.getByRole('alert')).toHaveTextContent('开始日期不能晚于结束日期')
  await user.click(screen.getByRole('button', { name: '删除' }))
  expect(screen.queryByLabelText('事件1名称')).not.toBeInTheDocument()
})
