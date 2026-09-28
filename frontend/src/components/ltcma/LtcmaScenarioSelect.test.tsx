import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import LtcmaScenarioSelect from './LtcmaScenarioSelect'

it('explains unavailable choices on hover, keyboard focus and touch without selecting them', () => {
  const change = vi.fn()
  render(<LtcmaScenarioSelect label="历史情景" value="" onChange={change} choices={[
    { value: 'monthly', name: '月频划分的区间', available: true, reasons: [] },
    { value: 'future', name: '未来区间', available: false, reasons: ['区间使用了研究日之后的数据。'] },
  ]} />)
  const trigger = screen.getByRole('button', { name: '历史情景' })
  fireEvent.click(trigger)
  const future = screen.getByRole('menuitemradio', { name: /未来区间/ })
  fireEvent.mouseEnter(future)
  expect(screen.getByRole('tooltip')).toHaveTextContent('区间使用了研究日之后的数据。')
  fireEvent.click(future)
  expect(change).not.toHaveBeenCalled()
  fireEvent.keyDown(future, { key: 'End' })
  expect(future).toHaveFocus()
  expect(future).toHaveAttribute('aria-describedby', screen.getByRole('tooltip').id)
  fireEvent.keyDown(future, { key: 'Escape' })
  expect(trigger).toHaveFocus()
  expect(screen.queryByRole('menu')).not.toBeInTheDocument()
  fireEvent.keyDown(trigger, { key: 'ArrowDown' })
  fireEvent.keyDown(screen.getByRole('menu'), { key: 'ArrowDown' })
  const allowed = screen.getByRole('menuitemradio', { name: '月频划分的区间' })
  expect(allowed).toHaveFocus()
  fireEvent.click(allowed)
  expect(change).toHaveBeenCalledWith('monthly')
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
})
