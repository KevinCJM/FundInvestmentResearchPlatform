import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { expect, it, vi } from 'vitest'
import CommandPalette from './CommandPalette'

vi.mock('../i18n/runtime', () => ({
  useI18n: () => ({ s: (key: string) => key, version: 0 }),
  systemText: (key: string, _parameters?: unknown, fallback?: string) => fallback ?? key,
}))
vi.mock('../i18n/navigation', () => ({ localizeStage: (stage: unknown) => stage }))

function Probe() {
  return <p>当前路径 {useLocation().pathname}</p>
}

const mount = () => render(
  <MemoryRouter initialEntries={['/pre-investment']}>
    <CommandPalette />
    <Routes><Route path="*" element={<Probe />} /></Routes>
  </MemoryRouter>,
)

it('opens with the keyboard shortcut and lists entries taken from the process registry', async () => {
  const user = userEvent.setup()
  mount()
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

  await user.keyboard('{Meta>}k{/Meta}')

  const listbox = within(screen.getByRole('dialog')).getByRole('listbox')
  expect(within(listbox).getByRole('option', { name: /投资目标与约束/ })).toBeInTheDocument()
  expect(within(listbox).getByRole('option', { name: /LTCMA 中心/ })).toBeInTheDocument()
})

it('filters on any of label, stage, description and path', async () => {
  const user = userEvent.setup()
  mount()
  await user.click(screen.getByRole('button', { name: /navigation.search/ }))

  await user.type(screen.getByRole('combobox'), 'ltcma')

  const listbox = within(screen.getByRole('listbox'))
  // 描述里提到 LTCMA 的节点也算命中，所以不要求每行的可见文字都含关键词，只要求无关条目消失。
  expect(listbox.getByRole('option', { name: /LTCMA 中心/ })).toBeInTheDocument()
  expect(listbox.queryByRole('option', { name: /组合财务报表与净值核验/ })).not.toBeInTheDocument()
  expect(listbox.getAllByRole('option').length).toBeLessThan(20)
})

it('moves the visual focus with the arrow keys and opens the active entry on Enter', async () => {
  const user = userEvent.setup()
  mount()
  await user.keyboard('{Meta>}k{/Meta}')
  const input = screen.getByRole('combobox')
  await user.type(input, '选择研究路径')

  // DOM 焦点全程留在输入框，选中项只由 aria-activedescendant 表示。
  await user.keyboard('{ArrowDown}')
  expect(input).toHaveFocus()
  expect(input.getAttribute('aria-activedescendant')).toBe('command-palette-option-0')

  await user.keyboard('{Enter}')
  expect(screen.getByText('当前路径 /pre-investment/product-pool')).toBeInTheDocument()
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

it('closes on Escape and hands focus back to the trigger', async () => {
  const user = userEvent.setup()
  mount()
  const trigger = screen.getByRole('button', { name: /navigation.search/ })
  await user.click(trigger)
  expect(screen.getByRole('dialog')).toBeInTheDocument()

  await user.keyboard('{Escape}')

  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(trigger).toHaveFocus()
})

it('explains an empty result instead of showing a blank list', async () => {
  const user = userEvent.setup()
  mount()
  await user.keyboard('{Meta>}k{/Meta}')

  await user.type(screen.getByRole('combobox'), '不存在的功能名称')

  expect(within(screen.getByRole('listbox')).queryAllByRole('option')).toHaveLength(0)
  expect(screen.getByRole('status')).toHaveTextContent('navigation.searchEmpty')
})
