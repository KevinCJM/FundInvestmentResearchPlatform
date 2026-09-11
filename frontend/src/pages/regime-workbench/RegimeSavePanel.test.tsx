import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import RegimeSavePanel from './RegimeSavePanel'
import * as api from '../../services/regimeGraph'

vi.mock('../../services/regimeGraph', async importOriginal => ({
  ...await importOriginal<typeof api>(), createRegimeGraphDefinition: vi.fn(), updateRegimeGraphDefinition: vi.fn(),
  prepareRegimeGraph: vi.fn(), enableRegimeResearchVersion: vi.fn(),
}))
const draft = { ...api.createBlankRegimeDefinition(), name: '美林时钟', description: '', states: [{ id: 'recovery', label: '复苏', color: '#22c55e' }] }
const saved = { ...draft, id: 'clock', revision: 1, default_mode: 'retrospective' as const }
const version: api.RegimeResearchVersion = { run_id: 'r1', publication_id: 'p1', definition_id: 'clock', revision: 1,
  name: '美林时钟', mode: 'retrospective', as_of: null, available_for: ['product_research', 'research_display'],
  series_summary: { row_count: 259, first_observation_date: '2005-01-31', last_observation_date: '2026-07-31' } }
const onViewResult = vi.fn()

function Host({ initial = draft, initiallyDirty = true }: { initial?: api.RegimeGraphDefinition; initiallyDirty?: boolean }) {
  const [definition, setDefinition] = useState(initial)
  const [dirty, setDirty] = useState(initiallyDirty)
  return <RegimeSavePanel definition={definition} dirty={dirty} valid mode="retrospective" asOf="" onBusy={() => {}}
    onSaved={value => { setDefinition(value); setDirty(false) }} onViewResult={onViewResult}><button>运行已保存版本</button></RegimeSavePanel>
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.createRegimeGraphDefinition).mockResolvedValue(saved)
  vi.mocked(api.updateRegimeGraphDefinition).mockImplementation(async definition => ({ ...definition, revision: (definition.revision || 0) + 1 }))
  vi.mocked(api.prepareRegimeGraph).mockResolvedValue({ compile_token: 'token' } as api.PreparedRegimeGraph)
  vi.mocked(api.enableRegimeResearchVersion).mockResolvedValue(version)
})

it('一次保存算法并启用研究，默认不展示技术管理', async () => {
  const user = userEvent.setup(); render(<Host />)
  expect(screen.queryByRole('button', { name: '运行已保存版本' })).not.toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: '保存并用于研究' }))
  await screen.findByText('美林时钟 · v1 已可选用')
  expect(api.createRegimeGraphDefinition).toHaveBeenCalledTimes(1)
  expect(api.prepareRegimeGraph).toHaveBeenCalledWith(saved)
  expect(api.enableRegimeResearchVersion).toHaveBeenCalledWith(saved, 'token', 'retrospective', '')
  expect(screen.getByText(/2005-01-31 — 2026-07-31/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '前往产品研究' })).toHaveAttribute('href', '/product-research/products')
  await user.click(screen.getByRole('button', { name: '查看情景结果' }))
  expect(onViewResult).toHaveBeenCalledWith('r1')
  await user.click(screen.getByText('高级管理'))
  await screen.findByRole('button', { name: '运行已保存版本' })
})

it('算法已保存但计算失败时，可重试且不会另建算法或修订', async () => {
  vi.mocked(api.enableRegimeResearchVersion).mockRejectedValueOnce(new Error('CPI数据缺失'))
  const user = userEvent.setup(); render(<Host />)
  await user.click(screen.getByRole('button', { name: '保存并用于研究' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('算法已保存为 v1，尚未完成研究准备。CPI数据缺失')
  await user.click(screen.getByRole('button', { name: '重试生成研究结果' }))
  await screen.findByText('美林时钟 · v1 已可选用')
  expect(api.createRegimeGraphDefinition).toHaveBeenCalledTimes(1)
  expect(api.updateRegimeGraphDefinition).not.toHaveBeenCalled()
})

it('未修改的算法不新增版本，修改名称后使用乐观修订保存', async () => {
  const user = userEvent.setup(); render(<Host initial={saved} initiallyDirty={false} />)
  await user.click(screen.getByRole('button', { name: '保存并用于研究' }))
  await screen.findByText('美林时钟 · v1 已可选用')
  expect(api.updateRegimeGraphDefinition).not.toHaveBeenCalled()
  await user.clear(screen.getByRole('textbox', { name: '情景名称' }))
  await user.type(screen.getByRole('textbox', { name: '情景名称' }), '美林时钟调整版')
  await user.click(screen.getByRole('button', { name: '保存并用于研究' }))
  await waitFor(() => expect(api.updateRegimeGraphDefinition).toHaveBeenCalledWith(expect.objectContaining({ id: 'clock', revision: 1, name: '美林时钟调整版' })))
})

it('保存时阻止重复提交和修改输入', async () => {
  let finish!: (value: api.RegimeGraphDefinition) => void
  vi.mocked(api.createRegimeGraphDefinition).mockReturnValue(new Promise(resolve => { finish = resolve }))
  render(<Host />)
  const button = screen.getByRole('button', { name: '保存并用于研究' })
  fireEvent.click(button); fireEvent.click(button)
  expect(api.createRegimeGraphDefinition).toHaveBeenCalledTimes(1)
  expect(screen.getByRole('textbox', { name: '情景名称' })).toBeDisabled()
  finish(saved)
  await screen.findByText('美林时钟 · v1 已可选用')
})
