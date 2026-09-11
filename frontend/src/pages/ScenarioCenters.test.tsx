import { useState, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { act, render as renderUI, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ScenarioCenters from './ScenarioCenters'

vi.mock('./HistoricalRegimeWorkbench', () => ({ default: function MockWorkbench() { const [count, setCount] = useState(0); return <><h2>时序算法工作台内容</h2><button onClick={() => setCount(value => value + 1)}>草稿修改 {count}</button></> } }))
vi.mock('./PublishedScenarioCenter', () => ({ default: () => <h2>情景模拟与压测内容</h2> }))
const render = (node: ReactNode) => renderUI(<MemoryRouter>{node}</MemoryRouter>)

describe('ScenarioCenters', () => {
  it('从产品页深链接直达模拟中心，不启动历史识别工作台', () => {
    renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=simulation']}><ScenarioCenters /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '情景模拟与压测内容' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '时序算法工作台内容' })).not.toBeInTheDocument()
  })
  it('默认进入历史识别，并可切换到独立的模拟压测中心', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)

    expect(screen.getByRole('tab', { name: /历史情景识别/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: '时序算法工作台内容' })).toBeInTheDocument()

    await act(async () => { await user.click(screen.getByRole('tab', { name: /情景模拟与压测/ })) })
    expect(screen.getByRole('tab', { name: /情景模拟与压测/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: '情景模拟与压测内容' })).toBeInTheDocument()
  })

  it('支持键盘左右键切换两类情景中心', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)
    const historical = screen.getByRole('tab', { name: /历史情景识别/ })
    historical.focus()
    await act(async () => { await user.keyboard('{ArrowRight}') })
    expect(screen.getByRole('tab', { name: /情景模拟与压测/ })).toHaveFocus()
    expect(screen.getByRole('heading', { name: '情景模拟与压测内容' })).toBeInTheDocument()
  })
  it('切换模拟中心再返回时保留情景草稿', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)
    await user.click(screen.getByRole('button', { name: '草稿修改 0' }))
    await user.click(screen.getByRole('tab', { name: /情景模拟与压测/ }))
    expect(screen.queryByRole('button', { name: '草稿修改 1' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /历史情景识别/ }))
    expect(screen.getByRole('button', { name: '草稿修改 1' })).toBeInTheDocument()
  })

})
