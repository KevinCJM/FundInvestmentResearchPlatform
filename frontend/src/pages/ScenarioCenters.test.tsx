import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ScenarioCenters from './ScenarioCenters'

vi.mock('./HistoricalRegimeDirectory', () => ({ default: () => <h2>V2 算法目录内容</h2> }))
vi.mock('./ScenarioAlgorithmCenter', () => ({ default: () => <h2>情景模拟与压测内容</h2> }))

describe('ScenarioCenters', () => {
  it('默认进入历史识别，并可切换到独立的模拟压测中心', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)

    expect(screen.getByRole('tab', { name: /历史情景识别/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: 'V2 算法目录内容' })).toBeInTheDocument()

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
})
