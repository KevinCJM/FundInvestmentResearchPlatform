import { useState, type ReactNode } from 'react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { act, render as renderUI, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ScenarioCenters from './ScenarioCenters'

vi.mock('./HistoricalRegimeWorkbench', () => ({
  default: function MockWorkbench({ purpose, taskFocus, onNextResearchStep }: { purpose?: string; taskFocus?: string; onNextResearchStep?: () => void }) {
    const [count, setCount] = useState(0)
    const title = purpose === 'historical_reference' ? '历史参考工作台内容' : taskFocus === 'validation' ? '识别验证工作台内容' : '实时识别工作台内容'
    return <><h2>{title}</h2><button onClick={() => setCount(value => value + 1)}>草稿修改 {count}</button>{onNextResearchStep && <button onClick={onNextResearchStep}>下一步</button>}</>
  },
}))
vi.mock('./regime-workbench/GlobalEventCenter', () => ({ default: () => <h2>全球历史事件库内容</h2> }))
vi.mock('./PublishedScenarioCenter', () => ({ default: () => <h2>情景模拟与压测内容</h2> }))
const render = (node: ReactNode) => renderUI(<MemoryRouter>{node}</MemoryRouter>)
const stepTabs = () => within(screen.getByRole('tablist', { name: '市场状态研究步骤' }))

describe('ScenarioCenters', () => {
  it('默认进入市场状态研究，并按历史参考→实时识别→验证三步切换且保留草稿', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)
    expect(screen.getByRole('tab', { name: /市场状态研究/ })).toHaveAttribute('aria-selected', 'true')
    expect(stepTabs().getByRole('tab', { name: /定义历史参考/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: '历史参考工作台内容' })).toBeVisible()
    await act(async () => { await user.click(screen.getByRole('button', { name: '草稿修改 0' })) })
    await act(async () => { await user.click(stepTabs().getByRole('tab', { name: /建立实时识别/ })) })
    expect(screen.getByRole('heading', { name: '实时识别工作台内容' })).toBeVisible()
    await act(async () => { await user.click(screen.getByRole('button', { name: '草稿修改 0' })) })
    await act(async () => { await user.click(stepTabs().getByRole('tab', { name: /验证识别能力/ })) })
    expect(screen.getByRole('heading', { name: '识别验证工作台内容' })).toBeVisible()
    expect(screen.getByRole('button', { name: '草稿修改 1' })).toBeVisible()
    await act(async () => { await user.click(stepTabs().getByRole('tab', { name: /定义历史参考/ })) })
    expect(screen.getByRole('button', { name: '草稿修改 1' })).toBeVisible()
  })

  it('旧 historical/realtime 深链接兼容为市场状态研究步骤，并保留精确版本', () => {
    function Location() { const value = useLocation(); return <output data-testid="location">{value.search}</output> }
    const first = renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=historical&mode=realtime&definition=model&revision=7']}><ScenarioCenters /><Location /></MemoryRouter>)
    expect(screen.getByRole('tab', { name: /市场状态研究/ })).toHaveAttribute('aria-selected', 'true')
    expect(stepTabs().getByRole('tab', { name: /建立实时识别/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: '实时识别工作台内容' })).toBeVisible()
    expect(screen.getByTestId('location')).toHaveTextContent('definition=model&revision=7')
    first.unmount()
    renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=historical&definition=reference&revision=2']}><ScenarioCenters /><Location /></MemoryRouter>)
    expect(stepTabs().getByRole('tab', { name: /定义历史参考/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByTestId('location')).toHaveTextContent('definition=reference&revision=2')
  })

  it('跨历史/实时步骤清理另一工作区版本，实时→验证共用同一草稿', async () => {
    function Location() { const value = useLocation(); return <output data-testid="location">{value.search}</output> }
    const user = userEvent.setup()
    renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=market-state&stage=historical&definition=history&revision=3']}><ScenarioCenters /><Location /></MemoryRouter>)
    await act(async () => { await user.click(stepTabs().getByRole('tab', { name: /建立实时识别/ })) })
    expect(screen.getByTestId('location')).not.toHaveTextContent('definition=')
    await act(async () => { await user.click(screen.getByRole('button', { name: '草稿修改 0' })) })
    await act(async () => { await user.click(stepTabs().getByRole('tab', { name: /验证识别能力/ })) })
    expect(screen.getByRole('heading', { name: '识别验证工作台内容' })).toBeVisible()
    expect(screen.getByRole('button', { name: '草稿修改 1' })).toBeVisible()
  })

  it('事件库和模拟压测保持独立一级模块，键盘可切换', async () => {
    const user = userEvent.setup()
    render(<ScenarioCenters />)
    const marketState = screen.getByRole('tab', { name: /市场状态研究/ })
    marketState.focus()
    await act(async () => { await user.keyboard('{ArrowRight}') })
    expect(screen.getByRole('tab', { name: /全球历史事件库/ })).toHaveFocus()
    expect(screen.getByRole('heading', { name: '全球历史事件库内容' })).toBeVisible()
    await act(async () => { await user.keyboard('{ArrowRight}') })
    expect(screen.getByRole('tab', { name: /情景模拟与压测/ })).toHaveFocus()
    expect(screen.getByRole('heading', { name: '情景模拟与压测内容' })).toBeVisible()
  })

  it('从产品页深链接直达模拟中心，不启动市场状态工作台', () => {
    renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=simulation']}><ScenarioCenters /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '情景模拟与压测内容' })).toBeVisible()
    expect(screen.queryByRole('heading', { name: '历史参考工作台内容' })).not.toBeInTheDocument()
  })
})

it('clicking the active center keeps the validation stage and exact deep link', async () => {
  function Location() { return <output data-testid="same-center-location">{useLocation().search}</output> }
  const user = userEvent.setup()
  renderUI(<MemoryRouter initialEntries={['/settings/scenario-algorithms?center=market-state&stage=validation&definition=model&revision=7']}><ScenarioCenters /><Location /></MemoryRouter>)
  const before = screen.getByTestId('same-center-location').textContent
  await act(async () => { await user.click(screen.getByRole('tab', { name: /市场状态研究/ })) })
  expect(screen.getByTestId('same-center-location').textContent).toBe(before)
  expect(screen.getByRole('heading', { name: '识别验证工作台内容' })).toBeVisible()
})
