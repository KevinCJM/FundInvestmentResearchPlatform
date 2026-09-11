import { StrictMode } from 'react'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProcessHome from '../pages/ProcessHome'
import { chooseLocale, i18n, LANGUAGE_KEY, systemText } from '../i18n/runtime'
import { homeModules, researchExamples } from './catalog'
import { recordRecentVisit } from './history'

const mount = () => render(<StrictMode><MemoryRouter><ProcessHome /></MemoryRouter></StrictMode>)
describe('research landing homepage', () => {
  beforeEach(async () => {
    localStorage.clear()
    await chooseLocale('zh-CN')
    // JSDOM lacks native dialog layout/focus; actual focus containment is tested in Chromium.
    HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', '') }
    HTMLDialogElement.prototype.close = function () { this.removeAttribute('open') }
  })
  afterEach(async () => { cleanup(); vi.restoreAllMocks(); await chooseLocale('zh-CN') })
  it('renders one current homepage with real workflow, tool and data links', () => {
    mount()
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    expect(screen.getByRole('link', { name: '开始研究' })).toHaveAttribute('href', '/product-research')
    expect(within(screen.getByRole('region', { name: '完整的投研流程' })).getAllByRole('link')).toHaveLength(5)
    expect(within(screen.getByRole('region', { name: '核心能力' })).getAllByRole('link')).toHaveLength(4)
    expect(screen.getByRole('link', { name: '探索数据源' })).toHaveAttribute('href', '/settings/source-center')
    expect(screen.queryByText('未连接业务服务')).not.toBeInTheDocument()
    expect(screen.getByText(/研究示例非投资建议/)).toBeInTheDocument()
  })
  it('search uses reviewed destinations and descriptions, with honest empty results', async () => {
    const user = userEvent.setup(); mount()
    await user.click(screen.getByRole('button', { name: '搜索平台功能' }))
    expect(screen.getByRole('dialog')).toHaveAttribute('open')
    expect(document.body.style.overflow).toBe('hidden')
    const input = screen.getByRole('searchbox')
    expect(input).toHaveAttribute('maxlength', '100')
    await user.type(input, '指标')
    expect(within(screen.getByRole('dialog')).getAllByRole('link')).toHaveLength(1)
    expect(within(screen.getByRole('dialog')).getByRole('link')).toHaveAttribute('href', '/settings/indicators-models')
    await user.clear(input); await user.type(input, '<script>alert(1)</script>')
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(within(screen.getByRole('dialog')).queryByRole('link')).not.toBeInTheDocument()
    fireEvent(screen.getByRole('dialog'), new Event('cancel', { bubbles: true, cancelable: true }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(document.body.style.overflow).not.toBe('hidden')
  })
  it('shortcut opens search and is removed when the homepage unmounts', () => {
    const view = mount()
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true })
    expect(screen.getByRole('searchbox')).toBeInTheDocument()
    view.unmount()
    const event = new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, cancelable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(false)
    expect(document.body.style.overflow).not.toBe('hidden')
  })
  it('tour supports forward/back/end without generating business results', async () => {
    const user = userEvent.setup(); mount()
    await user.click(screen.getByRole('button', { name: '查看平台导览' }))
    expect(screen.getByRole('button', { name: '上一步' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '下一步' }))
    expect(screen.getByText('2 / 3')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '上一步' }))
    expect(screen.getByText('1 / 3')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '下一步' }))
    await user.click(screen.getByRole('button', { name: '下一步' }))
    expect(screen.getByText('3 / 3')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: systemText('landing.tourDone') }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
  it('examples explain boundaries and link to existing workspaces', async () => {
    const user = userEvent.setup(); mount()
    const examples = within(screen.getByRole('tabpanel')).getAllByRole('button')
    for (const [index, example] of researchExamples.entries()) {
      await user.click(examples[index])
      expect(screen.getByText(systemText('landing.exampleNotice'))).toBeInTheDocument()
      expect(screen.getByRole('link', { name: systemText('landing.openWorkspace') })).toHaveAttribute('href', example.path)
      await user.click(screen.getByRole('button', { name: '关闭' }))
    }
  })
  it('recent tab has no invented entries, supports keyboard switching and scoped clear', async () => {
    const user = userEvent.setup(); const view = mount()
    await user.click(screen.getByRole('tab', { name: '最近打开' }))
    expect(within(screen.getByRole('tabpanel')).queryByRole('link')).not.toBeInTheDocument()
    view.unmount(); recordRecentVisit('/settings/indicators-models'); mount()
    screen.getByRole('tab', { name: '研究示例' }).focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: '最近打开' })).toHaveFocus()
    expect(within(screen.getByRole('tabpanel')).getByRole('link', { name: '指标中心' })).toHaveAttribute('href', '/settings/indicators-models')
    await user.click(screen.getByRole('button', { name: systemText('landing.clearHistory') }))
    expect(within(screen.getByRole('tabpanel')).queryByRole('link')).not.toBeInTheDocument()
  })
  it('shares the app language preference and translates all dynamic catalog keys', async () => {
    const user = userEvent.setup(); mount()
    await act(async () => { await user.selectOptions(screen.getByRole('combobox'), 'en-US') })
    expect(localStorage.getItem(LANGUAGE_KEY)).toBe('en-US')
    expect(screen.getByRole('link', { name: 'Start research' })).toBeInTheDocument()
    for (const lng of ['zh-CN', 'en-US']) for (const item of homeModules) {
      expect(i18n.exists(`landing.module.${item.id}`, { lng })).toBe(true)
      expect(i18n.exists(`landing.summary.${item.id}`, { lng })).toBe(true)
    }
  })
  it('more navigation closes on Escape and outside press', async () => {
    const user = userEvent.setup(); mount()
    const button = screen.getByRole('button', { name: '更多' })
    await user.click(button)
    expect(screen.getByRole('link', { name: '基金会计' })).toHaveAttribute('href', '/fund-accounting')
    await user.keyboard('{Escape}')
    expect(button).toHaveAttribute('aria-expanded', 'false')
    expect(button).toHaveFocus()
    await user.click(button); await user.click(screen.getByRole('heading', { level: 1 }))
    expect(button).toHaveAttribute('aria-expanded', 'false')
  })
})
