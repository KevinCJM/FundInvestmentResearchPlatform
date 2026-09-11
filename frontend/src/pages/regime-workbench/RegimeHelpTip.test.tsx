import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import RegimeHelpTip from './RegimeHelpTip'
import RegimeWorkbenchDrawer from './RegimeWorkbenchDrawer'

it('说明脱离侧栏滚动容器显示，Escape 只收起说明而不关闭侧栏', async () => {
  const close = vi.fn()
  render(<RegimeWorkbenchDrawer title="节点参数" onClose={close}><RegimeHelpTip label="参数说明" text="完整说明" /></RegimeWorkbenchDrawer>)
  const trigger = screen.getByRole('button', { name: '参数说明' })
  await userEvent.hover(trigger)
  const tip = screen.getByRole('tooltip')
  expect(tip.parentElement).toBe(document.body)
  expect(tip).toHaveTextContent('完整说明')
  expect(trigger).toHaveAttribute('aria-describedby', tip.id)
  await userEvent.keyboard('{Escape}')
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  expect(close).not.toHaveBeenCalled()
  expect(screen.getByRole('dialog')).toBeVisible()
})

it('支持键盘与点击打开，滚动和卸载后清理悬浮说明', async () => {
  const { unmount } = render(<label>参数<RegimeHelpTip label="参数说明" text="完整说明" /><input aria-label="参数值" /></label>)
  const trigger = screen.getByRole('button', { name: '参数说明' })
  await userEvent.tab()
  expect(screen.getByRole('tooltip')).toBeVisible()
  fireEvent.scroll(document)
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  await userEvent.keyboard(' ')
  expect(screen.getByRole('tooltip')).toBeVisible()
  await userEvent.click(trigger)
  expect(screen.getByRole('textbox')).not.toHaveFocus()
  unmount()
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
})
