import { fireEvent, render, screen, within } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { DataTable, ErrorPanel, LoadingPanel } from './ui'

interface Row { code: string; name: string; amount: number }

const rows: Row[] = [
  { code: 'A1', name: '沪深300ETF', amount: 1250 },
  { code: 'A2', name: '中债综合', amount: 830 },
]

const columns = [
  { header: '代码', nowrap: true, cell: (row: Row) => row.code },
  { header: '名称', cell: (row: Row) => row.name },
  { header: '规模', numeric: true, cell: (row: Row) => row.amount },
]

const table = (override: Partial<Parameters<typeof DataTable<Row>>[0]> = {}) => (
  <DataTable caption="持仓明细" columns={columns} rows={rows} rowKey={(row) => row.code} empty="当前筛选没有持仓。换一个数据域再看。" {...override} />
)

it('names the table with its caption and marks every header as a column header', () => {
  render(table())

  const found = screen.getByRole('table', { name: '持仓明细' })
  const headers = within(found).getAllByRole('columnheader')
  expect(headers.map((header) => header.textContent)).toEqual(['代码', '名称', '规模'])
  expect(headers.every((header) => header.getAttribute('scope') === 'col')).toBe(true)
})

it('right-aligns numeric columns so digits line up, and leaves text columns alone', () => {
  render(table())

  expect(screen.getByRole('columnheader', { name: '规模' })).toHaveClass('text-right')
  expect(screen.getByRole('columnheader', { name: '名称' })).toHaveClass('text-left')
  expect(screen.getByRole('cell', { name: '1250' })).toHaveClass('text-right')
  expect(screen.getByRole('cell', { name: '沪深300ETF' })).not.toHaveClass('text-right')
})

it('explains an empty result instead of rendering an empty body', () => {
  render(table({ rows: [] }))

  expect(screen.getByRole('cell', { name: '当前筛选没有持仓。换一个数据域再看。' })).toBeInTheDocument()
  expect(screen.queryByRole('cell', { name: '沪深300ETF' })).not.toBeInTheDocument()
})

it('keeps the header and the row shape while loading, and announces the wait', () => {
  render(table({ loading: '正在读取持仓…' }))

  // 骨架占住行位，数据回来时布局不跳；表头始终在，不是整块消失换一行文字。
  expect(screen.getAllByRole('columnheader')).toHaveLength(3)
  expect(screen.queryByRole('cell', { name: '沪深300ETF' })).not.toBeInTheDocument()
  expect(screen.getByRole('status')).toHaveTextContent('正在读取持仓…')
})

it('offers the horizontal scroll area to the keyboard when columns overflow', () => {
  render(table({ minWidth: '900px' }))

  const region = screen.getByLabelText('持仓明细', { selector: 'div' })
  expect(region).toHaveAttribute('tabindex', '0')
  expect(region).toHaveClass('overflow-x-auto')
})

it('caps a long list at maxHeight and pins the header above the rows that scroll under it', () => {
  render(table({ maxHeight: '24rem' }))

  const region = screen.getByLabelText('持仓明细', { selector: 'div' })
  expect(region).toHaveStyle({ maxHeight: '24rem' })
  expect(region).toHaveClass('overflow-auto')
  // 表头吸顶要自带底色：行从它下面滚过去，透明表头会透出行内容。
  expect(screen.getByRole('columnheader', { name: '名称' })).toHaveClass('sticky', 'top-0', 'bg-slate-50')
})

it('centres the working mascot in a reserved area while a panel waits for data', () => {
  const view = render(<LoadingPanel text="正在读取 LTCMA…" />)

  const waiting = screen.getByRole('status')
  expect(waiting).toHaveTextContent('正在读取 LTCMA…')
  // 8.1 的整面板例外：预留高度居中，不铺灰底色块。装饰性形象 alt=""，只能按资产查询。
  expect(waiting.className).toContain('place-items-center')
  expect(waiting.className).toContain('min-h-56')
  expect(view.container.querySelector('img[src*="mascot-working"]')).toBeInTheDocument()
  expect(view.container.querySelector('.animate-pulse')).toBeNull()
})

it('drops the mascot on request so one screen never shows two of them', () => {
  const view = render(<LoadingPanel text="正在读取 LTCMA…" mascot={false} />)

  expect(screen.getByRole('status')).toHaveTextContent('正在读取 LTCMA…')
  expect(view.container.querySelector('img[src*="mascot-working"]')).toBeNull()
})

it('centres the error mascot with the failure reason and a retry when a panel cannot load', () => {
  const view = render(<ErrorPanel message="Failed to fetch" action={<button type="button">重试</button>} />)

  // 8.3：读取失败要就近说明并给重试；形象与等待态占同一块预留高度，互斥出现。
  const failed = screen.getByRole('alert')
  expect(failed).toHaveTextContent('Failed to fetch')
  expect(failed.className).toContain('place-items-center')
  expect(failed.className).toContain('min-h-56')
  expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument()
  expect(view.container.querySelector('img[src*="mascot-error"]')).toBeInTheDocument()
})

it('drops the error mascot on request so one screen never shows two of them', () => {
  const view = render(<ErrorPanel message="Failed to fetch" mascot={false} />)

  expect(screen.getByRole('alert')).toHaveTextContent('Failed to fetch')
  expect(view.container.querySelector('img[src*="mascot-error"]')).toBeNull()
})


it('provides a readable default failure and retries without changing the existing action API', () => {
  const onRetry = vi.fn()
  const view = render(<ErrorPanel onRetry={onRetry} />)
  expect(screen.getByRole('alert')).toHaveTextContent('暂时无法读取数据，请重试。')
  fireEvent.click(screen.getByRole('button', { name: '重试' }))
  expect(onRetry).toHaveBeenCalledTimes(1)
  view.rerender(<ErrorPanel onRetry={onRetry} action={<button>返回列表</button>} />)
  expect(screen.queryByRole('button', { name: '重试' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '返回列表' })).toBeInTheDocument()
})
