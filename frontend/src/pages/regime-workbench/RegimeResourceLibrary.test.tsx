import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import RegimeResourceLibrary from './RegimeResourceLibrary'
import type { RegimeNodeSchema } from '../../services/regimeGraph'

const make = (id: string, label: string, category: string): RegimeNodeSchema => ({ id, label, category, inputs: [], outputs: [] })
it('只保留一个分类下拉，指标独立于数据源，删除的入口不会出现', async () => {
  const indicator = { ...make('indicator.calc_test', '布林带', 'indicator_calculation'), indicator_reference: { id: 'bands', revision: 2, definition_hash: 'hash', result_kind: 'time_series' }, inputs: [{ id: 'market_close', label: '收盘价' }], outputs: [{ id: 'upper' }, { id: 'middle' }, { id: 'lower' }] }
  const add = vi.fn()
  render(<RegimeResourceLibrary schemas={[make('source.inline', '手工输入时序', 'source'), make('source.indicator', '引用指标', 'source'), make('source.index', '指数行情', 'source'), make('source.etf', 'ETF行情', 'source'), make('source.fund', '公募基金行情', 'source'), make('source.constant', '常量', 'source'), make('filter.kalman', '卡尔曼滤波', 'filter'), indicator, { ...make('indicator.calc_bad', '组合收益', 'indicator_calculation'), available: false, unavailable_reason: '需要组合上下文' }]} onAdd={add} onOpenDataLab={vi.fn()} />)
  expect(screen.getAllByRole('combobox')).toHaveLength(1)
  expect(screen.queryByText('手工输入时序')).not.toBeInTheDocument()
  expect(screen.queryByText('引用指标')).not.toBeInTheDocument()
  await userEvent.selectOptions(screen.getByLabelText('节点类型'), 'source')
  expect(screen.getByRole('button', { name: '添加指数行情' })).toBeVisible()
  expect(screen.getByRole('button', { name: '添加ETF行情' })).toBeVisible()
  expect(screen.getByRole('button', { name: '添加公募基金行情' })).toBeVisible()
  expect(screen.queryByText('布林带')).not.toBeInTheDocument()
  await userEvent.selectOptions(screen.getByLabelText('节点类型'), 'indicator_calculation')
  expect(screen.getByText('· 3 个输出')).toBeVisible()
  expect(screen.getByRole('button', { name: '添加组合收益' })).toBeDisabled()
  await userEvent.type(screen.getByLabelText('搜索计算节点'), '布林')
  expect(screen.queryByText('组合收益')).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '添加布林带 第2版' }))
  expect(add).toHaveBeenCalledWith(indicator)
})
