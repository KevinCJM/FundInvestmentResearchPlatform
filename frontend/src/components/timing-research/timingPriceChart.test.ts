import { describe, expect, it } from 'vitest'
import { timingPriceOption } from './timingPriceChart'
import { timingRunFixture } from './timingFixtures'

const product = timingRunFixture.products[0]
const options = (trades = product.trades!) => timingPriceOption(product.curve!, trades) as any

describe('择时买卖点图表', () => {
  it('用上下分离的 B/S 小标签代替水滴，保留真实成交价格和日期', () => {
    const [line, buy, sell] = options().series
    expect(line.markPoint).toBeUndefined()
    expect(buy).toMatchObject({ type: 'scatter', symbol: 'circle', symbolSize: 5, label: { formatter: 'B', position: 'bottom' }, labelLayout: { hideOverlap: true } })
    expect(sell).toMatchObject({ type: 'scatter', label: { formatter: 'S', position: 'top' }, labelLayout: { hideOverlap: true } })
    expect(buy.data[0]).toMatchObject({ name: '2024-01-03', value: ['2024-01-03', 1] })
    expect(sell.data[0]).toMatchObject({ name: '2024-01-04', value: ['2024-01-04', 1.04] })
  })

  it('密集交易不抽样或丢弃成交点，提供缩放和悬停恢复标签', () => {
    const trades = Array.from({ length: 300 }, () => ({ ...product.trades![0] }))
    const option = options(trades)
    expect(option.series[1].data).toHaveLength(300)
    expect(option.series[2].data).toHaveLength(300)
    expect(option.series[1].emphasis.label.show).toBe(true)
    expect(option.dataZoom.map((item: any) => item.type)).toEqual(['inside', 'slider'])
    expect(option.dataZoom[0].zoomOnMouseWheel).toBe('ctrl')
  })

  it('买入和卖出悬停分别说明成交价、信号日与退出原因', () => {
    const [, buy, sell] = options().series
    expect(buy.tooltip.formatter({ data: buy.data[0] })).toBe('B · 买入\n2024-01-03\n模拟成交价  1\n信号日  2024-01-02')
    expect(sell.tooltip.formatter([{ data: sell.data[0] }])).toContain('退出原因  退出条件触发\n本笔净收益  4.00%')
    expect(options().tooltip.renderMode).toBe('richText')
  })

  it('只标记有有效成交日期和价格的端点，不伪造区间外或未知成交', () => {
    const trade = product.trades![0]
    const [, buy, sell] = options([
      { ...trade, entry_date: '2023-12-31' },
      { ...trade, entry_price: Number.NaN, exit_price: Number.POSITIVE_INFINITY },
    ]).series
    expect(buy.data).toHaveLength(0)
    expect(sell.data).toHaveLength(1)
  })

  it('无成交和空曲线正常显示，曲线断点不填充，输入结果不被修改', () => {
    const before = structuredClone(product)
    expect(options([]).series[1].data).toEqual([])
    expect((timingPriceOption([], product.trades!) as any).series[2].data).toEqual([])
    const curve = [{ ...product.curve![0], close: null }]
    const line = (timingPriceOption(curve, []) as any).series[0]
    expect(line.data).toEqual([null])
    expect(line.connectNulls).toBe(false)
    expect(product).toEqual(before)
  })
})
