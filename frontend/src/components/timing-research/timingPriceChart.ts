import type { EChartsOption, ScatterSeriesOption } from 'echarts'
import { timingNumber, timingPercent, timingReason, type TimingPoint, type TimingTrade } from '../../services/timingResearch'

function tradeSeries(trades: TimingTrade[], side: 'buy' | 'sell', dates: Set<string>): ScatterSeriesOption {
  const buying = side === 'buy'
  const letter = buying ? 'B' : 'S'
  const name = buying ? '买入' : '卖出'
  const color = buying ? '#be123c' : '#047857'
  return {
    id: `timing-${side}`, name, type: 'scatter', z: 5,
    symbol: 'circle', symbolSize: 5,
    itemStyle: { color, opacity: 1, borderColor: '#fff', borderWidth: 1 },
    label: {
      show: true, formatter: letter, position: buying ? 'bottom' : 'top', distance: 5,
      color, fontSize: 11, fontWeight: 700, backgroundColor: buying ? '#fff1f2' : '#ecfdf5',
      borderColor: buying ? '#fecdd3' : '#a7f3d0', borderWidth: 1, borderRadius: 4, padding: [2, 4],
    },
    // Only colliding text is hidden. Every actual execution point stays on the
    // chart, and zoom/hover reveals the corresponding letter and trade details.
    labelLayout: { hideOverlap: true },
    emphasis: { scale: 2, label: { show: true } },
    tooltip: {
      trigger: 'item', confine: true,
      formatter: params => {
        const item = Array.isArray(params) ? params[0] : params
        const point = item.data as { value: [string, number]; trade: TimingTrade }
        return [`${letter} · ${name}`, point.value[0], `模拟成交价  ${timingNumber(point.value[1], 4)}`,
          ...(buying ? [`信号日  ${point.trade.signal_date}`] : [`退出原因  ${timingReason(point.trade.reason)}`, `本笔净收益  ${timingPercent(point.trade.net_return)}`]),
        ].join('\n')
      },
    },
    data: trades.flatMap(trade => {
      const date = buying ? trade.entry_date : trade.exit_date
      const price = buying ? trade.entry_price : trade.exit_price
      return dates.has(date) && Number.isFinite(price) ? [{ name: date, value: [date, price], trade }] : []
    }),
  }
}

export function timingPriceOption(curve: TimingPoint[], trades: TimingTrade[]): EChartsOption {
  const dates = curve.map(point => point.date)
  const visibleDates = new Set(dates)
  return {
    animation: false, aria: { enabled: true },
    tooltip: { trigger: 'axis', confine: true, renderMode: 'richText', valueFormatter: value => timingNumber(typeof value === 'number' ? value : null, 4) },
    grid: { left: 50, right: 20, top: 38, bottom: 72 },
    xAxis: { type: 'category', data: dates, boundaryGap: true, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', scale: true, name: '研究价格', boundaryGap: ['12%', '12%'] },
    dataZoom: [{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }, { type: 'slider', xAxisIndex: 0, bottom: 8, height: 20 }],
    series: [
      { name: '价格', type: 'line', showSymbol: false, connectNulls: false, data: curve.map(point => point.close), itemStyle: { color: '#475569' }, lineStyle: { width: 1.5 } },
      tradeSeries(trades, 'buy', visibleDates), tradeSeries(trades, 'sell', visibleDates),
    ],
  }
}
