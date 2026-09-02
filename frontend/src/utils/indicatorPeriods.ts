export function indicatorPeriodLabel(period: string): string {
  const value = String(period || '').trim().toUpperCase()
  if (value === 'ALL') return '成立以来'
  const rolling = value.match(/^(\d+)([WMY])$/)
  if (rolling) {
    const unit = rolling[2] === 'W' ? '周' : rolling[2] === 'M' ? '月' : '年'
    return `近 ${rolling[1]} ${unit}`
  }
  const calendar = value.match(/^([WMY])(\d+)$/)
  if (!calendar) return value
  const offset = Number(calendar[2])
  if (calendar[1] === 'W') return offset === 1 ? '上周' : offset === 2 ? '上上周' : `${offset} 周前`
  if (calendar[1] === 'M') return offset === 1 ? '上月' : offset === 2 ? '上上月' : `${offset} 个月前`
  return offset === 1 ? '去年' : offset === 2 ? '前年' : `${offset} 年前`
}

export function indicatorPeriodOptionLabel(period: string): string {
  const value = String(period || '').trim().toUpperCase()
  return `${indicatorPeriodLabel(value)}（${value}）`
}
