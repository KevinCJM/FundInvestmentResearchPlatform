import type { IndicatorOperator, IndicatorOperatorParameter, IndicatorVariable } from '../../services/customIndicators'
import type { GraphValueType } from '../../services/indicatorGraph'
import { businessText, systemText } from '../../i18n/runtime'

/** Presentation only. Never feed these labels back into graph bindings or DSL source. */
const AXIS_LABELS: Record<string, string> = {
  time: '时间', asset: '资产', factor: '因子', scenario: '情景', group: '分组',
  observation: '观察值', window: '窗口内观察', component: '分量', row: '行', column: '列',
}
const PARAMETER_LABELS: Record<string, string> = {
  x: '输入值', a: '输入 A', b: '输入 B', left: '输入 A', right: '输入 B',
  numerator: '分子', denominator: '分母', base: '底数', exponent: '指数',
  window: '窗口期数', ddof: '自由度修正', min_periods: '最少有效观察数',
  mask: '判断条件', condition: '判断条件', axis: '计算维度',
}
const TYPE_TOKEN = /\b(?:scalar|series|vector|matrix|window|mask|tensor|tuple|float64|bool|boolean|unknown)\b(?:\s*<[^>]*>)?(?:\s*\[[^\]]*\])?/gi
const typeText = (key: string, fallback: string) => businessText(`valueTypes.${key}`, fallback)

function typeAxes(value: GraphValueType | string): string[] {
  if (typeof value !== 'string' && value.axes?.length) return value.axes.map(axis => axis.toLowerCase())
  const display = typeof value === 'string' ? value : value.display || ''
  return display.match(/<([^>]+)>/)?.[1].split(',').map(axis => axis.trim().toLowerCase()) || []
}

export function graphTypeLabel(value?: GraphValueType | string | null): string {
  if (!value) return typeText('unknown', '类型待检查')
  const display = typeof value === 'string' ? value : value.display || value.kind
  if (typeof value === 'string' && value.includes('|')) {
    return [...new Set(value.split('|').map(type => graphTypeLabel(type.trim())))].join('、')
  }
  if (/^same\s*\(/i.test(display)) return typeText('same', '与输入相同的数据类型')
  if (/^one_dimensional$/i.test(display)) return typeText('vector', '一维数组')
  const kind = (typeof value === 'string' ? value.match(/^\s*([a-z_]+)/i)?.[1] : value.kind)?.toLowerCase()
  const axes = typeAxes(value)
  const boolean = kind === 'mask' || kind === 'bool' || kind === 'boolean'
    || (typeof value !== 'string' && value.dtype === 'bool')
  if (boolean) {
    if (kind === 'scalar' || kind === 'bool' || kind === 'boolean') return typeText('scalar.boolean', '单个判断值（是／否）')
    if (axes.length > 1 || kind === 'matrix') return typeText('matrixMask', '条件矩阵（是／否）')
    if (axes[0] === 'time' || kind === 'series') return typeText('timeMask', '时间条件序列（是／否）')
    if (axes[0] === 'asset') return typeText('assetMask', '资产条件序列（是／否）')
    return typeText('mask', '条件判断结果（是／否）')
  }
  switch (kind) {
    case 'scalar': case 'float64': return typeText('scalar.numeric', '单个数值')
    case 'series': return typeText('series', '时间序列')
    case 'vector': return axes[0] === 'asset' ? typeText('assetVector', '资产向量') : typeText('vector', '一维数组')
    case 'window': return typeText('rollingWindow', '滚动窗口集合（中间结果）')
    case 'matrix':
      if (axes[0] === 'time' && axes[1] === 'asset') return typeText('timeAsset', '时间—资产矩阵')
      if (axes[0] === 'asset' && axes[1] === 'time') return typeText('assetTime', '资产—时间矩阵')
      if (axes[0] === 'asset' && axes[1] === 'asset') return typeText('assetSquare', '资产方阵')
      return typeText('matrix', '矩阵（二维数据）')
    case 'tensor': return typeText('tensor', '多维数组')
    case 'record': return typeText('record', '多个命名结果')
    case 'tuple': return typeText('tuple', '结构化值（兼容）')
    default: return typeText('unknown', '类型待检查')
  }
}

export function graphAxesLabel(value?: GraphValueType): string {
  if (!value) return ''
  return typeAxes(value).map((axis, index) => businessText(`axes.${axis}`, AXIS_LABELS[axis] || systemText('graph.dimension', { index: index + 1 }, '第 {{index}} 维'))).join('、')
}

export function graphConstantLabel(value: number | boolean | null): string {
  return value === null ? systemText('graph.valueRequired', {}, '待填写') : typeof value === 'boolean' ? systemText(value ? 'common.yes' : 'common.no') : String(value)
}

export function graphVariableLabel(id: string, variables: IndicatorVariable[]): string {
  const label = variables.find(variable => variable.name === id)?.label
  return businessText(`variables.${id}.label`, label || systemText('graph.inputNameMissing', {}, '输入数据（名称待补充）'))
}

export function graphOperatorLabel(id: string, operators: IndicatorOperator[]): string {
  const label = operators.find(operator => operator.name === id)?.label
  return businessText(`operators.${id}.label`, label || systemText('graph.operatorNameMissing', {}, '计算步骤（名称待补充）'))
}

export function graphParameterLabel(parameter: Pick<IndicatorOperatorParameter, 'name' | 'label'>, index?: number, operatorId?: string): string {
  const fallback = parameter.label?.replace(/\s*[（(](?:ddof|min_periods)[）)]/gi, '') || PARAMETER_LABELS[parameter.name] || (index === undefined ? systemText('graph.parameter', {}, '输入参数') : systemText('graph.inputNumber', { index: index + 1 }, '输入 {{index}}'))
  const generic = businessText(`parameters.${parameter.name}.label`, fallback)
  return operatorId ? businessText(`operators.${operatorId}.parameters.${parameter.name}.label`, generic) : generic
}

/** Translate metadata prose, not formulas, user names or user-authored step notes. */
export function createGraphTextFormatter(variables: IndicatorVariable[], operators: IndicatorOperator[]) {
  const labels = new Map<string, string>(Object.keys(PARAMETER_LABELS).map(name => [name, graphParameterLabel({ name })]))
  for (const variable of variables) {
    const label = graphVariableLabel(variable.name, variables)
    labels.set(variable.name, label)
    for (const alias of variable.aliases || []) labels.set(alias, label)
  }
  for (const operator of operators) {
    const label = graphOperatorLabel(operator.name, operators)
    labels.set(operator.name, label)
    for (const alias of operator.aliases || []) labels.set(alias, label)
  }
  return (value: unknown): string => {
    if (value === null || value === undefined || value === '') return ''
    const raw = String(value)
      .replace(/\s*[（(](?:ddof|min_periods)[）)]/gi, '')
      .replace(/same\([^)]*\)/gi, '与输入相同的数据类型')
      .replace(/one_dimensional/gi, '一维数组')
      .replace(TYPE_TOKEN, token => graphTypeLabel(token))
      .replace(/布尔\s*(?=条件判断结果)/g, '')
      .replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/g, token => labels.get(token) || ({
        NaN: '缺失值', nan: '缺失值', Inf: '无穷值', inf: '无穷值',
        True: '是', true: '是', False: '否', false: '否',
      } as Record<string, string>)[token] || token)
      .replace(/(?:单个数值|时间序列|资产向量|一维数组|矩阵（二维数据）)(?:\s*\|\s*(?:单个数值|时间序列|资产向量|一维数组|矩阵（二维数据）))+/g, alternatives => alternatives.replace(/\s*\|\s*/g, '、'))
    // Keep the server's output contract explanation: scalar and series roots differ.
    return raw.replace(/^\s*(?:\[[A-Z][A-Z0-9_]+\]|[A-Z][A-Z0-9_]{3,}\s*[:：-])\s*/u, '')
  }
}
