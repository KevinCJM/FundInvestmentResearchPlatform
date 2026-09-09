import type { IndicatorDraft, IndicatorOperator, InferenceResponse, OperatorScalarPort, ScalarOutputDefinition } from '../../services/customIndicators'
import { newScalarOutput, scalarBundlePatch } from './scalarOutputDraft'

/** UI-only recognition of direct named projections. The server validates the full DSL. */
export function namedPortSource(expression: string) {
  const text = expression.trim()
  const suffix = /\.([A-Za-z_][A-Za-z0-9_]*)$/.exec(text)
  if (!suffix) return null
  const source = text.slice(0, suffix.index).trim()
  const name = /^(?:\\operatorname\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))\s*(?:\\left)?\(/.exec(source)
  if (!name || !source.endsWith(')')) return null
  // Reject separate top-level expressions; never pretend that A(x)+B(y) is one call.
  const opening = source.indexOf('(', name.index)
  let depth = 0
  for (let index = opening; index < source.length; index += 1) {
    if (source[index] === '(') depth += 1
    else if (source[index] === ')' && --depth === 0 && index !== source.length - 1) return null
  }
  if (depth !== 0) return null
  return { operator: name[1] || name[2], expression: source, port: suffix[1] }
}

export function sharedOutputBinding(draft: IndicatorDraft, operators: IndicatorOperator[]) {
  if (draft.result_kind !== 'scalar_bundle' || !draft.scalar_outputs?.length) return null
  const parsed = draft.scalar_outputs.map(output => namedPortSource(output.expression))
  const first = parsed[0]
  if (!first || !parsed.every(item => item?.operator === first.operator && item.expression === first.expression)) return null
  const operator = operators.find(item => item.name === first.operator && item.output_ports?.length)
  if (!operator || parsed.some(item => !operator.output_ports?.some(port => port.id === item!.port))) return null
  return { operator, expression: first.expression, ports: parsed.map(item => item!.port) }
}

export type ScalarOutputChoice = OperatorScalarPort & { expression: string }

export function outputFromPort(port: ScalarOutputChoice): ScalarOutputDefinition {
  return {
    ...newScalarOutput(port.label), expression: port.expression, label: port.label,
    description: port.description, unit: port.unit, display_format: port.display_format,
    precision: port.precision, direction: port.direction, output_measure: port.output_measure,
  }
}

/** Apply authoritative compose output contracts, never infer tuple positions in the browser. */
export function composedScalarOutputs(
  draft: IndicatorDraft, response: InferenceResponse, activeId: string | undefined, preserveShared: boolean,
): { patch: Partial<IndicatorDraft>; selectedId: string } {
  const ports = response.output_ports
  if (response.shape !== 'record' || !ports?.length || ports.length > 8 || new Set(ports.map(port => port.id)).size !== ports.length) {
    throw new Error('计算未返回完整的具名结果，请重新解析。')
  }
  const previous = draft.scalar_outputs ?? []
  let outputs: ScalarOutputDefinition[]
  if (preserveShared) {
    outputs = previous.map(output => {
      const source = namedPortSource(output.expression)
      const port = ports.find(item => item.id === source?.port)
      if (!port) throw new Error('新的计算不再提供已有结果，已保留原草稿。')
      return { ...output, expression: port.expression, editable_latex: undefined }
    })
  } else {
    const added = ports.map(outputFromPort)
    const index = previous.findIndex(output => output.id === activeId)
    outputs = draft.result_kind === 'scalar_bundle' && index >= 0
      ? [...previous.slice(0, index), ...added, ...previous.slice(index + 1)]
      : added
    if (outputs.length > 8) throw new Error('展开后超过 8 个结果，请先减少结果数量。')
    activeId = added[0].id
  }
  return { patch: scalarBundlePatch(outputs), selectedId: activeId && outputs.some(item => item.id === activeId) ? activeId : outputs[0].id }
}
