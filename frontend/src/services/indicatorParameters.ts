import { apiRequest, type IndicatorDraft, type SeriesParameterDefinition } from './customIndicators'

export interface ParameterCandidate {
  id: string
  output_id: string
  output_label: string
  operator_id: string
  argument: string
  label: string
  value: number
  parameter_id: string | null
  source_expression?: string
  position?: number
  type: 'integer' | 'number'
  minimum: number
  maximum: number
  step: number
}
export interface ParameterInspection { contract_version: '1.0'; candidates: ParameterCandidate[] }
export const parameterDefinitionKey = (draft: IndicatorDraft) => JSON.stringify({
  outputs: draft.series_outputs, schema: draft.parameter_schema,
  version: draft.parameter_contract_version, registry: draft.operator_registry_version,
})
const requestDefinition = (draft: IndicatorDraft) => ({ ...draft, name: draft.name || '未保存指标' })
export const inspectIndicatorParameters = (draft: IndicatorDraft) => apiRequest<ParameterInspection>(
  '/api/custom-indicators/parameters/inspect', { method: 'POST', body: JSON.stringify({ definition: requestDefinition(draft) }) },
)
export const bindIndicatorParameter = (draft: IndicatorDraft, action: { candidate_id?: string; parameter_id?: string; fixed_parameter_id?: string }) =>
  apiRequest<ParameterInspection & { definition: IndicatorDraft }>('/api/custom-indicators/parameters/bind', {
    method: 'POST', body: JSON.stringify({ definition: requestDefinition(draft), ...action }),
  })

export function parameterInputIssue(spec: SeriesParameterDefinition, raw: string): 'required' | 'number' | 'integer' | 'range' | 'step' | null {
  if (!raw.trim()) return 'required'
  const value = Number(raw)
  if (!Number.isFinite(value)) return 'number'
  if (spec.type === 'integer' && !Number.isInteger(value)) return 'integer'
  if (value < spec.minimum || value > spec.maximum) return 'range'
  const units = (value - spec.minimum) / spec.step
  if (!Number.isFinite(units) || Math.abs(units - Math.round(units)) > 1e-6) return 'step'
  return null
}
