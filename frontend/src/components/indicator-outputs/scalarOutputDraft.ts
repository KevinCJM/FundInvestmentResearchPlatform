import type { IndicatorDraft, ScalarOutputDefinition } from '../../services/customIndicators'

export const newScalarOutput = (label: string): ScalarOutputDefinition => ({
  id: `result_${crypto.randomUUID().replace(/-/g, '')}`,
  label, expression: '', description: '', unit: '', display_format: 'number', precision: 2, direction: 'neutral',
})

export const scalarOutputsOf = (draft: IndicatorDraft): ScalarOutputDefinition[] => draft.result_kind === 'scalar_bundle'
  ? draft.scalar_outputs ?? []
  : [{ id: 'value', label: draft.name || '指标值', expression: draft.expression, description: '', unit: draft.unit, display_format: draft.display_format, precision: draft.precision, direction: draft.direction }]

export const scalarBundlePatch = (outputs: ScalarOutputDefinition[]): Partial<IndicatorDraft> => ({
  result_kind: 'scalar_bundle', output_contract: 'scalar_bundle', output_schema_version: 1,
  scalar_outputs: outputs, expression: '', series_outputs: [], direction: 'neutral',
  rolling_source: null, rolling_transform: null,
})

export function normalizeScalarBundleDraft(draft: IndicatorDraft): IndicatorDraft {
  const { periods: _periods, ...rest } = draft
  return {
    ...rest, ...scalarBundlePatch((draft.scalar_outputs ?? []).map(({ editable_latex: _latex, ...output }) => ({
      ...output, id: output.id.trim(), label: output.label.trim(), expression: output.expression.trim(),
      unit: output.unit.trim(), precision: Math.min(8, Math.max(0, Number.isFinite(output.precision) ? output.precision : 2)),
      direction: output.direction || 'neutral',
    }))),
    name: draft.name.trim() || '未命名指标', description: draft.description.trim(),
    dsl_version: draft.dsl_version || '2.3.0', operator_registry_version: draft.operator_registry_version || '2.3.0',
    context_kind: draft.context_kind || 'single_product', parameter_schema: [], fixed_parameters: [],
  }
}
