import type { EvaluationResult, IndicatorDefinition, ScalarOutputReference } from '../services/customIndicators'

/** Browser selection identity only. API and saved plans use structural refs. */
export const scalarSelectionKey = (indicatorId: string, outputId?: string | null): string => outputId ? `${indicatorId}::${outputId}` : indicatorId

export function scalarSelectionRef(key: string): ScalarOutputReference {
  const delimiter = key.indexOf('::')
  return delimiter < 0 ? { indicator_id: key } : { indicator_id: key.slice(0, delimiter), output_id: key.slice(delimiter + 2) }
}

/** Lightweight view options; never create separately persisted child indicators. */
export function scalarOutputOptions(items: IndicatorDefinition[]): IndicatorDefinition[] {
  return items.flatMap(indicator => {
    if (indicator.result_kind !== 'scalar_bundle') return [indicator]
    return (indicator.scalar_outputs ?? []).map(output => {
      const name = `${indicator.name} · ${output.label}`
      const option: IndicatorDefinition = {
        ...indicator, ...output,
        id: scalarSelectionKey(indicator.id, output.id),
        output_id: output.id, parent_indicator_id: indicator.id, parent_indicator_name: indicator.name,
        name, expression: output.editable_latex || output.expression,
        result_kind: 'scalar', output_contract: 'scalar', scalar_outputs: [],
        display_latex: output.display_latex,
        math_notation_version: output.math_notation_version,
      }
      if (indicator.presentation) option.presentation = {
        ...indicator.presentation, ...output,
        indicator_id: indicator.id, output_id: output.id,
        parent_indicator_name: indicator.name, name, result_kind: 'scalar', scalar_outputs: [],
        output_measure: output.output_measure || 'dimensionless',
        value_scale: output.display_format === 'percent' ? 100 : 1,
      }
      return option
    })
  })
}

/** Adapt a selected output response to existing metric-grid selection keys. */
export const selectedOutputView = (result: EvaluationResult): EvaluationResult => result.output_id && result.indicator_id
  ? { ...result, parent_indicator_id: result.indicator_id, indicator_id: scalarSelectionKey(result.indicator_id, result.output_id) }
  : result
