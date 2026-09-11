import type { IndicatorMeta, IndicatorOperatorParameter } from '../services/customIndicators'
import { businessText } from './runtime'

export function localizeParameter(parameter: IndicatorOperatorParameter, operatorId?: string): IndicatorOperatorParameter {
  return {
    ...parameter,
    label: operatorId
      ? businessText(`operators.${operatorId}.parameters.${parameter.name}.label`, businessText(`parameters.${parameter.name}.label`, parameter.label || parameter.name))
      : businessText(`parameters.${parameter.name}.label`, parameter.label || parameter.name),
    description: parameter.description ? businessText(`parameters.${parameter.name}.description`, parameter.description) : parameter.description,
  }
}

/** Only presentation fields are projected. Never call this on a saved definition. */
export function localizeIndicatorMeta(meta: IndicatorMeta | null): IndicatorMeta | null {
  if (!meta) return meta
  return {
    ...meta,
    variables: meta.variables.map(variable => ({
      ...variable,
      label: businessText(`variables.${variable.name}.label`, variable.label),
      description: variable.description ? businessText(`variables.${variable.name}.description`, variable.description) : variable.description,
    })),
    operators: meta.operators.map(operator => ({
      ...operator,
      label: businessText(`operators.${operator.name}.label`, operator.label),
      mathematical_essence: operator.mathematical_essence ? businessText(`operators.${operator.name}.description`, operator.mathematical_essence) : operator.mathematical_essence,
      semantic: operator.semantic ? businessText(`operators.${operator.name}.description`, operator.semantic) : operator.semantic,
      parameters: operator.parameters?.map(parameter => localizeParameter(parameter, operator.name)),
      parameter_sets: operator.parameter_sets?.map(set => ({ ...set, parameters: set.parameters.map(parameter => localizeParameter(parameter, operator.name)) })),
    })),
    indicator_types: meta.indicator_types?.map(item => ({ ...item, label: businessText(`indicatorTypes.${item.id}`, item.label) })),
    indicator_categories: meta.indicator_categories?.map(item => ({ ...item, label: businessText(`indicatorTypes.${item.id}`, item.label) })),
    series_output_measures: meta.series_output_measures?.map(item => ({ ...item, label: businessText(`measures.${item.id}`, item.label) })),
  }
}
