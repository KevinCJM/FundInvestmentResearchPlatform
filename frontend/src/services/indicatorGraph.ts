import { apiRequest, type IndicatorDefinition, type IndicatorDraft, type IndicatorDag, type IndicatorDiagnostic, type SeriesOutputDefinition } from './customIndicators'
import type { GraphViewport } from '../components/computation-graph/types'

export type GraphBinding = { source: 'node'; node_id: string; port_id?: string } | { source: 'constant'; value: number | boolean | null }
export type AuthoringNode =
  | { id: string; kind: 'variable'; variable_id: string; label?: string }
  | { id: string; kind: 'constant'; value: number | boolean | null; label?: string }
  | { id: string; kind: 'parameter'; parameter_id: string; label?: string }
  | { id: string; kind: 'operator'; operator_id: string; arity?: number | null; arguments: Record<string, GraphBinding>; label?: string }
export type OperatorNode = Extract<AuthoringNode, { kind: 'operator' }>
export interface GraphOutput extends Pick<SeriesOutputDefinition, 'id' | 'label' | 'unit' | 'display_format' | 'precision' | 'output_measure'> {
  node_id: string | null
  port_id?: string
  direction?: 'neutral' | 'higher_better' | 'lower_better'
  description?: string
}
export interface AuthoringGraph { graph_version: 1; nodes: AuthoringNode[]; outputs: GraphOutput[] }
export interface GraphDocument { graph: AuthoringGraph; positions: Record<string, { x: number; y: number }>; viewport?: GraphViewport | null }
export interface GraphValueType { kind: string; display?: string; dtype?: string; axes?: string[]; semantic_dimension?: string; price_basis?: string | null; fields?: Record<string, GraphValueType> }
export interface GraphDiagnostic extends IndicatorDiagnostic { editor_node_id?: string; parameter_id?: string; severity?: 'error' | 'warning' }
export interface GraphResolution {
  valid: boolean; draft_revision: number; diagnostics: GraphDiagnostic[]; graph: AuthoringGraph | null
  expressions?: Record<string, string>; editable_latex?: Record<string, string>; display_latex?: Record<string, string>
  node_types?: Record<string, GraphValueType>; editor_to_compiled?: Record<string, number>
  dag?: IndicatorDag; dependencies?: string[]; definition_fingerprint?: string; compile_status: 'not_requested'
}
export interface IndicatorEditorState {
  indicator_id: string; definition_revision: number; editor_revision: number
  definition_fingerprint?: string; state: GraphDocument | null
}
export const graphContext = (draft: IndicatorDraft) => ({
  ...(draft.result_kind === 'time_series' && draft.parameter_contract_version === '1.0'
    ? { parameter_contract_version: draft.parameter_contract_version, parameter_schema: draft.parameter_schema ?? [] }
    : {}),
  context_kind: draft.context_kind || 'single_product', result_kind: draft.result_kind || 'scalar',
  dsl_version: draft.dsl_version, operator_registry_version: draft.operator_registry_version,
  numeric_kernel_version: draft.numeric_kernel_version, variable_registry_version: draft.variable_registry_version,
  data_contract_version: draft.data_contract_version, context_schema_version: draft.context_schema_version,
})
export const definitionExpressions = (draft: IndicatorDraft): Record<string, string> =>
  draft.result_kind === 'time_series'
      ? Object.fromEntries((draft.series_outputs ?? []).map(output => [output.id, output.expression]))
      : { result: draft.expression }
export const definitionSourceKey = (draft: IndicatorDraft) => JSON.stringify({
  ...graphContext(draft), expressions: definitionExpressions(draft),
  outputs: draft.series_outputs?.map(({ expression: _expression, editable_latex: _latex, ...output }) => output),
})
export const resolveIndicatorFormula = (draft: IndicatorDraft, revision: number, signal?: AbortSignal) =>
  apiRequest<GraphResolution>('/api/custom-indicators/graph/resolve', {
    method: 'POST', signal,
    body: JSON.stringify({ ...graphContext(draft), source_kind: 'formula', expressions: definitionExpressions(draft), draft_revision: revision }),
  })
export const resolveIndicatorGraph = (draft: IndicatorDraft, graph: AuthoringGraph, revision: number, signal?: AbortSignal) =>
  apiRequest<GraphResolution>('/api/custom-indicators/graph/resolve', {
    method: 'POST', signal,
    body: JSON.stringify({ ...graphContext(draft), source_kind: 'graph', graph, draft_revision: revision }),
  })
export const getIndicatorEditorState = (indicator: Pick<IndicatorDefinition, 'id' | 'revision'>, signal?: AbortSignal) =>
  apiRequest<IndicatorEditorState>(`/api/custom-indicators/${encodeURIComponent(indicator.id)}/editor-state?revision=${indicator.revision}`, { signal })
export const saveIndicatorEditorState = (indicator: Pick<IndicatorDefinition, 'id' | 'revision'>, document: GraphDocument, expectedEditorRevision: number) =>
  apiRequest<IndicatorEditorState>(`/api/custom-indicators/${encodeURIComponent(indicator.id)}/editor-state?revision=${indicator.revision}`, {
    method: 'PUT', body: JSON.stringify({ ...document, expected_editor_revision: expectedEditorRevision }),
  })
