import type { RegimeGraphDefinition, RegimeStudy } from '../../services/regimeGraph'
import type { IndexSnapshotBindings } from '../../services/regimeProspective'

export type MarketStateStage = 'historical' | 'realtime' | 'validation'
export type ScenarioCenterArea = 'market-state' | 'events' | 'simulation'

export function marketStateStageFromQuery(params: URLSearchParams): MarketStateStage {
  const center = params.get('center')
  if (center === 'market-state') {
    const stage = params.get('stage')
    if (stage === 'realtime' || stage === 'validation') return stage
    return 'historical'
  }
  // Backward-compatible deep links from the old four-tab center.
  if (center === 'realtime' || (center === 'historical' && params.get('mode') === 'realtime')) return 'realtime'
  return 'historical'
}

export function scenarioAreaFromQuery(params: URLSearchParams): ScenarioCenterArea {
  const center = params.get('center')
  if (center === 'events' || center === 'simulation') return center
  return 'market-state'
}

export function scenarioCenterFromQuery(params: URLSearchParams): 'historical' | 'realtime' | 'events' | 'simulation' {
  const area = scenarioAreaFromQuery(params)
  if (area === 'events' || area === 'simulation') return area
  return marketStateStageFromQuery(params) === 'historical' ? 'historical' : 'realtime'
}

export const studyMode = (purpose: RegimeStudy['purpose']) => purpose === 'historical_reference' ? 'retrospective' : 'realtime'

// Only new drafts adopt the task contract automatically. Stored revisions stay untouched.
export function studyDraft(definition: RegimeGraphDefinition, purpose?: RegimeStudy['purpose']): RegimeGraphDefinition {
  return purpose && !definition.id ? { ...definition, default_mode: studyMode(purpose), study: { purpose, family: definition.study?.family || 'custom' } } : definition
}

export function bindStudyReference(definition: RegimeGraphDefinition, reference?: RegimeStudy['reference'], states?: RegimeGraphDefinition['states']): RegimeGraphDefinition {
  // Empty new models inherit the taxonomy; existing rules keep explicit mapping.
  const adoptStates = reference && states && !definition.id && !definition.graph.nodes.length
  return { ...definition, ...(adoptStates ? { states: states.map(state => ({ ...state })) } : {}), default_mode: 'realtime', study: { purpose: 'realtime_recognition', family: definition.study?.family || 'custom', ...(reference ? { reference } : {}) } }
}

export function studyMappingIssue(definition: RegimeGraphDefinition, states: Array<{ id: string }>) {
  const ids = new Set(states.map(state => state.id))
  const mapping = definition.study?.state_mapping
  if (!mapping && (definition.states.length !== ids.size || definition.states.some(state => !ids.has(state.id)))) return '状态集合不同，请显式配置每个状态的对应关系。'
  return definition.states.some(state => !ids.has(mapping ? mapping[state.id] : state.id))
    ? '请为每个识别状态明确选择参考状态；不会按名称或顺序自动对应。' : ''
}

/** Drop both authorities without adding null/undefined keys to immutable old payloads. */
export function invalidateStudyQualification(definition: RegimeGraphDefinition): RegimeGraphDefinition {
  if (!definition.study || (!('calibration_id' in definition.study) && !('qualification_id' in definition.study))) return definition
  const { calibration_id: _calibration, qualification_id: _qualification, ...study } = definition.study
  return { ...definition, study }
}

export function adoptStudyQualification(definition: RegimeGraphDefinition, calibrationId: string, qualificationId?: string, sources?: IndexSnapshotBindings): RegimeGraphDefinition {
  const clean = invalidateStudyQualification(definition)
  if (!clean.study) return clean
  const keys = ['snapshot_id', 'snapshot_generation', 'source_file', 'file_checksum']
  if (sources && (!qualificationId || !Object.keys(sources).length || Object.entries(sources).some(([id, binding]) =>
    !clean.graph.nodes.some(node => node.id === id && node.type === 'source.index') ||
    Object.keys(binding).length !== keys.length || keys.some(key => typeof binding[key as keyof typeof binding] !== 'string' || !binding[key as keyof typeof binding]) || Object.keys(binding).some(key => !keys.includes(key))))) throw new Error('续接来源不属于当前模型，不能采用。')
  const graph = sources ? { ...clean.graph, nodes: clean.graph.nodes.map(node => sources[node.id] ? { ...node, parameters: { ...node.parameters, ...sources[node.id] } } : node) } : clean.graph
  return { ...clean, graph, study: { ...clean.study, calibration_id: calibrationId, ...(qualificationId ? { qualification_id: qualificationId } : {}) } }
}
