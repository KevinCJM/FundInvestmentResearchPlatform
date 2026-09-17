import type { RegimeGraphDefinition, RegimeGraphTemplate } from '../../services/regimeGraph'

export type RegimeWorkspace = 'historical' | 'events'
export const MANUAL_EVENT_TEMPLATE = 'manual-historical-events-v1'
export const MANUAL_EVENT_NODE = 'annotation.manual_events'

export function isManualEventDefinition(definition?: Pick<RegimeGraphDefinition, 'graph'>) {
  return definition?.graph.nodes.some(node => node.type === MANUAL_EVENT_NODE) ?? false
}

export function isManualEventTemplate(template: RegimeGraphTemplate) {
  return template.id === MANUAL_EVENT_TEMPLATE || isManualEventDefinition(template.definition)
}

export function eventStudyHref(target: { id?: string; revision?: number; template?: string } = {}) {
  const query = new URLSearchParams({ center: 'events', event_view: 'manual' })
  if (target.id) {
    query.set('definition', target.id)
    if (target.revision) query.set('revision', String(target.revision))
  } else if (target.template) query.set('template', target.template)
  query.set('mode', 'retrospective')
  return `/settings/scenario-algorithms?${query}`
}
