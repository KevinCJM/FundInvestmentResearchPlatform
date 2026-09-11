import { getStage, type StageId } from '../app/processRegistry'
import { routeTranslationKey } from './catalogs'
import { systemText, useI18n } from './runtime'

/** Project stable route identities onto localized labels without mutating registry data. */
export function localizeStage(stage: ReturnType<typeof getStage>): ReturnType<typeof getStage> {
  const text = (path: string, field: 'label' | 'description', fallback: string) => systemText(`${routeTranslationKey(path)}${field === 'description' ? '.description' : ''}`, {}, fallback)
  return {
    ...stage,
    label: text(stage.path, 'label', stage.label),
    description: text(stage.path, 'description', stage.description),
    nodes: stage.nodes.map(node => ({ ...node, label: text(node.path, 'label', node.label), description: text(node.path, 'description', node.description) })),
    tools: stage.tools?.map(tool => ({ ...tool, label: text(tool.path, 'label', tool.label), description: text(tool.path, 'description', tool.description) })),
  }
}
export function useLocalizedStage(stageId: StageId) {
  useI18n()
  return localizeStage(getStage(stageId))
}
