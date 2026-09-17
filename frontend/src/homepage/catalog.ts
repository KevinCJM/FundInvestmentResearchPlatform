import { getStage, type StageId } from '../app/processRegistry'

export type HomeIcon = 'search' | 'chart' | 'send' | 'pie' | 'repeat' | 'database' | 'network' | 'market' | 'cube' | 'grid' | 'settings' | 'accounting'
export interface HomeModule { id: string; path: string; icon: HomeIcon; tone: string; stage?: StageId }
// 色调只在 processRegistry 定义一次，首页与工作台共用，不在这里第二次声明。
const stage = (id: string, stageId: StageId, icon: HomeIcon): HomeModule => {
  const definition = getStage(stageId)
  return { id, stage: stageId, path: definition.path, icon, tone: definition.tone }
}

/** Only real application destinations belong here. No demo router or editable URLs. */
export const homeModules: HomeModule[] = [
  stage('products', 'product-research', 'search'),
  stage('decision', 'pre-investment', 'chart'),
  stage('portfolio', 'portfolio-center', 'cube'),
  stage('execution', 'investment-execution', 'send'),
  stage('post', 'post-investment', 'pie'),
  stage('feedback', 'feedback', 'repeat'),
  { id: 'data', path: '/settings/source-center', icon: 'database', tone: 'blue' },
  { id: 'metrics', path: '/settings/indicators-models', icon: 'chart', tone: 'teal' },
  { id: 'scenarios', path: '/settings/scenario-algorithms', icon: 'network', tone: 'purple' },
  { id: 'market', path: '/product-research/panorama', icon: 'market', tone: 'blue' },
  stage('accounting', 'fund-accounting', 'accounting'),
  stage('solutions', 'portfolio-solutions', 'grid'),
  stage('settings', 'settings', 'settings'),
]
export const homeModule = (id: string) => homeModules.find(item => item.id === id)!
export const workflowModules = ['products', 'decision', 'execution', 'post', 'feedback'].map(homeModule)
export const coreModules = ['data', 'metrics', 'scenarios', 'market'].map(homeModule)
export const quickModules = ['portfolio', 'metrics', 'scenarios', 'data', 'solutions'].map(homeModule)
export const primaryModules = ['products', 'decision', 'portfolio', 'execution', 'post'].map(homeModule)
export const moreModules = ['accounting', 'feedback', 'solutions', 'settings'].map(homeModule)
export const researchExamples = [
  { id: 'regime', module: 'scenarios', path: '/settings/scenario-algorithms/workbench' },
  { id: 'evaluation', module: 'products', path: '/product-research/evaluation' },
  { id: 'allocation', module: 'decision', path: '/pre-investment/saa' },
] as const
