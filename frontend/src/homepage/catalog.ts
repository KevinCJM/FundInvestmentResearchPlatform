import { getStage, type StageId } from '../app/processRegistry'

export type HomeIcon = 'search' | 'chart' | 'send' | 'pie' | 'repeat' | 'database' | 'network' | 'market' | 'cube' | 'grid' | 'settings' | 'accounting'
export interface HomeModule { id: string; path: string; icon: HomeIcon; tone: string; stage?: StageId }
const stage = (id: string, stageId: StageId, icon: HomeIcon, tone: string): HomeModule => ({ id, stage: stageId, path: getStage(stageId).path, icon, tone })

/** Only real application destinations belong here. No demo router or editable URLs. */
export const homeModules: HomeModule[] = [
  stage('products', 'product-research', 'search', 'blue'),
  stage('decision', 'pre-investment', 'chart', 'teal'),
  stage('portfolio', 'portfolio-center', 'cube', 'blue'),
  stage('execution', 'investment-execution', 'send', 'orange'),
  stage('post', 'post-investment', 'pie', 'purple'),
  stage('feedback', 'feedback', 'repeat', 'pink'),
  { id: 'data', path: '/settings/source-center', icon: 'database', tone: 'blue' },
  { id: 'metrics', path: '/settings/indicators-models', icon: 'chart', tone: 'teal' },
  { id: 'scenarios', path: '/settings/scenario-algorithms', icon: 'network', tone: 'purple' },
  { id: 'market', path: '/product-research/panorama', icon: 'market', tone: 'blue' },
  stage('accounting', 'fund-accounting', 'accounting', 'teal'),
  stage('solutions', 'portfolio-solutions', 'grid', 'blue'),
  stage('settings', 'settings', 'settings', 'blue'),
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
