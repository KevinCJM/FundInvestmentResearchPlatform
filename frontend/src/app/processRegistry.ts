import navigation from '../../../shared/platform-navigation.json'

export type StageId =
  | 'product-research'
  | 'pre-investment'
  | 'portfolio-center'
  | 'investment-execution'
  | 'fund-accounting'
  | 'post-investment'
  | 'feedback'
  | 'portfolio-solutions'
  | 'settings'

export type CapabilityStatus = 'available' | 'partial' | 'prototype'

/** 阶段色调。取值与首页 homepage/catalog.ts 的 home-tone-* 同名，两处共用这一份定义。 */
export type StageTone = 'blue' | 'teal' | 'orange' | 'purple' | 'pink'
export interface StageAccent { badge: string; soft: string; text: string }

// Tailwind 只扫描字面量类名，拼接出来的类名会被 purge，所以这里必须写全。
// 文字色一律取 700：在白底上都不低于 4.7:1，过 WCAG AA。
const TONE_CLASSES: Record<StageTone, { soft: string; text: string }> = {
  blue: { soft: 'bg-accent-50', text: 'text-accent-700' },
  teal: { soft: 'bg-teal-50', text: 'text-teal-700' },
  orange: { soft: 'bg-orange-50', text: 'text-orange-700' },
  purple: { soft: 'bg-violet-50', text: 'text-violet-700' },
  pink: { soft: 'bg-pink-50', text: 'text-pink-700' },
}

// 色调只给图标底和眉标上色。徽章保持中性，主按钮与链接一律用 accent：
// 九个阶段各有一种主按钮颜色时，颜色就不再表示"这是主操作"；而且原来的
// emerald / rose / amber 三种阶段色会与成功 / 错误 / 警告状态撞色。
const withAccent = (stage: Omit<StageDefinition, 'accent'>): StageDefinition => ({
  ...stage,
  accent: { badge: 'bg-slate-100 text-slate-700', ...TONE_CLASSES[stage.tone] },
})

export interface ProcessNodeDefinition {
  id: string
  label: string
  description: string
  path: string
  status: CapabilityStatus
}

export interface StageToolDefinition {
  label: string
  description: string
  path: string
}

export interface StageDefinition {
  id: StageId
  order?: number
  label: string
  eyebrow: string
  description: string
  path: string
  /** 阶段色调，与首页 home-tone-* 同名。 */
  tone: StageTone
  accent: StageAccent
  nodes: ProcessNodeDefinition[]
  tools?: StageToolDefinition[]
}

// Navigation facts are shared with the server's capability catalog.
export const allStages = (navigation as Array<Omit<StageDefinition, 'accent'>>).map(withAccent)
const stage = (id: StageId) => allStages.find(item => item.id === id)!
export const businessStages = ['product-research', 'pre-investment', 'investment-execution', 'post-investment', 'feedback'].map(id => stage(id as StageId))
export const portfolioCenterStage = stage('portfolio-center')
export const accountingStage = stage('fund-accounting')
export const supportingStages = ['portfolio-center', 'portfolio-solutions', 'settings'].map(id => stage(id as StageId))

export const getStage = (stageId: StageId) => {
  const stage = allStages.find((item) => item.id === stageId)
  if (!stage) throw new Error(`Unknown process stage: ${stageId}`)
  return stage
}

export const statusLabels: Record<CapabilityStatus, string> = {
  available: '已有功能',
  partial: '部分具备',
  prototype: '静态演示',
}
