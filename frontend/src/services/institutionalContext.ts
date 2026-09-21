import type { NativeNumericalExecutionAudit } from '../utils/fixedNjitExecution'
export const reviewTopics = [['tax', '税务'], ['regulation', '监管'], ['currency_hedging', '币种与对冲'], ['leverage', '杠杆'], ['special_liquidity', '特殊流动性']] as const
export type ReviewTopic = typeof reviewTopics[number][0]
export interface ReviewItem {
  topic: ReviewTopic; status: 'not_assessed' | 'pending' | 'researcher_checked' | 'not_applicable'
  reason: string; evidence: string; reviewed_on: string | null; valid_until: string | null
}
export interface BalanceSheet {
  as_of: string; currency: string; source: string; investable_assets: number | null
  outside_assets: number | null; confirmed_liabilities: number | null; uncalled_commitments: number | null
}
export interface InstitutionalContext {
  investor_type: 'personal' | 'family_office' | 'asset_manager' | 'corporate_treasury'
  purpose: string; cash_reserve_weight: number; balance_sheet: BalanceSheet | null; review_items: ReviewItem[]
}
export interface InstitutionalDiagnostics {
  balance_sheet: null | { as_of: string; currency: string; source: string; total_assets: number | null
    net_assets_after_confirmed_liabilities: number | null; uncalled_commitments: number | null; status: 'research_snapshot' }
  cash_reserve_weight: number; review_blockers: string[]; current_review_blockers: string[]
  automated_compliance: 'not_modelled'; independent_approval: false; execution: NativeNumericalExecutionAudit
}
export function newInstitution(type: InstitutionalContext['investor_type']): InstitutionalContext {
  return { investor_type: type, purpose: '', cash_reserve_weight: 0, balance_sheet: null,
    review_items: reviewTopics.map(([topic]) => ({ topic, status: 'not_assessed', reason: '', evidence: '', reviewed_on: null, valid_until: null })) }
}
export function institutionalIssue(context: InstitutionalContext | null | undefined, day: string, currency: string): string {
  if (!context) return ''
  if (context.purpose.trim().length < 3) return '请说明机构资金用途，至少3个字符。'
  if (!Number.isFinite(context.cash_reserve_weight) || context.cash_reserve_weight < 0 || context.cash_reserve_weight > 1) return '现金用途下限须在0%至100%之间。'
  const sheet = context.balance_sheet
  if (sheet && (sheet.as_of !== day || sheet.currency !== currency || sheet.source.trim().length < 3
    || [sheet.investable_assets, sheet.outside_assets, sheet.confirmed_liabilities, sheet.uncalled_commitments].some(n => n !== null && (!Number.isFinite(n) || n < 0)))) return '经济状况须同研究日、同币种，并填写来源；未提供金额保持空白。'
  if (context.review_items.some(item => (item.reviewed_on && item.reviewed_on > day)
    || (['researcher_checked', 'not_applicable'].includes(item.status) && (!item.reviewed_on || !item.valid_until
      || item.valid_until <= item.reviewed_on || item.reason.trim().length < 3 || item.evidence.trim().length < 3)))) return '已核对或不适用事项须有理由、证据、核验日及有效期；核验日不能在研究日之后。'
  return ''
}
