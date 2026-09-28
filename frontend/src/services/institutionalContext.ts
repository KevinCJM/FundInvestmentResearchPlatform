import { systemText } from '../i18n/runtime'
import type { NativeNumericalExecutionAudit } from '../utils/fixedNjitExecution'
export const reviewTopics = [['tax', systemText('preInvestment.institutionalContext.tax')], ['regulation', systemText('preInvestment.institutionalContext.regulation')], ['currency_hedging', systemText('preInvestment.institutionalContext.currencyAndHedging')], ['leverage', systemText('preInvestment.institutionalContext.leverage')], ['special_liquidity', systemText('preInvestment.institutionalContext.specialLiquidity')]] as const
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
  if (context.purpose.trim().length < 3) return systemText('preInvestment.institutionalContext.describeTheInstitutionalFundingPurposeInAt')
  if (!Number.isFinite(context.cash_reserve_weight) || context.cash_reserve_weight < 0 || context.cash_reserve_weight > 1) return systemText('preInvestment.institutionalContext.theCashPurposeFloorMustBeBetween')
  const sheet = context.balance_sheet
  if (sheet && (sheet.as_of !== day || sheet.currency !== currency || sheet.source.trim().length < 3
    || [sheet.investable_assets, sheet.outside_assets, sheet.confirmed_liabilities, sheet.uncalled_commitments].some(n => n !== null && (!Number.isFinite(n) || n < 0)))) return systemText('preInvestment.institutionalContext.economicInformationMustUseTheResearchDate')
  if (context.review_items.some(item => (item.reviewed_on && item.reviewed_on > day)
    || (['researcher_checked', 'not_applicable'].includes(item.status) && (!item.reviewed_on || !item.valid_until
      || item.valid_until <= item.reviewed_on || item.reason.trim().length < 3 || item.evidence.trim().length < 3)))) return systemText('preInvestment.institutionalContext.reviewedOrInapplicableItemsNeedRationaleEvidence')
  return ''
}
