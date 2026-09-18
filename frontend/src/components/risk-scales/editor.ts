import type { ReferenceInputRequest, ReferenceVersion, RiskScaleDefinition } from '../../services/riskScales'
import { RiskScaleError } from '../../services/riskScales'

export interface RiskEditor {
  definition: RiskScaleDefinition
  reference: ReferenceInputRequest
  referenceVersion: ReferenceVersion | null
  step: number
  sourceLabels: Record<string, string>
}

export const uniqueKey = () => crypto.randomUUID()
const today = () => new Date().toISOString().slice(0, 10)

export function newRiskEditor(researchDay = today()): RiskEditor {
  return {
    definition: {
      name: '', description: '', scheme_id: `scheme-${uniqueKey()}`, base_currency: 'CNY',
      risk_basis_id: 'annualized-periodic-volatility-v1', research_as_of: researchDay,
      review_due_at: null, purpose: '', reference_input_ref: { id: '', content_hash: '' },
      constraint_profile: { asset_limits: {}, group_limits: [], allow_short: false, allow_leverage: false, gross_exposure: 1 },
      segmentation: { algorithm_id: 'frontier_shape_dp_v2' },
    },
    step: 0, sourceLabels: {}, referenceVersion: null,
    reference: {
      name: '', currency: 'CNY', as_of: researchDay, calendar: 'SSE', frequency: 'daily', periods_per_year: 252,
      return_basis: 'selected_index_and_adjusted_product_total_return', fee_basis: 'source_embedded_no_additional_fee',
      fx_basis: 'same_currency_no_conversion', assets: [],
    },
  }
}

export function copyRiskEditor(definition: RiskScaleDefinition, researchDay = today()): RiskEditor {
  const editor = newRiskEditor(researchDay)
  return { ...editor, definition: { ...structuredClone(definition), scheme_id: editor.definition.scheme_id, research_as_of: researchDay,
    review_due_at: null, reference_input_ref: { id: '', content_hash: '' } } }
}

export function editRiskEditor(definition: RiskScaleDefinition, researchDay = today()): RiskEditor {
  const editor = newRiskEditor(researchDay)
  return { ...editor, definition: { ...structuredClone(definition), research_as_of: researchDay,
    reference_input_ref: { id: '', content_hash: '' } } }
}

function draftObject(value: unknown): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  return value as Record<string, any>
}
function draftStrings(value: Record<string, any>, keys: string[]) {
  if (keys.some(key => typeof value[key] !== 'string')) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
}
function draftArray(value: unknown): any[] {
  if (!Array.isArray(value)) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  return value
}

export function restoreReferenceDefinition(value: unknown): ReferenceInputRequest {
  const raw = draftObject(value)
  draftStrings(raw, ['name', 'currency', 'as_of', 'return_basis'])
  const assets = draftArray(raw.assets)
  assets.forEach(asset => {
    const item = draftObject(asset)
    draftStrings(item, ['id', 'name', 'asset_type'])
    if (item.rationale !== undefined && typeof item.rationale !== 'string') throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
    item.rationale ??= ''
    delete item.role
    draftArray(item.components).forEach(component => {
      const member = draftObject(component)
      draftStrings(member, ['kind', 'series_id', 'field'])
      delete member.product_pool_ref
    })
  })
  return raw as ReferenceInputRequest
}

export function restoreRiskEditor(value: unknown, researchDay = today()): RiskEditor {
  const raw = draftObject(value), defaults = newRiskEditor(researchDay)
  const definition = { ...defaults.definition, ...draftObject(raw.definition), research_as_of: researchDay }
  draftStrings(definition, ['name', 'scheme_id', 'base_currency', 'risk_basis_id', 'research_as_of'])
  if (definition.purpose !== undefined && typeof definition.purpose !== 'string') throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  if (definition.description !== undefined && typeof definition.description !== 'string') throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  if (definition.review_due_at !== undefined && definition.review_due_at !== null && typeof definition.review_due_at !== 'string') throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  definition.purpose ??= ''; definition.description ??= ''; definition.review_due_at ??= null
  const referenceRef = draftObject(definition.reference_input_ref ?? { id: '', content_hash: '' })
  draftStrings(referenceRef, ['id', 'content_hash']); definition.reference_input_ref = { id: referenceRef.id, content_hash: referenceRef.content_hash }
  const profile = draftObject(definition.constraint_profile), segmentation = draftObject(definition.segmentation)
  Object.values(draftObject(profile.asset_limits ?? {})).forEach(draftObject)
  draftArray(profile.group_limits ?? []).forEach(group => {
    draftStrings(draftObject(group), ['id'])
    if (draftArray(group.assets).some(asset => typeof asset !== 'string')) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  })
  if (segmentation.manual_caps != null) draftArray(segmentation.manual_caps)
  const reference = restoreReferenceDefinition({ ...defaults.reference, ...(raw.reference === undefined ? {} : draftObject(raw.reference)), as_of: researchDay })
  let referenceVersion = raw.referenceVersion ?? null
  if (referenceVersion != null) {
    const version = draftObject(referenceVersion); draftStrings(version, ['id', 'content_hash'])
    if (draftArray(version.ordered_asset_ids).some(id => typeof id !== 'string')) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
    if (String(version.definition && draftObject(version.definition).as_of) !== researchDay) {
      referenceVersion = null
      definition.reference_input_ref = { id: '', content_hash: '' }
    }
  }
  const sourceLabels = raw.sourceLabels === undefined ? {} : draftObject(raw.sourceLabels)
  if (Object.values(sourceLabels).some(label => typeof label !== 'string')) throw new RiskScaleError('DRAFT_SCHEMA_UNSUPPORTED', '')
  return { definition, reference, referenceVersion, sourceLabels,
    step: typeof raw.step === 'number' && Number.isInteger(raw.step) && raw.step >= 0 && raw.step <= 4 ? Math.min(raw.step, 3) : 0,
  }
}

export function basicsIssue(value: RiskScaleDefinition): string {
  if (!value.name.trim()) return 'basicsRequired'
  const now = today()
  if (!value.research_as_of || value.research_as_of > now) return 'datesInvalid'
  if (value.review_due_at && value.review_due_at <= value.research_as_of) return 'reviewDateInvalid'
  return ''
}

export function definitionIssue(value: RiskScaleDefinition): string {
  const base = basicsIssue(value); if (base) return base
  if (!value.reference_input_ref.id || !value.reference_input_ref.content_hash) return 'referenceRequired'
  for (const limit of Object.values(value.constraint_profile?.asset_limits ?? {})) {
    const lo = limit.min_weight ?? NaN, hi = limit.max_weight ?? NaN
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < 0 || hi > 1 || lo > hi) return 'boundsInvalid'
  }
  const groups = value.constraint_profile?.group_limits ?? []
  if (new Set(groups.map(group => group.id)).size !== groups.length || groups.some(group => !group.id.trim() || !group.assets.length
    || !Number.isFinite(group.lo) || !Number.isFinite(group.hi) || group.lo! < 0 || group.hi! > 1 || group.lo! > group.hi!)) return 'groupsInvalid'
  const adjusted = value.segmentation?.adjusted_caps
  if (adjusted && (adjusted.length !== 5 || adjusted.some((cap, index) => !Number.isFinite(cap) || cap < 0 || (index > 0 && cap <= adjusted[index - 1])))) return 'capsInvalid'
  if (value.segmentation?.algorithm_id === 'manual_volatility_bands_v1') {
    const caps = value.segmentation.manual_caps
    if (!caps || caps.length !== 5 || caps.some((cap, index) => !Number.isFinite(cap) || cap < 0 || (index > 0 && cap <= caps[index - 1]))) return 'capsInvalid'
    if ((value.segmentation.rationale ?? '').trim().length < 5) return 'rationaleRequired'
  }
  return ''
}

export function referenceIssue(value: ReferenceInputRequest): string {
  if (!value.assets.length || value.assets.some(asset => !asset.name.trim())) return 'assetsRequired'
  if (!value.assets.some(asset => asset.asset_type === 'market')) return 'marketAssetRequired'
  if (value.assets.filter(asset => asset.asset_type === 'cash').length > 1) return 'cashCountInvalid'
  if (value.assets.some(asset => asset.asset_type === 'cash'
    && (!Number.isFinite(asset.cash_return) || asset.cash_return! < -.5 || asset.cash_return! > 1 || asset.components.length > 0 || asset.rebalance != null))) return 'cashAssetInvalid'
  if (value.assets.some(asset => asset.asset_type === 'market'
    && (asset.cash_return != null || !asset.components.length || !asset.rebalance))) return 'marketAssetInvalid'
  if (value.assets.some(asset => asset.asset_type === 'market' && (asset.components.some(component => !Number.isFinite(component.weight)
    || component.weight < 0 || component.weight > 1) || Math.abs(asset.components.reduce((total, component) => total + component.weight, 0) - 1) > 1e-10))) return 'weightsInvalid'
  return ''
}
