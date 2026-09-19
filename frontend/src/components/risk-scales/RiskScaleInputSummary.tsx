import type { RiskScaleDefinition } from '../../services/riskScales'
import { pct, useRiskText } from './shared'

export function RiskScaleInputSummary({ definition, assetNames = {} }: { definition: RiskScaleDefinition; assetNames?: Record<string, string> }) {
  const { t } = useRiskText()
  const fields = ['name', 'purpose', 'description', 'research_as_of', 'review_due_at', 'valid_until'] as const
  const label = (id: string) => assetNames[id] || id
  return <div className="space-y-4"><dl className="grid gap-3 text-sm sm:grid-cols-2">{fields.map(key => <div key={key}><dt className="font-medium">{t(`definition.${key}`)}</dt><dd className="break-words">{String(definition[key] ?? t('unavailable'))}</dd></div>)}</dl><p className="text-sm">{t('segmentationMethod')}: {t(`algorithm.${definition.segmentation?.algorithm_id ?? 'frontier_shape_dp_v2'}`)}</p>{definition.segmentation?.rationale && <p className="text-sm">{t('manualRationale')}: {definition.segmentation.rationale}</p>}<p className="text-xs text-slate-600">{t('constraintHint')}</p><dl className="grid gap-3 text-sm sm:grid-cols-2">{Object.entries(definition.constraint_profile?.asset_limits ?? {}).map(([id, bounds]) => <div key={id}><dt>{label(id)}</dt><dd>{pct(bounds.min_weight)} / {pct(bounds.max_weight)}</dd></div>)}{definition.constraint_profile?.group_limits?.map(group => <div key={group.id}><dt>{group.id}: {group.assets.map(label).join(', ')}</dt><dd>{pct(group.lo)} / {pct(group.hi)}</dd></div>)}</dl></div>
}
