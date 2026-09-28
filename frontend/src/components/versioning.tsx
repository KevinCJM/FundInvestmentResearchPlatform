/** 投前产物的版本与上游展示，口径见 docs/pre-investment/versioning.md；可用性以后端结果为准。 */
import { Link } from 'react-router-dom'
import { useI18n } from '../i18n/runtime'
import type { UpstreamRef, Usability, UsabilityReason, Versioned, VersionInfo } from '../services/versioning'
import { Badge } from './ui'

const linkClass = 'font-medium text-accent-800 underline decoration-accent-300 underline-offset-2 hover:text-accent-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500'

export function upstreamHref(ref: Pick<UpstreamRef, 'kind' | 'id'>) {
  const id = encodeURIComponent(ref.id ?? '')
  switch (ref.kind) {
    case 'mandate': return `/pre-investment/objectives/new?view=${id}`
    case 'strategic_scope': return `/pre-investment/product-pool/new?scope=strategic&strategic_universe=${id}`
    case 'product_scope': return `/pre-investment/product-pool/new?universe=${id}`
    case 'cma': return `/pre-investment/ltcma/${id}`
    case 'saa_policy': return `/pre-investment/saa/policy?baseline=${id}`
    case 'taa_decision': return `/pre-investment/taa?decision=${id}`
    default: return null
  }
}

export const isUsable = (item?: Versioned) => !item?.usable || item.usable.status === 'ready'
export const upstreamOf = (item: Versioned | undefined, kind: UpstreamRef['kind']) => item?.upstream?.find(ref => ref.kind === kind)

export function useVersionText() {
  const { s } = useI18n()
  const t = (key: string, values?: Record<string, string | number>) => s(`versioning.${key}`, values)
  const kind = (value?: string | null) => t(`kind.${value ?? 'unknown'}`)
  const reason = (item: UsabilityReason) => {
    const name = item.name ? t('quoted', { name: item.name }) : ''
    if (item.code === 'self_retired') return t('reason.selfRetired')
    if (item.code === 'self_superseded') return t('reason.selfSuperseded', { latest: item.latest_number ?? '' })
    if (item.code === 'upstream_deleted') return t('reason.upstreamDeleted', { kind: kind(item.kind), name })
    // 版本号相同说明上游本身未改版，而是它的上游改版，需要沿链更新。
    return item.number != null && item.number === item.latest_number
      ? t('reason.upstreamStale', { kind: kind(item.kind), name })
      : t('reason.upstreamSuperseded', { kind: kind(item.kind), name, latest: item.latest_number ?? '' })
  }
  return { t, kind, reason }
}

/** 自身版本号；非当前版本同时注明状态。 */
export function VersionTag({ version }: { version?: VersionInfo | null }) {
  const { t } = useVersionText()
  if (!version) return null
  return <span className="inline-flex flex-wrap items-center gap-1 align-middle">
    <span className="text-xs font-semibold tabular-nums text-slate-500">{t('number', { number: version.number })}</span>
    {version.status === 'superseded' && <Badge tone="warning">{version.latest_number ? t('newer', { latest: version.latest_number }) : t('superseded')}</Badge>}
    {(version.status === 'retired' || version.status === 'missing') && <Badge tone="warning">{t('deleted')}</Badge>}
  </span>
}

/** 一条上游：名称链接、钉住的版本号与状态。 */
export function UpstreamLink({ item, fallback }: { item?: UpstreamRef | null; fallback?: string }) {
  const { t } = useVersionText()
  if (!item?.id) return <span className="text-slate-600">{fallback ?? t('unlinked')}</span>
  const href = item.status === 'missing' ? null : upstreamHref(item)
  const name = item.name ?? item.id
  return <span className="inline-flex flex-wrap items-center gap-x-1.5 gap-y-1">
    {href ? <Link className={linkClass} to={href}>{name}</Link> : <span className="text-slate-700">{name}</span>}
    {item.number != null && <span className="text-xs font-semibold tabular-nums text-slate-500">{t('number', { number: item.number })}</span>}
    {/* 只改说明的等价改版不影响可用性，用中性色提示。 */}
    {item.status === 'superseded' && <Badge tone={item.equivalent ? 'neutral' : 'warning'}>{item.latest_number ? t(item.equivalent ? 'newerEquivalent' : 'newer', { latest: item.latest_number }) : t('superseded')}</Badge>}
    {(item.status === 'retired' || item.status === 'missing') && <Badge tone="warning">{t('deleted')}</Badge>}
    {item.status === 'current' && item.usable !== 'ready' && <Badge tone="warning">{t(item.usable === 'blocked' ? 'blocked' : 'stale')}</Badge>}
  </span>
}

/** 可用性状态徽标。 */
export function UsabilityBadge({ usable }: { usable?: Usability | null }) {
  const { t } = useVersionText()
  if (!usable) return null
  return <Badge tone={usable.status === 'ready' ? 'success' : 'warning'}>{t(usable.status)}</Badge>
}

/** 不能接入新下游工作的原因；可用时不渲染。 */
export function UsabilityNote({ usable, id }: { usable?: Usability | null; id?: string }) {
  const { reason } = useVersionText()
  if (!usable || usable.status === 'ready') return null
  return <ul id={id} className="mt-1 max-w-72 space-y-0.5 text-xs leading-5 text-amber-800">
    {usable.reasons.map((item, index) => <li key={index}>{reason(item)}</li>)}
  </ul>
}
