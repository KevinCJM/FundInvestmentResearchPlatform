import { useAllocationDraft } from '../../app/allocationJourney'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  batchUpdateProductPoolMembers,
  getProductPoolReviewData,
  type ProductPool,
  type ProductPoolMember,
  type ProductPoolMemberReviewUpdate,
  type ProductPoolResearchStatus,
  type ProductPoolReviewData,
  type ProductPoolReviewField,
  type ProductPoolUsageStatus,
} from '../../services/productPools'

const researchStatusLabels: Record<ProductPoolResearchStatus, string> = {
  pending: '待复核',
  approved: '已准入',
  watch: '观察',
  rejected: '排除',
}

const usageStatusLabels: Record<ProductPoolUsageStatus, string> = {
  normal: '正常使用',
  limited: '限额使用',
  no_new: '禁止新增',
  hold_only: '仅可继续持有',
  unavailable: '不可用',
}

const DEFAULT_BASIC_FIELDS = ['management']
const DEFAULT_SNAPSHOT_METRICS = ['return_1y']
const MAX_BASIC_FIELDS = 10
const MAX_SNAPSHOT_METRICS = 8

interface MemberDraft {
  research_status: ProductPoolResearchStatus
  usage_status: ProductPoolUsageStatus
  primary_plan_id: string
  max_weight: string
  reasons: string
  owner: string
  review_due_date: string
  valid_until: string
  substitute_group: string
}

type SortDirection = 'asc' | 'desc'
type SortKey = 'name' | 'rank' | 'score' | `basic:${string}` | `snapshot:${string}`

interface SortState {
  key: SortKey
  direction: SortDirection
}

interface CandidateReviewTableProps {
  pool: ProductPool
  busy: boolean
  setBusy: (value: boolean) => void
  onPool: (pool: ProductPool) => void
  onMessage: (message: string) => void
  onError: (message: string) => void
}

const numberFormatter = new Intl.NumberFormat('zh-CN', {
  maximumFractionDigits: 3,
})

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

function memberDraft(member: ProductPoolMember): MemberDraft {
  return {
    research_status: member.research_status,
    usage_status: member.usage_status,
    primary_plan_id: member.primary_plan_id ?? '',
    max_weight: member.max_weight == null ? '' : String(member.max_weight * 100),
    reasons: member.reasons.join('\n'),
    owner: member.owner,
    review_due_date: member.review_due_date ?? '',
    valid_until: member.valid_until ?? '',
    substitute_group: member.substitute_group,
  }
}

function draftSignature(draft: MemberDraft) {
  return JSON.stringify({
    ...draft,
    reasons: draft.reasons.split('\n').map((item) => item.trim()).filter(Boolean),
  })
}

function parseReasons(value: string) {
  return Array.from(new Set(value.split('\n').map((item) => item.trim()).filter(Boolean)))
}

function primaryEvidenceFor(member: ProductPoolMember, planId: string) {
  return member.evidences.find((item) => item.plan_id === planId) ?? member.evidences[0]
}

function orderedEvidences(member: ProductPoolMember, primaryPlanId: string) {
  const primary = member.evidences.find((item) => item.plan_id === primaryPlanId)
  if (!primary) return member.evidences
  return [primary, ...member.evidences.filter((item) => item !== primary)]
}

function formatReviewValue(
  value: number | string | null | undefined,
  field: ProductPoolReviewField,
) {
  if (value === null || value === undefined || value === '') return '—'
  if (field.data_type !== 'number') return String(value)
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return '—'
  if (field.display_format === 'percent') return `${numberFormatter.format(numeric * 100)}%`
  if (field.unit === 'percent_points') return `${numberFormatter.format(numeric)}%`
  if (field.unit === 'project_normalized_wan') {
    return Math.abs(numeric) >= 10_000
      ? `${numberFormatter.format(numeric / 10_000)} 亿`
      : `${numberFormatter.format(numeric)} 万`
  }
  if (field.unit === 'years') return `${numberFormatter.format(numeric)} 年`
  return numberFormatter.format(numeric)
}

function isBlank(value: unknown) {
  return value === null || value === undefined || value === ''
}

function compareValues(
  left: unknown,
  right: unknown,
  dataType: ProductPoolReviewField['data_type'],
  direction: SortDirection,
) {
  const leftBlank = isBlank(left)
  const rightBlank = isBlank(right)
  if (leftBlank || rightBlank) {
    if (leftBlank && rightBlank) return 0
    return leftBlank ? 1 : -1
  }
  let result = 0
  if (dataType === 'number') {
    result = Number(left) - Number(right)
  } else {
    result = String(left).localeCompare(String(right), 'zh-CN', {
      numeric: true,
      sensitivity: 'base',
    })
  }
  return direction === 'asc' ? result : -result
}

function SortableHeader({
  label,
  sortKey,
  sort,
  onSort,
  className = '',
}: {
  label: string
  sortKey: SortKey
  sort: SortState
  onSort: (key: SortKey) => void
  className?: string
}) {
  const active = sort.key === sortKey
  return <th scope="col" className={`sticky top-0 z-20 whitespace-nowrap bg-slate-50 px-3 py-3 ${className}`} aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}>
    <button type="button" onClick={() => onSort(sortKey)} className="inline-flex items-center gap-1 font-semibold text-slate-600 hover:text-slate-900">
      {label}<span aria-hidden="true" className={active ? 'text-emerald-700' : 'text-slate-600'}>{active ? (sort.direction === 'asc' ? '↑' : '↓') : '↕'}</span>
    </button>
  </th>
}

function ReviewValueCell({
  field,
  value,
  date,
  status,
  warning,
}: {
  field: ProductPoolReviewField
  value: number | string | null | undefined
  date?: string | null
  status?: string | null
  warning?: string | null
}) {
  const unavailable = status && !['available', 'ok', 'ready'].includes(status.toLowerCase())
  return <td className="px-3 py-3 align-top">
    <div className="whitespace-nowrap font-medium text-slate-800">{formatReviewValue(value, field)}</div>
    {field.source === 'snapshot' && <div className={`mt-1 max-w-40 text-xs ${unavailable ? 'text-amber-700' : 'text-slate-600'}`} title={warning ?? undefined}>
      {unavailable ? status : date || '快照日期未知'}
    </div>}
  </td>
}

export default function CandidateReviewTable({
  pool,
  busy,
  setBusy,
  onPool,
  onMessage,
  onError,
}: CandidateReviewTableProps) {
  const [compact, setCompact] = useState(() => window.innerWidth < 640)
  useEffect(() => {
    const update = () => setCompact(window.innerWidth < 640)
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])
  const originalDrafts = useMemo(
    () => Object.fromEntries(pool.members.map((member) => [member.key, memberDraft(member)])),
    [pool.members],
  )
  const [drafts, setDrafts] = useAllocationDraft<Record<string, MemberDraft>>(`pool-review:${pool.id}:${pool.revision}`, () => originalDrafts)
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(() => new Set())
  const [query, setQuery] = useState('')
  const [selectedBasicFields, setSelectedBasicFields] = useState<string[]>(DEFAULT_BASIC_FIELDS)
  const [selectedSnapshotMetrics, setSelectedSnapshotMetrics] = useState<string[]>(DEFAULT_SNAPSHOT_METRICS)
  const [reviewData, setReviewData] = useState<ProductPoolReviewData | null>(null)
  const [reviewLoading, setReviewLoading] = useState(false)
  const [sort, setSort] = useState<SortState>({ key: 'rank', direction: 'asc' })

  const [bulkResearchStatus, setBulkResearchStatus] = useState<'' | ProductPoolResearchStatus>('')
  const [bulkUsageStatus, setBulkUsageStatus] = useState<'' | ProductPoolUsageStatus>('')
  const [bulkMaxWeight, setBulkMaxWeight] = useState('')
  const [bulkOwner, setBulkOwner] = useState('')
  const [bulkReviewDate, setBulkReviewDate] = useState('')
  const [bulkSubstituteGroup, setBulkSubstituteGroup] = useState('')
  const [bulkReason, setBulkReason] = useState('')
  const selectAllRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setReviewLoading(true)
    getProductPoolReviewData(pool.id, {
      basicFields: selectedBasicFields,
      snapshotMetrics: selectedSnapshotMetrics,
      signal: controller.signal,
    }).then((payload) => {
      if (payload.pool_revision === pool.revision) setReviewData(payload)
    }).catch((reason) => {
      if ((reason as DOMException)?.name !== 'AbortError') {
        onError(messageOf(reason, '基础信息和快照指标加载失败。'))
      }
    }).finally(() => {
      if (!controller.signal.aborted) setReviewLoading(false)
    })
    return () => controller.abort()
  }, [onError, pool.id, pool.revision, selectedBasicFields, selectedSnapshotMetrics])

  const reviewRows = useMemo(
    () => new Map((reviewData?.rows ?? []).map((row) => [row.key, row])),
    [reviewData],
  )
  const basicFieldMap = useMemo(
    () => new Map((reviewData?.basic_fields ?? []).map((field) => [field.field, field])),
    [reviewData],
  )
  const snapshotFieldMap = useMemo(
    () => new Map((reviewData?.snapshot_metric_fields ?? []).map((field) => [field.field, field])),
    [reviewData],
  )
  const displayedBasicFields = selectedBasicFields.flatMap((field) => {
    const definition = basicFieldMap.get(field)
    return definition ? [definition] : []
  })
  const displayedSnapshotFields = selectedSnapshotMetrics.flatMap((field) => {
    const definition = snapshotFieldMap.get(field)
    return definition ? [definition] : []
  })

  const dirtyKeys = useMemo(() => new Set(
    pool.members.flatMap((member) => {
      const draft = drafts[member.key]
      const original = originalDrafts[member.key]
      return draft && original && draftSignature(draft) !== draftSignature(original)
        ? [member.key]
        : []
    }),
  ), [drafts, originalDrafts, pool.members])

  const sortField = useMemo(() => {
    if (sort.key.startsWith('basic:')) return basicFieldMap.get(sort.key.slice(6))
    if (sort.key.startsWith('snapshot:')) return snapshotFieldMap.get(sort.key.slice(9))
    return undefined
  }, [basicFieldMap, snapshotFieldMap, sort.key])

  const visibleMembers = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    const filtered = keyword
      ? pool.members.filter((member) => `${member.name} ${member.code} ${member.product_id}`.toLowerCase().includes(keyword))
      : pool.members
    const indexed = filtered.map((member, index) => ({ member, index }))
    const valueOf = (member: ProductPoolMember): unknown => {
      const draft = drafts[member.key] ?? originalDrafts[member.key]
      const evidence = primaryEvidenceFor(member, draft?.primary_plan_id ?? '')
      if (sort.key === 'name') return member.name || member.code
      if (sort.key === 'rank') return evidence?.rank
      if (sort.key === 'score') return evidence?.score
      const row = reviewRows.get(member.key)
      if (sort.key.startsWith('basic:')) return row?.basic_values[sort.key.slice(6)]
      return row?.snapshot_values[sort.key.slice(9)]
    }
    const dataType = sort.key === 'rank' || sort.key === 'score'
      ? 'number'
      : sort.key === 'name'
        ? 'text'
        : sortField?.data_type ?? 'text'
    indexed.sort((left, right) => {
      const compared = compareValues(valueOf(left.member), valueOf(right.member), dataType, sort.direction)
      return compared || left.index - right.index
    })
    return indexed.map((item) => item.member)
  }, [drafts, originalDrafts, pool.members, query, reviewRows, sort, sortField])

  const allVisibleSelected = visibleMembers.length > 0 && visibleMembers.every((member) => selectedKeys.has(member.key))
  const someVisibleSelected = visibleMembers.some((member) => selectedKeys.has(member.key))
  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = someVisibleSelected && !allVisibleSelected
    }
  }, [allVisibleSelected, someVisibleSelected])

  const requestSort = (key: SortKey) => {
    setSort((current) => {
      if (current.key === key) {
        return { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
      }
      const field = key.startsWith('basic:')
        ? basicFieldMap.get(key.slice(6))
        : key.startsWith('snapshot:')
          ? snapshotFieldMap.get(key.slice(9))
          : undefined
      const direction: SortDirection = key === 'rank' || key === 'name' || field?.data_type !== 'number' ? 'asc' : 'desc'
      return { key, direction }
    })
  }

  const updateDraft = (key: string, patch: Partial<MemberDraft>) => {
    setDrafts((current) => ({
      ...current,
      [key]: { ...current[key], ...patch },
    }))
  }

  const toggleSelected = (key: string) => {
    setSelectedKeys((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const toggleAllVisible = () => {
    setSelectedKeys((current) => {
      const next = new Set(current)
      if (allVisibleSelected) visibleMembers.forEach((member) => next.delete(member.key))
      else visibleMembers.forEach((member) => next.add(member.key))
      return next
    })
  }

  const toggleColumn = (source: 'basic' | 'snapshot', field: string) => {
    const selected = source === 'basic' ? selectedBasicFields : selectedSnapshotMetrics
    const setter = source === 'basic' ? setSelectedBasicFields : setSelectedSnapshotMetrics
    const limit = source === 'basic' ? MAX_BASIC_FIELDS : MAX_SNAPSHOT_METRICS
    if (selected.includes(field)) {
      setter(selected.filter((item) => item !== field))
      return
    }
    if (selected.length >= limit) {
      onError(`${source === 'basic' ? '基础信息' : '快照指标'}最多展示 ${limit} 列。`)
      return
    }
    setter([...selected, field])
  }

  function applyBulk(saveImmediately: boolean) {
    const keys = Array.from(selectedKeys)
    if (keys.length === 0) {
      onError('请先选择需要批量处理的产品。')
      return
    }
    const hasPatch = Boolean(
      bulkResearchStatus
      || bulkUsageStatus
      || bulkMaxWeight.trim()
      || bulkOwner.trim()
      || bulkReviewDate
      || bulkSubstituteGroup.trim()
      || bulkReason.trim(),
    )
    if (!hasPatch) {
      onError('请至少填写一个批量设置项。')
      return
    }
    const next = { ...drafts }
    const changedKeys: string[] = []
    for (const key of keys) {
      const before = next[key]
      if (!before) continue
      const current = { ...before }
      if (bulkResearchStatus) current.research_status = bulkResearchStatus
      if (bulkUsageStatus) current.usage_status = bulkUsageStatus
      if (bulkMaxWeight.trim()) current.max_weight = bulkMaxWeight.trim()
      if (bulkOwner.trim()) current.owner = bulkOwner.trim()
      if (bulkReviewDate) current.review_due_date = bulkReviewDate
      if (bulkSubstituteGroup.trim()) current.substitute_group = bulkSubstituteGroup.trim()
      if (bulkReason.trim()) {
        current.reasons = Array.from(new Set([...parseReasons(current.reasons), bulkReason.trim()])).join('\n')
      }
      next[key] = current
      if (draftSignature(current) !== draftSignature(before)) changedKeys.push(key)
    }
    setDrafts(next)
    if (changedKeys.length === 0) {
      onMessage(`已选 ${keys.length} 个产品均已满足当前批量设置，无需保存。`)
      return
    }
    if (saveImmediately) {
      void saveKeys(changedKeys, next)
    } else {
      onMessage(`已对 ${changedKeys.length}/${keys.length} 个产品应用批量设置，尚未保存。`)
    }
  }

  function buildUpdate(member: ProductPoolMember, draft: MemberDraft) {
    if (!draft.primary_plan_id) throw new Error(`${member.name || member.code} 未选择所属评价方案。`)
    const maxWeight = draft.max_weight.trim() ? Number(draft.max_weight) / 100 : null
    if (maxWeight !== null && (!Number.isFinite(maxWeight) || maxWeight <= 0 || maxWeight > 1)) {
      throw new Error(`${member.name || member.code} 的最大权重必须大于 0 且不超过 100%。`)
    }
    const reasons = parseReasons(draft.reasons)
    if ((draft.research_status === 'watch' || draft.research_status === 'rejected' || draft.usage_status !== 'normal') && reasons.length === 0) {
      throw new Error(`${member.name || member.code} 为观察、排除或受限状态时必须填写原因。`)
    }
    const payload: Pick<ProductPoolMember, 'kind' | 'product_id'> & ProductPoolMemberReviewUpdate = {
      kind: member.kind,
      product_id: member.product_id,
      research_status: draft.research_status,
      usage_status: draft.usage_status,
      primary_plan_id: draft.primary_plan_id,
      max_weight: maxWeight,
      reasons,
      owner: draft.owner.trim(),
      review_due_date: draft.review_due_date || null,
      valid_until: draft.valid_until || null,
      substitute_group: draft.substitute_group.trim(),
    }
    return payload
  }

  async function saveKeys(keys: string[], sourceDrafts: Record<string, MemberDraft> = drafts) {
    const keySet = new Set(keys)
    const members = pool.members.filter((member) => keySet.has(member.key))
    if (members.length === 0) {
      onError('没有可保存的产品。')
      return
    }
    let items: Array<Pick<ProductPoolMember, 'kind' | 'product_id'> & ProductPoolMemberReviewUpdate>
    try {
      items = members.map((member) => buildUpdate(member, sourceDrafts[member.key]))
    } catch (reason) {
      onError(messageOf(reason, '复核数据校验失败。'))
      return
    }
    setBusy(true)
    onError('')
    try {
      const next = await batchUpdateProductPoolMembers(pool.id, {
        revision: pool.revision,
        items,
      })
      onPool(next)
      onMessage(`已保存 ${items.length} 个产品的复核结果。`)
    } catch (reason) {
      onError(messageOf(reason, '批量保存复核结果失败。'))
    } finally {
      setBusy(false)
    }
  }

  return <section className="min-w-0 max-w-full rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">候选产品人工复核</h2>
        <p className="mt-1 text-sm text-slate-600">
          待复核 {pool.members.filter((item) => item.research_status === 'pending').length}
          {' · '}已准入 {pool.members.filter((item) => item.research_status === 'approved').length}
          {' · '}共 {pool.members.length}
        </p>
      </div>
      <p className="text-xs text-slate-600">不同评价方案的综合分数不做横向比较；排名和得分排序按主评价证据。</p>
    </div>

    <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
      <div className="flex min-w-0 w-full flex-wrap items-center gap-2 sm:w-auto">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="搜索候选产品"
          placeholder="搜索产品名称或代码"
          className="min-w-0 w-full sm:w-60 rounded-lg border border-slate-300 px-3 py-2 text-sm"
        />
        <span className="text-xs text-slate-600">显示 {visibleMembers.length} 条</span>
        {reviewLoading && <span className="text-xs text-slate-600">正在加载扩展字段…</span>}
      </div>
      <details className="relative">
        <summary className="cursor-pointer list-none rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700">
          显示字段（{displayedBasicFields.length + displayedSnapshotFields.length}）
        </summary>
        <div className="absolute right-0 z-40 mt-2 w-[520px] max-w-[90vw] rounded-xl border border-slate-200 bg-white p-4 shadow-xl">
          <div>
            <div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-slate-900">基础信息</h3><span className="text-xs text-slate-600">最多 {MAX_BASIC_FIELDS} 列</span></div>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {(reviewData?.basic_fields ?? []).map((field) => <label key={field.field} className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" checked={selectedBasicFields.includes(field.field)} onChange={() => toggleColumn('basic', field.field)} />{field.label}</label>)}
            </div>
          </div>
          <div className="mt-4 border-t border-slate-100 pt-4">
            <div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-slate-900">快照指标</h3><span className="text-xs text-slate-600">最多 {MAX_SNAPSHOT_METRICS} 列</span></div>
            <div className="mt-2 grid max-h-56 gap-2 overflow-y-auto sm:grid-cols-2">
              {(reviewData?.snapshot_metric_fields ?? []).map((field) => <label key={field.field} title={field.description} className={`flex items-center gap-2 text-xs ${field.available ? 'text-slate-700' : 'text-slate-600'}`}><input type="checkbox" checked={selectedSnapshotMetrics.includes(field.field)} onChange={() => toggleColumn('snapshot', field.field)} />{field.label}</label>)}
            </div>
          </div>
        </div>
      </details>
    </div>

    <details open={!compact || selectedKeys.size > 0} className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50/60 p-3">
      <summary className="cursor-pointer text-sm font-semibold text-emerald-900 sm:hidden">批量复核 · 已选 {selectedKeys.size} 只</summary>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold text-emerald-950">批量处理 · 已选 {selectedKeys.size}</div>
        <div className="flex items-center gap-3 text-xs">
          <button type="button" onClick={toggleAllVisible} className="font-medium text-emerald-800 hover:underline">{allVisibleSelected ? '取消选择当前结果' : '选择当前结果'}</button>
          {selectedKeys.size > 0 && <button type="button" onClick={() => setSelectedKeys(new Set())} className="font-medium text-slate-600 hover:underline">清空选择</button>}
        </div>
      </div>
      <div className="mt-3 grid min-w-0 gap-2 md:grid-cols-2 xl:grid-cols-4 [&>*]:min-w-0">
        <select aria-label="批量研究结论" value={bulkResearchStatus} onChange={(event) => setBulkResearchStatus(event.target.value as '' | ProductPoolResearchStatus)} className="rounded-lg border border-emerald-200 bg-white px-2 py-2 text-sm"><option value="">研究结论：不修改</option>{Object.entries(researchStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
        <select aria-label="批量使用状态" value={bulkUsageStatus} onChange={(event) => setBulkUsageStatus(event.target.value as '' | ProductPoolUsageStatus)} className="rounded-lg border border-emerald-200 bg-white px-2 py-2 text-sm"><option value="">使用状态：不修改</option>{Object.entries(usageStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
        <input aria-label="批量最大权重" type="number" min="0.1" max="100" step="0.1" value={bulkMaxWeight} onChange={(event) => setBulkMaxWeight(event.target.value)} placeholder="最大权重（%），空白不修改" className="rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm" />
        <input aria-label="批量研究负责人" value={bulkOwner} onChange={(event) => setBulkOwner(event.target.value)} placeholder="研究负责人，空白不修改" className="rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm" />
        <input aria-label="批量复审日期" type="date" value={bulkReviewDate} onChange={(event) => setBulkReviewDate(event.target.value)} className="rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm" />
        <input aria-label="批量替代组" value={bulkSubstituteGroup} onChange={(event) => setBulkSubstituteGroup(event.target.value)} placeholder="替代组，空白不修改" className="rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm" />
        <input aria-label="批量复核原因" value={bulkReason} onChange={(event) => setBulkReason(event.target.value)} placeholder="追加一条复核原因" className="rounded-lg border border-emerald-200 bg-white px-3 py-2 text-sm xl:col-span-2" />
      </div>
      <div className="mt-3 flex flex-wrap justify-end gap-2">
        <button type="button" disabled={busy || selectedKeys.size === 0} onClick={() => applyBulk(false)} className="rounded-lg border border-emerald-700 bg-white px-3 py-2 text-sm font-semibold text-emerald-800 disabled:opacity-40">应用到已选</button>
        <button type="button" disabled={busy || selectedKeys.size === 0} onClick={() => applyBulk(true)} className="rounded-lg bg-emerald-800 px-3 py-2 text-sm font-semibold text-white disabled:bg-emerald-300">应用并保存</button>
        <button type="button" disabled={busy || dirtyKeys.size === 0} onClick={() => void saveKeys(Array.from(dirtyKeys))} className="rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white disabled:bg-slate-400">保存全部修改（{dirtyKeys.size}）</button>
      </div>
    </details>

    {compact && <div data-testid="candidate-review-cards" className="mt-4 space-y-3">{visibleMembers.map(member => {
      const draft = drafts[member.key] ?? memberDraft(member)
      const evidences = orderedEvidences(member, draft.primary_plan_id)
      const row = reviewRows.get(member.key)
      return <article key={member.key} className="min-w-0 rounded-xl border border-slate-200 p-3">
        <label className="flex min-w-0 items-start gap-2"><input type="checkbox" checked={selectedKeys.has(member.key)} onChange={() => toggleSelected(member.key)} aria-label={`选择 ${member.name}`} className="mt-1" /><span className="min-w-0 text-sm font-semibold text-slate-900">{member.name}<span className="mt-1 block font-mono text-xs font-normal text-slate-600">{member.code} · {researchStatusLabels[draft.research_status]}{dirtyKeys.has(member.key) ? ' · 未保存' : ''}</span></span></label>
        <details className="mt-3 text-sm"><summary className="cursor-pointer font-medium text-emerald-800">查看证据与复核</summary>
          <div className="mt-3 space-y-3">
            {evidences.map(evidence => <div key={evidence.plan_id} className="rounded-lg bg-slate-50 p-2 text-xs"><b>{evidence.plan_name}</b><p className="mt-1">排名 {evidence.rank ?? '未排名'} · 得分 {evidence.score == null ? '不可用' : evidence.score.toFixed(3)} · 截至 {evidence.as_of || '未记录'}</p></div>)}
            <dl className="grid grid-cols-2 gap-2 text-xs">{displayedBasicFields.map(field => <div key={field.field}><dt className="text-slate-600">{field.label}</dt><dd className="break-words">{formatReviewValue(row?.basic_values[field.field], field)}</dd></div>)}{displayedSnapshotFields.map(field => <div key={field.field}><dt className="text-slate-600">{field.label}</dt><dd>{formatReviewValue(row?.snapshot_values[field.field], field)}<span className="block text-slate-600">{row?.snapshot_statuses[field.field] && !['available', 'ok', 'ready'].includes(row.snapshot_statuses[field.field] ?? '') ? row.snapshot_statuses[field.field] : row?.snapshot_value_dates[field.field] || '日期未记录'}</span></dd></div>)}</dl>
            <div className="grid grid-cols-2 gap-2"><label className="min-w-0 text-xs">研究结论<select aria-label={`${member.name} 研究结论`} value={draft.research_status} onChange={event => updateDraft(member.key, { research_status: event.target.value as ProductPoolResearchStatus })} className="mt-1 w-full rounded-lg border p-2">{Object.entries(researchStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label className="min-w-0 text-xs">使用状态<select aria-label={`${member.name} 使用状态`} value={draft.usage_status} onChange={event => updateDraft(member.key, { usage_status: event.target.value as ProductPoolUsageStatus })} className="mt-1 w-full rounded-lg border p-2">{Object.entries(usageStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label></div>
            <label className="block text-xs">复核原因<textarea aria-label={`${member.name} 复核原因`} value={draft.reasons} onChange={event => updateDraft(member.key, { reasons: event.target.value })} rows={2} className="mt-1 w-full rounded-lg border p-2" /></label>
            <div className="grid grid-cols-2 gap-2">{(['max_weight', 'owner', 'review_due_date', 'substitute_group'] as const).map(field => {
              const label = { max_weight: '最大权重（%）', owner: '负责人', review_due_date: '复审日期', substitute_group: '替代组' }[field]
              return <label key={field} className="min-w-0 text-xs">{label}<input aria-label={`${member.name} ${label}`} type={field === 'review_due_date' ? 'date' : field === 'max_weight' ? 'number' : 'text'} value={draft[field]} onChange={event => updateDraft(member.key, { [field]: event.target.value })} className="mt-1 min-w-0 w-full rounded-lg border p-2" /></label>
            })}</div>
            <button type="button" disabled={busy || !dirtyKeys.has(member.key)} onClick={() => void saveKeys([member.key])} className="w-full rounded-lg bg-slate-900 px-3 py-2 font-semibold text-white disabled:bg-slate-300">保存复核</button>
          </div>
        </details>
      </article>
    })}{!visibleMembers.length && <p className="p-4 text-sm text-slate-600">没有匹配的候选产品。</p>}</div>}

    {!compact && <div data-testid="candidate-review-scroll" className="mt-4 min-w-0 max-w-full max-h-[680px] overflow-auto rounded-xl border border-slate-200">
      <table className="min-w-[1450px] w-full text-sm">
        <thead className="text-left text-xs text-slate-600">
          <tr>
            <th scope="col" className="sticky left-0 top-0 z-30 w-11 bg-slate-50 px-3 py-3"><input ref={selectAllRef} type="checkbox" aria-label="选择全部当前候选" checked={allVisibleSelected} onChange={toggleAllVisible} /></th>
            <SortableHeader label="产品" sortKey="name" sort={sort} onSort={requestSort} className="left-11 z-30 min-w-48" />
            <th scope="col" className="sticky top-0 z-20 min-w-56 bg-slate-50 px-3 py-3">评价方案</th>
            <SortableHeader label="排名" sortKey="rank" sort={sort} onSort={requestSort} />
            <SortableHeader label="得分" sortKey="score" sort={sort} onSort={requestSort} />
            {displayedBasicFields.map((field) => <SortableHeader key={`basic:${field.field}`} label={field.label} sortKey={`basic:${field.field}`} sort={sort} onSort={requestSort} />)}
            {displayedSnapshotFields.map((field) => <SortableHeader key={`snapshot:${field.field}`} label={field.label} sortKey={`snapshot:${field.field}`} sort={sort} onSort={requestSort} />)}
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">研究结论</th>
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">使用状态</th>
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">最大权重 / 替代组</th>
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">原因与责任人</th>
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">复审</th>
            <th scope="col" className="sticky top-0 z-20 bg-slate-50 px-3 py-3">操作</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {visibleMembers.map((member) => {
            const draft = drafts[member.key] ?? memberDraft(member)
            const evidences = orderedEvidences(member, draft.primary_plan_id)
            const row = reviewRows.get(member.key)
            const selected = selectedKeys.has(member.key)
            const dirty = dirtyKeys.has(member.key)
            const rowTone = selected ? 'bg-emerald-50/60' : draft.research_status === 'pending' ? 'bg-amber-50/40' : 'bg-white'
            return <tr key={member.key} className={rowTone}>
              <td className="sticky left-0 z-10 bg-inherit px-3 py-3 align-top"><input type="checkbox" aria-label={`选择 ${member.name || member.code}`} checked={selected} onChange={() => toggleSelected(member.key)} /></td>
              <td className="sticky left-11 z-10 min-w-48 bg-inherit px-3 py-3 align-top"><div className="font-medium text-slate-900">{member.name || member.code}{dirty && <span className="ml-1 text-emerald-700" title="有未保存修改">●</span>}</div><div className="mt-1 font-mono text-xs text-slate-600">{member.code} · {member.kind === 'etf' ? 'ETF' : '公募基金'}</div>{member.manual_exception && <span className="mt-2 inline-block rounded-lg bg-amber-100 px-2 py-0.5 text-xs text-amber-800">人工例外</span>}</td>
              <td className="min-w-64 px-3 py-3 align-top">
                <div className="space-y-1.5" aria-label={`${member.name} 对应评价方案`}>
                  {evidences.map((item) => {
                    const isPrimary = item.plan_id === draft.primary_plan_id
                    return <div key={`${member.key}:${item.plan_id}`} className="min-h-10 rounded-lg bg-accent-50 px-2.5 py-1.5 ring-1 ring-accent-100">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-medium leading-5 text-slate-800">{item.plan_name}</span>
                        {isPrimary && evidences.length > 1 && <span className="shrink-0 rounded-lg bg-accent-100 px-1.5 py-0.5 text-xs font-semibold text-accent-700">主证据</span>}
                      </div>
                      <div className="mt-0.5 text-xs text-slate-600">方案 v{item.plan_revision}{item.source === 'manual_exception' ? ' · 人工例外' : ''}</div>
                    </div>
                  })}
                </div>
              </td>
              <td className="px-3 py-3 align-top text-right font-mono">
                <div className="space-y-1.5" aria-label={`${member.name} 评价方案排名`}>
                  {evidences.map((item) => <div key={`${member.key}:${item.plan_id}:rank`} title={item.plan_name} className={`flex min-h-10 items-center justify-end ${item.plan_id === draft.primary_plan_id ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>{item.rank ?? '—'}</div>)}
                </div>
              </td>
              <td className="px-3 py-3 align-top text-right font-mono">
                <div className="space-y-1.5" aria-label={`${member.name} 评价方案得分`}>
                  {evidences.map((item) => <div key={`${member.key}:${item.plan_id}:score`} title={item.plan_name} className={`flex min-h-10 items-center justify-end ${item.plan_id === draft.primary_plan_id ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>{item.score == null ? '—' : item.score.toFixed(3)}</div>)}
                </div>
              </td>
              {displayedBasicFields.map((field) => <ReviewValueCell key={`basic:${member.key}:${field.field}`} field={field} value={row?.basic_values[field.field]} />)}
              {displayedSnapshotFields.map((field) => <ReviewValueCell key={`snapshot:${member.key}:${field.field}`} field={field} value={row?.snapshot_values[field.field]} date={row?.snapshot_value_dates[field.field]} status={row?.snapshot_statuses[field.field]} warning={row?.snapshot_warnings[field.field]} />)}
              <td className="px-3 py-3 align-top"><select aria-label={`${member.name} 研究结论`} value={draft.research_status} onChange={(event) => updateDraft(member.key, { research_status: event.target.value as ProductPoolResearchStatus })} className="w-28 rounded-lg border border-slate-300 px-2 py-1.5">{Object.entries(researchStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></td>
              <td className="px-3 py-3 align-top"><select aria-label={`${member.name} 使用状态`} value={draft.usage_status} onChange={(event) => updateDraft(member.key, { usage_status: event.target.value as ProductPoolUsageStatus })} className="w-36 rounded-lg border border-slate-300 px-2 py-1.5">{Object.entries(usageStatusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></td>
              <td className="px-3 py-3 align-top"><div className="flex items-center gap-1"><input aria-label={`${member.name} 最大权重`} type="number" min="0.1" max="100" step="0.1" value={draft.max_weight} onChange={(event) => updateDraft(member.key, { max_weight: event.target.value })} className="w-20 rounded-lg border border-slate-300 px-2 py-1.5" /><span className="text-slate-600">%</span></div><input aria-label={`${member.name} 替代组`} value={draft.substitute_group} onChange={(event) => updateDraft(member.key, { substitute_group: event.target.value })} placeholder="替代组" className="mt-2 w-28 rounded-lg border border-slate-300 px-2 py-1.5 text-xs" /></td>
              <td className="px-3 py-3 align-top"><textarea aria-label={`${member.name} 复核原因`} rows={2} value={draft.reasons} onChange={(event) => updateDraft(member.key, { reasons: event.target.value })} placeholder="每行一个原因" className="w-56 rounded-lg border border-slate-300 px-2 py-1.5 text-xs" /><input aria-label={`${member.name} 负责人`} value={draft.owner} onChange={(event) => updateDraft(member.key, { owner: event.target.value })} placeholder="研究负责人" className="mt-2 w-56 rounded-lg border border-slate-300 px-2 py-1.5 text-xs" /></td>
              <td className="px-3 py-3 align-top"><input aria-label={`${member.name} 复审日期`} type="date" value={draft.review_due_date} onChange={(event) => updateDraft(member.key, { review_due_date: event.target.value })} className="rounded-lg border border-slate-300 px-2 py-1.5 text-xs" /></td>
              <td className="px-3 py-3 align-top"><button type="button" disabled={busy || !dirty} onClick={() => void saveKeys([member.key])} className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white disabled:bg-slate-300">保存</button></td>
            </tr>
          })}
        </tbody>
      </table>
      {visibleMembers.length === 0 && <p className="p-5 text-center text-sm text-slate-600">没有匹配的候选产品。</p>}
    </div>}
  </section>
}
