import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useResearchDay } from '../app/ResearchContext'
import { updateAllocationJourney, useAllocationDraft, useAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'
import {
  createInvestableUniverseSnapshot,
  getInvestableUniverse,
  listProductPoolVersions,
  poolVersionDataAsOf,
  type InvestableUniverseSnapshot,
  type ProductPoolVersion,
} from '../services/productPools'
import { replayPoolVersion, type PoolReplayResult } from '../services/pit'

const today = () => new Date().toISOString().slice(0, 10)

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

export default function ProductPoolSelection() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [journey] = useAllocationJourney()
  const platformAsOf = useResearchDay()
  const requestedVersion = params.get('version') || ''
  const requestedUniverse = params.get('universe') || ''
  const [draft, setDraft] = useAllocationDraft(`pool:${requestedUniverse ? `universe:${requestedUniverse}` : requestedVersion || 'resume'}`, { researchDate: '', name: '投前研究可投资域', selectedIds: [] as string[], snapshotId: '', selectionEdited: false })
  const researchDate = draft.researchDate || platformAsOf || journey.researchDate || today()
  const name = draft.name
  const selectedIds = draft.selectedIds
  const [showHistory, setShowHistory] = useState(false)
  const setName = (value: string) => setDraft(current => ({ ...current, name: value }))
  const setResearchDate = (value: string) => { setDraft(current => ({ ...current, researchDate: value, snapshotId: '', selectedIds: [], selectionEdited: true })); setSnapshot(null) }
  const [versions, setVersions] = useState<ProductPoolVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [snapshot, setSnapshot] = useState<InvestableUniverseSnapshot | null>(null)
  const [replays, setReplays] = useState<Record<string, PoolReplayResult | string>>({})

  // A version whose data cut is later than the research day was screened with
  // information that day did not have. Reproducible is not the same as causal,
  // and this is the one place the difference is still fixable.
  const replay = async (versionId: string) => {
    setReplays((current) => ({ ...current, [versionId]: '回放中…' }))
    try {
      const result = await replayPoolVersion(versionId, researchDate)
      setReplays((current) => ({ ...current, [versionId]: result }))
    } catch (reason) {
      setReplays((current) => ({ ...current, [versionId]: messageOf(reason, '回放失败。') }))
    }
  }

  useEffect(() => {
    let active = true
    setLoading(true); setError(''); setReplays({})
    listProductPoolVersions({ activeOn: researchDate })
      .then((response) => {
        if (!active) return
        setVersions(response.items)
        setDraft(current => ({ ...current, selectedIds: current.selectedIds.length ? current.selectedIds.filter(id => response.items.some(item => item.id === id)) : requestedVersion && response.items.some(item => item.id === requestedVersion) ? [requestedVersion] : [] }))
      })
      .catch((reason) => { if (active) setError(messageOf(reason, '无法加载有效产品池版本。')) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [researchDate, requestedVersion])

  const restoreId = draft.snapshotId || (!draft.selectionEdited ? requestedUniverse || (!requestedVersion ? journey.universeId : '') : '')
  useEffect(() => {
    setSnapshot(null)
    if (!restoreId) return
    let active = true
    getInvestableUniverse(restoreId).then(result => {
      if (!active) return
      setSnapshot(result)
      setDraft(current => ({ ...current, snapshotId: result.id, name: current.snapshotId === result.id ? current.name : result.name, researchDate: result.research_date, selectedIds: result.version_ids ?? current.selectedIds, selectionEdited: false }))
      updateAllocationJourney({ universeId: result.id, name: result.name, researchDate: result.research_date, poolVersionIds: result.version_ids })
    }).catch(() => { if (active) setError('之前锁定的产品范围无法恢复，请重新选择。') })
    return () => { active = false }
  }, [restoreId])

  const toggle = (versionId: string) => {
    const target = versions.find(item => item.id === versionId)
    setDraft(current => ({ ...current, snapshotId: '', selectionEdited: true, selectedIds: current.selectedIds.includes(versionId)
      ? current.selectedIds.filter(id => id !== versionId)
      : [...current.selectedIds.filter(id => versions.find(item => item.id === id)?.pool_id !== target?.pool_id), versionId] }))
    setSnapshot(null)
  }
  const visibleVersions = showHistory ? versions : versions.filter(version => selectedIds.includes(version.id) || !versions.some(other => other.pool_id === version.pool_id && other.version > version.version))

  const createSnapshot = async () => {
    if (!name.trim() || selectedIds.length === 0) {
      setError('请填写名称并至少选择一个产品池版本。')
      return
    }
    setCreating(true); setError('')
    try {
      const result = await createInvestableUniverseSnapshot({
        name: name.trim(),
        research_date: researchDate,
        version_ids: selectedIds,
      })
      setSnapshot(result)
      const savedDraft = { ...draft, snapshotId: result.id, selectionEdited: false, researchDate: result.research_date, selectedIds: result.version_ids ?? selectedIds }
      if (requestedUniverse && requestedUniverse !== result.id) {
        writeAllocationDraft(`pool:universe:${result.id}`, savedDraft)
        setParams({ universe: result.id }, { replace: true })
      } else setDraft(savedDraft)
      updateAllocationJourney({ name: result.name, researchDate: result.research_date, universeId: result.id, poolVersionIds: result.version_ids })
    } catch (reason) {
      setError(messageOf(reason, '生成可投资域快照失败。'))
    } finally {
      setCreating(false)
    }
  }

  return <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" aria-busy={loading || creating}>
    <header className="rounded-2xl bg-slate-900 px-5 py-6 text-white sm:px-7">
      <p className="text-sm font-medium text-emerald-300">投前研究 · 研究边界</p>
      <h1 className="mt-1 text-2xl font-semibold">选择产品池版本</h1>
      <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">选择产品范围并命名本次研究，再继续构建大类。每个产品池选一个发布版本，之后各步骤使用同一范围。</p>
    </header>

    {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="grid gap-4 md:grid-cols-[180px_minmax(0,1fr)_auto] md:items-end">
        <label className="text-sm text-slate-700">研究日期<input type="date" value={researchDate} onChange={(event) => setResearchDate(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <label className="text-sm text-slate-700">研究名称<input value={name} onChange={(event) => setName(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <button type="button" disabled={creating || selectedIds.length === 0} onClick={() => void createSnapshot()} className="rounded-lg bg-emerald-800 px-5 py-2 text-sm font-semibold text-white disabled:bg-emerald-300">生成锁定快照</button>
      </div>
      <p className="mt-3 text-xs text-slate-500">{platformAsOf ? `平台知识截止 ${platformAsOf}；本页日期用于筛选版本生效范围，不等于历史时点认证。` : '平台使用全部磁盘数据；产品池研究日只定义范围，历史可得性仍须单独验证。'}</p>
      {platformAsOf && researchDate > platformAsOf && <p role="alert" className="mt-2 text-sm text-amber-800">产品范围研究日晚于平台知识截止。后续计算受平台口径限制，请先统一日期。</p>}
    </section>

    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-end justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-900">有效产品池版本</h2><p className="mt-1 text-sm text-slate-500">已选择 {selectedIds.length} 个版本</p></div><span className="text-xs text-slate-500">研究日：{researchDate}</span></div>
      <label className="mt-3 flex items-center gap-2 text-sm text-slate-600"><input type="checkbox" checked={showHistory} onChange={event => setShowHistory(event.target.checked)} />显示历史发布版本</label>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{visibleVersions.map((version) => {
        const selected = selectedIds.includes(version.id)
        const dataAsOf = poolVersionDataAsOf(version)
        const lookahead = Boolean(dataAsOf && dataAsOf > researchDate)
        const replayed = replays[version.id]
        return <div key={version.id} className={`rounded-xl border p-4 ${selected ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200'}`}>
          <label className="flex cursor-pointer items-start gap-3"><input type="checkbox" checked={selected} onChange={() => toggle(version.id)} className="mt-1 h-4 w-4" /><div className="min-w-0"><h3 className="font-semibold text-slate-900">{version.pool_name} · V{version.version}</h3><p className="mt-1 text-xs text-slate-500">{version.effective_from} ～ {version.effective_to || '持续有效'}</p><p className="mt-2 text-sm text-slate-700">{version.evaluation_plans.length} 套评价方案 · {version.investable_count} 个可投资产品</p><details className="mt-2 text-xs text-slate-500"><summary className="cursor-pointer">版本标识</summary><p className="break-all font-mono">{version.id}</p></details></div></label>
          <div className="mt-3 border-t border-slate-100 pt-2 text-[11px] leading-5">
            <span className={`rounded border px-1.5 py-0.5 font-semibold ${lookahead ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
              评价数据截至 {dataAsOf ?? '未记录'}
            </span>
            {lookahead && <span className="ml-2 text-rose-700">晚于研究日 {researchDate}，名单含未来信息</span>}
            <button type="button" onClick={() => void replay(version.id)} className="ml-2 underline hover:no-underline">按研究日回放</button>
            {typeof replayed === 'string' && <p className="mt-1 text-slate-500">{replayed}</p>}
            {replayed && typeof replayed !== 'string' && (
              <p className="mt-1 text-slate-700">
                回放到 {replayed.as_of}：保留 {replayed.summary.kept} · 新增 {replayed.summary.added} · 移出 {replayed.summary.removed}
                {replayed.summary.manual_only > 0 && ` · 人工准入 ${replayed.summary.manual_only}（不可回放）`}
              </p>
            )}
          </div>
        </div>
      })}</div>
      {!loading && versions.length === 0 && <p className="mt-4 rounded-lg bg-slate-50 p-6 text-center text-sm text-slate-500">该日期没有已生效产品池版本。<Link to="/product-research/pools" className="ml-2 text-emerald-800 underline">去创建或发布产品池</Link></p>}
    </section>

    {snapshot && <section className="space-y-5 rounded-2xl border border-emerald-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-sm font-medium text-emerald-700">可投资域快照已锁定</p><h2 className="mt-1 text-xl font-semibold text-slate-900">{snapshot.name}</h2></div><div className="flex flex-wrap gap-2"><button type="button" onClick={() => navigate(`/pre-investment/saa/auto-classification?universe=${encodeURIComponent(snapshot.id)}`)} className="rounded-lg bg-violet-700 px-4 py-2 text-sm font-semibold text-white">进入自动构建大类</button><button type="button" onClick={() => navigate(`/pre-investment/saa/asset-classes?universe=${encodeURIComponent(snapshot.id)}`)} className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white">进入手动构建大类</button></div></div>
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-4"><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">产品池版本</dt><dd className="mt-1 text-lg font-semibold">{snapshot.version_ids?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">评价方案分组</dt><dd className="mt-1 text-lg font-semibold">{snapshot.groups?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">可投资产品</dt><dd className="mt-1 text-lg font-semibold">{snapshot.product_count}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">研究日期</dt><dd className="mt-1 text-sm font-semibold">{snapshot.research_date}</dd></div></dl>
      <div className="grid gap-4 xl:grid-cols-2">{(snapshot.groups ?? []).map((group) => <div key={`${group.evaluation_plan_id}:${group.evaluation_plan_revision}`} className="rounded-xl border border-slate-200 p-4"><h3 className="font-semibold text-slate-900">{group.evaluation_plan_name}</h3><p className="mt-1 text-xs text-slate-500">评价方案 v{group.evaluation_plan_revision} · {group.product_count} 个产品</p><ul className="mt-3 max-h-56 divide-y divide-slate-100 overflow-auto">{group.products.map((product) => <li key={product.key} className="flex items-center justify-between gap-3 py-2 text-sm"><span><b>{product.name || product.code}</b><span className="ml-2 font-mono text-xs text-slate-500">{product.code}</span></span><span className="text-xs text-slate-500">{product.usage_status === 'limited' ? `限额 ${product.max_weight == null ? '—' : `${(product.max_weight * 100).toFixed(1)}%`}` : '正常'}</span></li>)}</ul></div>)}</div>
    </section>}
  </div>
}
