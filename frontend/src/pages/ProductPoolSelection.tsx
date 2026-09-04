import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  createInvestableUniverseSnapshot,
  listProductPoolVersions,
  type InvestableUniverseSnapshot,
  type ProductPoolVersion,
} from '../services/productPools'

const today = () => new Date().toISOString().slice(0, 10)

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

export default function ProductPoolSelection() {
  const navigate = useNavigate()
  const [researchDate, setResearchDate] = useState(today())
  const [name, setName] = useState('投前研究可投资域')
  const [versions, setVersions] = useState<ProductPoolVersion[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [snapshot, setSnapshot] = useState<InvestableUniverseSnapshot | null>(null)

  useEffect(() => {
    let active = true
    setLoading(true); setError(''); setSnapshot(null)
    listProductPoolVersions({ activeOn: researchDate })
      .then((response) => {
        if (!active) return
        setVersions(response.items)
        setSelectedIds((current) => current.filter((id) => response.items.some((item) => item.id === id)))
      })
      .catch((reason) => { if (active) setError(messageOf(reason, '无法加载有效产品池版本。')) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [researchDate])

  const toggle = (versionId: string) => {
    setSelectedIds((current) => current.includes(versionId)
      ? current.filter((id) => id !== versionId)
      : [...current, versionId])
    setSnapshot(null)
  }

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
      sessionStorage.setItem('investableUniverseSnapshotId', result.id)
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
      <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">选择研究日有效的一个或多个产品池版本，生成不可变的可投资域快照。后续大类构建与产品配置应只使用该快照中的产品。</p>
    </header>

    {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="grid gap-4 md:grid-cols-[180px_minmax(0,1fr)_auto] md:items-end">
        <label className="text-sm text-slate-700">研究日期<input type="date" value={researchDate} onChange={(event) => setResearchDate(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <label className="text-sm text-slate-700">可投资域名称<input value={name} onChange={(event) => setName(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <button type="button" disabled={creating || selectedIds.length === 0} onClick={() => void createSnapshot()} className="rounded-lg bg-emerald-800 px-5 py-2 text-sm font-semibold text-white disabled:bg-emerald-300">生成锁定快照</button>
      </div>
    </section>

    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-end justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-900">有效产品池版本</h2><p className="mt-1 text-sm text-slate-500">已选择 {selectedIds.length} 个版本</p></div><span className="text-xs text-slate-500">研究日：{researchDate}</span></div>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{versions.map((version) => {
        const selected = selectedIds.includes(version.id)
        return <label key={version.id} className={`cursor-pointer rounded-xl border p-4 ${selected ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200 hover:bg-slate-50'}`}>
          <div className="flex items-start gap-3"><input type="checkbox" checked={selected} onChange={() => toggle(version.id)} className="mt-1 h-4 w-4" /><div className="min-w-0"><h3 className="font-semibold text-slate-900">{version.pool_name} · V{version.version}</h3><p className="mt-1 text-xs text-slate-500">{version.effective_from} ～ {version.effective_to || '持续有效'}</p><p className="mt-2 text-sm text-slate-700">{version.evaluation_plans.length} 套评价方案 · {version.investable_count} 个可投资产品</p><p className="mt-2 truncate font-mono text-[11px] text-slate-400" title={version.id}>{version.id}</p></div></div>
        </label>
      })}</div>
      {!loading && versions.length === 0 && <p className="mt-4 rounded-lg bg-slate-50 p-6 text-center text-sm text-slate-500">该日期没有已生效产品池版本。</p>}
    </section>

    {snapshot && <section className="space-y-5 rounded-2xl border border-emerald-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-sm font-medium text-emerald-700">可投资域快照已锁定</p><h2 className="mt-1 text-xl font-semibold text-slate-900">{snapshot.name}</h2><p className="mt-1 font-mono text-xs text-slate-400">{snapshot.id}</p></div><div className="flex flex-wrap gap-2"><button type="button" onClick={() => navigate(`/pre-investment/saa/auto-classification?universe=${encodeURIComponent(snapshot.id)}`)} className="rounded-lg bg-violet-700 px-4 py-2 text-sm font-semibold text-white">进入自动构建大类</button><button type="button" onClick={() => navigate(`/pre-investment/saa/asset-classes?universe=${encodeURIComponent(snapshot.id)}`)} className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white">进入手动构建大类</button></div></div>
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-4"><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">产品池版本</dt><dd className="mt-1 text-lg font-semibold">{snapshot.version_ids?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">评价方案分组</dt><dd className="mt-1 text-lg font-semibold">{snapshot.groups?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">可投资产品</dt><dd className="mt-1 text-lg font-semibold">{snapshot.product_count}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">研究日期</dt><dd className="mt-1 text-sm font-semibold">{snapshot.research_date}</dd></div></dl>
      <div className="grid gap-4 xl:grid-cols-2">{(snapshot.groups ?? []).map((group) => <div key={`${group.evaluation_plan_id}:${group.evaluation_plan_revision}`} className="rounded-xl border border-slate-200 p-4"><h3 className="font-semibold text-slate-900">{group.evaluation_plan_name}</h3><p className="mt-1 text-xs text-slate-500">评价方案 v{group.evaluation_plan_revision} · {group.product_count} 个产品</p><ul className="mt-3 max-h-56 divide-y divide-slate-100 overflow-auto">{group.products.map((product) => <li key={product.key} className="flex items-center justify-between gap-3 py-2 text-sm"><span><b>{product.name || product.code}</b><span className="ml-2 font-mono text-xs text-slate-500">{product.code}</span></span><span className="text-xs text-slate-500">{product.usage_status === 'limited' ? `限额 ${product.max_weight == null ? '—' : `${(product.max_weight * 100).toFixed(1)}%`}` : '正常'}</span></li>)}</ul></div>)}</div>
    </section>}
  </div>
}
