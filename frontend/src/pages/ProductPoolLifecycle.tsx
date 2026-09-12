import { useEffect, useMemo, useState } from 'react'
import {
  diffProductPoolVersions,
  listProductPoolVersions,
  type ProductPoolVersion,
  type ProductPoolVersionDiff,
} from '../services/productPools'

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

export default function ProductPoolLifecycle() {
  const [versions, setVersions] = useState<ProductPoolVersion[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [againstId, setAgainstId] = useState('')
  const [diff, setDiff] = useState<ProductPoolVersionDiff | null>(null)
  const [loading, setLoading] = useState(true)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    listProductPoolVersions()
      .then((response) => {
        if (!active) return
        setVersions(response.items)
        setSelectedId(response.items[0]?.id ?? '')
        setAgainstId(response.items[1]?.id ?? '')
      })
      .catch((reason) => { if (active) setError(messageOf(reason, '无法加载产品池版本。')) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  const selected = versions.find((item) => item.id === selectedId) ?? null
  const comparable = useMemo(
    () => versions.filter((item) => item.id !== selectedId && (!selected || item.pool_id === selected.pool_id)),
    [selected, selectedId, versions],
  )

  useEffect(() => {
    if (againstId && comparable.some((item) => item.id === againstId)) return
    setAgainstId(comparable[0]?.id ?? '')
    setDiff(null)
  }, [againstId, comparable])

  const compare = async () => {
    if (!selectedId || !againstId) return
    setComparing(true); setError('')
    try {
      setDiff(await diffProductPoolVersions(selectedId, againstId))
    } catch (reason) {
      setError(messageOf(reason, '版本比较失败。'))
    } finally {
      setComparing(false)
    }
  }

  return <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" aria-busy={loading || comparing}>
    <header className="rounded-xl bg-slate-900 px-5 py-6 text-white sm:px-7">
      <p className="text-sm font-medium text-emerald-300">产品研究 · 版本治理</p>
      <h1 className="mt-1 text-2xl font-semibold">产品池版本与生命周期</h1>
      <p className="mt-2 text-sm text-slate-200">已发布版本不可修改；产品变更通过新版本生效，并保留完整差异和评价方案来源。</p>
    </header>

    {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="font-semibold text-slate-900">已发布版本</h2>
      <div className="mt-4 overflow-x-auto"><table className="min-w-[920px] w-full text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr><th scope="col" className="px-3 py-3">产品池</th><th scope="col" className="px-3 py-3">版本</th><th scope="col" className="px-3 py-3">有效期</th><th scope="col" className="px-3 py-3">评价方案</th><th scope="col" className="px-3 py-3">可投资产品</th><th scope="col" className="px-3 py-3">发布时间</th><th scope="col" className="px-3 py-3">查看</th></tr></thead><tbody className="divide-y divide-slate-100">{versions.map((version) => <tr key={version.id} className={selectedId === version.id ? 'bg-emerald-50/50' : ''}><td className="px-3 py-3 font-medium text-slate-900">{version.pool_name}</td><td className="px-3 py-3">V{version.version}</td><td className="px-3 py-3">{version.effective_from} ～ {version.effective_to || '持续有效'}</td><td className="px-3 py-3">{version.evaluation_plans.length}</td><td className="px-3 py-3">{version.investable_count}</td><td className="px-3 py-3 text-xs text-slate-600">{version.created_at.slice(0, 19).replace('T', ' ')}</td><td className="px-3 py-3"><button type="button" onClick={() => { setSelectedId(version.id); setDiff(null) }} className="text-emerald-700 underline">查看</button></td></tr>)}</tbody></table>{!loading && versions.length === 0 && <p className="p-6 text-center text-sm text-slate-600">尚未发布产品池版本。</p>}</div>
    </section>

    {selected && <section className="grid gap-5 xl:grid-cols-[1fr_1fr]">
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-900">{selected.pool_name} · V{selected.version}</h2><p className="mt-1 font-mono text-xs text-slate-600">{selected.id}</p></div><span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800">不可变</span></div>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm"><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">生效日</dt><dd className="mt-1 font-medium">{selected.effective_from}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">可投资产品</dt><dd className="mt-1 font-medium">{selected.investable_count}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">已准入</dt><dd className="mt-1 font-medium">{selected.member_counts.approved ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">观察 / 排除</dt><dd className="mt-1 font-medium">{selected.member_counts.watch ?? 0} / {selected.member_counts.rejected ?? 0}</dd></div></dl>
        <h3 className="mt-5 text-sm font-semibold text-slate-800">评价方案即产品分组</h3><ul className="mt-2 space-y-2">{selected.evaluation_plans.map((plan) => <li key={`${plan.plan_id}:${plan.plan_revision}`} className="rounded-lg border border-slate-200 px-3 py-2 text-sm"><b>{plan.plan_name}</b><span className="ml-2 text-xs text-slate-600">v{plan.plan_revision} · {plan.imported_count} 个候选 · 截止 {plan.as_of || '—'}</span></li>)}</ul>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">版本差异</h2>
        <div className="mt-3 flex gap-2"><select aria-label="对比版本" value={againstId} onChange={(event) => { setAgainstId(event.target.value); setDiff(null) }} className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"><option value="">选择同产品池历史版本</option>{comparable.map((item) => <option key={item.id} value={item.id}>{item.pool_name} · V{item.version}</option>)}</select><button type="button" disabled={!againstId || comparing} onClick={() => void compare()} className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:bg-slate-400">比较</button></div>
        {!diff && <p className="mt-5 rounded-lg bg-slate-50 p-4 text-sm text-slate-600">选择历史版本后查看新增、移除和状态变化。</p>}
        {diff && <div className="mt-5 grid grid-cols-3 gap-3 text-center"><div className="rounded-lg bg-emerald-50 p-3"><p className="text-xs text-emerald-700">新增</p><p className="mt-1 text-xl font-semibold text-emerald-900">{diff.added.length}</p></div><div className="rounded-lg bg-rose-50 p-3"><p className="text-xs text-rose-700">移除</p><p className="mt-1 text-xl font-semibold text-rose-900">{diff.removed.length}</p></div><div className="rounded-lg bg-amber-50 p-3"><p className="text-xs text-amber-700">变更</p><p className="mt-1 text-xl font-semibold text-amber-900">{diff.changed.length}</p></div></div>}
        {diff && <div className="mt-4 max-h-80 space-y-2 overflow-auto text-sm">{diff.added.map((item) => <p key={`add-${item.key}`} className="rounded-lg bg-emerald-50 px-3 py-2 text-emerald-900">新增：{item.name || item.code}</p>)}{diff.removed.map((item) => <p key={`remove-${item.key}`} className="rounded-lg bg-rose-50 px-3 py-2 text-rose-900">移除：{item.name || item.code}</p>)}{diff.changed.map((item) => <div key={`change-${item.key}`} className="rounded-lg bg-amber-50 px-3 py-2 text-amber-950"><b>变更：{item.name || item.key}</b><p className="mt-1 text-xs">{Object.keys(item.changes).join('、')}</p></div>)}</div>}
      </div>
    </section>}
  </div>
}
