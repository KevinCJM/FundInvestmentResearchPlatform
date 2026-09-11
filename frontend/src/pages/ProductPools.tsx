import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useAllocationDraft } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import CandidateReviewTable from '../components/product-pools/CandidateReviewTable'
import ProductPoolHeader, { type CreateProductPoolInput } from '../components/product-pools/ProductPoolHeader'
import {
  listEvaluationPlans,
  type EvaluationPlan,
} from '../services/customIndicators'
import {
  addManualPoolMember,
  attachEvaluationPlan,
  createProductPool,
  listProductPools,
  publishProductPool,
  removeEvaluationPlan,
  updateProductPool,
  type EvaluationPlanSelectionMode,
  type ProductPool,
} from '../services/productPools'

const today = () => new Date().toISOString().slice(0, 10)

const selectionLabels: Record<EvaluationPlanSelectionMode, string> = {
  all_ranked: '全部已排名产品',
  top_n: '排名前 N',
  top_percent: '排名前百分比',
}

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

export default function ProductPools() {
  const [pools, setPools] = useState<ProductPool[]>([])
  const [evaluationPlans, setEvaluationPlans] = useState<EvaluationPlan[]>([])
  const [params] = useSearchParams()
  const [poolDraft, setPoolDraft] = useAllocationDraft(`pool-manager:${params.get('pool') || 'resume'}`, { selectedPoolId: params.get('pool') || '' })
  const selectedPoolId = poolDraft.selectedPoolId
  const setSelectedPoolId = (value: string | ((current: string) => string)) => setPoolDraft(current => ({ selectedPoolId: typeof value === 'function' ? value(current.selectedPoolId) : value }))
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const selectedPool = pools.find((item) => item.id === selectedPoolId) ?? null

  useEffect(() => {
    let active = true
    setLoading(true)
    Promise.all([listProductPools(), listEvaluationPlans()])
      .then(([poolResponse, planResponse]) => {
        if (!active) return
        setPools(poolResponse.items)
        setEvaluationPlans(planResponse.items)
        setSelectedPoolId((current) => current || poolResponse.items[0]?.id || '')
      })
      .catch((reason) => {
        if (active) setError(messageOf(reason, '无法加载产品池。'))
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => { active = false }
  }, [])

  const applyPool = useCallback((next: ProductPool) => {
    setPools((current) => [next, ...current.filter((item) => item.id !== next.id)])
    setSelectedPoolId(next.id)
  }, [])

  const showMessage = useCallback((value: string) => {
    setMessage(value)
    setError('')
  }, [])

  const showError = useCallback((value: string) => {
    setError(value)
    setMessage('')
  }, [])

  const createPool = async (input: CreateProductPoolInput) => {
    setBusy(true); setError(''); setMessage('')
    try {
      const created = await createProductPool({
        ...input,
        description: '',
      })
      applyPool(created)
      setMessage('产品池草稿已创建。')
    } catch (reason) {
      throw new Error(messageOf(reason, '创建产品池失败。'))
    } finally {
      setBusy(false)
    }
  }

  return <div className="mx-auto min-w-0 w-full max-w-[1680px] space-y-6 p-4 sm:p-6" aria-busy={loading || busy}>
    <ProductPoolHeader
      pools={pools}
      selectedPool={selectedPool}
      selectedPoolId={selectedPoolId}
      loading={loading}
      busy={busy}
      onSelectPool={(poolId) => {
        setSelectedPoolId(poolId)
        setError('')
        setMessage('')
      }}
      onCreatePool={createPool}
    />

    {(message || error) && <div role={error ? 'alert' : 'status'} className={`rounded-xl border px-4 py-3 text-sm ${error ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-emerald-200 bg-emerald-50 text-emerald-800'}`}>{error || message}</div>}

    <main data-testid="product-pool-workspace" className="min-w-0 w-full">
      {selectedPool ? <PoolWorkspace
        key={`${selectedPool.id}:${selectedPool.revision}`}
        pool={selectedPool}
        evaluationPlans={evaluationPlans}
        busy={busy}
        setBusy={setBusy}
        onPool={applyPool}
        onMessage={showMessage}
        onError={showError}
      /> : <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-slate-500">
        {loading ? '正在加载产品池…' : '点击表头中的“新建产品池”创建草稿后开始配置。'}
      </div>}
    </main>
  </div>
}

function PoolWorkspace({ pool, evaluationPlans, busy, setBusy, onPool, onMessage, onError }: {
  pool: ProductPool
  evaluationPlans: EvaluationPlan[]
  busy: boolean
  setBusy: (value: boolean) => void
  onPool: (pool: ProductPool) => void
  onMessage: (message: string) => void
  onError: (message: string) => void
}) {
  const [name, setName] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:name`, pool.name)
  const [description, setDescription] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:description`, pool.description)
  const [purpose, setPurpose] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:purpose`, pool.purpose)
  const [owner, setOwner] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:owner`, pool.owner)

  const [planId, setPlanId] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:planId`, '')
  const [asOf, setAsOf] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:asOf`, '')
  const platformAsOf = useResearchDay()
  const [selectionMode, setSelectionMode] = useAllocationDraft<EvaluationPlanSelectionMode>(`pool-form:${pool.id}:${pool.revision}:selectionMode`, 'all_ranked')
  const [selectionValue, setSelectionValue] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:selectionValue`, '10')

  const [manualPlanId, setManualPlanId] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:manualPlanId`, '')
  const [manualKind, setManualKind] = useAllocationDraft<'etf' | 'fund'>(`pool-form:${pool.id}:${pool.revision}:manualKind`, 'etf')
  const [manualCode, setManualCode] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:manualCode`, '')
  const [manualName, setManualName] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:manualName`, '')
  const [manualReason, setManualReason] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:manualReason`, '')

  const [effectiveFrom, setEffectiveFrom] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:effectiveFrom`, today())
  const [effectiveTo, setEffectiveTo] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:effectiveTo`, '')
  const [publicationNote, setPublicationNote] = useAllocationDraft(`pool-form:${pool.id}:${pool.revision}:publicationNote`, '')

  const attachedPlanIds = useMemo(
    () => new Set(pool.evaluation_plans.map((item) => item.plan_id)),
    [pool.evaluation_plans],
  )
  const candidatePlans = evaluationPlans
  const pendingCount = pool.members.filter((item) => item.research_status === 'pending').length
  const approvedCount = pool.members.filter((item) => item.research_status === 'approved').length

  useEffect(() => {
    if (planId && candidatePlans.some((item) => item.id === planId)) return
    const nextPlan = candidatePlans.find((item) => !attachedPlanIds.has(item.id)) ?? candidatePlans[0]
    setPlanId(nextPlan?.id ?? '')
  }, [attachedPlanIds, candidatePlans, planId])

  useEffect(() => {
    if (!manualPlanId && pool.evaluation_plans[0]) {
      setManualPlanId(pool.evaluation_plans[0].plan_id)
      setManualKind(pool.evaluation_plans[0].product_kind)
    }
  }, [manualPlanId, pool.evaluation_plans])

  const runAction = async (action: () => Promise<ProductPool>, success: string) => {
    setBusy(true); onError('')
    try {
      const next = await action()
      onPool(next)
      onMessage(success)
    } catch (reason) {
      onError(messageOf(reason, '操作失败。'))
    } finally {
      setBusy(false)
    }
  }

  const saveMetadata = () => runAction(
    () => updateProductPool(pool.id, { revision: pool.revision, name, description, purpose, owner }),
    '产品池基本信息已保存。',
  )

  const attachPlan = () => {
    if (!planId) { onError('请选择评价方案。'); return }
    const value = selectionMode === 'all_ranked' ? null : Number(selectionValue)
    void runAction(
      () => attachEvaluationPlan(pool.id, {
        revision: pool.revision,
        plan_id: planId,
        as_of: asOf || null,
        selection_mode: selectionMode,
        selection_value: Number.isFinite(value) ? value : null,
      }),
      '评价方案已运行并导入候选产品。',
    )
  }

  const addManual = () => {
    if (!manualPlanId || !manualCode.trim() || !manualReason.trim()) {
      onError('人工加入必须填写评价方案、产品代码和例外原因。')
      return
    }
    void runAction(
      () => addManualPoolMember(pool.id, {
        revision: pool.revision,
        plan_id: manualPlanId,
        kind: manualKind,
        product_id: manualCode.trim(),
        code: manualCode.trim(),
        name: manualName.trim() || manualCode.trim(),
        reason: manualReason.trim(),
      }),
      '人工例外产品已加入候选列表。',
    )
  }

  const publish = async () => {
    setBusy(true); onError('')
    try {
      const result = await publishProductPool(pool.id, {
        revision: pool.revision,
        effective_from: effectiveFrom,
        effective_to: effectiveTo || null,
        publication_note: publicationNote,
      })
      onPool(result.pool)
      onMessage(`产品池版本 V${result.version.version} 已发布。`)
    } catch (reason) {
      onError(messageOf(reason, '发布产品池失败。'))
    } finally {
      setBusy(false)
    }
  }

  return <div className="min-w-0 space-y-5">
    <section className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
      <div><h2 className="font-semibold text-emerald-950">本池产品与研究进度</h2><p className="mt-1 text-sm text-emerald-900">可用 {approvedCount} 只 · 待复核 {pendingCount} 只 · {pool.current_version_id ? '有已发布版本可用于研究' : '复核完成后发布，才能继续配置研究'}</p></div>
      {pool.current_version_id && <Link to={`/pre-investment/product-pool?version=${encodeURIComponent(pool.current_version_id)}`} className="rounded-lg bg-emerald-800 px-4 py-2 text-sm font-semibold text-white">使用已发布版本开展配置研究 →</Link>}
    </section>
    <CandidateReviewTable
      pool={pool}
      busy={busy}
      setBusy={setBusy}
      onPool={onPool}
      onMessage={onMessage}
      onError={onError}
    />

    <details open={pool.members.length === 0 ? true : undefined} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <summary className="cursor-pointer font-semibold text-slate-800">产品池规则与基本信息</summary>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h2 className="text-lg font-semibold text-slate-900">产品池规则与基本信息</h2><p className="mt-1 text-sm text-slate-500">编辑修订 {pool.revision} · {pool.current_version_id ? '已发布版本保持不变' : '尚未发布'}</p></div>
        <button type="button" disabled={busy} onClick={() => void saveMetadata()} className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:bg-slate-400">保存基本信息</button>
      </div>
      <div className="mt-4 grid min-w-0 gap-3 [&>label]:min-w-0 sm:grid-cols-2">
        <label className="text-sm text-slate-700">名称<input value={name} onChange={(event) => setName(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <label className="text-sm text-slate-700">负责人<input value={owner} onChange={(event) => setOwner(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <label className="text-sm text-slate-700 sm:col-span-2">用途<input value={purpose} onChange={(event) => setPurpose(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        <label className="text-sm text-slate-700 sm:col-span-2">说明<textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={2} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      </div>
    </details>

    <details open={pool.members.length === 0 ? true : undefined} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <summary className="cursor-pointer font-semibold text-slate-800">评价方案与候选来源</summary>
      <h2 className="text-lg font-semibold text-slate-900">评价方案与候选来源</h2>
      <p className="mt-1 text-sm text-slate-500">评价方案就是产品分组。关联时运行当前锁定版本，并保存运行结果编号作为入池证据。</p>
      <div className="mt-4 grid min-w-0 gap-3 [&>label]:min-w-0 lg:grid-cols-[minmax(0,1fr)_150px_130px_150px_auto]">
        <label className="text-xs text-slate-600">评价方案<select aria-label="待关联评价方案" value={planId} onChange={(event) => setPlanId(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"><option value="">请选择</option>{candidatePlans.map((plan) => <option key={plan.id} value={plan.id}>{plan.name} · v{plan.revision}{attachedPlanIds.has(plan.id) ? '（已关联，可重新运行）' : ''}</option>)}</select></label>
        <label className="text-xs text-slate-600">评价截止日<input type="date" value={asOf} onChange={(event) => setAsOf(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
          {/* 空着不等于「用全部数据」——它跟随平台研究日，和产品研究页同一个口径。 */}
          <span className="mt-1 block text-[11px] text-slate-500">{asOf ? '仅本次运行生效' : platformAsOf === undefined ? '平台 PIT 口径尚未确认，以服务端结果为准' : platformAsOf ? `跟随平台研究日 ${platformAsOf}` : '磁盘全部数据'}</span>
        </label>
        <label className="text-xs text-slate-600">导入方式<select value={selectionMode} onChange={(event) => setSelectionMode(event.target.value as EvaluationPlanSelectionMode)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-2 py-2 text-sm">{Object.entries(selectionLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="text-xs text-slate-600">N / 百分比<input aria-label="N / 百分比" type={selectionMode === 'all_ranked' ? 'text' : 'number'} inputMode={selectionMode === 'all_ranked' ? undefined : 'decimal'} disabled={selectionMode === 'all_ranked'} value={selectionMode === 'all_ranked' ? '-' : selectionValue} onChange={(event) => setSelectionValue(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-500" /></label>
        <button type="button" disabled={busy || !planId} onClick={attachPlan} className="self-end rounded-lg bg-violet-700 px-4 py-2 text-sm font-semibold text-white disabled:bg-violet-300">运行并关联</button>
      </div>
      <div className="mt-4 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-800">已关联评价方案（{pool.evaluation_plans.length}）</h3>
        <span className="text-xs text-slate-500">可继续关联其他方案，也可重新运行已关联方案。</span>
      </div>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        {pool.evaluation_plans.map((binding) => <div key={binding.plan_id} className="rounded-xl border border-slate-200 p-4">
          <div className="flex items-start justify-between gap-3"><div><h3 className="font-semibold text-slate-900">{binding.plan_name}</h3><p className="mt-1 text-xs text-slate-500">方案 v{binding.plan_revision} · {binding.product_kind === 'etf' ? 'ETF' : '公募基金'} · 截止 {binding.as_of || '运行日'}</p></div><button type="button" aria-label={`删除关联：${binding.plan_name}`} disabled={busy} onClick={() => { if (window.confirm(`删除评价方案“${binding.plan_name}”与当前产品池的关联？`)) void runAction(() => removeEvaluationPlan(pool.id, binding.plan_id, pool.revision), '评价方案及其候选证据已移除。') }} className="text-sm text-rose-700 underline">删除关联</button></div>
          <dl className="mt-3 grid grid-cols-3 gap-2 text-center text-xs"><div className="rounded bg-slate-50 p-2"><dt className="text-slate-500">已排名</dt><dd className="mt-1 font-semibold">{binding.ranked_count}</dd></div><div className="rounded bg-slate-50 p-2"><dt className="text-slate-500">未排名</dt><dd className="mt-1 font-semibold">{binding.excluded_count}</dd></div><div className="rounded bg-slate-50 p-2"><dt className="text-slate-500">已导入</dt><dd className="mt-1 font-semibold">{binding.imported_count}</dd></div></dl>
          <p className="mt-2 truncate font-mono text-[11px] text-slate-400" title={binding.result_id}>结果：{binding.result_id}</p>
        </div>)}
        {pool.evaluation_plans.length === 0 && <p className="rounded-lg bg-slate-50 p-4 text-sm text-slate-500">尚未关联评价方案。</p>}
      </div>
    </details>

    <details open={pool.members.length === 0 ? true : undefined} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <summary className="cursor-pointer font-semibold text-slate-800">人工补充例外产品</summary>
      <h2 className="text-lg font-semibold text-slate-900">人工补充例外产品</h2>
      <div className="mt-4 grid min-w-0 gap-3 [&>label]:min-w-0 md:grid-cols-2 xl:grid-cols-[1fr_110px_150px_1fr_1.4fr_auto]">
        <label className="text-xs text-slate-600">所属评价方案<select value={manualPlanId} onChange={(event) => { const id = event.target.value; setManualPlanId(id); const binding = pool.evaluation_plans.find((item) => item.plan_id === id); if (binding) setManualKind(binding.product_kind) }} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-2 py-2 text-sm"><option value="">请选择</option>{pool.evaluation_plans.map((item) => <option key={item.plan_id} value={item.plan_id}>{item.plan_name}</option>)}</select></label>
        <label className="text-xs text-slate-600">类型<select value={manualKind} onChange={(event) => setManualKind(event.target.value as 'etf' | 'fund')} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-2 py-2 text-sm"><option value="etf">ETF</option><option value="fund">公募基金</option></select></label>
        <label className="text-xs text-slate-600">产品代码<input value={manualCode} onChange={(event) => setManualCode(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" /></label>
        <label className="text-xs text-slate-600">产品名称<input value={manualName} onChange={(event) => setManualName(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" /></label>
        <label className="text-xs text-slate-600">例外原因<input value={manualReason} onChange={(event) => setManualReason(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" /></label>
        <button type="button" disabled={busy || pool.evaluation_plans.length === 0} onClick={addManual} className="self-end rounded-lg border border-slate-900 px-4 py-2 text-sm font-semibold text-slate-800 disabled:opacity-40">加入候选</button>
      </div>
    </details>



    <section className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5">
      <h2 className="text-lg font-semibold text-emerald-950">复核完成，发布研究用版本</h2>
      <p className="mt-1 text-sm text-emerald-800">发布前必须处理全部待复核产品。历史版本不会被后续修改覆盖。</p>
      <div className="mt-4 grid min-w-0 gap-3 [&>label]:min-w-0 md:grid-cols-[160px_160px_minmax(0,1fr)_auto]">
        <label className="text-xs text-emerald-900">生效日<input type="date" value={effectiveFrom} onChange={(event) => setEffectiveFrom(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-emerald-300 bg-white px-3 py-2 text-sm" /></label>
        <label className="text-xs text-emerald-900">失效日（可选）<input type="date" value={effectiveTo} onChange={(event) => setEffectiveTo(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-emerald-300 bg-white px-3 py-2 text-sm" /></label>
        <label className="text-xs text-emerald-900">发布说明<input value={publicationNote} onChange={(event) => setPublicationNote(event.target.value)} className="mt-1 min-w-0 w-full rounded-lg border border-emerald-300 bg-white px-3 py-2 text-sm" /></label>
        <button type="button" disabled={busy || pendingCount > 0 || approvedCount === 0} onClick={() => void publish()} className="self-end rounded-lg bg-emerald-800 px-5 py-2 text-sm font-semibold text-white disabled:bg-emerald-300">发布版本</button>
      </div>
    </section>
  </div>
}
