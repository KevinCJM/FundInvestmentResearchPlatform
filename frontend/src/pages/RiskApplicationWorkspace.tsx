import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import PublishedRiskPanel, { RiskImpactResult } from '../components/risk-models/PublishedRiskPanel'
import { buttonClass, Empty, Feedback, sectionClass } from '../components/risk-models/ResearchUI'
import { getRiskImpact, type RiskImpact } from '../services/riskModels'

export default function RiskApplicationWorkspace() {
  const [params] = useSearchParams()
  const impactId = params.get('impact_id') ?? ''
  const [result, setResult] = useState<RiskImpact | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    setResult(null)
    setError('')
    if (!impactId) {
      setLoading(false)
      return () => controller.abort()
    }
    // Compatibility only for immutable results created before the persist-on-publish rule.
    setLoading(true)
    getRiskImpact(impactId, controller.signal)
      .then(value => { if (!controller.signal.aborted) setResult(value) })
      .catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '历史结果读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [impactId])

  return <div className="min-w-0 space-y-5">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h2 className="text-xl font-semibold">产品与组合情景应用</h2><p className="mt-2 text-sm text-slate-600">把已发布的情景和敏感度组合使用。计算结果只保留在当前页面，不自动写入磁盘。</p></div>
      <Link className={buttonClass} to="/settings/scenario-algorithms?center=simulation">返回情景中心</Link>
    </div>
    <Feedback error={error} />
    {impactId ? loading ? <p role="status" className={sectionClass}>正在读取旧版历史结果，不会重新计算…</p> : result ? <><p className="text-sm text-slate-600">这是旧版本曾保存的历史结果，仅作兼容查看。当前版本的新压测不会自动保存。</p><RiskImpactResult result={result} /><Link className={buttonClass} to="/settings/scenario-algorithms/apply">新建一次临时压测</Link></> : <Empty title="没有可展示的历史结果"><p>当前版本不会自动保存压测结果。请重新选择已发布模型和情景进行计算。</p></Empty> : <PublishedRiskPanel key={`${params.get('product_key') ?? ''}:${params.get('portfolio_run_id') ?? ''}:${params.get('exposure_release_id') ?? ''}:${params.get('scenario_release_id') ?? ''}:${params.get('target_type') ?? ''}`} productKey={params.get('product_key') ?? undefined} portfolioRunId={params.get('portfolio_run_id') ?? undefined} portfolioOnly={params.get('target_type') === 'portfolio_run'} />}
  </div>
}
