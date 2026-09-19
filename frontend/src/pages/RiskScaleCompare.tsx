import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Card, EmptyState } from '../components/ui'
import { RiskScaleInputSummary } from '../components/risk-scales/RiskScaleInputSummary'
import { RiskScaleResults } from '../components/risk-scales/RiskScaleResults'
import { ErrorNotice, Loading, pct, TaskHeader, useRiskTask, useRiskText } from '../components/risk-scales/shared'
import { metadata, riskScales, type CompareResponse } from '../services/riskScales'

export default function RiskScaleCompare() {
  const { t } = useRiskText(), [params] = useSearchParams(), task = useRiskTask(), [result, setResult] = useState<CompareResponse | null>(null)
  const left = params.get('left'), right = params.get('right'), valid = Boolean(left && right && left !== right)
  const load = () => { if (valid) void task.run(signal => riskScales.compare(left!, right!, signal), setResult) }
  useEffect(() => { setResult(null); load(); return task.invalidate }, [left, right])
  return <div className="min-w-0 space-y-3"><TaskHeader title={t('compareVersions')} />{!valid && <EmptyState mascot={false} title={t('exactlyTwo')} hint={t('compareFromList')} />}<ErrorNotice error={task.error} retry={load} />{task.busy && <Loading />}{result && <><Card className="!p-4 space-y-2"><p className="text-sm text-slate-600">{t(result.compatible ? 'compatibleComparison' : 'incompatibleComparison')}</p>{result.differences.map((difference, index) => <p className="break-words text-sm" key={index}>{t('difference', { field: t(`definition.${difference}`) })}</p>)}{result.compatible && result.boundary_differences && <dl className="grid gap-3 text-sm sm:grid-cols-5">{result.boundary_differences.map((difference, index) => <div key={index}><dt>C{index + 1}</dt><dd className="tabular-nums">{pct(difference)}</dd></div>)}</dl>}</Card><div className="grid min-w-0 gap-3 xl:grid-cols-2">{[result.left, result.right].map(version => <Card key={version.id} className="min-w-0 !p-4 space-y-2"><h2 className="text-base font-semibold">{version.name} · v{version.version_number}</h2><RiskScaleInputSummary definition={version.preview.request_echo.definition} assetNames={metadata(version.preview.result.diagnostics?.asset_names) as Record<string, string>} /><RiskScaleResults preview={version.preview} compact /></Card>)}</div></>}</div>
}
