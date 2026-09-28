import { useEffect, useId, useState } from 'react'
import { Button, Skeleton } from '../ui'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import { ltcma } from '../../services/ltcma'
import { niwPriorObservationBounds, type CmaSampleRequest, type CmaSampleSummary } from '../../services/ltcmaContract.generated'
import { validNiwPriorObservations } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import { control, useLtcmaTask, useLtcmaText } from './shared'

export default function LtcmaNiwStrengths({ value, onChange }: { value: CmaDraft; onChange: (value: CmaDraft) => void }) {
  const { t, locale } = useLtcmaText(), helpId = useId(), task = useLtcmaTask()
  const model = value.model?.method === 'bayesian_niw' ? value.model : null
  const body: CmaSampleRequest | null = model ? {
    alloc_name: value.alloc_name ?? null, strategic_universe_id: value.strategic_universe_id ?? null,
    model: { asset_ids: model.asset_ids, as_of: model.as_of, currency: model.currency, source: model.source,
      return_basis: model.return_basis, window: model.window, proxy_inputs: model.proxy_inputs,
      observation_frequency: model.observation_frequency, periods_per_year: model.periods_per_year },
  } : null
  // Only evidence inputs identify this reference; changing prior strengths keeps it useful.
  const key = JSON.stringify(body)
  const [result, setResult] = useState<{ key: string; sample: CmaSampleSummary } | null>(null)
  useEffect(() => { task.invalidate(); setResult(null) }, [key, task.invalidate])
  const sample = result?.key === key ? result.sample : null
  if (!model || model.prior_mode === 'continue') return null
  const number = (n: number) => new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(n)
  const multiple = (n: number) => new Intl.NumberFormat(locale, { maximumSignificantDigits: 3 }).format(n)
  const mean = model.mean_prior_observations, risk = model.covariance_prior_observations
  const limit = t('niwStrengthRange', { max: number(niwPriorObservationBounds.maximum) })
  const invalid = [mean, risk].some(n => n != null && Number.isFinite(n) && !validNiwPriorObservations(n))

  return <div className="space-y-4">
    <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="font-semibold">{t('niwSampleTitle')}</p>
        <Button disabled={task.busy || !body || !(body.alloc_name || body.strategic_universe_id)} onClick={() => {
          if (!body) return
          setResult(null)
          void task.run(signal => ltcma.sample(body, signal), next => setResult({ key, sample: next }))
        }}>{t(task.busy ? 'niwSampleLoading' : sample ? 'niwSampleRefresh' : 'niwSampleLoad')}</Button>
      </div>
      <p className="text-sm leading-6 text-slate-600">{t('niwSampleHint')}</p>
      {task.busy && <div aria-busy="true" className="space-y-2"><Skeleton className="h-6 w-1/2" /><Skeleton className="h-5 w-full" /></div>}
      {task.error && <p role="alert" className="text-sm leading-6 text-rose-700">{task.error}</p>}
      {sample && <div role="status" className="space-y-2 text-sm leading-6 text-slate-700 tabular-nums">
        <p className="font-semibold">{t('niwSampleCount', { count: number(sample.observations) })}</p>
        <p>{t('niwSampleDates', { start: sample.actual_start, end: sample.actual_end, nav: number(sample.observations + 1) })}</p>
        <p>{t('niwSampleExamples', { quarter: number(sample.observations / 4), equal: number(sample.observations), four: number(sample.observations * 4) })}</p>
        <p>{t('niwSampleLimits')}</p>
      </div>}
    </div>
    <p id={`${helpId}-range`} className="text-sm leading-6 text-slate-600">{limit}</p>
    <div className="grid gap-3 sm:grid-cols-2">
      <div><Field label={t('meanStrength')}><NumberInput aria-describedby={`${helpId}-mean ${helpId}-range`} aria-valuemax={niwPriorObservationBounds.maximum} className={control} value={mean ?? NaN} onValueChange={mean_prior_observations => onChange({ ...value, model: { ...model, mean_prior_observations } })} /></Field>
        <p id={`${helpId}-mean`} className="mt-1 text-sm leading-6 text-slate-600">{t('meanStrengthHint')}</p>
        {sample && validNiwPriorObservations(mean) && <p className="mt-1 text-sm leading-6 font-medium text-accent-800 tabular-nums">{t('niwMeanShare', {
          multiple: multiple(mean / sample.observations), prior: number(100 * mean / (mean + sample.observations)), evidence: number(100 * sample.observations / (mean + sample.observations)),
        })}</p>}
      </div>
      <div><Field label={t('riskStrength')}><NumberInput aria-describedby={`${helpId}-risk ${helpId}-range`} aria-valuemax={niwPriorObservationBounds.maximum} className={control} value={risk ?? NaN} onValueChange={covariance_prior_observations => onChange({ ...value, model: { ...model, covariance_prior_observations } })} /></Field>
        <p id={`${helpId}-risk`} className="mt-1 text-sm leading-6 text-slate-600">{t('riskStrengthHint')}</p>
        {sample && validNiwPriorObservations(risk) && <p className="mt-1 text-sm leading-6 font-medium text-accent-800 tabular-nums">{t('niwRiskRatio', { multiple: multiple(risk / sample.observations) })}</p>}
      </div>
    </div>
    {invalid && <p role="alert" className="text-sm leading-6 text-rose-700">{limit}</p>}
    <p className="text-sm leading-6 text-slate-600">{t('priorHint')}</p>
    <details className="border-l-2 border-slate-200 pl-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('niwExampleTitle')}</summary>
      <p className="text-sm leading-6 text-slate-600 tabular-nums">{t('niwExample')}</p>
    </details>
  </div>
}
