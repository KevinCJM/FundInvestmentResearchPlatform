import { useEffect, useState } from 'react'
import { Button } from '../ui'
import { Feedback, Field, NumberInput } from '../risk-models/ResearchUI'
import { riskReference, type CmaDraft, type EconomicRole, type RiskReferenceRequest } from '../../services/strategicAllocation'
import { control, RateInput, useLtcmaTask, useLtcmaText } from './shared'

const roles: EconomicRole[] = ['growth', 'rates', 'inflation', 'credit', 'liquidity', 'diversifier']
type Props = { value: CmaDraft; onChange: (next: CmaDraft) => void }

export default function LtcmaAssetFields({ value, onChange }: Props) {
  const { t } = useLtcmaText(), manual = !value.model
  const change = (index: number, patch: Partial<CmaDraft['assets'][number]>) => onChange({
    ...value, assets: value.assets.map((asset, i) => i === index ? { ...asset, ...patch } : asset),
    ...('annual_volatility' in patch ? { risk_origin: 'manual' as const, risk_reference: null, risk_reference_hash: null } : {}),
  })
  const correlation = (i: number, j: number, number: number) => onChange({ ...value,
    correlation: value.assets.map((_, r) => value.assets.map((__, c) => r === i && c === j || r === j && c === i ? number : value.correlation[r]?.[c] ?? NaN)),
    risk_origin: 'manual', risk_reference: null, risk_reference_hash: null,
  })
  return <section className="space-y-4" aria-label={t('assetInputs')}>
    <h2 className="text-lg font-semibold">{t('assetInputs')}</h2>
    <div className="divide-y divide-slate-200">{value.assets.map((asset, index) => <fieldset key={asset.id} className="min-w-0 space-y-3 py-4">
      <legend className="text-base font-semibold">{asset.id}</legend>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <Field label={`${asset.id} · ${t('role')}`}><select className={control} value={asset.role} disabled={Boolean(value.strategic_universe_id)} onChange={event => change(index, { role: event.target.value as EconomicRole })}>
          <option value="">{t('choose')}</option>{roles.map(role => <option key={role} value={role}>{t(`role.${role}`)}</option>)}
        </select></Field>
        <Field label={`${asset.id} · ${t('liquidity')}`}><select className={control} value={asset.liquidity} disabled={Boolean(value.strategic_universe_id)} onChange={event => change(index, { liquidity: event.target.value as 'liquid' | 'illiquid' })}>
          <option value="">{t('choose')}</option><option value="liquid">{t('liquid')}</option><option value="illiquid">{t('illiquid')}</option>
        </select></Field>
        <Field label={`${asset.id} · ${t('rationale')}`}><input className={control} value={asset.rationale} maxLength={1000} onChange={event => change(index, { rationale: event.target.value })} /></Field>
        {manual && <><Field label={`${asset.id} · ${t('return')}`}><RateInput value={asset.annual_return} onChange={annual_return => change(index, { annual_return })} /></Field>
          <Field label={`${asset.id} · ${t('volatility')}`}><RateInput value={asset.annual_volatility} onChange={annual_volatility => change(index, { annual_volatility })} /></Field></>}
      </div>
      {(!value.model || value.model.method === 'black_litterman' || value.model.method === 'scenario_mixture') && <details><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('uncertainty')}</summary>
        <Field label={`${asset.id} · ${t('uncertainty')}`} hint={t('uncertaintyHint')}><RateInput value={asset.mean_uncertainty} onChange={mean_uncertainty => change(index, { mean_uncertainty })} /></Field>
      </details>}
    </fieldset>)}</div>
    {manual && <details open={value.risk_origin !== 'historical_reference'}><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('correlation')}</summary><p className="text-xs leading-5 text-slate-600">{t('matrixHint')}</p>
      <div className="overflow-x-auto"><table className="w-full text-sm" aria-label={t('correlation')}><thead><tr><th scope="col" className="p-2 text-left">{t('asset')}</th>{value.assets.map(asset => <th key={asset.id} scope="col" className="min-w-28 p-2 text-right">{asset.id}</th>)}</tr></thead>
        <tbody>{value.assets.map((asset, i) => <tr key={asset.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{asset.id}</th>{value.assets.map((other, j) => <td key={other.id} className="p-2 text-right tabular-nums">
          {j > i ? <NumberInput aria-label={`${t('correlation')}: ${asset.id} / ${other.id}`} className={`${control} min-w-24 text-right`} min={-1} max={1} value={value.correlation[i]?.[j] ?? NaN} onValueChange={number => correlation(i, j, number)} /> : Number.isFinite(value.correlation[i]?.[j]) ? value.correlation[i][j].toFixed(3) : '—'}
        </td>)}</tr>)}</tbody>
      </table></div>
    </details>}
  </section>
}

export function LtcmaRiskReference({ value, onChange, coverage }: Props & { coverage?: { start_date?: string; end_date?: string } }) {
  const { t } = useLtcmaText(), task = useLtcmaTask()
  const [request, setRequest] = useState<RiskReferenceRequest>(() => value.risk_reference ?? {
    alloc_name: value.alloc_name ?? '', as_of: value.as_of, start_date: coverage?.start_date ?? '',
    end_date: [value.as_of, coverage?.end_date].filter((x): x is string => Boolean(x)).sort()[0], shrinkage: .1, periods_per_year: 252,
  })
  const [notice, setNotice] = useState('')
  useEffect(() => { task.invalidate(); setNotice('') }, [value, request])
  const load = () => {
    const query = { ...request, alloc_name: value.alloc_name ?? '', as_of: value.as_of }
    void task.run(signal => riskReference(query, signal), result => {
      if (result.assets.join('\u0000') !== value.assets.map(asset => asset.id).join('\u0000')) throw new Error('LTCMA_RISK_AXIS')
      onChange({ ...value, assets: value.assets.map((asset, i) => ({ ...asset, annual_volatility: result.volatility[i] })),
        correlation: result.correlation, risk_origin: 'historical_reference', risk_reference: query, risk_reference_hash: result.preview_hash })
      setNotice(t('riskLoaded', { count: result.observations }))
    })
  }
  return <section className="space-y-3" aria-label={t('riskReference')}>
    <p className="text-sm leading-6 text-slate-600">{t('riskReferenceHint')}</p>
    <div className="grid gap-3 sm:grid-cols-3"><Field label={t('start')}><input type="date" className={control} value={request.start_date} onChange={event => setRequest({ ...request, start_date: event.target.value })} /></Field>
      <Field label={t('end')}><input type="date" className={control} max={value.as_of} value={request.end_date} onChange={event => setRequest({ ...request, end_date: event.target.value })} /></Field>
      <Field label={t('shrinkage')}><RateInput value={request.shrinkage} onChange={shrinkage => setRequest({ ...request, shrinkage })} /></Field>
    </div>
    <Button disabled={task.busy || !request.start_date || !request.end_date || request.start_date >= request.end_date || request.end_date > value.as_of || !Number.isFinite(request.shrinkage)} onClick={load}>{t('riskReference')}</Button>
    {value.risk_origin === 'historical_reference' && <p className="text-xs text-slate-600">{t('riskCertified')}</p>}
    <Feedback error={task.error} notice={notice} />
  </section>
}
