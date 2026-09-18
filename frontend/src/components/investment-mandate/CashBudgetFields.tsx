import { Button } from '../ui'
import { Field, inputClass, NumberInput } from '../risk-models/ResearchUI'
import type { CashBudget, FundingFlow, FundingSummary, MandateDefinition } from '../../services/strategicAllocation'
import { amountText, lastPaymentMonth, modelMonthLabel, newCashBudget } from './model'
import { useMandateText } from './text'

export default function CashBudgetFields({ value, onChange, funding }: {
  value: MandateDefinition; onChange: (patch: Partial<MandateDefinition>) => void; funding?: FundingSummary | null
}) {
  const { t, locale } = useMandateText()
  const cash = value.cash_budget
  const patch = (update: Partial<CashBudget>) => cash && onChange({ cash_budget: { ...cash, ...update } })
  const flows = (items: FundingFlow[]) => patch({ flows: items })
  const required = value.objective_kind === 'funding_goal'
  // The ledger is filled in the order it happens: capital first, then the end balance it
  // has to reach — the objective's target for a funding goal, otherwise a floor.
  const terminal = required ? value.funding_target ?? null : value.cash_protection?.terminal_floor ?? null
  const setTerminal = (amount: number, basis: 'nominal' | 'real' = terminal?.amount_basis ?? 'nominal') =>
    onChange(required ? { funding_target: { amount, amount_basis: basis } }
      : { cash_protection: Number.isFinite(amount) && amount > 0
          ? { mode: 'payments_and_terminal_floor', terminal_floor: { amount, amount_basis: basis } } : null })
  // Restated from the server's own funding numbers; no schedule arithmetic runs here.
  const net = funding ? funding.investable_capital + funding.total_contributions - funding.total_withdrawals : null
  const gap = funding && net !== null ? (funding.nominal_terminal_target ?? 0) - net : null
  const real = Boolean(cash) && (cash!.amount_basis === 'real' || terminal?.amount_basis === 'real')
  const months = Number.isInteger(value.horizon_years) && value.horizon_years >= 1 && value.horizon_years <= 30 ? value.horizon_years * 12 : 0
  // 结束月 is derived, never chosen: a stream runs to its last payment inside the horizon,
  // and 仅一次 is the same month as 开始月.
  const budgetDay = cash?.balance_as_of ?? value.as_of
  const monthText = (month: number) => locale === 'zh-CN' ? modelMonthLabel(budgetDay, month)
    : `Month ${month} · ${modelMonthLabel(budgetDay, month).slice(-11, -1)}`
  const monthOption = (month: number) => <option key={month} value={month}>{monthText(month)}</option>
  return <section className="min-w-0 space-y-4 border-t border-slate-200 pt-5" aria-label={t('cashBudget')}>
    <h4 className="text-base font-semibold">{t('sectionCash')}</h4>
    <label className="flex min-h-10 items-start gap-2 text-sm leading-6">
      <input className="mt-1.5" type="checkbox" checked={Boolean(cash)} disabled={required}
        onChange={event => onChange({ cash_budget: event.target.checked ? newCashBudget(value.as_of) : null, cash_protection: null })} />
      <span>{required ? t('cashRequiredForFunding') : t('useCashBudget')}</span>
    </label>
    {!cash ? <p className="text-sm leading-6 text-slate-600">{t('cashOptionalSimple')}</p> : <>
      {cash.balance_as_of !== value.as_of && <div className="space-y-2 rounded-lg border border-amber-200 bg-amber-50 p-3">
        <p role="status" className="text-sm text-amber-900">{t('cashRollRequired', { from: cash.balance_as_of, to: value.as_of })}</p>
        <Button onClick={() => patch({ balance_as_of: value.as_of })}>{t('confirmCashRoll')}</Button>
      </div>}
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label={`${t('capital')}（${value.currency}）`} hint={t('capitalHint')}><NumberInput className={inputClass} value={cash.total_capital} min={0} onValueChange={n => patch({ total_capital: n })} /></Field>
        <Field label={`${t(required ? 'terminalTarget' : 'terminalFloor')}（${value.currency}）`} hint={t(required ? 'terminalTargetHint' : 'terminalFloorHint')}>
          <NumberInput className={inputClass} value={terminal?.amount ?? NaN} min={0} onValueChange={setTerminal} /></Field>
      </div>
      {gap !== null && funding && <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">
        {t('fundingBridge', { capital: amountText(funding.investable_capital), contributions: amountText(funding.total_contributions),
          withdrawals: amountText(funding.total_withdrawals), net: amountText(net), target: amountText(funding.nominal_terminal_target ?? 0) })}
        {gap > 0 ? t('fundingBridgeGap', { gap: amountText(gap) }) : t('fundingBridgeCovered')}</p>}
      <p className="text-xs leading-5 text-slate-600">{t('cashModelConvention')}</p>
      <div>
        <h5 className="text-base font-semibold">{t('cashflows')}</h5>
        <p className="mt-1 text-xs leading-5 text-slate-600">{t('cashflowSimpleHint')}</p>
      </div>
      {!cash.flows.length && <p className="text-sm text-slate-600">{t('noFlows')}</p>}
      <div className="divide-y divide-slate-200">{cash.flows.map((flow, index) => {
        const update = (next: Partial<FundingFlow>) => flows(cash.flows.map((item, i) => i === index ? { ...item, ...next } : item))
        const label = (key: string) => `${t('flowNumber', { n: index + 1 })} · ${t(key)}`
        const once = flow.last_month === flow.first_month
        return <fieldset key={index} className="min-w-0 space-y-3 py-4"><legend className="text-sm font-medium">{t('flowNumber', { n: index + 1 })}</legend>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label={label('name')}><input className={inputClass} maxLength={120} value={flow.name} onChange={e => update({ name: e.target.value })} /></Field>
            <Field label={label('direction')}><select className={inputClass} value={flow.kind} onChange={e => update({ kind: e.target.value as FundingFlow['kind'] })}><option value="contribution">{t('contribution')}</option><option value="withdrawal">{t('payment')}</option></select></Field>
            <Field label={`${label('amount')}（${value.currency}）`}><NumberInput className={inputClass} value={flow.amount} min={0} onValueChange={n => update({ amount: n })} /></Field>
            <Field label={label('first_month')}><select className={inputClass} value={flow.first_month} onChange={e => { const first = Number(e.target.value); update({ first_month: first, last_month: once ? first : lastPaymentMonth(first, flow.every_months, months) }) }}>
              {Array.from({ length: months }, (_, month) => monthOption(month + 1))}
            </select></Field>
            <Field label={label('last_month')} hint={t('lastMonthHint')}><input className={inputClass} readOnly value={monthText(flow.last_month)} /></Field>
            <Field label={label('frequency')}><select className={inputClass} value={once ? 'once' : String(flow.every_months)}
              onChange={e => update(e.target.value === 'once' ? { every_months: 1, last_month: flow.first_month }
                : { every_months: Number(e.target.value) as 1 | 3 | 12, last_month: lastPaymentMonth(flow.first_month, Number(e.target.value), months) })}>
              {['once', '1', '3', '12'].map(key => <option key={key} value={key}>{t(key === 'once' ? 'everyOnce' : `every${key}`)}</option>)}
            </select></Field>
          </div>
          <Button onClick={() => flows(cash.flows.filter((_, i) => i !== index))}>{t('removeFlow', { n: index + 1 })}</Button>
        </fieldset>
      })}</div>
      <Button disabled={cash.flows.length >= 24} onClick={() => flows([...cash.flows, { name: '', kind: 'withdrawal', amount: NaN, first_month: 1, last_month: 1, every_months: 1 }])}>{t('addFlow')}</Button>
      <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('cashAdvanced')}</summary>
        <p className="mb-3 text-xs leading-5 text-slate-600">{t('cashAdvancedHint')}</p>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={`${t('outsideReserve')}（${value.currency}）`} hint={t('outsideReserveHint')}><NumberInput className={inputClass} value={cash.outside_reserve} min={0} onValueChange={n => patch({ outside_reserve: n })} /></Field>
          <Field label={t('additionalFee')} hint={t('feeHint')}><NumberInput className={inputClass} value={cash.annual_fee * 100} min={0} max={10} onValueChange={n => patch({ annual_fee: n / 100 })} /></Field>
        </div>
        {/* 通胀只有按购买力填写时才是一个真实参数，所以它跟着口径选择出现，就在旁边。 */}
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <Field label={t('flowBasis')} hint={t('basisHint')}><select className={inputClass} value={cash.amount_basis} onChange={e => patch({ amount_basis: e.target.value as CashBudget['amount_basis'] })}><option value="nominal">{t('nominal')}</option><option value="real">{t('real')}</option></select></Field>
          {terminal && <Field label={t('targetBasis')} hint={t('basisHint')}><select className={inputClass} value={terminal.amount_basis}
            onChange={e => setTerminal(terminal.amount, e.target.value as 'nominal' | 'real')}><option value="nominal">{t('nominal')}</option><option value="real">{t('real')}</option></select></Field>}
          {real && <Field label={t('inflation')} hint={t('inflationHint')}><NumberInput className={inputClass} value={cash.inflation * 100} min={-5} max={20} onValueChange={n => patch({ inflation: n / 100 })} /></Field>}
        </div>
        {real && <p className="mt-2 text-xs leading-5 text-slate-600">{t('inflationActive', { years: value.horizon_years, factor: ((1 + cash.inflation) ** value.horizon_years).toFixed(4) })}</p>}
      </details>
    </>}
  </section>
}
