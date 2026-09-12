import { useState } from 'react'
import { factorApi, type FactorDefinition, type FactorFields } from '../../services/factorResearch'
import { buttonClass, Card, Field, inputClass, secondaryClass, type Action } from './shared'

const empty: FactorFields = { name: '', description: '', operator: 'momentum', window: 126, skip: 21, direction: 1, product_kinds: ['etf', 'fund'] }
const names = { momentum: '动量收益', volatility: '收益波动', drawdown: '窗口最大回撤', reversal: '收益反转' }
export default function FactorLibrary({ factors, action, busy, onChanged }: { factors: FactorDefinition[]; action: Action; busy: boolean; onChanged: () => Promise<void> }) {
  const [fields, setFields] = useState<FactorFields>(empty)
  const [current, setCurrent] = useState<FactorDefinition>()
  const [kind, setKind] = useState('all')
  const select = (factor: FactorDefinition, copy = false) => {
    setFields({ name: factor.name + (copy ? ' · 自定义' : ''), description: factor.description, operator: factor.operator, window: factor.window, skip: factor.skip, direction: factor.direction, product_kinds: factor.product_kinds })
    setCurrent(copy ? undefined : factor)
  }
  return <div className="grid min-w-0 gap-5 xl:grid-cols-[1.3fr_1fr]">
    <Card title="共享因子库">
      <div className="mb-4 flex flex-wrap gap-2">{[['all', '全部'], ['etf', 'ETF'], ['fund', '场外基金'], ['stock', '股票']].map(([value, label]) => <button type="button" key={value} aria-pressed={kind === value} onClick={() => setKind(value)} className={kind === value ? buttonClass : secondaryClass}>{label}</button>)}</div>
      {kind === 'stock' && <p className="mb-4 text-sm text-amber-800">股票研究共用定义框架；当前缺少完整股票行情与财务时点数据。</p>}
      <div className="space-y-3">{factors.filter(item => kind === 'all' || item.product_kinds.includes(kind as 'etf')).map(factor => <article key={factor.id} className="rounded-lg border border-slate-200 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2"><h4 className="font-semibold text-slate-900">{factor.name}</h4><span className="text-xs text-slate-600">v{factor.revision} · {factor.read_only ? '内置模板' : '自定义'}</span></div>
        <p className="mt-2 text-sm leading-6 text-slate-600">{factor.description}</p>
        <p className="mt-2 text-xs text-slate-600">{names[factor.operator]} · {factor.window} 日窗口 · 跳过 {factor.skip} 日 · {factor.direction > 0 ? '越大越好' : '越小越好'}</p>
        <div className="mt-3 flex gap-2"><button className={secondaryClass} onClick={() => select(factor, true)}>复制构建</button>{!factor.read_only && <button className={secondaryClass} onClick={() => select(factor)}>编辑版本</button>}</div>
      </article>)}</div>
    </Card>
    <Card title={current ? '编辑因子版本' : '构建因子'}>
      <form onSubmit={event => { event.preventDefault(); void action('保存因子', async () => { await factorApi.saveFactor(fields, current); await onChanged(); setCurrent(undefined); setFields(empty) }) }}>
        <fieldset disabled={busy} className="space-y-4">
          <Field label="因子名称"><input className={inputClass} required maxLength={80} value={fields.name} onChange={e => setFields({ ...fields, name: e.target.value })} /></Field>
          <Field label="研究假设"><textarea className={inputClass} rows={3} value={fields.description} onChange={e => setFields({ ...fields, description: e.target.value })} /></Field>
          <Field label="计算算子"><select className={inputClass} value={fields.operator} onChange={e => setFields({ ...fields, operator: e.target.value as FactorFields['operator'] })}>{Object.entries(names).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></Field>
          <div className="grid grid-cols-2 gap-3"><Field label="窗口（交易日）"><input className={inputClass} type="number" min={2} max={504} required value={fields.window} onChange={e => setFields({ ...fields, window: Number(e.target.value) })} /></Field><Field label="跳过最近交易日"><input className={inputClass} type="number" min={0} max={126} required value={fields.skip} onChange={e => setFields({ ...fields, skip: Number(e.target.value) })} /></Field></div>
          <Field label="得分方向"><select className={inputClass} value={fields.direction} onChange={e => setFields({ ...fields, direction: Number(e.target.value) as -1 | 1 })}><option value={1}>原始值越大越好</option><option value={-1}>原始值越小越好</option></select></Field>
          <div><p className="text-sm font-medium text-slate-700">适用产品</p><div className="mt-2 flex flex-wrap gap-4">{[['etf', 'ETF'], ['fund', '场外基金'], ['stock', '股票']].map(([value, label]) => <label className="flex min-h-10 items-center gap-2 text-sm" key={value}><input type="checkbox" checked={fields.product_kinds.includes(value as 'etf')} onChange={e => setFields({ ...fields, product_kinds: e.target.checked ? [...fields.product_kinds, value as 'etf'] : fields.product_kinds.filter(x => x !== value) })} />{label}</label>)}</div></div>
          <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-600">输入：公告后可用的复权净值。{fields.operator === 'momentum' ? '公式：期末净值 ÷ 期初净值 − 1。' : fields.operator === 'reversal' ? '公式：短期净值收益的相反数。' : fields.operator === 'volatility' ? '公式：日收益样本标准差 × √252。' : '公式：窗口内净值相对历史峰值的最小跌幅。'}横截面标准化和权重在研究方案中设置。</p>
          <div className="flex flex-wrap gap-2"><button className={buttonClass} type="submit">{current ? '保存新修订' : '保存因子'}</button><button className={secondaryClass} type="button" onClick={() => { setCurrent(undefined); setFields(empty) }}>新建</button></div>
        </fieldset>
      </form>
    </Card>
  </div>
}
