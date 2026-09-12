import { Button } from '../ui'
import { Field, inputClass, NumberInput, sectionClass } from '../risk-models/ResearchUI'
import { percentInputValue, type CmaDraft, type EconomicRole, type RiskReferenceRequest } from '../../services/strategicAllocation'

const roles: Array<[EconomicRole, string]> = [
  ['growth', '增长参与'], ['rates', '利率防御'], ['inflation', '通胀分散'],
  ['credit', '信用收益'], ['liquidity', '流动性储备'], ['diversifier', '其他分散用途'],
]
const pct = percentInputValue

export default function AssumptionEditor({ value, onChange, reference, onReferenceChange, onLoadReference, busy }: {
  value: CmaDraft; onChange: (value: CmaDraft) => void; reference: RiskReferenceRequest
  onReferenceChange: (value: RiskReferenceRequest) => void; onLoadReference: () => void; busy: boolean
}) {
  function patchAsset(index: number, patch: Partial<CmaDraft['assets'][number]>) {
    const editedRisk = 'annual_volatility' in patch
    onChange({ ...value, assets: value.assets.map((asset, i) => i === index ? { ...asset, ...patch } : asset),
      ...(editedRisk ? { risk_origin: 'manual', risk_reference: null, risk_reference_hash: null } as const : {}) })
  }
  function setCorrelation(row: number, column: number, number: number) {
    const correlation = value.correlation.map((values, i) => values.map((old, j) => (i === row && j === column) || (i === column && j === row) ? number : old))
    onChange({ ...value, correlation, risk_origin: 'manual', risk_reference: null, risk_reference_hash: null })
  }
  return <div className="space-y-5">
    <section className={`${sectionClass} space-y-4`} aria-label="长期假设口径">
      <h2 className="text-lg font-semibold">长期假设：先说明依据，再填写数值</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="假设版本名称"><input className={inputClass} value={value.name} maxLength={120} onChange={e => onChange({ ...value, name: e.target.value })} /></Field>
        <Field label="假设研究日"><input type="date" className={inputClass} value={value.as_of} onChange={e => onChange({ ...value, as_of: e.target.value, risk_origin: 'manual', risk_reference: null, risk_reference_hash: null })} /></Field>
      </div>
      <p className="text-sm text-slate-600">计价币种：{value.currency} · 期限：{value.horizon_years} 年 · 年化算术总收益（不是几何复合收益率）。</p>
      <Field label="预测来源与主要假设" hint="记录研究依据、估值/增长假设或外部模型版本；不填造机构预测。"><textarea className={inputClass} rows={3} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field>
    </section>
    <section className={`${sectionClass} space-y-4`} aria-label="经济大类与预期">
      <h2 className="text-lg font-semibold">每个大类承担什么角色？</h2>
      <p className="text-sm leading-6 text-slate-600">相关聚类只能辅助判断，不能自动定义经济风险。这里确认大类用途与流动性；具体产品沿用已保存的分类，不改变产品归属。</p>
      {value.assets.map((asset, index) => <fieldset key={asset.id} className="min-w-0 border-t border-slate-200 pt-4">
        <legend className="px-1 text-sm font-semibold text-slate-900">{asset.id}</legend>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label={`${asset.id}经济角色`}><select className={inputClass} value={asset.role} onChange={e => patchAsset(index, { role: e.target.value as EconomicRole })}><option value="">请选择用途</option>{roles.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></Field>
          <Field label={`${asset.id}流动性`}><select className={inputClass} value={asset.liquidity} onChange={e => patchAsset(index, { liquidity: e.target.value as 'liquid' | 'illiquid' })}><option value="">请确认可变现性</option><option value="liquid">可提供流动性</option><option value="illiquid">非流动性资产</option></select></Field>
          <Field label={`${asset.id}分类与代理理由`}><input className={inputClass} value={asset.rationale} onChange={e => patchAsset(index, { rationale: e.target.value })} /></Field>
          <Field label={`${asset.id}预期年收益（%）`}><NumberInput className={inputClass} value={pct(asset.annual_return)} onValueChange={number => patchAsset(index, { annual_return: number / 100 })} /></Field>
          <Field label={`${asset.id}年化波动（%）`}><NumberInput className={inputClass} value={pct(asset.annual_volatility)} onValueChange={number => patchAsset(index, { annual_volatility: number / 100 })} /></Field>
          <Field label={`${asset.id}均值不确定半宽（百分点）`} hint="例如填 2，表示预期均值可能偏低或偏高 2 个百分点；不是资产波动率。"><NumberInput className={inputClass} value={pct(asset.mean_uncertainty)} min={0} onValueChange={number => patchAsset(index, { mean_uncertainty: number / 100 })} /></Field>
        </div>
      </fieldset>)}
    </section>
    <section className={`${sectionClass} space-y-4`} aria-label="风险假设参考">
      <h2 className="text-lg font-semibold">风险参考与相关性</h2>
      <p className="text-sm leading-6 text-slate-600">可以填写前瞻风险判断，也可显式读取历史风险参考。读取只回填波动和相关性，不会把历史均值当成未来预期。历史参考固定要求 SSE 开放日连续日频，并按 252 期年化；任一开放日整组缺失都会拒绝计算。</p>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="风险样本开始"><input type="date" className={inputClass} value={reference.start_date} onChange={e => onReferenceChange({ ...reference, start_date: e.target.value })} /></Field>
        <Field label="风险样本结束"><input type="date" className={inputClass} max={value.as_of} value={reference.end_date} onChange={e => onReferenceChange({ ...reference, end_date: e.target.value })} /></Field>
        <Field label="对角收缩强度（%）" hint="明确设定的研究参数，不是自动估计值。"><NumberInput className={inputClass} value={pct(reference.shrinkage)} min={0} max={100} onValueChange={number => onReferenceChange({ ...reference, shrinkage: number / 100 })} /></Field>
      </div>
      <Button disabled={busy || !reference.start_date || !reference.end_date || !Number.isFinite(reference.shrinkage)} onClick={onLoadReference}>读取历史风险参考</Button>
      <p role="status" className="text-xs leading-5 text-slate-600">{value.risk_origin === 'historical_reference' ? `已引用 ${value.risk_reference?.start_date} 至 ${value.risk_reference?.end_date} 的历史风险；手动修改波动或相关性会解除此引用。` : '当前为人工风险假设；未读取或已编辑历史参考。'}</p>
      <div className="overflow-x-auto rounded-lg border border-slate-200"><table aria-label="大类相关矩阵" className="w-full min-w-[400px] text-sm">
        <caption className="p-3 text-left text-xs leading-5 text-slate-600">编辑上三角，下三角同步；服务端检查整个矩阵是否有效，不会自动修正。</caption>
        <thead><tr><th scope="col" className="p-3 text-left">资产类别</th>{value.assets.map(asset => <th scope="col" key={asset.id} className="p-3 text-right">{asset.id}</th>)}</tr></thead>
        <tbody>{value.assets.map((asset, row) => <tr key={asset.id} className="border-t border-slate-200"><th scope="row" className="whitespace-nowrap p-3 text-left font-medium">{asset.id}</th>{value.assets.map((other, column) => <td key={other.id} className="min-w-24 p-2 text-right tabular-nums">{column > row ? <NumberInput aria-label={`${asset.id}与${other.id}相关系数`} className={`${inputClass} text-right tabular-nums`} value={value.correlation[row]?.[column] ?? NaN} min={-1} max={1} onValueChange={number => setCorrelation(row, column, number)} /> : Number.isFinite(value.correlation[row]?.[column]) ? value.correlation[row][column].toFixed(3) : '—'}</td>)}</tr>)}</tbody>
      </table></div>
      <label className="flex items-start gap-3 text-sm leading-6 text-slate-800"><input type="checkbox" className="mt-1" checked={value.basis_confirmed} onChange={e => onChange({ ...value, basis_confirmed: e.target.checked })} /><span>我已确认：所有假设采用相同计价币种与年化算术总收益口径；经济角色和流动性已经复核。系统未代我验证汇率对冲或宏观因果关系。</span></label>
    </section>
  </div>
}
