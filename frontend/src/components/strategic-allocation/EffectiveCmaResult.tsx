import { percentText } from '../risk-models/ResearchUI'
import type { CmaPreview } from '../../services/strategicAllocation'

export default function EffectiveCmaResult({ value }: { value: CmaPreview }) {
  const effective = value.effective_assumptions
  if (!effective || !value.model_result) return null
  return <div className="space-y-3" aria-label="实际用于政策求解的模型结果">
    <h3 className="text-sm font-semibold">实际用于政策求解的有效假设</h3>
    <p className="text-sm text-slate-600">以下均值与风险将进入政策比较、资金目标诊断及冻结政策的 TAA 风险检查。</p>
    <div className="overflow-x-auto"><table className="w-full text-sm" aria-label="有效收益与风险"><thead><tr>{['资产', '预期年收益', '年化波动', '显式稳健半宽'].map(label => <th scope="col" key={label} className="p-2 text-right">{label}</th>)}</tr></thead><tbody>{effective.assets.map(asset => <tr key={asset.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left">{asset.id}</th>{[asset.annual_return, asset.annual_volatility, asset.mean_uncertainty].map((n, i) => <td key={i} className="p-2 text-right tabular-nums">{percentText(n)}</td>)}</tr>)}</tbody></table></div>
    {value.model_result.model_audit.limitations.map((text, i) => <p key={i} className="text-sm text-amber-800">{text}</p>)}
    <details><summary className="cursor-pointer text-sm">有效协方差与模型证据</summary><pre className="max-w-full overflow-x-auto p-2 text-xs tabular-nums">{JSON.stringify({ covariance: value.covariance, model: value.model_result.model_audit, hash: value.model_result.content_hash }, null, 2)}</pre></details>
  </div>
}
