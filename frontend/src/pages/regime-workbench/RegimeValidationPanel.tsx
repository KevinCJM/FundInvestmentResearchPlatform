import type { RegimeGraphDefinition, RegimeGraphNode } from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'
import { regimeEnumLabel, regimeParameterLabel } from './regimeDisplay'

function sourceKind(node: RegimeGraphNode) {
  return node.type.startsWith('source.') ? node.type.slice('source.'.length) : node.type
}

function targetId(node: RegimeGraphNode, index: number, occupied: Set<string>) {
  const normalized = node.id.replace(/[^A-Za-z0-9_-]/g, '_')
  const base = /^[A-Za-z]/.test(normalized) ? `target_${normalized}`.slice(0, 58) : `target_${index + 1}`
  let candidate = base
  let sequence = 2
  while (occupied.has(candidate)) candidate = `${base}_${sequence++}`.slice(0, 64)
  return candidate
}

function sourceSummary(source: Record<string, unknown>) {
  const keys = ['kind', 'name', 'ts_code', 'series_id', 'indicator_id', 'artifact_id', 'field', 'frequency']
  const sourceKinds: Record<string, string> = { index: '指数', macro: '宏观', indicator: '指标中心版本', upload: '上传数据', inline: '内联数据' }
  const parts = keys.flatMap((key) => {
    if (source[key] == null || source[key] === '') return []
    const label = key === 'kind' ? '类型' : regimeParameterLabel(key)
    const raw = String(source[key])
    const value = key === 'kind' ? sourceKinds[raw] || '研究序列' : ['field', 'frequency'].includes(key) ? regimeEnumLabel(raw) : raw
    return [`${label}：${value}`]
  })
  return parts.join(' · ') || '来源参数已保存'
}

export default function RegimeValidationPanel({ definition, onChange }: { definition: RegimeGraphDefinition; onChange: (next: RegimeGraphDefinition) => void }) {
  const sourceNodes = definition.graph.nodes.filter((node) => node.type.startsWith('source.') && node.type !== 'source.constant')
  const validation = definition.validation || {}
  const patchValidation = (patch: Record<string, unknown>) => onChange({ ...definition, validation: { ...validation, ...patch } })
  const addTarget = (node: RegimeGraphNode) => {
    const index = definition.evaluation_targets.length
    const source = { kind: sourceKind(node), ...JSON.parse(JSON.stringify(node.parameters)) as Record<string, unknown> }
    onChange({
      ...definition,
      evaluation_targets: [...definition.evaluation_targets, {
        id: targetId(node, index, new Set(definition.evaluation_targets.map((target) => String(target.id || '')))),
        name: String(node.parameters.name || node.label || '评估序列'),
        source,
        primary: definition.evaluation_targets.length === 0,
      }],
    })
  }

  return <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="评估目标与验证规则">
    <div><h3 className="text-sm font-bold text-slate-950">评估目标与验证 <RegimeHelpTip label="评估目标与验证说明" text="识别输入用于划分状态；评估目标只衡量各状态下资产表现，两者必须分开，避免用评价结果反向定义状态。" /></h3><p className="mt-1 text-xs leading-4 text-slate-600">评估序列不参与状态识别，只用于检验各状态下的市场表现；验证规则进入正式运行。</p></div>
    <div className="mt-4 rounded-xl border border-slate-200 p-3">
      <div className="flex items-center justify-between gap-2"><h4 className="text-xs font-bold text-slate-900">表现评估序列 <RegimeHelpTip label="表现评估序列说明" text="例如用宏观指标识别美林时钟，同时用股票、债券、商品指数比较各状态下的收益、波动和回撤。" /></h4><span className="text-xs font-bold text-slate-600">{definition.evaluation_targets.length}/20</span></div>
      <div className="mt-2 flex flex-wrap gap-2">{sourceNodes.map((node) => <button key={node.id} type="button" disabled={definition.evaluation_targets.length >= 20} onClick={() => addTarget(node)} className="min-h-8 rounded-lg border border-accent-200 px-2 text-xs font-bold text-accent-700 disabled:opacity-40">添加 {String(node.parameters.name || node.label || '数据序列')}</button>)}{!sourceNodes.length ? <p className="text-xs text-slate-600">先加入指数、宏观、指标中心、上传或内联时序等可评价数据源。</p> : null}</div>
      <div className="mt-3 space-y-2">{definition.evaluation_targets.map((target, index) => <article key={`${target.id}-${index}`} className="rounded-lg bg-slate-50 p-2"><div className="grid gap-2 sm:grid-cols-[minmax(100px,.8fr)_minmax(140px,1fr)_auto_auto]"><input aria-label={`评估目标${index + 1} ID`} value={String(target.id || '')} onChange={(event) => onChange({ ...definition, evaluation_targets: definition.evaluation_targets.map((item, itemIndex) => itemIndex === index ? { ...item, id: event.target.value } : item) })} className="min-h-9 min-w-0 rounded-lg border border-slate-300 px-2 text-xs" placeholder="target_id" /><input aria-label={`评估目标${index + 1}名称`} value={String(target.name || '')} onChange={(event) => onChange({ ...definition, evaluation_targets: definition.evaluation_targets.map((item, itemIndex) => itemIndex === index ? { ...item, name: event.target.value } : item) })} className="min-h-9 min-w-0 rounded-lg border border-slate-300 px-2 text-xs" placeholder="序列名称" /><label className="flex min-h-9 items-center gap-1 text-xs font-bold text-slate-600"><input type="radio" name="primary-evaluation-target" checked={target.primary === true} onChange={() => onChange({ ...definition, evaluation_targets: definition.evaluation_targets.map((item, itemIndex) => ({ ...item, primary: itemIndex === index })) })} />主评估</label><button type="button" onClick={() => { const remaining = definition.evaluation_targets.filter((_, itemIndex) => itemIndex !== index); const hasPrimary = remaining.some((item) => item.primary === true); onChange({ ...definition, evaluation_targets: remaining.map((item, itemIndex) => ({ ...item, primary: hasPrimary ? item.primary === true : itemIndex === 0 })) }) }} className="min-h-9 rounded-lg px-2 text-xs font-bold text-rose-600">删除</button></div><p className="mt-1 truncate text-xs text-slate-600" title={sourceSummary(target.source as Record<string, unknown>)}>{sourceSummary(target.source as Record<string, unknown>)}</p></article>)}</div>
    </div>
    <div className="mt-3 grid gap-3 rounded-xl border border-slate-200 p-3 sm:grid-cols-2 xl:grid-cols-3">
      <label className="flex min-h-10 items-center gap-2 text-xs font-bold text-slate-700"><input aria-label="启用走步验证" type="checkbox" checked={validation.walk_forward !== false} onChange={(event) => patchValidation({ walk_forward: event.target.checked })} />启用走步验证<RegimeHelpTip label="走步验证说明" text="按时间顺序用过去样本训练、在后续样本验证，避免把未来数据泄露给模型。正式回测和 TAA 应保持开启。" /></label>
      <label className="text-xs font-bold text-slate-600">折数（2–12）<RegimeHelpTip label="走步验证折数说明" text="把历史顺序切成多少个训练—验证区段。折数越多，稳定性观察更细，但计算时间更长。" /><input aria-label="走步验证折数" type="number" min={2} max={12} value={Number(validation.folds ?? 4)} onChange={(event) => patchValidation({ folds: Math.max(2, Math.min(12, Math.trunc(Number(event.target.value) || 4))) })} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-xs font-normal" /></label>
      <label className="text-xs font-bold text-slate-600">稳定性扰动（0–0.5）<RegimeHelpTip label="稳定性扰动说明" text="对可实验参数施加小幅变化，检查状态划分是否剧烈翻转。数值越大，压力越强。" /><input aria-label="稳定性扰动" type="number" min={0} max={0.5} step={0.01} value={Number(validation.stability_perturbation ?? 0.1)} onChange={(event) => patchValidation({ stability_perturbation: Math.max(0, Math.min(0.5, Number(event.target.value) || 0)) })} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-xs font-normal" /></label>
      <p className="text-xs leading-4 text-slate-600 sm:col-span-2 xl:col-span-3">验证截止日由试算命令栏的 as-of 控制；状态最短持续期请在图中加入“最短持续期”节点，因此不会出现只记录但不生效的验证参数。</p>
      <div className="rounded-lg border border-accent-200 bg-accent-50 p-3 text-xs leading-5 text-accent-950 sm:col-span-2 xl:col-span-3"><strong>图谱不绑定使用意图 <RegimeHelpTip label="图谱复用说明" text="同一个定义或模板可以用于产品研究、组合回测、TAA 和情景模拟。系统会在具体发布或绑定时，按目标场景重新执行相应门禁。" /></strong><span className="mt-1 block">实验和模板只描述“如何识别状态”。保存后可在产品研究、组合回测、战术资产配置或情景模拟中复用；正式引用时再检查数据时点、因果性、样本外稳定性和高性能计算审计。</span></div>
    </div>
  </section>
}
