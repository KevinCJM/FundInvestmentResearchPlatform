import type { RegimeGraphDefinition, RegimeNodeSchema } from '../../services/regimeGraph'

/** Presentation only: the server remains authoritative for graph inference. */
export default function RegimeGuidedFormulaPanel({ definition, schemas, outputId, onBuild, onResources }: {
  definition: RegimeGraphDefinition; schemas: RegimeNodeSchema[]; outputId: string; onBuild: (nodeId?: string) => void; onResources: () => void
}) {
  const root = definition.graph.outputs[outputId]
  const seen = new Set<string>()
  const steps: typeof definition.graph.nodes = []
  const visit = (id: string) => {
    if (seen.has(id)) return
    seen.add(id)
    const node = definition.graph.nodes.find(item => item.id === id)
    if (!node) return
    Object.values(node.inputs).forEach(input => visit(input.node_id))
    steps.push(node)
  }
  if (root) visit(root.node_id)
  const label = (id: string) => {
    const node = definition.graph.nodes.find(item => item.id === id)
    return node?.label || schemas.find(item => (item.id || item.type) === node?.type)?.label || id
  }
  const outputLabel = definition.graph.channel_metadata?.[outputId]?.label || (outputId === 'state' ? '市场状态' : outputId)
  return <section aria-label="情景公式构建向导" className="mt-4 space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
    <div className="flex items-center justify-between gap-3"><p className="text-xs font-semibold text-slate-600">当前结果：{outputLabel}</p><span className="text-xs text-slate-600">{root ? '已生成' : '待构建'}</span></div>
    {root ? <>
      <p className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-600">选择计算步骤可修改参数和输入；完整结构与画布、高级公式保持同步。</p>
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2"><p className="text-xs font-semibold text-emerald-800">已识别公式的计算结构</p><p className="mt-0.5 text-xs text-emerald-700">最后一步：{label(root.node_id)} · 共 {steps.length} 个计算步骤</p></div>
      <div aria-label="当前输出计算逻辑" className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white p-4">{steps.map((step, i) => <button type="button" key={step.id} onClick={() => onBuild(step.id)} className="min-h-10 rounded-lg border border-accent-100 bg-accent-50 px-3 py-2 text-sm text-accent-900">{i + 1}. {label(step.id)}</button>)}</div>
    </> : <div className="rounded-lg border border-dashed border-accent-200 bg-white p-4 text-center"><p className="font-semibold text-slate-800">从变量、数学算子或已有指标开始</p><p className="mt-1 text-xs leading-5 text-slate-600">组合输入与计算步骤，再将结果连接到输出通道。</p></div>}
    <div className="flex flex-wrap gap-2"><button type="button" onClick={onResources} className="min-h-11 flex-1 rounded-lg bg-accent-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-accent-700">浏览公式构建资源</button>{definition.graph.nodes.length > 0 && <button type="button" onClick={() => onBuild(root?.node_id)} className="min-h-11 rounded-lg border border-accent-200 bg-white px-4 text-sm font-semibold text-accent-700">修改计算逻辑</button>}</div>
  </section>
}
