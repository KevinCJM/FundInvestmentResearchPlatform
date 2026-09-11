import type { TimingCatalog, TimingDefinition, TimingTraining } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'

const actionClass = 'min-h-11 rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-700 hover:bg-slate-50 disabled:opacity-40'
export const trainingModeLabels = { global: '全训练期选一个方案', state: '按条件状态选方案', month: '按自然月选方案', quarter: '按自然季度选方案' }
export function conditionPorts(definition: TimingDefinition, catalog: TimingCatalog) {
  return definition.nodes.flatMap(node => (catalog.operators.find(operator => operator.id === node.op)?.outputs || []).filter(port => port.type === 'condition').map(port => ({ value: `${node.id}.${port.name}`, label: `${node.label} · ${port.label}` })))
}
export function timingTrainingError(definition: TimingDefinition, catalog: TimingCatalog): string {
  const training = definition.training
  if (!training) return ''
  const refs = new Set(conditionPorts(definition, catalog).map(port => port.value))
  if (!training.actions.length || training.actions.length > 8) return '训练需要 1–8 个候选动作。'
  if (!training.actions.some(action => action.entry !== null)) return '训练至少需要一个非空仓候选动作。'
  if (training.actions.some(action => !action.label.trim() || (action.entry !== null && !refs.has(action.entry)))) return '请为训练动作填写名称并连接有效的条件；空仓需显式选择。'
  if (training.state_refs.length > 3 || (training.mode === 'state' && !training.state_refs.length) || training.state_refs.some(ref => !refs.has(ref))) return '请连接 1–3 个有效的条件状态。'
  if (training.mode !== 'state' && training.state_refs.length) return '只有状态选择模式使用状态条件。'
  if (training.min_trades < 2 || training.min_trades > 10000 || !Number.isInteger(training.min_trades) || training.embargo_bars < 0 || training.embargo_bars > 250 || !Number.isInteger(training.embargo_bars)) return '最少已完成信号数须为 2–10000 的整数，隔离交易日须为 0–250 的整数。'
  if (![training.confidence, training.risk_penalty, training.min_utility].every(value => Number.isFinite(value) && value >= 0 && value <= 5)) return '置信惩罚、风险惩罚与最低效用须在 0–5 之间。'
  if (training.search_space.length > 6) return '最多设置 6 个搜索维度。'
  let combinations = 1
  for (const dimension of training.search_space) {
    if (!dimension.label.trim() || !dimension.choices.length) return '搜索维度需要名称和至少一个参数方案。'
    if (dimension.choices.length > 12) return '每个搜索维度最多 12 个方案；非空仓动作 × 参数组合最多 108 个。'
    combinations *= dimension.choices.length
    for (const choice of dimension.choices) {
      if (!choice.length || choice.length > 4) return '每个参数方案需要 1–4 项联动参数；不搜索时请移除搜索维度。'
      for (const patch of choice) {
        const node = definition.nodes.find(item => item.id === patch.node)
        const parameter = catalog.operators.find(operator => operator.id === node?.op)?.parameters.find(item => item.name === patch.parameter)
        if (!parameter || (parameter.type !== 'string' && (typeof patch.value !== 'number' || !Number.isFinite(patch.value)))) return '搜索参数存在缺失步骤或无效数值，请重新选择。'
        if (parameter.options?.length && !parameter.options.some(option => option.value === patch.value)) return '请从目录提供的选项选择搜索参数值。'
        if (typeof patch.value === 'number' && ((parameter.minimum != null && patch.value < parameter.minimum) || (parameter.maximum != null && patch.value > parameter.maximum) || (parameter.type === 'integer' && !Number.isInteger(patch.value)))) return '搜索参数超出目录允许范围。'
      }
    }
  }
  if (training.actions.filter(action => action.entry !== null).length * combinations > 108) return '非空仓动作 × 参数组合最多 108 个，请减少搜索方案。'
  return ''
}

export default function TimingTrainingEditor({ definition, catalog, onChange }: { definition: TimingDefinition; catalog: TimingCatalog; onChange: (next: TimingDefinition) => void }) {
  const training = definition.training
  const ports = conditionPorts(definition, catalog)
  const patch = (next: Partial<TimingTraining>) => training && onChange({ ...definition, training: { ...training, ...next } })
  const defaultPatch = () => {
    const node = definition.nodes.find(item => catalog.operators.find(operator => operator.id === item.op)?.parameters.length)
    const parameter = catalog.operators.find(operator => operator.id === node?.op)?.parameters[0]
    return { node: node?.id || '', parameter: parameter?.name || '', value: parameter?.default ?? 0 }
  }
  const selectReference = (value: string | null, onSelect: (next: string | null) => void, label: string, cash = false) => <select aria-label={label} value={value === null ? '__cash__' : value} className={timingField} onChange={event => onSelect(event.target.value === '__cash__' ? null : event.target.value)}><option value="">请选择条件</option>{cash && <option value="__cash__">空仓，不发出买入信号</option>}{value && !ports.some(port => port.value === value) && <option value={value}>缺失连接：{value}</option>}{ports.map(port => <option key={port.value} value={port.value}>{port.label}</option>)}</select>
  return <section aria-label="训练与方案选择" className="rounded-xl border border-slate-200 bg-white p-4">
    <label className="flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-800"><input type="checkbox" className="h-4 w-4" checked={!!training} onChange={event => {
      if (!event.target.checked) { const { training: omitted, ...rest } = definition; onChange(rest); return }
      onChange({ ...definition, training: { mode: 'global', state_refs: [], actions: [{ id: 'entry', label: '当前买入规则', entry: definition.entry }, { id: 'cash', label: '空仓', entry: null }], search_space: [], min_trades: 5, confidence: 1, risk_penalty: .1, min_utility: 0, embargo_bars: 0 } })
    }} />用训练期选择方案</label>
    <p className="mt-1 text-xs leading-5 text-slate-500">{training ? `${trainingModeLabels[training.mode]}；${training.actions.length} 个动作。` : '关闭时直接执行下方买入条件，不做自动选择。'} 只使用样本外开始前已完成的信号；之后冻结，不滚动重训。</p>
    {training && <details className="mt-3"><summary className="cursor-pointer py-2 text-sm font-medium text-indigo-700">编辑训练规则与候选动作</summary><div className="mt-3 space-y-4">
      <div className="grid gap-3 sm:grid-cols-2"><label className="text-xs text-slate-600">选择方式<select className={timingField} value={training.mode} onChange={event => patch({ mode: event.target.value as TimingTraining['mode'], state_refs: event.target.value === 'state' ? training.state_refs : [] })}>{Object.entries(trainingModeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label className="text-xs text-slate-600">最少已完成信号数<input type="number" className={timingField} min={2} max={10000} step={1} value={training.min_trades} onChange={event => patch({ min_trades: Number(event.target.value) })} /></label></div>
      {(training.mode === 'month' || training.mode === 'quarter') && <p className="text-xs leading-5 text-amber-800">在训练区间内，按历史同自然月或同季度统计。不是每月 / 每季重新训练；样本不足时不强行选择。</p>}
      {training.mode === 'state' && <div className="space-y-2"><p className="text-xs text-slate-600">用条件真假组合状态，最多 3 个；缺失仍为未知。</p>{training.state_refs.map((ref, index) => <div key={index} className="flex items-end gap-2"><label className="min-w-0 flex-1 text-xs text-slate-600">状态条件 {index + 1}{selectReference(ref, value => patch({ state_refs: training.state_refs.map((item, i) => i === index ? value || '' : item) }), `状态条件 ${index + 1}`)}</label><button type="button" className={actionClass} onClick={() => patch({ state_refs: training.state_refs.filter((_, i) => i !== index) })}>移除</button></div>)}<button type="button" className={actionClass} disabled={training.state_refs.length >= 3} onClick={() => patch({ state_refs: [...training.state_refs, ''] })}>添加状态条件</button></div>}
      <fieldset className="space-y-3"><legend className="text-sm font-medium text-slate-800">候选动作</legend>{training.actions.map((action, index) => <div key={action.id} className="rounded-lg bg-slate-50 p-3"><div className="grid gap-3 sm:grid-cols-2"><label className="text-xs text-slate-600">动作 {index + 1} 名称<input className={timingField} value={action.label} maxLength={80} onChange={event => patch({ actions: training.actions.map((item, i) => i === index ? { ...item, label: event.target.value } : item) })} /></label><label className="min-w-0 text-xs text-slate-600">动作 {index + 1} 买入条件{selectReference(action.entry, entry => patch({ actions: training.actions.map((item, i) => i === index ? { ...item, entry } : item) }), `动作 ${index + 1} 买入条件`, true)}</label></div><button type="button" className="mt-2 min-h-11 px-2 text-xs text-rose-700" onClick={() => patch({ actions: training.actions.filter((_, i) => i !== index) })}>移除动作 {index + 1}</button></div>)}<button type="button" className={actionClass} disabled={training.actions.length >= 8} onClick={() => {
        let index = training.actions.length + 1
        while (training.actions.some(action => action.id === `action_${index}`)) index += 1
        patch({ actions: [...training.actions, { id: `action_${index}`, label: `候选动作 ${index}`, entry: '' }] })
      }}>添加候选动作</button></fieldset>
      <details><summary className="cursor-pointer py-2 text-xs font-medium text-slate-600">置信与风险约束</summary><div className="mt-2 grid gap-3 sm:grid-cols-2">{([['confidence', '置信惩罚系数'], ['risk_penalty', '风险惩罚系数'], ['min_utility', '最低效用'], ['embargo_bars', '冻结前隔离交易日']] as const).map(([key, label]) => <label key={key} className="text-xs text-slate-600">{label}<input type="number" className={timingField} min={0} max={key === 'embargo_bars' ? 250 : 5} step={key === 'embargo_bars' ? 1 : 'any'} value={training[key]} onChange={event => patch({ [key]: Number(event.target.value) })} /></label>)}</div></details>
      <details><summary className="cursor-pointer py-2 text-xs font-medium text-slate-600">参数搜索空间 · {training.search_space.length ? `${training.search_space.length} 个维度` : '不搜索'}</summary><p className="mt-2 text-xs leading-5 text-slate-500">同一方案的参数一起替换，不同维度组合。仅在训练期比较；非空仓动作 × 参数组合最多 108 个。</p><div className="mt-3 space-y-3">{training.search_space.map((dimension, dimensionIndex) => <div key={dimensionIndex} className="space-y-3 rounded-lg border border-slate-200 p-3">
        <label className="block text-xs text-slate-600">搜索维度 {dimensionIndex + 1} 名称<input className={timingField} value={dimension.label} onChange={event => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, label: event.target.value } : item) })} /></label>
        {dimension.choices.map((choice, choiceIndex) => <div key={choiceIndex} className="space-y-2 rounded-lg bg-slate-50 p-3"><p className="text-xs font-medium text-slate-700">方案 {choiceIndex + 1}</p>{choice.map((parameterPatch, patchIndex) => {
          const node = definition.nodes.find(item => item.id === parameterPatch.node)
          const parameters = catalog.operators.find(operator => operator.id === node?.op)?.parameters || []
          const parameter = parameters.find(item => item.name === parameterPatch.parameter)
          const update = (next: typeof parameterPatch) => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, choices: item.choices.map((items, j) => j === choiceIndex ? items.map((value, k) => k === patchIndex ? next : value) : items) } : item) })
          const fieldPrefix = `搜索 ${dimensionIndex + 1} 方案 ${choiceIndex + 1} 参数 ${patchIndex + 1}`
          return <div key={patchIndex} className="grid gap-2 sm:grid-cols-3"><label className="text-xs text-slate-600">步骤<select aria-label={`${fieldPrefix} 步骤`} className={timingField} value={parameterPatch.node} onChange={event => { const nextNode = definition.nodes.find(item => item.id === event.target.value); const nextParameter = catalog.operators.find(operator => operator.id === nextNode?.op)?.parameters[0]; update({ node: event.target.value, parameter: nextParameter?.name || '', value: nextParameter?.default ?? 0 }) }}><option value="">选择步骤</option>{!node && parameterPatch.node && <option value={parameterPatch.node}>缺失：{parameterPatch.node}</option>}{definition.nodes.filter(item => catalog.operators.find(operator => operator.id === item.op)?.parameters.length).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label className="text-xs text-slate-600">参数<select aria-label={`${fieldPrefix} 参数名`} className={timingField} value={parameterPatch.parameter} onChange={event => { const next = parameters.find(item => item.name === event.target.value); update({ ...parameterPatch, parameter: event.target.value, value: next?.default ?? 0 }) }}><option value="">选择参数</option>{parameters.map(item => <option key={item.name} value={item.name}>{item.label}</option>)}</select></label><label className="text-xs text-slate-600">候选值{parameter?.options?.length ? <select aria-label={`${fieldPrefix} 值`} className={timingField} value={parameterPatch.value} onChange={event => update({ ...parameterPatch, value: parameter.type === 'string' ? event.target.value : Number(event.target.value) })}>{parameter.options.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select> : <input aria-label={`${fieldPrefix} 值`} className={timingField} type={parameter?.type === 'string' ? 'text' : 'number'} min={parameter?.minimum} max={parameter?.maximum} step={parameter?.type === 'integer' ? 1 : 'any'} value={parameterPatch.value} onChange={event => update({ ...parameterPatch, value: parameter?.type === 'string' ? event.target.value : Number(event.target.value) })} />}</label><button type="button" className="min-h-11 text-left text-xs text-rose-700" aria-label={`${fieldPrefix} 移除`} onClick={() => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, choices: item.choices.map((items, j) => j === choiceIndex ? items.filter((_, k) => k !== patchIndex) : items) } : item) })}>移除参数</button></div>
        })}<div className="flex flex-wrap gap-2"><button type="button" className={actionClass} onClick={() => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, choices: item.choices.map((items, j) => j === choiceIndex ? [...items, defaultPatch()] : items) } : item) })}>添加联动参数</button><button type="button" className={actionClass} onClick={() => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, choices: item.choices.filter((_, j) => j !== choiceIndex) } : item) })}>移除方案 {choiceIndex + 1}</button></div></div>)}<div className="flex flex-wrap gap-2"><button type="button" className={actionClass} onClick={() => patch({ search_space: training.search_space.map((item, i) => i === dimensionIndex ? { ...item, choices: [...item.choices, [defaultPatch()]] } : item) })}>添加参数方案</button><button type="button" className={actionClass} onClick={() => patch({ search_space: training.search_space.filter((_, i) => i !== dimensionIndex) })}>移除搜索维度 {dimensionIndex + 1}</button></div>
      </div>)}<div className="flex flex-wrap gap-2"><button type="button" className={actionClass} onClick={() => patch({ search_space: [...training.search_space, { label: `参数维度 ${training.search_space.length + 1}`, choices: [[defaultPatch()]] }] })}>添加搜索维度</button>{!!training.search_space.length && <button type="button" className={actionClass} onClick={() => patch({ search_space: [] })}>禁用参数搜索</button>}</div></div></details>
    </div></details>}
  </section>
}
