import type { EtlDefinition, EtlRunOptions } from '../../services/etl'
import { inputClass } from './EditorFields'
import EventUpdateSettings from './EventUpdateSettings'

export default function EtlRunOptionsEditor({ definition, value, onChange, disabled }: {
  definition: EtlDefinition; value: EtlRunOptions; onChange: (value: EtlRunOptions) => void; disabled: boolean
}) {
  const overrides = definition.steps.filter(step => ['download', 'task'].includes(step.kind) && step.mode !== 'inherit')
  return <fieldset disabled={disabled} className="space-y-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4" aria-label="本次运行设置">
    <legend className="px-1 text-sm font-bold">本次运行设置（不修改已保存流程）</legend>
    <label className="block text-sm font-semibold">本次运行模式<select aria-label="本次运行模式" className={inputClass} value={value.mode} onChange={event => {
      const { auto_baseline_run_id: baseline, ...rest } = value
      onChange({ ...rest, mode: event.target.value as EtlRunOptions['mode'], ...(event.target.value === 'auto_incremental' && baseline ? { auto_baseline_run_id: baseline } : {}) })
    }}>
      <option value="auto_incremental">自动增量：依据活跃快照推断日期，无需填写区间</option>
      <option value="incremental">增量更新：从成功断点补充，无断点则初始化</option>
      <option value="full">全量重取：重新下载指定范围，不删除已有数据</option>
    </select></label>
    <p className="text-xs leading-5 text-slate-600">{value.mode === 'auto_incremental' ? '适用于数据集节点：日期和更新策略以服务端预览为准，忽略流程中固定的日期与全量策略。基线缺失时阻止运行，不会自动拉全历史。单接口流程请使用手动模式。' : '运行设置不修改保存的流程。快照按完整输入重新计算，不是只算新增几天。'}</p>
    {overrides.length && value.mode !== 'auto_incremental' ? <p className="text-xs leading-5 text-slate-600">固定策略步骤：{overrides.map(step => `${step.name}（${step.mode === 'full' ? '每次全量刷新' : '固定增量'}）`).join('、')}。</p> : null}
    <div className="grid gap-3 sm:grid-cols-2">{(definition.parameters ?? []).filter(p => value.mode !== 'auto_incremental' || p.data_type !== 'date').map(parameter => <label key={parameter.id} className="text-xs font-semibold">{parameter.label}{parameter.required ? '（必填）' : ''}
      <input aria-label={parameter.label} aria-required={parameter.required} className={inputClass} type={parameter.data_type === 'date' ? 'date' : 'text'} maxLength={200} value={value.parameters[parameter.id] ?? parameter.default} onChange={event => onChange({ ...value, parameters: { ...value.parameters, [parameter.id]: event.target.value } })} />
      {parameter.description ? <span className="mt-1 block font-normal leading-5 text-slate-600">{parameter.description}</span> : null}
    </label>)}</div>
    {value.mode === 'auto_incremental' ? <EventUpdateSettings value={value} onChange={onChange} disabled={disabled} /> : null}
  </fieldset>
}
