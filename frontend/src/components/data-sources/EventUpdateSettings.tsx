import type { EtlRunOptions } from '../../services/etl'
import { inputClass } from './EditorFields'

export default function EventUpdateSettings({ value, onChange, disabled = false }: {
  value: EtlRunOptions; onChange: (value: EtlRunOptions) => void; disabled?: boolean
}) {
  return <details className="rounded-lg border border-slate-200 p-3 text-sm">
    <summary className="cursor-pointer font-semibold">采集基线与披露修订策略</summary>
    <fieldset disabled={disabled} className="mt-3 grid gap-3 sm:grid-cols-2">
      <label>已完成下载的用途<select className={inputClass} value={value.auto_baseline_scope ?? 'missing_only'} onChange={e => onChange({ ...value, auto_baseline_scope: e.target.value as EtlRunOptions['auto_baseline_scope'] })}>
        <option value="acquisition">作为采集基线，复用已下载区间</option><option value="missing_only">仅补足活跃快照缺少的表</option>
      </select></label>
      <label>披露更新方式<select className={inputClass} value={value.event_update_purpose ?? 'update'} onChange={e => onChange({ ...value, event_update_purpose: e.target.value as EtlRunOptions['event_update_purpose'] })}>
        <option value="update">自动更新：补缺口，按计划复核修订</option><option value="recheck">本次主动复核历史修订</option>
      </select></label>
      <label>修订复核间隔（天）<input className={inputClass} type="number" min={1} max={90} value={value.event_revision_interval_days ?? 7} onChange={e => onChange({ ...value, event_revision_interval_days: Number(e.target.value) })} /></label>
      <label>修订回查范围（自然日）<input className={inputClass} type="number" min={1} max={366} value={value.event_revision_window_days ?? 90} onChange={e => onChange({ ...value, event_revision_window_days: Number(e.target.value) })} /></label>
    </fieldset>
    <p className="mt-2 text-xs leading-6 text-slate-600">持仓/分红按公告日记录查询覆盖，已复核空响应也计入覆盖；净值仍按交易日回查。历史缺口请单独手动补齐。采集基线须明确选择，不会切换研究快照或自动发布；修改后须重新预览。</p>
  </details>
}
