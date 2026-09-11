import type { AutoIncrementalPlan, EtlDefinition } from '../../services/etl'
import { buttonClass, inputClass } from './EditorFields'

const dateText = (value: string) => /^\d{8}$/.test(value) ? `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6)}` : value

export default function AutoPlanSummary({ plan, busy = false, onBaseline, onExclude }: {
  plan: AutoIncrementalPlan; busy?: boolean; onBaseline?: (id: string) => void; onExclude?: (definition: EtlDefinition) => void
}) {
  const proposal = plan.exclusion_proposal
  return <section aria-label="自动增量计划" className="min-w-0 space-y-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-sm">
    <h3 className="font-bold">{plan.ready ? '本次实际下载计划' : '尚未启动：请处理增量计划中的拦截项'}</h3>
    <p className="break-all text-xs">基线快照：{plan.snapshot}</p>
    <p className="text-xs leading-6">请求截止：{plan.cutoff_date}（上海时区昨日，不代表上游已全部披露）。净值/行情回查最近 {plan.lookback_trade_days} 个交易日；持仓/分红按已核验查询覆盖与修订策略规划。写入私有候选，不覆盖正式快照。</p>
    {onBaseline && Boolean(plan.baseline_choices?.length) ? <div className="space-y-2 rounded-lg bg-white p-3">
      <label className="block font-semibold">补足缺失基线<select aria-label="补足缺失基线" className={inputClass} disabled={busy} value={plan.supplemental_baseline?.run_id ?? ''} onChange={e => onBaseline(e.target.value)}>
        <option value="">仅使用活跃快照</option>
        {plan.baseline_choices?.map(item => <option key={item.run_id} value={item.run_id}>{item.name.replace(/(?:（恢复）)+/g, '')} · {item.finished_at ? new Date(item.finished_at).toLocaleString() : item.run_id.slice(0, 8)} · 可补 {item.files.length} 张表</option>)}
      </select></label>
      <p className="text-xs leading-5 text-slate-600">{plan.supplemental_baseline?.scope === 'acquisition' ? '选择已完成下载作为本次采集基线，复用对应数据与查询覆盖；不使用比活跃快照更旧的表。' : '选择已完成下载，只补活跃快照缺少的表。'}不会覆盖现有研究数据或自动发布；启动前核对文件校验和，损坏则拒绝下载。</p>
    </div> : null}
    {plan.warnings?.map(message => <p key={message} className="rounded-lg bg-amber-50 p-3 text-xs text-amber-900">{message}</p>)}
    <details open={!plan.ready}><summary className="cursor-pointer font-semibold">数据集计划 · {plan.steps.length} 个步骤</summary><div className="mt-2 grid gap-2 lg:grid-cols-2">{plan.steps.map(step => <article key={step.id} className="min-w-0 rounded-lg bg-white p-3">
      <h4 className="font-semibold">{step.name}</h4>
      <p className="mt-1 text-xs">{step.strategy === 'blocked' ? '不可自动增量' : step.strategy === 'incremental' ? `已有最新：${step.latest_date ?? '未知'}；请求区间：${dateText(step.start_date)} 至 ${dateText(step.end_date)}` : step.strategy === 'derive' ? '本地派生，无网络下载' : '目录、日历或披露表刷新'}</p>
      <p className="mt-2 text-xs leading-5 text-slate-600">{step.message}</p>
      {step.query_dates ? <dl className="mt-2 grid gap-1 text-xs text-slate-700">
        <div><dt className="inline">已查询至：</dt><dd className="inline">{step.coverage_through ?? '暂无可验证记录'}</dd></div>
        <div><dt className="inline">最近复核：</dt><dd className="inline">{step.last_checked_at ? new Date(step.last_checked_at).toLocaleString('zh-CN') : '未知'}</dd></div>
        <div><dt className="inline">本次：</dt><dd className="inline">{step.query_dates.length ? `${step.new_query_days} 天新增/缺口 · ${step.revision_query_days} 天修订复核 · ${step.reused_query_days} 天复用` : '区间已覆盖，无需重复查询（新增基金将单独补齐）'}</dd></div>
        {step.request_estimate ? <div><dt className="inline">请求量参考：</dt><dd className="inline">至少 {step.request_estimate.minimum} 次，单轮分页上限 {step.request_estimate.page_ceiling.toLocaleString('zh-CN')} 次。{step.request_estimate.note}</dd></div> : null}
      </dl> : null}
    </article>)}</div></details>
    {plan.errors.map((error, i) => <p key={i} className="text-rose-800">{error.message}</p>)}
    {proposal && onExclude ? <div className="space-y-2 border-t border-indigo-200 pt-3">
      <p className="text-xs leading-6">可从本次草稿排除：{proposal.excluded.map(s => `${s.name}${s.reason === 'dependency' ? '（真实依赖受影响）' : ''}`).join('、')}。排除不等于下载成功；这些数据需单独处理。</p>
      {proposal.rebuild_dependencies ? <p className="text-xs text-amber-900">旧串行流程将按服务端任务合同重新梳理依赖，只有顺序关系的节点可继续；请确认后再应用。</p> : null}
      <button type="button" className={buttonClass} disabled={busy} onClick={() => {
        if (window.confirm(`仅保留可增量步骤？\n将排除：${proposal.excluded.map(s => s.name).join('、')}。\n${proposal.rebuild_dependencies ? '同时按任务合同重新梳理旧串行连线。\n' : ''}只修改草稿，可以撤销；不会启动下载或修改历史运行。`)) onExclude(proposal.definition)
      }}>仅保留可增量步骤</button>
    </div> : null}
  </section>
}
