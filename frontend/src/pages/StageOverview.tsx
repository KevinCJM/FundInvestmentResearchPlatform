import { Link } from 'react-router-dom'
import { getStage, statusLabels, type StageId } from '../app/processRegistry'

const statusStyles = {
  available: 'bg-emerald-100 text-emerald-800',
  partial: 'bg-amber-100 text-amber-900',
  prototype: 'bg-sky-100 text-sky-800',
}

export default function StageOverview({ stageId }: { stageId: StageId }) {
  const stage = getStage(stageId)

  return (
    <div className="space-y-5">
      <div className={`rounded-2xl border ${stage.accent.border} ${stage.accent.soft} p-5 sm:p-6`}>
        <p className={`text-xs font-semibold uppercase tracking-[0.2em] ${stage.accent.text}`}>阶段工作台</p>
        <h2 className="mt-2 text-2xl font-bold text-slate-950">{stage.label}流程目录</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">选择一个流程节点进入对应工作区。已有能力直接复用，静态演示节点使用预置示例数据呈现未来功能与交互边界。</p>
      </div>

      <ol className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {stage.nodes.map((node, index) => (
          <li key={node.id}>
            <Link to={node.path} className="group flex h-full min-h-44 flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-indigo-500">
              <div className="flex items-start justify-between gap-3">
                <span className={`rounded-lg px-2.5 py-1 text-xs font-bold ${stage.accent.badge}`}>{String(index + 1).padStart(2, '0')}</span>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyles[node.status]}`}>{statusLabels[node.status]}</span>
              </div>
              <h3 className="mt-4 text-base font-semibold text-slate-900 group-hover:text-indigo-700">{node.label}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">{node.description}</p>
              <span className={`mt-auto pt-4 text-sm font-semibold ${stage.accent.text}`}>进入节点 →</span>
            </Link>
          </li>
        ))}
      </ol>
    </div>
  )
}
