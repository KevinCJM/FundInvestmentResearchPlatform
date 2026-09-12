import { Link } from 'react-router-dom'
import { statusLabels, type StageId } from '../app/processRegistry'
import { useLocalizedStage } from '../i18n/navigation'
import { useI18n } from '../i18n/runtime'

const statusStyles = {
  available: 'bg-emerald-100 text-emerald-800',
  partial: 'bg-amber-100 text-amber-900',
  prototype: 'bg-accent-100 text-accent-800',
}

export default function StageOverview({ stageId }: { stageId: StageId }) {
  const { s } = useI18n()
  const stage = useLocalizedStage(stageId)

  return (
    <div className="space-y-5">
      <div className={`rounded-xl border border-slate-200 ${stage.accent.soft} p-5 sm:p-6`}>
        <p className={`text-xs font-semibold ${stage.accent.text}`}>{s('navigation.stageWorkbench', {}, '阶段工作台')}</p>
        <h2 className="mt-2 text-2xl font-bold text-slate-950">{s('navigation.directory', { stage: stage.label }, '{{stage}}流程目录')}</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{s('navigation.directoryHint', {}, '选择一个流程节点进入对应工作区。已有能力直接复用，静态演示节点使用预置示例数据呈现未来功能与交互边界。')}</p>
      </div>

      <ol className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {stage.nodes.map((node, index) => (
          <li key={node.id}>
            <Link to={node.path} className="group flex h-full min-h-44 flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-accent-500">
              <div className="flex items-start justify-between gap-3">
                <span className={`rounded-lg px-2.5 py-1 text-xs font-bold ${stage.accent.badge}`}>{String(index + 1).padStart(2, '0')}</span>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyles[node.status]}`}>{s(`common.${node.status}`, {}, statusLabels[node.status])}</span>
              </div>
              <h3 className="mt-4 text-base font-semibold text-slate-900 group-hover:text-accent-700">{node.label}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{node.description}</p>
              {/* 可点击的东西全站只有一种颜色；阶段色调只标身份，不标交互。 */}
              <span className="mt-auto pt-4 text-sm font-semibold text-accent-700">{s('navigation.enterNode', {}, '进入节点')} →</span>
            </Link>
          </li>
        ))}
      </ol>
    </div>
  )
}
