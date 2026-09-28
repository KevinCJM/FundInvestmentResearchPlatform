// 续接条。地址栏是页面上下文的唯一事实来源：没带身份就按未选择渲染，
// 上次研究只在这里当书签提示，由用户点一下才生效，导航本身从不写 journey。
import { Link, useLocation } from 'react-router-dom'
import { actionClass } from './ui'
import { useI18n } from '../i18n/runtime'

/** 只有身份参数值得提示；scope、fresh 这类显示开关不算"上次选过什么"。 */
export const RESUME_IDENTITY_KEYS = ['view', 'mandate', 'universe', 'strategic_universe', 'mapping', 'version', 'alloc', 'cma', 'baseline', 'decision'] as const
export const resumeLabelKey = (param: string) => `navigation.resume.${param}`

export default function ResumeResearch({ to }: { to: string }) {
  const { s } = useI18n()
  const location = useLocation()
  const [path, search = ''] = to.split('?')
  const current = new URLSearchParams(location.search)
  // 只列地址栏还没有的身份；已经带齐时这条就没有信息量，不占版面。
  const missing = [...new URLSearchParams(search)].filter(([key, value]) => (RESUME_IDENTITY_KEYS as readonly string[]).includes(key) && !current.getAll(key).includes(value))
  if (path !== location.pathname || !missing.length) return null
  return (
    <div className="mb-3 flex flex-col gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm sm:flex-row sm:items-center sm:justify-between">
      <p className="min-w-0">
        <span className="font-semibold text-slate-800">{s('navigation.resumeLead', { items: [...new Set(missing.map(([key]) => s(resumeLabelKey(key))))].join(' · ') })}</span>
        <span className="mt-0.5 block text-xs text-slate-600">{s('navigation.resumeHint')}</span>
      </p>
      <Link to={to} className={actionClass('secondary', 'shrink-0')}>{s('navigation.resumeAction')}</Link>
    </div>
  )
}
