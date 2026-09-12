import { NavLink } from 'react-router-dom'

const destinations = [
  { path: '/settings/data-sources', label: '同步数据', shortLabel: '同步数据', description: '选择内容，启动或恢复下载' },
  { path: '/settings/source-center', label: '数据源与映射', shortLabel: '来源映射', description: '首次接入或修改配置' },
  { path: '/settings/data-quality', label: '数据质量', shortLabel: '数据质量', description: '检查覆盖范围与异常' },
]

export default function DataWorkspaceNav({ beforeNavigate }: { beforeNavigate?: () => boolean }) {
  return <nav aria-label="数据工作区" className="grid grid-cols-3 gap-1 rounded-xl border border-slate-200 bg-white p-2 sm:gap-2">
    {destinations.map(item => <NavLink key={item.path} to={item.path} end
      onClick={event => { if (beforeNavigate && !beforeNavigate()) event.preventDefault() }}
      className={({ isActive }) => `min-w-0 rounded-xl px-2 py-3 sm:px-4 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${isActive ? 'bg-accent-50 text-accent-900 ring-1 ring-accent-200' : 'text-slate-600 hover:bg-slate-50'}`}>
      <span className="block text-center text-xs font-semibold sm:hidden">{item.shortLabel}</span>
      <span className="hidden text-sm font-semibold sm:block">{item.label}</span>
      <span className="mt-1 hidden text-xs text-slate-600 sm:block">{item.description}</span>
    </NavLink>)}
  </nav>
}
