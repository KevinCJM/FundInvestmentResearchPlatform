import { Link } from 'react-router-dom'
import DataHealthRefreshPanel from '../components/dashboard/DataHealthRefreshPanel'

export default function DataManagement() {
  return (
    <div className="space-y-5">
      <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">Data operations</p>
            <h1 className="mt-2 text-2xl font-bold text-slate-950">数据下载与更新</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">配置 Tushare 数据源，选择更新方式与下载范围，并跟踪后台任务状态。</p>
          </div>
          <Link
            to="/settings/data-quality"
            className="inline-flex min-h-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            查看数据质量 →
          </Link>
        </div>
      </header>
      <DataHealthRefreshPanel />
    </div>
  )
}
