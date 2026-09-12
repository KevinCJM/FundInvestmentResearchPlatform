import { useState } from 'react'
import { Link } from 'react-router-dom'
import DataDownloadWorkspace from '../components/data-sources/DataDownloadWorkspace'
import DataHealthRefreshPanel from '../components/dashboard/DataHealthRefreshPanel'
import DataWorkspaceNav from '../components/data-sources/DataWorkspaceNav'
import DataStoragePanel from '../components/data-sources/DataStoragePanel'

export default function DataManagement() {
  const [legacyOpen, setLegacyOpen] = useState(false)
  return (
    <div className="space-y-5">
      <DataWorkspaceNav />
      <header className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-600">Data operations</p>
            <h1 className="mt-2 text-2xl font-bold text-slate-950">数据下载与更新</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">先选择数据源与下载内容，再执行全量或增量更新。需要跨来源、控制先后顺序或安排快照时，使用 ETL 任务编排。</p>
          </div>
          <Link
            to="/settings/data-quality"
            className="inline-flex min-h-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:border-accent-200 hover:bg-accent-50 hover:text-accent-800 focus:outline-none focus:ring-2 focus:ring-accent-500"
          >
            查看数据质量 →
          </Link>
        </div>
      </header>
      <DataStoragePanel />
      <DataDownloadWorkspace />
      <details className="rounded-xl border border-slate-200 bg-white p-4" onToggle={event => setLegacyOpen(event.currentTarget.open)}>
        <summary className="cursor-pointer text-sm font-semibold">原 Tushare 全市场任务与旧快照维护</summary>
        <p className="my-3 text-xs leading-6 text-amber-900">兼容原模块级下载、恢复及正式旧快照维护。这里与上面的标准候选 ETL 是不同输出路径，请勿把候选结果当作已发布数据。</p>
        {legacyOpen ? <DataHealthRefreshPanel /> : null}
      </details>
    </div>
  )
}
