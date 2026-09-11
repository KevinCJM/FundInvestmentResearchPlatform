import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { DataModelCatalog } from '../../services/dataModel'
import type { ConfigRecord, InterfaceConfig, SourceConfig } from '../../services/dataSources'
import { buttonClass, primaryClass } from './EditorFields'

export default function SourceOverview({ source, interfaces, total, targets, editingEnabled, onConnect, onAddInterface, onReview, onOpen, onClear }: {
  source: ConfigRecord<SourceConfig>
  interfaces: ConfigRecord<InterfaceConfig>[]
  total: number
  targets: DataModelCatalog
  editingEnabled: boolean
  onConnect: () => void
  onAddInterface: () => void
  onReview: () => void
  onOpen: (id: string) => void
  onClear: () => void
}) {
  const [page, setPage] = useState(0)
  const needsCredential = source.config.transport === 'tushare' || source.config.auth_mode !== 'none'
  const connected = source.config.enabled && (!needsCredential || source.credential_configured)
  const pageSize = 12
  const lastPage = Math.max(0, Math.ceil(interfaces.length / pageSize) - 1)
  const currentPage = Math.min(page, lastPage)
  const visible = interfaces.slice(currentPage * pageSize, (currentPage + 1) * pageSize)

  return <section aria-label="数据源使用引导" className="min-w-0 space-y-5">
    <div className="rounded-2xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h2 className="text-xl font-bold text-slate-950">{source.config.name} · 先选你需要的数据</h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">已有接口不用重新创建。选择下方数据，查看字段对应并验证；日常更新直接进入下载页。</p>
        </div>
        <button type="button" className={buttonClass} onClick={onConnect}>{connected ? '查看连接设置' : '配置连接凭据'}</button>
      </div>
      <div className={`mt-4 rounded-xl p-3 text-sm ${connected ? 'bg-slate-50 text-slate-700' : 'bg-amber-50 text-amber-900'}`}>
        <strong>{connected ? '连接配置已准备' : '请先完成连接配置'}：</strong>
        {!source.config.enabled ? '来源已停用，请先在设置中启用。' : needsCredential ? source.credential_configured ? '凭据已保存；接口权限需要通过实际请求验证。' : '尚未保存凭据，暂时无法调用接口。可先查看字段和离线预览。' : '此来源无需认证；是否可访问仍需实际验证。'}
      </div>
    </div>

    <div className="rounded-2xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div><h3 className="font-bold text-slate-950">可接入的数据</h3><p className="mt-1 text-xs text-slate-500">显示 {interfaces.length} / {total} 个接口。状态表示配置进度，不表示已经下载。</p></div>
        <button type="button" className={buttonClass} onClick={onReview}>查看待完善映射</button>
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {visible.map(item => {
          const mapped = item.config.mappings.filter(mapping => mapping.enabled)
          const state = !item.config.enabled ? '接口已停用' : item.validation?.ready ? '定义已校验 · 待验证实际数据' : '映射待完善'
          return <button type="button" key={item.config.id} onClick={() => onOpen(item.config.id)} className="min-w-0 rounded-xl border border-slate-200 p-4 text-left hover:border-indigo-400 hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
            <span className="block text-base font-semibold text-slate-950">{item.config.name}</span>
            <span className="mt-2 block text-sm text-slate-600">存入：{mapped.length ? mapped.map(mapping => targets.tables.find(table => table.table_id === mapping.target_table)?.label ?? mapping.target_table).join('、') : '尚未选择系统数据表'}</span>
            <span className={`mt-3 block text-xs ${item.config.enabled && item.validation?.ready ? 'text-slate-500' : 'text-amber-800'}`}>{state}</span>
            <span className="mt-3 block text-sm font-semibold text-indigo-700">查看字段与验证 →</span>
          </button>
        })}
      </div>
      {!interfaces.length ? <div className="mt-4 rounded-xl bg-slate-50 p-5 text-sm text-slate-600">
        <p>{total ? '没有符合条件的数据。试试清空搜索或切换分类。' : '此来源还没有接口。下一步添加接口，告诉系统如何读取数据。'}</p>
        <button type="button" className={`${buttonClass} mt-3`} disabled={!total && !editingEnabled} onClick={total ? onClear : onAddInterface}>{total ? '清空筛选' : '配置新接口'}</button>
      </div> : null}
      {lastPage > 0 ? <nav aria-label="接口列表分页" className="mt-4 flex items-center justify-between gap-3 text-sm">
        <button type="button" className={buttonClass} disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>上一页</button>
        <span>第 {currentPage + 1} / {lastPage + 1} 页</span>
        <button type="button" className={buttonClass} disabled={currentPage === lastPage} onClick={() => setPage(currentPage + 1)}>下一页</button>
      </nav> : null}
    </div>

    <div className="rounded-2xl border border-indigo-200 bg-indigo-50 p-5">
      <h3 className="text-sm font-bold text-slate-950">只想更新数据？不需要重新配置映射</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">进入统一下载页选择本来源支持的数据及更新方式；配置和样本预览不会下载完整历史，也不会自动发布研究数据。</p>
      {connected ? <Link className={`${primaryClass} mt-3 inline-flex items-center`} to={`/settings/data-sources?source=${encodeURIComponent(source.config.id)}`}>进入 {source.config.name} 下载与更新 →</Link> : <p className="mt-3 text-xs font-semibold text-amber-900">完成连接配置后再下载。</p>}
    </div>
  </section>
}
