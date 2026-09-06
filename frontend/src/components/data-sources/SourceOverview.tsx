import { Link } from 'react-router-dom'
import type { ConfigRecord, InterfaceConfig, SourceConfig } from '../../services/dataSources'
import { buttonClass, primaryClass } from './EditorFields'

export default function SourceOverview({ source, interfaces, editingEnabled, onConnect, onAddInterface, onReview }: {
  source: ConfigRecord<SourceConfig>
  interfaces: ConfigRecord<InterfaceConfig>[]
  editingEnabled: boolean
  onConnect: () => void
  onAddInterface: () => void
  onReview: () => void
}) {
  const tushare = source.config.transport === 'tushare'
  const needsCredential = tushare || source.config.auth_mode !== 'none'
  const connected = source.config.enabled && (!needsCredential || source.credential_configured)
  const incomplete = interfaces.filter(item => !item.validation?.ready).length
  return <section aria-label="数据源使用引导" className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-6">
    <div><h2 className="text-xl font-bold text-slate-950">{source.config.name} · 接下来怎么做</h2>
      <p className="mt-2 text-sm leading-6 text-slate-600">{'来源、接口与字段映射都是可编辑的已保存配置。选择接口后可校验、采样或下载，结果再按多源规则选取。'}</p>
    </div>
    <ol className="space-y-3">
      <li className="rounded-xl bg-slate-50 p-4"><h3 className="text-sm font-semibold">1. 连接数据源</h3>
        <p className="mt-1 text-sm text-slate-600">{!source.config.enabled ? '来源已停用，请先在设置中启用。' : needsCredential ? source.credential_configured ? '凭据已保存；接口权限需要通过实际请求验证。' : '尚未保存凭据，暂时无法调用接口。' : '此来源无需认证。'}</p>
        <button type="button" className={`${buttonClass} mt-3`} onClick={onConnect}>{connected ? '查看连接设置' : '配置连接凭据'}</button>
      </li>
      <li className="rounded-xl bg-slate-50 p-4"><h3 className="text-sm font-semibold">2. 检查需要的接口与映射</h3>
        <p className="mt-1 text-sm text-slate-600">共 {interfaces.length} 个接口，{incomplete} 个映射待完善。定义校验通过不代表真实数据已经下载。</p>
        <div className="mt-3 flex flex-wrap gap-2"><button type="button" className={buttonClass} onClick={onReview}>查看待完善映射</button><button type="button" className={buttonClass} disabled={!editingEnabled} onClick={onAddInterface}>配置新接口</button></div>
      </li>
      <li className="rounded-xl border border-indigo-200 bg-indigo-50 p-4"><h3 className="text-sm font-semibold">3. 选择接口并下载</h3>
        <p className="mt-1 text-sm leading-6 text-slate-600">{'进入统一下载页选择本来源支持的数据及更新方式；也可编排跨来源下载、字段映射、多源取值和快照计算。定时调度尚未开放。'}</p>
        {connected ? <Link className={`${primaryClass} mt-3 inline-flex items-center`} to={`/settings/data-sources?source=${source.config.id}`}>进入 {source.config.name} 下载与更新 →</Link> : <p className="mt-3 text-xs font-semibold text-amber-900">完成第 1 步后再下载。</p>}
      </li>
    </ol>
  </section>
}
