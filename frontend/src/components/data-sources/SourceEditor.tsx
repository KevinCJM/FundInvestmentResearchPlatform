import { useRef, useState } from 'react'
import type { ConfigRecord, SourceConfig } from '../../services/dataSources'
import { deleteSourceConfig, saveSource, saveSourceCredential } from '../../services/dataSources'
import { buttonClass, inputClass, PolicyEditor, primaryClass, TextField, validateEditor } from './EditorFields'

export default function SourceEditor({ record, editingEnabled, onSaved, onDirty, onCredentialChanged, onNext }: {
  record: ConfigRecord<SourceConfig>; editingEnabled: boolean; onSaved: (id?: string) => void; onDirty: (dirty: boolean) => void
  onCredentialChanged?: (configured: boolean) => void
  onNext?: () => void
}) {
  const [config, setConfig] = useState(record.config)
  const [credential, setCredential] = useState('')
  const [credentialConfigured, setCredentialConfigured] = useState(record.credential_configured === true)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const form = useRef<HTMLFormElement>(null)
  const patch = (next: SourceConfig) => { setConfig(next); onDirty(true); setMessage(''); setError('') }
  const perform = async (operation: () => Promise<void>) => {
    setBusy(true); setError(''); setMessage('')
    try { await operation() } catch (reason) { setError(reason instanceof Error ? reason.message : '操作未完成。') } finally { setBusy(false) }
  }
  return <form ref={form} className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5" onSubmit={event => {
    event.preventDefault()
    if (busy || !editingEnabled || !validateEditor(form.current)) return
    const destinationChanged = record.revision > 0 && ['base_url', 'transport', 'auth_mode', 'auth_header'].some(key => config[key as keyof SourceConfig] !== record.config[key as keyof SourceConfig])
    if (destinationChanged && !window.confirm('修改地址或认证协议后，已有凭据不会发往新地址；需要重新保存凭据。继续保存？')) return
    void perform(async () => { const saved = await saveSource(config, record.revision); onDirty(false); onSaved(saved.config.id) })
  }}>
    <div><h2 className="text-lg font-bold text-slate-950">{record.revision ? '数据源设置' : '新建数据源'}</h2><p className="mt-1 text-xs text-slate-500">凭据独立存储，不写入参数、映射或浏览器缓存。修改配置并保存后，下次任务使用新版本。</p></div>
    {error ? <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}</p> : null}
    {message ? <p role="status" className="text-sm text-emerald-800">{message}</p> : null}
    <fieldset disabled={busy || !editingEnabled} className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2">
        <TextField label="数据源 ID" value={config.id} required disabled={record.revision > 0} onChange={id => patch({ ...config, id })} />
        <TextField label="数据源名称" value={config.name} required onChange={name => patch({ ...config, name })} />
        <label className="text-xs font-semibold text-slate-600">连接协议<select className={inputClass} value={config.transport} onChange={e => { const transport = e.target.value as SourceConfig['transport']; patch({ ...config, transport, base_url: transport === 'akshare' ? '' : config.base_url || 'https://example.com', auth_mode: transport === 'akshare' ? 'none' : config.auth_mode }) }}><option value="http">HTTPS JSON / CSV</option><option value="tushare">Tushare API 协议</option><option value="akshare">AKShare SDK</option></select></label>
        {config.transport !== 'akshare' ? <><TextField label="HTTPS 基础地址" value={config.base_url} required onChange={base_url => patch({ ...config, base_url })} />
        <label className="text-xs font-semibold text-slate-600">认证方式<select className={inputClass} value={config.auth_mode} onChange={e => patch({ ...config, auth_mode: e.target.value as SourceConfig['auth_mode'] })}><option value="none">按接口协议 / 无额外认证</option><option value="bearer">Bearer Token</option><option value="header">自定义请求头</option></select></label></> : <p className="text-xs leading-6 text-slate-600">SDK 使用已登记函数访问公开数据，无需 Token。可在接口页修改参数和字段映射。</p>}
        {config.auth_mode === 'header' ? <TextField label="认证请求头名称" value={config.auth_header} required onChange={auth_header => patch({ ...config, auth_header })} /> : null}
      </div>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={config.enabled} onChange={e => patch({ ...config, enabled: e.target.checked })} />启用数据源</label>
      <details className="rounded-xl border border-slate-200 p-4"><summary className="cursor-pointer text-sm font-bold">高级：共享下载限制</summary><div className="mt-4"><PolicyEditor value={config.policy} onChange={policy => patch({ ...config, policy })} /></div></details>
      <TextField label="备注" value={config.notes} onChange={notes => patch({ ...config, notes })} />
      <div className="flex flex-wrap gap-3"><button type="submit" className={primaryClass}>保存数据源</button>{record.revision > 0 ? <button type="button" className={buttonClass} onClick={() => {
        if (window.confirm('删除数据源及其凭据？须先删除该来源下的接口，已下载数据不会删除。')) void perform(async () => { await deleteSourceConfig('source', config.id, record.revision); onDirty(false); onSaved() })
      }}>删除数据源</button> : null}</div>
    </fieldset>
    {record.revision > 0 && config.transport !== 'akshare' && (config.transport === 'tushare' || config.auth_mode !== 'none') ? <section className="space-y-3 border-t border-slate-200 pt-4" aria-label="数据源凭据">
      <p className="text-sm font-semibold">认证凭据：{credentialConfigured ? '已配置（不回显）' : '未配置'}</p>
      <label className="block text-xs text-slate-600">新的认证凭据<input className={inputClass} type="password" autoComplete="new-password" value={credential} disabled={busy || !editingEnabled} onChange={e => { setCredential(e.target.value); onDirty(true) }} /></label>
      <div className="flex gap-3"><button type="button" className={buttonClass} disabled={busy || !editingEnabled || !credential.trim() || JSON.stringify(config) !== JSON.stringify(record.config)} onClick={() => void perform(async () => {
        const result = await saveSourceCredential(record.config.id, credential); setCredential(''); setCredentialConfigured(result.credential_configured); onCredentialChanged?.(result.credential_configured); onDirty(JSON.stringify(config) !== JSON.stringify(record.config)); setMessage('凭据已保存。下一步选择接口进行验证；保存不代表已验证权限。')
      })}>保存凭据</button><button type="button" className={buttonClass} disabled={busy || !editingEnabled || !credentialConfigured} onClick={() => {
        if (window.confirm('清除该数据源的认证凭据？已下载数据不会删除。')) void perform(async () => { await saveSourceCredential(record.config.id, null); setCredentialConfigured(false); setCredential(''); onCredentialChanged?.(false); onDirty(JSON.stringify(config) !== JSON.stringify(record.config)); setMessage('凭据已清除。') })
      }}>清除凭据</button></div>
    </section> : <p className="text-xs text-slate-500">{record.revision ? '当前连接不需要认证凭据。' : '先保存数据源，再配置认证凭据和接口。'}</p>}
    {record.revision > 0 && onNext ? <div className="border-t border-slate-100 pt-4"><p className="mb-3 text-xs text-slate-500">连接配置完成后，可选择已有接口，或配置新接口。</p><button type="button" className={buttonClass} disabled={busy || !editingEnabled} onClick={onNext}>下一步：配置新接口</button></div> : null}
  </form>
}
