import { useState } from 'react'
import { listEtlTemplates, type EtlDefinition } from '../../services/etl'
import { buttonClass } from './EditorFields'

export default function EtlTemplatePicker({ onChoose, disabled }: { onChoose: (definition: EtlDefinition) => void; disabled: boolean }) {
  const [items, setItems] = useState<Awaited<ReturnType<typeof listEtlTemplates>>>([])
  const [opened, setOpened] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = async () => {
    setBusy(true); setError('')
    try { setItems(await listEtlTemplates()); setOpened(true) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '流程库暂时不可用。') }
    finally { setBusy(false) }
  }
  return <section className="space-y-3">
    <button type="button" className={buttonClass} disabled={disabled || busy} onClick={() => { if (opened) setOpened(false); else void load() }}>{busy ? '正在读取流程库…' : opened ? '收起流程库' : '从流程库创建'}</button>
    {error ? <p role="alert" className="text-xs text-rose-700">{error}</p> : null}
    {opened ? <div className="space-y-2">{items.map(item => <div key={item.id} className="rounded-lg border border-slate-200 p-3"><strong className="text-sm">{item.name}</strong><p className="mt-2 text-xs leading-5 text-slate-600">{item.description}</p><button type="button" className={`${buttonClass} mt-2`} disabled={disabled} onClick={() => onChoose(item.definition)}>使用此流程 · {item.name}</button></div>)}{!items.length ? <p className="text-xs text-slate-600">当前没有可用流程模板，可新建流程并从任务目录添加步骤。</p> : null}</div> : null}
  </section>
}
