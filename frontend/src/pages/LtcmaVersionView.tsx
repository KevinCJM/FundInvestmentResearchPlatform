import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card } from '../components/ui'
import { Feedback, Field } from '../components/risk-models/ResearchUI'
import LtcmaResults from '../components/ltcma/LtcmaResults'
import { control, linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { ltcma, ltcmaSaaPath, type LtcmaView } from '../services/ltcma'

export default function LtcmaVersionView() {
  const { versionId = '' } = useParams(), [params] = useSearchParams(), navigate = useNavigate()
  const { t } = useLtcmaText(), task = useLtcmaTask()
  const [item, setItem] = useState<LtcmaView | null>(null), [revision, setRevision] = useState(0)
  const [retiring, setRetiring] = useState(false), [reason, setReason] = useState(''), [confirmed, setConfirmed] = useState(false)
  useEffect(() => {
    setItem(null); setRetiring(false); setConfirmed(false)
    void task.run(signal => ltcma.view(versionId, signal), value => {
      if (value.version.id !== versionId) throw new Error('LTCMA_VERSION_MISMATCH')
      setItem(value)
    })
    return task.invalidate
  }, [versionId, revision])
  const retire = () => {
    if (!item || !confirmed || reason.trim().length < 5) return
    void task.run(signal => ltcma.retire(item.version, reason.trim(), signal), () => { setRetiring(false); setItem({ ...item, retired: true }) })
  }
  const apply = () => {
    if (!item || item.retired) return
    const path = ltcmaSaaPath(item.version), mandate = params.get('mandate')
    navigate(mandate ? `${path}&mandate=${encodeURIComponent(mandate)}` : path)
  }
  return <div className="min-w-0 space-y-4 text-slate-900">
    <Link className={linkClass} to="/pre-investment/ltcma">{t('back')}</Link>
    <header className="space-y-2"><h1 className="text-2xl font-bold">{item?.version.name ?? t('title')}</h1><p className="text-sm leading-6 text-slate-600">{t('readonly')}</p></header>
    <Feedback error={task.error} />{task.error && <Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>}
    {task.busy && <div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('loading')}</p><div className="h-24 animate-pulse rounded-lg bg-slate-100 motion-reduce:animate-none" /></div>}
    {item && <>
      <div className="flex flex-wrap items-center gap-3"><Badge>{t(item.version.definition.model?.method ?? 'manual')}</Badge><Badge tone={item.retired ? 'warning' : 'neutral'}>{t(item.retired ? 'retired' : 'confirmed')}</Badge>
        <Button tone="primary" disabled={item.retired || task.busy} onClick={apply}>{t('useSaa')}</Button>
        <Link className={linkClass} to={`/pre-investment/ltcma/new?copy=${encodeURIComponent(versionId)}`}>{t('copy')}</Link>
      </div>
      {item.retired && <p className="text-sm text-amber-800">{t('retireHint')}</p>}
      <Card><LtcmaResults value={item.version} /></Card>
      {!item.retired && <section className="space-y-3 border-t border-slate-200 pt-4">
        <Button tone="danger" disabled={task.busy} onClick={() => { setRetiring(true); setConfirmed(false) }}>{t('retire')}</Button>
        {retiring && <><p className="text-sm leading-6 text-slate-700">{t('retireHint')}</p><Field label={t('retireReason')}><textarea className={control} value={reason} onChange={event => { setReason(event.target.value); setConfirmed(false) }} /></Field>
          <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />{t('retireConfirm')}</label>
          <div className="flex gap-2"><Button disabled={task.busy} onClick={() => setRetiring(false)}>{t('cancel')}</Button><Button tone="danger" disabled={task.busy || !confirmed || reason.trim().length < 5} onClick={retire}>{t('retireConfirm')}</Button></div>
        </>}
      </section>}
    </>}
  </div>
}
