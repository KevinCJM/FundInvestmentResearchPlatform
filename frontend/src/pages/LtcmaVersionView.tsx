import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card, ErrorPanel, LoadingPanel } from '../components/ui'
import { Feedback, Field } from '../components/risk-models/ResearchUI'
import LtcmaResults from '../components/ltcma/LtcmaResults'
import { control, linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { ltcma, ltcmaSaaIssue, ltcmaSaaPath, type LtcmaView } from '../services/ltcma'

export default function LtcmaVersionView() {
  const { versionId = '' } = useParams(), [params] = useSearchParams(), navigate = useNavigate()
  const { t } = useLtcmaText(), task = useLtcmaTask()
  const [item, setItem] = useState<LtcmaView | null>(null), [revision, setRevision] = useState(0)
  const [retiring, setRetiring] = useState(false), [reason, setReason] = useState(''), [confirmed, setConfirmed] = useState(false)
  const saaIssue = item ? ltcmaSaaIssue(item.version) : null
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
    if (!item || item.retired || saaIssue) return
    const path = ltcmaSaaPath(item.version), mandate = params.get('mandate')
    navigate(mandate ? `${path}&mandate=${encodeURIComponent(mandate)}` : path)
  }
  const loadFailed = Boolean(task.error) && !task.busy && !item
  return <div className="min-w-0 space-y-4 text-slate-900">
    <Link className={linkClass} to="/pre-investment/ltcma">{t('back')}</Link>
    <header className="space-y-2"><h1 className="text-2xl font-bold">{item?.version.name ?? t('title')}</h1><p className="text-sm leading-6 text-slate-600">{t('readonly')}</p></header>
    {/* 版本没读出来时页面主体是空的，整块换成错误态；停用这类操作失败时版本还在屏幕上，仍是纯文字。 */}
    {!loadFailed && <><Feedback error={task.error} />{task.error && <Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>}</>}
    {task.busy && <LoadingPanel text={t('loading')} />}
    {loadFailed && <ErrorPanel message={task.error} action={<Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>} />}
    {item && <>
      <div className="flex flex-wrap items-center gap-3"><Badge>{t(item.version.definition.model?.method ?? 'manual')}</Badge><Badge tone={item.retired ? 'warning' : 'neutral'}>{t(item.retired ? 'retired' : 'confirmed')}</Badge>
        <Button tone="primary" disabled={item.retired || task.busy || Boolean(saaIssue)} onClick={apply}>{t('useSaa')}</Button>
        <Link className={linkClass} to={`/pre-investment/ltcma/new?copy=${encodeURIComponent(versionId)}`}>{t('copy')}</Link>
      </div>
      {item.retired && <p className="text-sm text-amber-800">{t('retireHint')}</p>}
      {saaIssue && <p role="status" className="text-sm text-amber-800">{saaIssue}</p>}
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
