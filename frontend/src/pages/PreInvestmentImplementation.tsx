import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card, EmptyState } from '../components/ui'
import { Feedback, Field } from '../components/risk-models/ResearchUI'
import CandidateFields, { initialCandidate } from '../components/implementation/CandidateFields'
import ImplementationReportView from '../components/implementation/Report'
import { control, linkClass, useImplementationText, useResearchTask } from '../components/implementation/shared'
import { implementation, operationKey, type ImplementationCandidate, type ImplementationCatalog, type ImplementationReport, type PackageView, type ResearchPackage } from '../services/implementation'

type Mode = 'implementation' | 'synthesis' | 'validation' | 'approval'
const routes: Record<Mode,string> = {implementation:'/pre-investment/product-allocation-timing', synthesis:'/pre-investment/portfolio-synthesis', validation:'/pre-investment/validation', approval:'/pre-investment/approval'}
export default function PreInvestmentImplementation({mode='implementation'}: {mode?:Mode}) {
  const t=useImplementationText(), task=useResearchTask(), navigate=useNavigate(), [params,setParams]=useSearchParams()
  const [catalog,setCatalog]=useState<ImplementationCatalog|null>(null), [packages,setPackages]=useState<ResearchPackage[]>([])
  const [candidate,setCandidate]=useState<ImplementationCandidate|null>(null), [view,setView]=useState<PackageView|null>(null), [preview,setPreview]=useState<ImplementationReport|null>(null)
  const [dirty,setDirty]=useState(false), [reload,setReload]=useState(0), [copiedFrom,setCopiedFrom]=useState<string|null>(null)
  const [reviewer,setReviewer]=useState(''), [reason,setReason]=useState(''), [reviewDate,setReviewDate]=useState(''), [accept,setAccept]=useState(false)
  const requestKeys=useRef(new Map<string,string>())
  const key=(operation:string, body:unknown)=>{const fingerprint=JSON.stringify([operation,body]);let value=requestKeys.current.get(fingerprint);if(!value){value=operationKey();requestKeys.current.set(fingerprint,value)}return value}
  const packageId=params.get('package'), copyId=params.get('copy'), sourceId=params.get('source')
  useEffect(()=>{
    task.invalidate();setPreview(null);setView(null);setCandidate(null);setDirty(false);setAccept(false)
    void task.run(async signal=>{
      const [options,saved,loaded]=await Promise.all([implementation.catalog(signal),implementation.list(signal),(packageId||copyId)?implementation.view((packageId||copyId)!,signal):Promise.resolve(null)])
      return {options,saved,loaded}
    },({options,saved,loaded})=>{
      setCatalog(options);setPackages(saved.items);setView(copyId?null:loaded);setCopiedFrom(copyId?loaded?.package.id??null:null)
      if(loaded){setCandidate(loaded.package.candidate);setDirty(Boolean(copyId))}
      else if(sourceId){const source=options.sources.find(s=>s.id===sourceId);if(source)setCandidate(initialCandidate(source,options.today))}
    })
    return task.invalidate
  },[packageId,copyId,sourceId,reload])
  const update=(next:ImplementationCandidate)=>{task.invalidate();setCandidate(next);setPreview(null);setDirty(true);setAccept(false)}
  const source=catalog?.sources.find(s=>s.id===candidate?.source.id)
  const current=view?.package??null
  const report=dirty?null:view?.report??preview
  const open=(nextMode:Mode,id:string)=>navigate(`${routes[nextMode]}?package=${encodeURIComponent(id)}`)
  const save=()=>candidate&&void task.run(signal=>implementation.save(candidate,current,key('save',[candidate,current?.revision,copiedFrom]),copiedFrom,signal),item=>open('validation',item.scheme_id))
  const validate=()=>current&&void task.run(signal=>implementation.validate(current,key('validate',[current.scheme_id,current.revision]),signal),()=>setReload(n=>n+1))
  const finalize=()=>current&&report&&void task.run(signal=>implementation.finalize(current,report,{reviewer,reason,review_due_at:reviewDate},key('finalize',[current.revision,report.content_hash,reviewer,reason,reviewDate]),signal),()=>setReload(n=>n+1))
  return <div className="min-w-0 space-y-4 text-slate-900">
    <header className="flex flex-col justify-between gap-3 sm:flex-row"><div><h1 className="text-2xl font-bold">{t(mode+'Title')}</h1><p className="mt-2 text-sm leading-6 text-slate-600">{t(mode+'Description')}</p></div><div className="flex flex-wrap gap-2">
      {mode!=='implementation'&&<Link className={linkClass} to={routes.implementation}>{t('newPackage')}</Link>}
      {mode==='implementation'&&<><Link className={linkClass} to="/pre-investment/product-allocation-timing/construction">{t('historicalConstruction')}</Link><Link className={linkClass} to="/pre-investment/product-allocation-timing/timing">{t('timingResearch')}</Link></>}
    </div></header>
    {current&&<nav aria-label={t('packageSteps')} className="flex flex-wrap gap-2">{(['implementation','synthesis','validation','approval'] as const).map(step=><Link key={step} className={`${linkClass} ${step===mode?'bg-accent-50':''}`} aria-current={step===mode?'step':undefined} to={`${routes[step]}?package=${encodeURIComponent(current.scheme_id)}`}>{t(step+'Short')}</Link>)}</nav>}
    <Feedback error={task.error}/>{task.error&&<Button onClick={()=>setReload(n=>n+1)}>{t('reload')}</Button>}
    {task.busy&&<div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('working')}</p><div className="h-20 animate-pulse rounded-xl bg-slate-200 motion-reduce:animate-none"/></div>}
    {view?.current_eligibility.status==='needs_review'&&<div role="alert" className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900"><p className="font-medium">{t('needsReview')}</p>{view.current_eligibility.reasons.map(value=><p key={value} className="mt-2 break-words">{value}</p>)}</div>}
    {mode==='implementation'&&catalog&&current?.stage!=='finalized'&&<fieldset disabled={task.busy} className="min-w-0 space-y-4">
      <Field label={t('allocationSource')}><select className={control} value={candidate?.source.id??''} onChange={e=>{const s=catalog.sources.find(x=>x.id===e.target.value);if(s){setView(null);setCopiedFrom(null);update(initialCandidate(s,catalog.today));setParams({source:s.id})}}}>
        <option value="">{t('selectSource')}</option>{catalog.sources.map(s=><option key={s.id} value={s.id}>{t(s.kind)} · {s.name} · {s.as_of}</option>)}</select></Field>
      {catalog.sources.length===0&&<EmptyState mascot={false} title={t('noSources')} hint={t('noSourcesHelp')} action={<Link className={linkClass} to="/pre-investment/saa/policy">{t('openSaa')}</Link>}/>}
      {candidate&&source&&<><CandidateFields candidate={candidate} source={source} catalog={catalog} onChange={update}/><div className="flex flex-wrap gap-2">
        <Button tone="primary" onClick={()=>void task.run(signal=>implementation.preview(candidate,signal),result=>setPreview(result))}>{t('preview')}</Button>
        <Button onClick={()=>void task.run(signal=>implementation.optimize(candidate,signal),result=>{setCandidate(result.candidate);setPreview(result.validation);setDirty(true)})}>{t('optimize')}</Button>
        <Button onClick={save}>{t('savePackage')}</Button></div><p className="text-xs leading-5 text-slate-600">{t('saveHelp')}</p></>}
      {candidate&&!source&&<p role="alert" className="text-sm text-amber-900">{t('sourceUnavailable')}</p>}
    </fieldset>}
    {dirty&&current&&<p role="status" className="text-sm text-amber-900">{t('unsaved')}</p>}
    {mode==='implementation'&&preview&&<ImplementationReportView report={preview}/>}
    {current&&<Card><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">{current.name}</h2><p className="mt-1 text-sm text-slate-600">{t('revision',{count:current.revision})} · {t(current.stage)}</p></div><div className="flex flex-wrap gap-2">
      <Link className={linkClass} to={`${routes.implementation}?copy=${encodeURIComponent(current.scheme_id)}`}>{t('copyResearch')}</Link>
      <a className={linkClass} href={implementation.exportUrl(current)}>{t('export')}</a>
    </div></div><p className="mt-3 break-all text-xs text-slate-600">{t('candidateFingerprint')}: {current.candidate_hash}</p></Card>}
    {current&&mode==='synthesis'&&<Card><h2 className="text-lg font-semibold">{t('sameCandidate')}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{t('sameCandidateHelp')}</p><div className="mt-3 flex flex-wrap gap-2"><Button tone="primary" onClick={()=>open('validation',current.scheme_id)}>{t('openValidation')}</Button>{current.stage!=='finalized'&&<Button onClick={()=>open('implementation',current.scheme_id)}>{t('editInputs')}</Button>}</div></Card>}
    {current&&mode==='validation'&&<div className="space-y-3"><div className="flex flex-wrap gap-2"><Button tone="primary" disabled={task.busy||current.stage==='finalized'||view?.current_eligibility.status==='needs_review'} onClick={validate}>{t('validate')}</Button>{report?.research_ready&&<Button onClick={()=>open('approval',current.scheme_id)}>{t('openApproval')}</Button>}</div><p className="text-sm text-slate-600">{t('validationHelp')}</p></div>}
    {current&&mode==='approval'&&current.stage!=='finalized'&&<Card><h2 className="text-lg font-semibold">{t('reviewRecord')}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{t('reviewHelp')}</p>
      <fieldset disabled={task.busy} className="mt-4 space-y-3"><div className="grid gap-3 sm:grid-cols-2"><Field label={t('reviewer')}><input className={control} value={reviewer} onChange={e=>setReviewer(e.target.value)}/></Field><Field label={t('reviewDate')}><input type="date" className={control} value={reviewDate} onChange={e=>setReviewDate(e.target.value)}/></Field></div>
        <Field label={t('decisionReason')}><textarea className={control} rows={3} value={reason} onChange={e=>setReason(e.target.value)}/></Field>
        <label className="flex items-start gap-2 text-sm leading-6"><input type="checkbox" className="mt-1" checked={accept} onChange={e=>setAccept(e.target.checked)}/>{t('acceptLimits')}</label>
        <Button tone="primary" disabled={!report?.research_ready||report.validation_mode!=='frozen_candidate_validation'||!accept||reviewer.trim().length<2||reason.trim().length<5||!reviewDate||view?.current_eligibility.status==='needs_review'} onClick={finalize}>{t('finalize')}</Button>
        {!report?.research_ready&&<p role="status" className="text-sm text-amber-900">{t('mustValidate')}</p>}
      </fieldset></Card>}
    {current?.stage==='finalized'&&<p role="status" className="rounded-xl bg-slate-50 p-4 text-sm font-medium">{t('finalizedHelp')}</p>}
    {mode!=='implementation'&&report&&<ImplementationReportView report={report}/>}
    {!current&&mode!=='implementation'&&!task.busy&&(packages.length?<Card><h2 className="mb-3 text-lg font-semibold">{t('selectPackage')}</h2><div className="divide-y divide-slate-200">{packages.map(item=><div key={item.scheme_id} className="flex flex-wrap items-center justify-between gap-3 py-3"><div><Link className={linkClass} to={`${routes[mode]}?package=${encodeURIComponent(item.scheme_id)}`}>{item.name}</Link><Badge>{t(item.stage)}</Badge></div><span className="text-xs text-slate-600">{t('revision',{count:item.revision})}</span></div>)}</div></Card>:<EmptyState mascot={false} title={t('noPackages')} hint={t('noPackagesHelp')} action={<Link className={linkClass} to={routes.implementation}>{t('newPackage')}</Link>}/>)}
  </div>
}
