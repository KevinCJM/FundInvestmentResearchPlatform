import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRightIcon, ArrowPathIcon, PlayCircleIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline'
import { useI18n } from '../i18n/runtime'
import { HomeBrand, HomeHeader } from '../homepage/HomeHeader'
import { HomeDialog } from '../homepage/HomeDialog'
import { HomeIcon } from '../homepage/HomeIcon'
import Mascot from '../components/Mascot'
import { coreModules, homeModule, homeModules, quickModules, researchExamples, workflowModules, type HomeModule } from '../homepage/catalog'
import { clearRecentVisits, readRecentVisits } from '../homepage/history'
import '../homepage/homepage.css'

type Dialog = 'search' | 'tour' | 'about' | 'help' | 'workspace' | typeof researchExamples[number]['id'] | null
const Arrow = () => <ArrowRightIcon className="home-icon" aria-hidden="true" />

function ModuleCard({ module, variant, index = 0 }: { module: HomeModule; variant: 'workflow' | 'core' | 'quick'; index?: number }) {
  const { s } = useI18n()
  return <Link to={module.path} className={`home-${variant}-card home-tone-${module.tone}`}>
    <div className="home-card-top"><span className="home-feature-icon"><HomeIcon name={module.icon} /></span>{variant === 'workflow' && <span className="home-workflow-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>}</div>
    <div><h3>{s(`landing.module.${module.id}`)}</h3>{variant !== 'quick' && <p>{s(`landing.summary.${module.id}`)}</p>}</div>
    {variant !== 'core' && <span className="home-card-link">{variant === 'workflow' && s('landing.enter')}<Arrow /></span>}
  </Link>
}

function Hero({ onTour }: { onTour: () => void }) {
  const { s } = useI18n()
  return <section className="home-hero" aria-labelledby="home-title"><div className="home-container home-hero-inner">
    <picture className="home-hero-visual"><source srcSet="/homepage/images/hero-bull-800.webp 800w, /homepage/images/hero-bull-1200.webp 1200w, /homepage/images/hero-bull-1672.webp 1672w" sizes="(max-width: 700px) 100vw, 1100px" /><img src="/homepage/images/hero-bull-1200.webp" width="1200" height="675" alt="" {...{ fetchpriority: 'high' }} /></picture>
    <div className="home-hero-copy">
      <p className="home-eyebrow">DATA × RESEARCH × BETTER INVESTMENT</p>
      <h1 id="home-title"><span>{s('landing.heroLine1')}</span><span>{s('landing.heroLine2')}<em>{s('landing.heroEmphasis')}</em></span></h1>
      <p className="home-hero-description">{s('landing.heroDescription1')}<br />{s('landing.heroDescription2')}</p>
      <div className="home-hero-actions"><Link className="home-button home-button-primary" to="/product-research">{s('landing.startResearch')}<Arrow /></Link><button type="button" className="home-button home-button-secondary" onClick={onTour}><PlayCircleIcon className="home-icon" aria-hidden="true" />{s('landing.tourButton')}</button></div>
      <dl className="home-facts" aria-label={s('landing.platformStructure')}>
        <div><dt>{workflowModules.length}<span>{s('landing.unitSteps')}</span></dt><dd>{s('landing.factStages')}</dd></div>
        <div><dt>{coreModules.length}<span>{s('landing.unitTypes')}</span></dt><dd>{s('landing.factTools')}</dd></div>
        <div><dt>{s('landing.factConnectedTitle')}</dt><dd>{s('landing.factConnected')}</dd></div>
        <div><dt>{s('landing.factEvidenceTitle')}</dt><dd>{s('landing.factEvidence')}</dd></div>
      </dl>
    </div>
    <p className="home-mascot-caption">{s('landing.mascotCaption')}</p>
  </div></section>
}

function ResearchWorkflow({ onTour }: { onTour: () => void }) {
  const { s } = useI18n()
  return <>
    <section className="home-workflow-section" aria-labelledby="home-workflow-title">
      <div className="home-section-heading"><div><h2 id="home-workflow-title">{s('landing.workflowTitle')}</h2><p>{s('landing.workflowDescription')}</p></div><button type="button" className="home-text-link" onClick={onTour}>{s('landing.tourButton')}<Arrow /></button></div>
      <div className="home-workflow-grid">{workflowModules.map((module, index) => <ModuleCard key={module.id} module={module} index={index} variant="workflow" />)}</div>
      <p className="home-workflow-note"><ArrowPathIcon className="home-icon" aria-hidden="true" />{s('landing.workflowNote')}</p>
    </section>
    <section className="home-core-section" aria-labelledby="home-core-title">
      <div><div className="home-section-heading"><div><h2 id="home-core-title">{s('landing.coreTitle')}</h2><p>{s('landing.coreDescription')}</p></div></div><div className="home-core-grid">{coreModules.map(module => <ModuleCard key={module.id} module={module} variant="core" />)}</div></div>
      <figure className="home-quote"><ChatBubbleLeftRightIcon aria-hidden="true" /><blockquote>{s('landing.quoteLine1')}<br />{s('landing.quoteLine2')}</blockquote><figcaption>{s('app.title')}</figcaption></figure>
    </section>
  </>
}

function GlobalData() {
  const { s } = useI18n()
  return <section className="home-data-section" aria-labelledby="home-data-title"><div className="home-data-panel">
    <picture className="home-globe-visual"><source srcSet="/homepage/images/data-globe-800.webp 800w, /homepage/images/data-globe-1200.webp 1200w, /homepage/images/data-globe-1672.webp 1672w" sizes="(max-width: 700px) 100vw, 1280px" /><img src="/homepage/images/data-globe-1200.webp" width="1200" height="675" alt="" loading="lazy" /></picture>
    <div className="home-data-copy"><p className="home-eyebrow">FROM DATA TO INSIGHT</p><h2 id="home-data-title">{s('landing.dataTitle1')}<br />{s('landing.dataTitle2')}</h2><p className="home-data-description">{s('landing.dataDescription1')}<br />{s('landing.dataDescription2')}</p><Link to="/settings/source-center" className="home-button home-button-blue">{s('landing.exploreData')}<Arrow /></Link>
      <div className="home-data-categories">{['Fund', 'Index', 'Macro'].map(kind => <div key={kind}><strong>{s(`landing.data${kind}`)}</strong><span>{s(`landing.data${kind}Sub`)}</span></div>)}</div>
    </div>
    <div className="home-data-callouts">
      <Link to="/settings/source-center" className="home-data-callout home-callout-global"><HomeIcon name="globe" /><span><strong>Global Data</strong><small>{s('landing.calloutGlobal')}</small></span></Link>
      <Link to="/settings/scenario-algorithms" className="home-data-callout home-callout-alternative"><HomeIcon name="network" /><span><strong>Different Perspectives</strong><small>{s('landing.calloutAlternative')}</small></span></Link>
      <Link to="/settings/research-data-lab" className="home-data-callout home-callout-alpha"><HomeIcon name="chart" /><span><strong>From Data to Alpha</strong><small>{s('landing.calloutAlpha')}</small></span></Link>
    </div>
  </div></section>
}

function QuickAccess({ onExample }: { onExample: (id: Dialog) => void }) {
  const { s } = useI18n()
  const [tab, setTab] = useState('examples')
  const [recent, setRecent] = useState(readRecentVisits)
  const [storageError, setStorageError] = useState(false)
  return <section className="home-work-section" aria-labelledby="home-quick-title">
    <div><div className="home-section-heading"><div><h2 id="home-quick-title">{s('landing.quickTitle')}</h2><p>{s('landing.quickDescription')}</p></div></div><div className="home-quick-grid">{quickModules.map(module => <ModuleCard key={module.id} module={module} variant="quick" />)}</div></div>
    <div className="home-research-lists">
      <div role="tablist" aria-label={s('landing.researchLists')} className="home-list-tabs">{['examples', 'recent'].map(id => <button type="button" role="tab" key={id} id={`home-tab-${id}`} aria-selected={tab === id} aria-controls={`home-panel-${id}`} tabIndex={tab === id ? 0 : -1} onClick={() => setTab(id)} onKeyDown={event => {
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); const next = event.key === 'Home' ? 'examples' : event.key === 'End' ? 'recent' : id === 'examples' ? 'recent' : 'examples'; setTab(next); document.getElementById(`home-tab-${next}`)?.focus() }
      }}>{s(id === 'examples' ? 'landing.researchExamples' : 'landing.recentlyOpened')}</button>)}</div>
      <div role="tabpanel" id={`home-panel-${tab}`} aria-labelledby={`home-tab-${tab}`} tabIndex={0}>
        {tab === 'examples' ? researchExamples.map(example => <button type="button" className={`home-research-item home-tone-${homeModule(example.module).tone}`} key={example.id} onClick={() => onExample(example.id)}><span className="home-feature-icon"><HomeIcon name={homeModule(example.module).icon} /></span><span>{s(`landing.example.${example.id}`)}</span><small>{s('landing.example')}</small></button>) : <>
          {recent.length ? <>{recent.map(id => <Link key={id} to={homeModule(id).path} className="home-research-item"><HomeIcon name={homeModule(id).icon} /><span>{s(`landing.module.${id}`)}</span><Arrow /></Link>)}<button type="button" className="home-text-link" onClick={() => { const ok = clearRecentVisits(); setStorageError(!ok); if (ok) setRecent([]) }}>{s('landing.clearHistory')}</button></> : <p className="home-empty">{s('landing.noHistory')}<small>{s('landing.historyHelp')}</small></p>}
          {storageError && <p role="alert">{s('landing.storageFailed')}</p>}
        </>}
      </div>
    </div>
  </section>
}

function SearchContent({ close }: { close: () => void }) {
  const { s } = useI18n()
  const [query, setQuery] = useState('')
  const matches = homeModules.filter(item => `${s(`landing.module.${item.id}`)} ${s(`landing.summary.${item.id}`)} ${item.id}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))
  return <><label className="home-search-label">{s('landing.searchInputLabel')}<input data-home-autofocus maxLength={100} value={query} onChange={event => setQuery(event.target.value)} placeholder={s('landing.searchPlaceholder')} type="search" /></label><p className="home-dialog-note">{s('landing.searchScope')}</p><div className="home-search-results">{matches.map(item => <Link key={item.id} to={item.path} onClick={close}><HomeIcon name={item.icon} /><span><strong>{s(`landing.module.${item.id}`)}</strong><small>{s(`landing.summary.${item.id}`)}</small></span><Arrow /></Link>)}{!matches.length && <div className="home-empty-state"><Mascot state="noresult" /><p role="status" className="home-empty">{s('landing.noSearch')}<small>{s('landing.noSearchHelp')}</small></p></div>}</div></>
}

function TourContent({ close }: { close: () => void }) {
  const { s } = useI18n()
  const [step, setStep] = useState(0)
  const ids = [['data', 'metrics', 'scenarios', 'products'], ['products', 'decision', 'solutions'], ['portfolio', 'execution', 'accounting', 'post', 'feedback']][step]
  return <><p className="home-dialog-note">{step + 1} / 3</p><div className="home-tour-step" aria-live="polite"><h3>{s(`landing.tour.${step}.title`)}</h3><p>{s(`landing.tour.${step}.body`)}</p></div><div className="home-tour-links">{ids.map(id => <Link key={id} to={homeModule(id).path} onClick={close}><HomeIcon name={homeModule(id).icon} />{s(`landing.module.${id}`)}<Arrow /></Link>)}</div><div className="home-dialog-actions"><button type="button" className="home-button home-button-secondary" disabled={step === 0} onClick={() => setStep(step - 1)}>{s('landing.tourPrev')}</button><button type="button" className="home-button home-button-primary" onClick={() => step < 2 ? setStep(step + 1) : close()}>{s(step < 2 ? 'landing.tourNext' : 'landing.tourDone')}</button></div></>
}

function LandingDialog({ dialog, close }: { dialog: Exclude<Dialog, null>; close: () => void }) {
  const { s } = useI18n()
  const example = researchExamples.find(item => item.id === dialog)
  const title = example ? s(`landing.example.${example.id}`) : s(`landing.${dialog}Title`)
  return <HomeDialog title={title} onClose={close}>
    {dialog === 'search' ? <SearchContent close={close} /> : dialog === 'tour' ? <TourContent close={close} /> : example ? <>
      <p className="home-example-notice">{s('landing.exampleNotice')}</p><p>{s(`landing.exampleBody.${example.id}`)}</p><Link className="home-button home-button-blue" to={example.path} onClick={close}>{s('landing.openWorkspace')}<Arrow /></Link>
    </> : <><p>{s(`landing.${dialog}Body`)}</p><p className="home-dialog-note">{s(`landing.${dialog}Note`)}</p>{dialog === 'workspace' && <Link to="/settings" className="home-button home-button-blue" onClick={close}>{s('landing.settings')}<Arrow /></Link>}</>}
  </HomeDialog>
}

export default function ProcessHome() {
  const { s } = useI18n()
  const [dialog, setDialog] = useState<Dialog>(null)
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); setDialog('search') }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  }, [])
  return <div className="quant-homepage" data-testid="quant-homepage">
    <HomeHeader onSearch={() => setDialog('search')} onWorkspace={() => setDialog('workspace')} />
    <div id="home-content" tabIndex={-1}><Hero onTour={() => setDialog('tour')} /><div className="home-container"><ResearchWorkflow onTour={() => setDialog('tour')} /><GlobalData /><QuickAccess onExample={setDialog} /></div></div>
    <footer className="home-footer"><div className="home-container home-footer-inner"><HomeBrand footer /><nav aria-label={s('landing.footerNav')}><button type="button" onClick={() => setDialog('about')}>{s('landing.about')}</button><button type="button" onClick={() => setDialog('help')}>{s('landing.help')}</button><Link to="/settings">{s('landing.settings')}</Link></nav><span>{s('landing.footerStatus')}</span><time>{new Date().getFullYear()}</time></div></footer>
    {dialog && <LandingDialog key={dialog} dialog={dialog} close={() => setDialog(null)} />}
  </div>
}
