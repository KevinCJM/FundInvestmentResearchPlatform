import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { type StageId } from '../app/processRegistry'
import { useLocalizedStage } from '../i18n/navigation'
import { useI18n, systemText } from '../i18n/runtime'
import ActualPortfolioSelector from '../components/ActualPortfolioSelector'
import ResumeResearch, { RESUME_IDENTITY_KEYS } from '../components/ResumeResearch'
import { allocationJourneyPath, productsHandoffReady, useAllocationJourney, type AllocationJourney, type AllocationJourneyStep } from '../app/allocationJourney'

export default function StageLayout({ stageId }: { stageId: StageId }) {
  const { s } = useI18n()
  const stage = useLocalizedStage(stageId)
  const location = useLocation()
  const [journey] = useAllocationJourney()
  const isStageOverview = location.pathname.replace(/\/$/, '') === stage.path
  const isPackageWorkspace = ['/pre-investment/product-allocation-timing', '/pre-investment/portfolio-synthesis', '/pre-investment/validation', '/pre-investment/approval'].includes(location.pathname.replace(/\/$/, ''))
  // 产品研究与投前决策是两个独立模块，流程条只在投前决策自己的页面里出现，不跨模块借用。
  // 总览是流程目录，不把上次研究的书签显示为当前方案或完成进度。
  const isAllocationJourney = !isStageOverview && !isPackageWorkspace && stageId === 'pre-investment'
  const strategicFirst = Boolean(journey.strategicUniverseId)
  const saaScopeReady = strategicFirst || Boolean(journey.universeId && journey.allocationName)
  // 新研究先确认 LTCMA；已有政策自身包含冻结假设，仍允许返回历史政策。
  const saaReady = saaScopeReady && Boolean(journey.ltcmaId || journey.baselineId)
  // 步骤名和编号都取自侧栏那份注册表：同一个页面不再两处各写一份中文，也不会再出现"侧栏 02、顶部第 1 步"。
  // 落在流程节点上的步骤沿用侧栏序号；工具页在侧栏本来就没有编号，这里也不编。
  const named = (path: string) => {
    const index = stage.nodes.findIndex(node => node.path === path)
    return index < 0
      ? { order: '', label: stage.tools?.find(tool => tool.path === path)?.label ?? '' }
      : { order: String(index + 1).padStart(2, '0'), label: stage.nodes[index].label }
  }
  // 06 亮不亮看有没有交接草稿：ready 与 allocationJourneyPath 用同一个判断，避免"亮着却被送回 TAA"。
  const handedOff = productsHandoffReady(journey)
  // These links resume saved references; backend gates still verify current eligibility.
  const bookmarkSteps: { key: AllocationJourneyStep; order: string; label: string; path?: string; ready: boolean; done: boolean; current: boolean }[] = [
    { key: 'objectives', ...named('/pre-investment/objectives'), ready: true, done: Boolean(journey.mandateId), current: location.pathname.includes('/objectives') },
    { key: 'pool', ...named('/pre-investment/product-pool'), ready: Boolean(journey.mandateId), done: strategicFirst || Boolean(journey.universeId), current: location.pathname.includes('product-pool') },
    // Strategy first 的产品映射就在范围页上完成，它不是独立节点，所以不编号。
    { key: 'classes', ...(strategicFirst ? { order: '', label: s('navigation.journeyMapping') } : named('/pre-investment/saa/asset-classes')), path: strategicFirst ? allocationJourneyPath('pool', journey) : undefined, ready: strategicFirst || Boolean(journey.universeId), done: strategicFirst ? Boolean(journey.implementationMappingId) : Boolean(journey.universeId && journey.allocationName), current: /asset-classes|auto-classification/.test(location.pathname) },
    { key: 'ltcma', ...named('/pre-investment/ltcma'), ready: true, done: Boolean(journey.ltcmaId), current: location.pathname.includes('/ltcma') },
    { key: 'saa', ...named('/pre-investment/saa'), ready: saaReady, done: Boolean(saaScopeReady && journey.baselineId), current: location.pathname.endsWith('/allocation-lab') || location.pathname.endsWith('/saa/policy') },
    { key: 'taa', ...named('/pre-investment/taa'), ready: Boolean(saaScopeReady && journey.baselineId), done: Boolean(journey.baselineId && journey.taaRunId && (strategicFirst || journey.universeId)), current: location.pathname.includes('/taa') },
    { key: 'products', ...named('/pre-investment/product-allocation-timing'), ready: handedOff, done: handedOff, current: location.pathname.includes('/product-allocation-timing') },
  ]
  // 流程条与页面同源：站在流程步上而地址栏一个身份都没带，就是没在任何一轮研究里，
  // 整条按未开始渲染（01 与 LTCMA 本来就无上游依赖），续接仍由 ResumeResearch 显式发起。
  // LTCMA 的身份在路径上（/ltcma/:id），不在查询串里。
  const detached = stageId === 'pre-investment' && bookmarkSteps.some(step => step.current)
    && !RESUME_IDENTITY_KEYS.some(key => new URLSearchParams(location.search).get(key))
    && !/^\/pre-investment\/ltcma\/.+/.test(location.pathname)
  const journeySteps = detached ? bookmarkSteps.map(step => ({ ...step, ready: step.key === 'objectives' || step.key === 'ltcma', done: false })) : bookmarkSteps
  const barJourney: AllocationJourney = detached ? {} : journey
  const currentStep = journeySteps.find(step => step.current)
  // 窄屏放不下全名，缩写另给；LTCMA / SAA / TAA 两种语言写法相同，不进词表。
  const shortLabels = { objectives: s('navigation.journeyObjectivesShort'), pool: s('navigation.journeyScopeShort'), classes: strategicFirst ? s('navigation.journeyMappingShort') : s('navigation.journeyClassesShort'), ltcma: 'LTCMA', saa: 'SAA', taa: 'TAA', products: s('navigation.journeyProductsShort') }
  const stepName = (step: { order: string }, text: string) => step.order ? `${step.order} ${text}` : text
  // 未就绪的原因各不相同：02 卡在还没选目标，06 卡在 TAA 还没交接，其余才是"上一步没做完"。
  const blockedHint = (key: AllocationJourneyStep) => key === 'pool' ? s('navigation.journeyBlockedMandate')
    : key === 'saa' && saaScopeReady ? s('navigation.journeyBlockedLtcma')
    : key === 'products' ? s('navigation.journeyHandoff') : s('navigation.journeyBlocked')
  const stateHint = (step: { key: AllocationJourneyStep; done: boolean; current: boolean }) => step.current ? s('navigation.journeyCurrent')
    : step.done ? s('navigation.journeySaved') : step.key === 'products' ? s('navigation.journeyHandoff') : s('navigation.journeyContinue')
  const [isStageNavOpen, setIsStageNavOpen] = useState(false)
  const stageNavRef = useRef<HTMLElement>(null)
  const stageNavTriggerRef = useRef<HTMLButtonElement>(null)
  const previousPathRef = useRef(location.pathname)
  const navigationOptions = [
    { label: s('navigation.overview', { stage: stage.label }), path: stage.path },
    ...stage.nodes.map((node) => ({ label: node.label, path: node.path })),
    ...(stage.tools ?? []).map((tool) => ({ label: s('navigation.tool', { name: tool.label }), path: tool.path })),
  ]
  const selectedOption = navigationOptions.find((item) => location.pathname === item.path)
    ?? [...navigationOptions]
      .sort((left, right) => right.path.length - left.path.length)
      .find((item) => location.pathname.startsWith(`${item.path}/`))
    ?? navigationOptions[0]
  const isCrossPortfolioWorkspace = location.pathname === '/investment-execution/trade-allocation' || location.pathname === '/fund-accounting/account-statements'
  const isFullBleedWorkspace = location.pathname.startsWith('/settings/scenario-algorithms/workbench')
  // 组合风险预警是整页工作台：不显示真实组合选择，阶段导航默认收起、展开为浮窗。

  useEffect(() => {
    if (previousPathRef.current !== location.pathname) {
      previousPathRef.current = location.pathname
      setIsStageNavOpen(false)
    }
  }, [location.pathname])

  useEffect(() => {
    if (!isStageNavOpen) return undefined

    const handlePointerDown = (event: PointerEvent) => {
      if (event.target instanceof Node && !stageNavRef.current?.contains(event.target) && !stageNavTriggerRef.current?.contains(event.target)) {
        setIsStageNavOpen(false)
      }
    }
    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [isStageNavOpen])

  const stageContext = stage.id === 'portfolio-solutions' ? (
    <div className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm">
      <span className="font-semibold text-accent-900">{systemText('preInvestment.stageLayout.displayedVersion')}</span>
      <span className="text-accent-700">{systemText('preInvestment.stageLayout.conservativeMultiAssetPlanV32')}</span>
    </div>
  ) : isAllocationJourney ? (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
      <span className="font-semibold text-slate-800">{systemText('preInvestment.stageLayout.currentResearchPlan')}</span>
      <span className="break-words text-slate-700">{barJourney.name || (barJourney.strategicUniverseId ? systemText('preInvestment.stageLayout.strategicScopeLoadedNamePending') : barJourney.universeId ? systemText('preInvestment.stageLayout.productScopeLoadedNamePending') : systemText('preInvestment.stageLayout.noProductScopeSelected'))}</span>
      {barJourney.researchDate && <span className="mt-1 block text-xs text-slate-600">{systemText('preInvestment.stageLayout.scopeResearchDate') + " "}{barJourney.researchDate}</span>}
    </div>
  ) : stage.id === 'portfolio-center' ? (
    <div className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-900">
      <span className="font-semibold">{systemText('preInvestment.stageLayout.masterDataScope')}</span>{systemText('preInvestment.stageLayout.actualPortfoliosAccountsAccountingEntitiesAndTarget')}</div>
  ) : isCrossPortfolioWorkspace ? (
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
      <span className="font-semibold">{systemText('preInvestment.stageLayout.currentBusinessScope')}</span>{systemText('preInvestment.stageLayout.crossPortfolioAccountsAndAllocationWorkspace')}</div>
  ) : ['investment-execution', 'fund-accounting', 'post-investment', 'feedback'].includes(stage.id) ? (
    <ActualPortfolioSelector compact />
  ) : null

  const stageNavigation = (
    <div
      onKeyDown={(event) => {
        if (event.key === 'Escape' && isStageNavOpen) {
          event.stopPropagation()
          setIsStageNavOpen(false)
          stageNavTriggerRef.current?.focus()
        }
      }}
      className="relative w-full sm:w-auto xl:hidden"
    >
      <button
        ref={stageNavTriggerRef}
        type="button"
        aria-expanded={isStageNavOpen}
        aria-controls="stage-subpage-navigation"
        aria-label={s('navigation.stageAria', { stage: stage.label })}
        onClick={() => setIsStageNavOpen((current) => !current)}
        className="inline-flex min-h-11 w-full items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1 text-left shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 sm:w-auto sm:max-w-[280px]"
      >
        <span aria-hidden="true" className={`grid h-6 w-6 shrink-0 place-items-center rounded-lg ${stage.accent.soft} ${stage.accent.text}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
            <path strokeLinecap="round" d="M5 7h14M5 12h14M5 17h14" />
          </svg>
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-semibold uppercase tracking-[0.14em] text-slate-600">{s('navigation.stage')}</span>
          <span className="block truncate text-sm font-semibold text-slate-800">{selectedOption.label}</span>
        </span>
        <svg aria-hidden="true" viewBox="0 0 20 20" fill="currentColor" className={`ml-1 h-4 w-4 shrink-0 text-slate-600 transition-transform motion-reduce:transition-none ${isStageNavOpen ? 'rotate-180' : ''}`}>
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.938a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z" clipRule="evenodd" />
        </svg>
      </button>

    </div>
  )

  const stageNavigationPanel = (
        <aside ref={stageNavRef} id="stage-subpage-navigation" className={`${isStageNavOpen ? 'block' : 'hidden'} absolute inset-x-4 top-0 z-40 max-h-[70vh] overflow-y-auto rounded-xl border border-slate-200 bg-white p-3 shadow-xl sm:left-auto sm:right-6 sm:w-[340px] xl:sticky xl:inset-auto xl:top-4 xl:block xl:max-h-[calc(100dvh-96px)] xl:w-auto xl:self-start xl:shadow-none`} aria-label={s('navigation.subpages', { stage: stage.label })} onKeyDown={(event) => {
          if (event.key === 'Escape' && isStageNavOpen) {
            event.stopPropagation()
            setIsStageNavOpen(false)
            stageNavTriggerRef.current?.focus()
          }
        }}>
          <Link to={stage.path} aria-current={stageId === 'pre-investment' && isStageOverview ? 'page' : undefined} onClick={() => setIsStageNavOpen(false)} className={`block rounded-xl px-3 py-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${stageId === 'pre-investment' && isStageOverview ? 'bg-accent-50' : 'hover:bg-slate-50'}`}>
            <span className={`text-sm font-semibold ${stageId === 'pre-investment' && isStageOverview ? 'text-accent-800' : 'text-slate-900'}`}>{s('navigation.overview', { stage: stage.label })}</span>
            <span className="mt-1 block text-xs text-slate-600">{s('navigation.viewAll')}</span>
          </Link>
          <div className="my-2 border-t border-slate-100" />
          <nav className="grid gap-1">
            {stage.nodes.map((node, index) => (
              <NavLink
                key={node.id}
                to={node.path}
                onClick={() => setIsStageNavOpen(false)}
                className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-500 ${
                  isActive ? `${stage.accent.soft} font-semibold ${stage.accent.text}` : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                }`}
              >
                <span className="mr-2 text-xs tabular-nums text-slate-600">{String(index + 1).padStart(2, '0')}</span>
                {node.label}
              </NavLink>
            ))}
          </nav>
          {stage.tools?.length ? (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <p className="px-3 text-xs font-semibold uppercase tracking-wide text-slate-600">{s('navigation.tools')}</p>
              <nav className="mt-2 grid gap-1">
                {stage.tools.map((tool) => (
                  <NavLink
                    key={tool.path}
                    to={tool.path}
                    onClick={() => setIsStageNavOpen(false)}
                    className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-500 ${
                      isActive ? `${stage.accent.soft} font-semibold ${stage.accent.text}` : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                    }`}
                  >
                    {tool.label}
                  </NavLink>
                ))}
              </nav>
            </div>
          ) : null}
        </aside>
  )

  return (
    <div className="min-h-[calc(100vh-64px)] bg-slate-50">
      {!isFullBleedWorkspace ? <section className="relative z-30 border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-[1600px] px-4 py-3 sm:px-6 lg:px-8">
          <nav aria-label={s('navigation.breadcrumb')} className="flex items-center gap-2 text-xs text-slate-600">
            <Link to="/" className="rounded-lg hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-accent-500">{s('navigation.workflowHome')}</Link>
            <span aria-hidden="true">/</span>
            <span className="font-medium text-slate-800">{stage.label}</span>
          </nav>
          <div className={`flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between ${isStageOverview ? 'mt-3' : 'mt-2'}`}>
            {isStageOverview && <div className="min-w-0">
              <p className={`text-xs font-semibold uppercase tracking-[0.22em] ${stage.accent.text}`}>{stage.eyebrow}</p>
              <h1 className="mt-1 text-2xl font-bold text-slate-950 sm:text-3xl">{stage.label}</h1>
              {stageId !== 'pre-investment' && <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-600">{stage.description}</p>}
            </div>}
            <div className="flex w-full min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center xl:w-auto">
              {stageContext}
              {stageNavigation}
            </div>
          </div>
        </div>
      </section> : null}

      {isAllocationJourney && <nav aria-label={s('navigation.allocationFlow')} className="border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
        {/* 注册表全名比原来长：七列到 1024px 才排得开，全名再等到 1280px，更窄时四列 + 缩写，不让每格折成三行。 */}
        <ol className="mx-auto grid max-w-[1536px] grid-cols-4 gap-1 sm:gap-2 lg:grid-cols-7">{journeySteps.map(step => <li key={step.key} className="min-w-0">
          {step.ready || step.current ? <Link to={step.path ?? allocationJourneyPath(step.key, barJourney)} aria-current={step.current ? 'step' : undefined} aria-label={stepName(step, step.label)} className={`block rounded-lg border px-1 py-2 text-xs sm:px-3 sm:text-sm ${step.current ? 'border-accent-500 bg-accent-50 font-semibold text-accent-900' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}>
            <span className="xl:hidden">{stepName(step, shortLabels[step.key])}</span><span className="hidden xl:inline">{stepName(step, step.label)}</span><span className="mt-0.5 hidden text-xs font-normal sm:block text-slate-600">{stateHint(step)}</span>
          </Link> : <span className="block rounded-lg border border-dashed border-slate-200 px-1 py-2 text-xs text-slate-600 sm:px-3 sm:text-sm" aria-label={`${stepName(step, step.label)} · ${blockedHint(step.key)}`} title={blockedHint(step.key)}><span className="xl:hidden">{stepName(step, shortLabels[step.key])}</span><span className="hidden xl:inline">{stepName(step, step.label)}</span><span className="mt-0.5 hidden text-xs sm:block">{blockedHint(step.key)}</span></span>}
        </li>)}</ol>
      </nav>}
      <div className={isFullBleedWorkspace ? 'w-full' : 'relative mx-auto grid max-w-[1600px] gap-5 px-4 py-4 sm:px-6 lg:px-8 xl:grid-cols-[220px_minmax(0,1fr)]'}>
        {!isFullBleedWorkspace && stageNavigationPanel}
        <section className="min-w-0" aria-label={s('navigation.workspace', { stage: stage.label })}>
          {/* 侧栏进来的页面地址栏是裸的，页面就按未选择渲染；上次研究只在这里提示一次，点了才续接。 */}
          {isAllocationJourney && currentStep && <ResumeResearch to={currentStep.path ?? allocationJourneyPath(currentStep.key, journey)} />}
          <Outlet />
        </section>
      </div>
    </div>
  )
}
