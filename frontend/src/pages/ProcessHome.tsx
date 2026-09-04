import { Link } from 'react-router-dom'
import { ArrowLongLeftIcon } from '@heroicons/react/24/outline'
import { businessStages, getStage } from '../app/processRegistry'

export default function ProcessHome() {
  const settings = getStage('settings')
  const portfolioCenter = getStage('portfolio-center')
  const portfolioSolutions = getStage('portfolio-solutions')
  const accounting = getStage('fund-accounting')

  return (
    <div className="min-h-[calc(100vh-64px)] bg-slate-950 text-white">
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="absolute -left-24 top-8 h-72 w-72 rounded-full bg-sky-500/20 blur-3xl" />
        <div className="absolute right-0 top-16 h-80 w-80 rounded-full bg-violet-500/20 blur-3xl" />
        <div className="relative mx-auto max-w-[1600px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-sky-300">Fund investment research platform</p>
          <h1 className="mt-4 max-w-4xl text-3xl font-bold leading-tight sm:text-5xl">公募基金量化投研流程</h1>
          <p className="mt-4 max-w-5xl text-sm leading-7 text-slate-300 sm:text-base">产品研究持续沉淀可复用产品池；投前形成研究目标组合，外部批准后在组合中心登记为真实组合；投中、基金会计和投后共享同一真实组合身份与业务事实。</p>
        </div>
      </section>

      <main className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
        <section aria-labelledby="main-process-title">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Single portfolio lifecycle</p>
              <h2 id="main-process-title" className="mt-1 text-2xl font-bold">投研主流程</h2>
            </div>
            <p className="text-sm text-slate-400">点击节点进入对应业务页面</p>
          </div>

          <ol className="mt-7 grid gap-3 xl:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] xl:items-stretch">
            {businessStages.map((stage, index) => (
              <li key={stage.id} className="contents">
                <Link to={stage.path} className="group flex min-h-64 flex-col rounded-2xl border border-white/10 bg-white/[0.06] p-5 transition hover:-translate-y-1 hover:border-white/25 hover:bg-white/[0.1] focus:outline-none focus:ring-2 focus:ring-sky-400">
                  <span className="text-3xl font-bold tabular-nums text-white/20">{String(stage.order).padStart(2, '0')}</span>
                  <p className="mt-6 text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">{stage.eyebrow}</p>
                  <h3 className="mt-2 text-xl font-bold text-white group-hover:text-sky-200">{stage.label}</h3>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{stage.description}</p>
                  <span className="mt-auto pt-5 text-sm font-semibold text-sky-300">进入{stage.label} →</span>
                </Link>
                {index < businessStages.length - 1 ? (
                  <div className="hidden items-center text-2xl text-white/30 xl:flex" aria-hidden="true">→</div>
                ) : null}
              </li>
            ))}
          </ol>

          <div className="mt-4 flex items-center gap-3 px-3 text-rose-200" aria-label="反馈与迭代回流至产品研究">
            <ArrowLongLeftIcon className="h-5 w-5 shrink-0" aria-hidden="true" />
            <div className="h-px flex-1 bg-rose-300/35" aria-hidden="true" />
            <span className="shrink-0 text-xs font-semibold tracking-wide">反馈与迭代回流至产品研究</span>
          </div>

          <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-300/[0.06] px-5 py-4 text-sm leading-6 text-rose-100">
            <span className="font-semibold">反馈闭环：</span>投资结果回流至产品复审、产品池版本、配置模型和流程规则，但不会要求每个组合重新开展完整产品研究。
          </div>

          <Link to={portfolioCenter.path} className="group mt-5 block rounded-2xl border border-indigo-300/25 bg-gradient-to-r from-indigo-950/75 via-slate-900 to-blue-950/55 p-5 transition hover:border-indigo-300/55 focus:outline-none focus:ring-2 focus:ring-indigo-300 sm:p-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-300">Actual portfolio registry · 投中至投后</p>
                <h3 className="mt-2 text-2xl font-bold">组合中心 · 真实组合库</h3>
                <p className="mt-2 max-w-5xl text-sm leading-6 text-slate-300">承接外部已批准的研究方案，统一登记真实组合、账户、核算主体、会计账簿与目标版本关系。这里创建的是平台主数据，不代表法律设立、开户或下单。</p>
              </div>
              <span className="shrink-0 text-sm font-semibold text-indigo-300 group-hover:text-indigo-200">进入组合中心 →</span>
            </div>
            <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-indigo-100/80">覆盖范围：真实组合登记 · 组合主数据 · 账户与账簿关系 · 研究方案及目标版本 · 生命周期状态</p>
          </Link>

          <Link to={accounting.path} className="group mt-5 block rounded-2xl border border-cyan-300/25 bg-gradient-to-r from-cyan-950/70 via-slate-900 to-emerald-950/50 p-5 transition hover:border-cyan-300/55 focus:outline-none focus:ring-2 focus:ring-cyan-300 sm:p-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-300">Parallel accounting lifecycle · 投中至投后</p>
                <h3 className="mt-2 text-2xl font-bold">基金会计与管理人账务</h3>
                <p className="mt-2 max-w-5xl text-sm leading-6 text-slate-300">Booking 是已发生业务事实入口，不等于会计凭证。规则按核算主体生成各自借贷平衡的凭证，经过账、估值、核对和关账形成 ABOR 与绩效数据。</p>
              </div>
              <span className="shrink-0 text-sm font-semibold text-cyan-300 group-hover:text-cyan-200">进入会计核算 →</span>
            </div>
            <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-cyan-100/80">核算链路：业务 Booking → 会计确认与分类 → 复式凭证 → 总账/明细账 → 估值与关账 → 报表与 PBOR</p>
          </Link>
        </section>

        <section className="mt-10 border-t border-white/10 pt-8" aria-labelledby="extended-capabilities-title">
          <div className="mb-5">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Beyond the lifecycle</p>
            <h2 id="extended-capabilities-title" className="mt-1 text-xl font-bold">研究成果应用与平台支撑</h2>
            <p className="mt-2 text-sm text-slate-400">以下能力独立于单组合生命周期：一个负责展示研究成果，一个负责提供平台公共能力。</p>
          </div>
          <div className="space-y-4" data-testid="extended-capabilities-stack">
            <Link to={portfolioSolutions.path} className="group block rounded-2xl border border-fuchsia-300/20 bg-gradient-to-r from-fuchsia-950/60 to-slate-900 p-6 transition hover:border-fuchsia-300/50 focus:outline-none focus:ring-2 focus:ring-fuchsia-300">
              <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-center">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-fuchsia-300">Portfolio solutions</p>
                  <h3 className="mt-2 text-2xl font-bold">组合方案展示中心</h3>
                  <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">将已定稿组合转换为标准化展示版本，统一呈现组合画像、实盘与回测、周期情景、算法规则和风险披露。</p>
                </div>
                <span className="text-sm font-semibold text-fuchsia-300 group-hover:text-fuchsia-200">查看组合方案 →</span>
              </div>
              <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-fuchsia-100/80">展示范围：方案目录 · 组合画像 · 收益与风险 · 回测与情景 · 披露版本</p>
            </Link>
            <Link to={settings.path} className="group block rounded-2xl border border-white/10 bg-gradient-to-r from-slate-900 to-slate-800 p-6 transition hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400">
              <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-center">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Platform foundation</p>
                  <h3 className="mt-2 text-2xl font-bold">设置 · 公共能力</h3>
                  <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">数据源与快照、指标与模型、情景算法、统一回测和系统参数在这里集中管理，为五个业务阶段提供一致能力。</p>
                </div>
                <span className="text-sm font-semibold text-sky-300 group-hover:text-sky-200">进入设置 →</span>
              </div>
              <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-slate-300">公共能力：数据与 PIT · 指标与模型 · 情景算法 · 统一回测 · 系统参数</p>
            </Link>
          </div>
        </section>
      </main>
    </div>
  )
}
