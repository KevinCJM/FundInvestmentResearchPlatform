import { useEffect, useRef, useState } from 'react'
import { factorApi, parseCodes, studyDraft, type FactorDefinition, type FactorRun, type Study, type StudyDraft } from '../../services/factorResearch'
import { buttonClass, Card, Field, inputClass, secondaryClass, type Action } from './shared'

export default function StudyEditor({ seed, current, factors, action, busy, onSaved, onRun }: { seed: StudyDraft; current?: Study; factors: FactorDefinition[]; action: Action; busy: boolean; onSaved: (study: Study) => void; onRun: (run: FactorRun) => void }) {
  const [draft, setDraft] = useState(() => studyDraft(seed))
  const [targets, setTargets] = useState(seed.targets.join(', '))
  const [query, setQuery] = useState('')
  const [products, setProducts] = useState<Array<{ ts_code: string; name: string }>>([])
  const [searchError, setSearchError] = useState('')
  const form = useRef<HTMLFormElement>(null)
  useEffect(() => { const next = studyDraft(current || seed); setDraft(next); setTargets(next.targets.join(', ')) }, [current, seed])
  useEffect(() => {
    if (!query.trim()) { setProducts([]); return }
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setSearchError('')
      factorApi.products(draft.product_kind, query, controller.signal).then(value => setProducts(value.items)).catch(error => { if (!controller.signal.aborted) setSearchError(error.message) })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [query, draft.product_kind])
  const change = <K extends keyof StudyDraft>(key: K, value: StudyDraft[K]) => setDraft(previous => ({ ...previous, [key]: value }))
  const save = (run: boolean) => {
    if (!form.current?.reportValidity()) return
    void action(run ? '保存并运行研究' : '保存研究方案', async () => {
      if (!draft.factors.length) throw new Error('至少选择一个因子。')
      if ((draft.ic_min_periods || 6) > (draft.ic_window || 12)) throw new Error('滚动 IC 最少有效截面不能超过统计窗口。')
      const saved = await factorApi.saveStudy({ ...draft, targets: parseCodes(targets) }, current)
      onSaved(saved)
      if (run) onRun(await factorApi.run(saved))
      return saved
    })
  }
  return <form ref={form} onSubmit={event => { event.preventDefault(); save(false) }}>
    <fieldset disabled={busy} className="space-y-5">
      <div className="grid min-w-0 gap-5 lg:grid-cols-2">
        <Card title="1 · 研究对象与样本">
          <div className="space-y-4">
            <Field label="研究方案名称"><input className={inputClass} required maxLength={80} value={draft.name} onChange={e => change('name', e.target.value)} /></Field>
            <div className="grid grid-cols-2 gap-3"><Field label="产品类型"><select className={inputClass} value={draft.product_kind} onChange={e => { change('product_kind', e.target.value as 'etf'); setTargets('') }}><option value="etf">ETF</option><option value="fund">场外基金</option><option value="stock" disabled>股票 · 待补数据</option></select></Field><Field label="底层资产"><select className={inputClass} value={draft.asset_class} onChange={e => change('asset_class', e.target.value as StudyDraft['asset_class'])}><option value="equity">权益</option><option value="bond">债券</option><option value="commodity">商品</option><option value="multi_asset">多资产</option></select></Field></div>
            <Field label="搜索产品"><input className={inputClass} placeholder="输入代码或名称，点击加入研究池" value={query} onChange={e => setQuery(e.target.value)} /></Field>
            {searchError && <p className="text-sm text-rose-700">{searchError}</p>}
            {products.length > 0 && <div className="max-h-44 overflow-y-auto rounded-lg border border-slate-200">{products.map(product => <button type="button" className="block min-h-10 w-full px-3 py-2 text-left text-sm hover:bg-accent-50" key={product.ts_code} onClick={() => { setTargets(parseCodes(targets + ', ' + product.ts_code).join(', ')); setQuery('') }}>{product.ts_code} · {product.name}</button>)}</div>}
            <Field label="研究产品代码" hint="3–120 个完整代码，用逗号或换行分隔。固定研究池需关注样本选择偏差。"><textarea className={inputClass} rows={4} required value={targets} onChange={e => setTargets(e.target.value)} /></Field>
            <Field label="研究池来源" hint="固定研究池填 manual_fixed；池版本填 pool_version:版本ID，后端会核对已批准成员。"><input className={inputClass} required value={draft.universe_source} onChange={e => change('universe_source', e.target.value)} /></Field>
            <div className="grid gap-3 sm:grid-cols-3"><Field label="研究起始日"><input className={inputClass} required type="date" value={draft.start_date} onChange={e => change('start_date', e.target.value)} /></Field><Field label="研究截止日"><input className={inputClass} required type="date" value={draft.end_date} onChange={e => change('end_date', e.target.value)} /></Field><Field label="样本外起始日"><input className={inputClass} required type="date" value={draft.oos_date} onChange={e => change('oos_date', e.target.value)} /></Field></div>
          </div>
        </Card>
        <Card title="2 · 基准、模型与数据">
          <div className="space-y-4">
            <Field label="比较基准类型"><select className={inputClass} value={draft.benchmark.kind} onChange={e => change('benchmark', { ...draft.benchmark, kind: e.target.value as 'etf' | 'index', code: '', return_basis: e.target.value === 'etf' ? 'adjusted_nav' : 'price_index' })}><option value="etf">ETF 复权净值代理</option><option value="index">指数</option></select></Field>
            <div className="grid gap-3 sm:grid-cols-2"><Field label="基准代码"><input className={inputClass} required value={draft.benchmark.code} onChange={e => change('benchmark', { ...draft.benchmark, code: e.target.value.trim() })} /></Field><Field label="基准名称"><input className={inputClass} required value={draft.benchmark.label} onChange={e => change('benchmark', { ...draft.benchmark, label: e.target.value })} /></Field></div>
            <Field label="基准收益口径"><select className={inputClass} value={draft.benchmark.return_basis} disabled={draft.benchmark.kind === 'etf'} onChange={e => change('benchmark', { ...draft.benchmark, return_basis: e.target.value as StudyDraft['benchmark']['return_basis'] })}>{draft.benchmark.kind === 'etf' ? <option value="adjusted_nav">复权净值</option> : <><option value="price_index">价格指数（不含分红）</option><option value="total_return_index">全收益指数（需选择对应指数代码）</option></>}</select></Field>
            <Field label="因子模型"><select className={inputClass} value="characteristic_composite" disabled><option value="characteristic_composite">特征因子组合 · 横截面筛选</option></select></Field>
            <Field label="因子输入数据集"><select className={inputClass} value="active_adjusted_nav" disabled><option value="active_adjusted_nav">当前活跃快照 · 复权净值与公告日</option></select></Field>
            <p className="rounded-lg bg-accent-50 p-3 text-sm leading-6 text-accent-900">研究市场：中国；币种：人民币。比较基准用于衡量表现。FF3 与 RBSA 属于收益归因模型，请在“因子收益率 → 收益归因”中选择对应数据。</p>
            <p className="text-xs leading-5 text-slate-600">按所选日、周或月频形成信号，下一交易日净值模拟入场。缺失不补零，公告日只有日期时延后一交易日使用；研究结果保留数据修订局限。</p>
          </div>
        </Card>
      </div>
      <Card title="3 · 因子组合">
        <div className="mb-4 max-w-sm"><Field label="横截面标准化"><select className={inputClass} value={draft.normalization} onChange={e => change('normalization', e.target.value as 'rank' | 'zscore')}><option value="rank">平均秩百分位 · 组合分数 0–100</option><option value="zscore">截面去极值 Z-score</option></select></Field></div>
        <div className="grid gap-3 md:grid-cols-2">{factors.filter(f => f.product_kinds.includes(draft.product_kind)).map(factor => {
          const chosen = draft.factors.find(f => f.factor_id === factor.id)
          return <div key={factor.id} className={'flex flex-wrap items-center gap-3 rounded-lg border p-3 ' + (chosen ? 'border-accent-200 bg-accent-50/40' : 'border-slate-200')}>
            <label className="flex min-h-10 min-w-0 flex-1 items-center gap-2 text-sm"><input type="checkbox" checked={Boolean(chosen)} onChange={e => change('factors', e.target.checked ? [...draft.factors, { factor_id: factor.id, revision: factor.revision, weight: 1 }] : draft.factors.filter(f => f.factor_id !== factor.id))} /><span>{factor.name}<span className="ml-1 text-xs text-slate-600">v{chosen?.revision ?? factor.revision}</span></span></label>
            {chosen && <label className="flex items-center gap-2 text-xs text-slate-600">相对权重<input aria-label={factor.name + '权重'} type="number" step="any" min={0.0001} max={100} required className={inputClass + ' max-w-[88px]'} value={chosen.weight} onChange={e => change('factors', draft.factors.map(f => f.factor_id === factor.id ? { ...f, weight: Number(e.target.value) } : f))} /></label>}
          </div>
        })}</div>
        <p className="mt-3 text-xs text-slate-600">权重在计算时归一化。方案锁定因子修订号；单个因子缺失时该产品不获得组合得分。</p>
      </Card>
      <Card title="4 · 检验与净值模拟">
        <div className="mb-4 grid gap-4 sm:grid-cols-3"><Field label="信号与调仓频率"><select className={inputClass} value={draft.signal_frequency || 'monthly'} onChange={event => change('signal_frequency', event.target.value as StudyDraft['signal_frequency'])}><option value="monthly">月频 · 完整月末</option><option value="weekly">周频 · 完整周末</option><option value="daily">日频 · 每个交易日</option></select></Field><Field label="滚动 IC 窗口（期）" hint="对最近多少期截面 IC 做统计，不是价格窗口。"><input className={inputClass} type="number" required min={3} max={252} step={1} value={draft.ic_window ?? 12} onChange={event => change('ic_window', Number(event.target.value))} /></Field><Field label="最少有效 IC 截面"><input className={inputClass} type="number" required min={3} max={draft.ic_window ?? 12} step={1} value={draft.ic_min_periods ?? 6} onChange={event => change('ic_min_periods', Number(event.target.value))} /></Field></div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{([['horizon', '标签窗口（交易日）', 1, 126], ['quantiles', '截面分组数', 2, 10], ['top_n', '调仓 Top N', 1, 120], ['cost_bps', '单边费用（bp）', 0, 500]] as const).map(([key, label, min, max]) => <Field label={label} key={key}><input className={inputClass} type="number" required min={min} max={max} step={key === 'cost_bps' ? 'any' : 1} value={draft[key]} onChange={e => change(key, Number(e.target.value))} /></Field>)}</div>
        <p className="mt-3 text-xs leading-5 text-slate-600">多头等权，边界同分产品共同纳入，买卖均扣费；不足三个有效产品时持现金，现金收益假设为 0。净值模拟不代表实际成交。</p>
      </Card>
      <div className="flex flex-wrap items-center gap-3"><button className={secondaryClass} type="submit">保存方案</button><button className={buttonClass} type="button" onClick={() => save(true)}>保存并运行检验</button>{current && <span className="break-all text-xs text-slate-600">当前方案 v{current.revision}</span>}</div>
    </fieldset>
  </form>
}
