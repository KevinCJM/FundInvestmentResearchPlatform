import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  getScenarioPreview, previewScenario, publishScenario, retireScenario, riskCatalog, riskReleases, scenarioReleases,
  type RiskCatalog, type RiskRelease, type RiskVariable, type ScenarioDraft, type ScenarioPreview, type ScenarioRelease,
} from '../services/riskModels'
import ResearchWorkbench from '../components/risk-models/ResearchWorkbench'
import { buttonClass, Empty, Feedback, Field, frequencyLabels, inputClass, numberText, primaryClass, sectionClass, statusLabels, Steps, NumberInput } from '../components/risk-models/ResearchUI'
import ScenarioSimulationCenter from './ScenarioAlgorithmCenter'

const blank = (): ScenarioDraft => ({ name: '', description: '', entry: 'market', frequency: 'monthly', event_template: 'custom', event_model_release_id: null, macro_model_release_id: null, input_ids: [], rows: [[]], shock_basis: 'period_change' })
const entryLabels = { event: '宏观事件', macro: '宏观变量', market: '市场风险因子' }

export function ScenarioPathTable({ variables, rows, title }: { variables: RiskVariable[]; rows: number[][]; title: string }) {
  const [page, setPage] = useState(0)
  useEffect(() => setPage(0), [rows])
  const offset = page * 24
  return <div className="min-w-0"><div className="max-h-80 overflow-auto"><table className="w-full min-w-[400px] text-sm"><caption className="mb-2 text-left font-medium">{title}（每期变动）</caption><thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="p-2 text-left">期数</th>{variables.map(variable => <th scope="col" key={variable.id} className="p-2 text-right">{variable.name}<br /><span className="font-normal">{variable.unit_label}</span></th>)}</tr></thead><tbody className="divide-y divide-slate-100">{rows.slice(offset, offset + 24).map((row, index) => <tr key={offset + index}><td className="p-2">{offset + index + 1}</td>{row.map((value, column) => <td className="p-2 text-right tabular-nums" key={variables[column].id}>{numberText(variables[column].unit === 'return' ? value * 100 : value, 4)}</td>)}</tr>)}</tbody></table></div>{rows.length > 24 && <div className="mt-2 flex items-center justify-between gap-2 text-xs"><button type="button" className={buttonClass} disabled={page === 0} onClick={() => setPage(value => value - 1)}>上一页</button><span>{offset + 1}—{Math.min(offset + 24, rows.length)} / {rows.length} 期</span><button type="button" className={buttonClass} disabled={offset + 24 >= rows.length} onClick={() => setPage(value => value + 1)}>下一页</button></div>}</div>
}

export default function PublishedScenarioCenter() {
  const [pane, setPane] = useState<'library' | 'build' | 'transmission' | 'advanced'>('library')
  const [visited, setVisited] = useState({ transmission: false, advanced: false })
  const [catalog, setCatalog] = useState<RiskCatalog | null>(null)
  const [models, setModels] = useState<RiskRelease[]>([])
  const [releases, setReleases] = useState<ScenarioRelease[]>([])
  const [draft, setDraft] = useState<ScenarioDraft>(blank)
  const [step, setStep] = useState(0)
  const [preview, setPreview] = useState<ScenarioPreview | null>(null)
  const [signature, setSignature] = useState('')
  const [publishedId, setPublishedId] = useState('')
  const [acknowledged, setAcknowledged] = useState(false)
  const [validDays, setValidDays] = useState(90)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true); const operation = useRef(0)
  useEffect(() => { alive.current = true; return () => { alive.current = false; operation.current++ } }, [])
  useEffect(() => {
    const controller = new AbortController(); setLoading(true)
    Promise.all([riskCatalog('transmission', controller.signal), riskReleases('transmission', {}, controller.signal), scenarioReleases(undefined, controller.signal)])
      .then(([nextCatalog, nextModels, nextReleases]) => { if (!controller.signal.aborted) { setCatalog(nextCatalog); setModels(nextModels); setReleases(nextReleases) } })
      .catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '情景目录加载失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [refresh])
  const activeModels = models.filter(item => item.status === 'active')
  const eventModel = models.find(item => item.id === draft.event_model_release_id)
  const macroModel = models.find(item => item.id === draft.macro_model_release_id)
  const inputs = draft.entry === 'market' ? (catalog?.variables.filter(item => draft.input_ids.includes(item.id)).sort((a, b) => draft.input_ids.indexOf(a.id) - draft.input_ids.indexOf(b.id)) ?? []) : draft.entry === 'event' ? eventModel?.inputs ?? [] : macroModel?.inputs ?? []
  const stale = signature !== JSON.stringify(draft)
  const validNumbers = draft.rows.every(row => row.length === inputs.length && row.every(Number.isFinite))
  const canConfigure = inputs.length > 0 && (draft.entry === 'market' || Boolean(macroModel && (draft.entry === 'macro' || eventModel)))
  function switchPane(next: typeof pane) { if (next === 'transmission' || next === 'advanced') setVisited(old => ({ ...old, [next]: true })); setPane(next); if (next === 'library' || next === 'build') setRefresh(old => old + 1) }
  function patch(value: Partial<ScenarioDraft>) { setDraft(old => ({ ...old, ...value })); setAcknowledged(false); setPublishedId(''); setError(''); setNotice('') }
  function newScenario() { operation.current++; setDraft(blank()); setPreview(null); setSignature(''); setStep(0); setPublishedId(''); setAcknowledged(false); setError(''); setNotice(''); setPane('build') }
  function chooseModel(id: string, stage: 'event' | 'macro') {
    const selected = models.find(item => item.id === id)
    if (stage === 'event') patch({ event_model_release_id: id || null, macro_model_release_id: null, frequency: (selected?.frequency as ScenarioDraft['frequency']) ?? draft.frequency, rows: draft.rows.map(() => selected?.inputs.map(() => 0) ?? []) })
    else patch({ macro_model_release_id: id || null, frequency: (selected?.frequency as ScenarioDraft['frequency']) ?? draft.frequency, ...(draft.entry === 'macro' ? { rows: draft.rows.map(() => selected?.inputs.map(() => 0) ?? []) } : {}) })
  }
  function selectFactor(id: string, checked: boolean) {
    const ids = checked ? [...draft.input_ids, id] : draft.input_ids.filter(key => key !== id)
    patch({ input_ids: ids, rows: draft.rows.map(row => ids.map(key => { const index = draft.input_ids.indexOf(key); return index >= 0 ? row[index] : 0 })) })
  }
  async function calculate() {
    const token = ++operation.current; setBusy('preview'); setError(''); setNotice(''); setAcknowledged(false); setPublishedId('')
    try {
      if (!draft.name.trim() || !canConfigure || !validNumbers) throw new Error('请填写情景名称，并补齐入口所需的因子或已发布传导模型。')
      const value = await previewScenario(draft)
      if (alive.current && token === operation.current) { setPreview(value); setSignature(JSON.stringify(draft)); setStep(2); setNotice('已生成冲击路径。当前预览不会写入磁盘；确认发布后才保存并供产品和组合使用。') }
    } catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '预览失败。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function publish() {
    if (!preview || stale || !preview.preview_hash || !acknowledged) return
    const token = ++operation.current; setBusy('publish'); setError('')
    try { const release = await publishScenario(draft, preview.preview_hash, validDays, note); if (alive.current && token === operation.current) { setPublishedId(release.id); setNotice('情景已确认发布并写入磁盘。产品和组合可以直接选择，不会重新运行宏观模型。'); setRefresh(old => old + 1) } }
    catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '发布失败。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function view(release: ScenarioRelease) {
    const token = ++operation.current; setBusy('read'); setError('')
    try { const value = await getScenarioPreview(release.preview_id); if (alive.current && token === operation.current) { setDraft(value.definition); setPreview(value); setSignature(JSON.stringify(value.definition)); setPublishedId(release.id); setStep(2); setPane('build') } }
    catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '读取失败。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function retire(release: ScenarioRelease) {
    if (!window.confirm(`确认停用「${release.name}」？历史压测结果会继续保留。`)) return
    setBusy('retire'); setError('')
    try { await retireScenario(release.id, '研究者在情景库中确认停用。'); if (alive.current) { setRefresh(old => old + 1); setNotice('情景已停用，历史结果保留。') } }
    catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : '停用失败。') }
    finally { if (alive.current) setBusy('') }
  }
  return <div className="min-w-0 space-y-5" data-testid="published-scenario-center">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-xl font-semibold text-slate-950">情景模拟与压测</h2><p className="mt-2 text-sm text-slate-600">从事件、经济变量或市场冲击开始，生成可复用情景，再应用于产品和组合。</p></div><button type="button" className={primaryClass} disabled={Boolean(busy)} onClick={newScenario}>新建情景</button></div>
    <nav aria-label="情景工作区" className="flex flex-wrap gap-2 border-b border-slate-200 pb-3">{([['library', '已发布情景'], ['build', '构建情景'], ['transmission', '宏观传导研究'], ['advanced', '高级计算实验']] as const).map(([id, label]) => <button type="button" key={id} disabled={Boolean(busy)} aria-current={pane === id ? 'page' : undefined} className={pane === id ? primaryClass : buttonClass} onClick={() => switchPane(id)}>{label}</button>)}</nav>
    <Feedback error={error} notice={notice} />
    {(pane === 'library' || pane === 'build') && <>{loading && !catalog ? <p role="status" className={sectionClass}>正在读取本地情景目录…</p> : !catalog ? <Empty title="情景目录暂不可用"><p>请检查后端和数据磁盘。</p><button type="button" className={`${buttonClass} mt-3`} onClick={() => setRefresh(old => old + 1)}>重新加载</button></Empty> : <>
      {pane === 'library' && (!releases.length ? <Empty title="还没有已发布情景"><p>最快的方式：新建情景 → 选择市场风险因子 → 填写变化 → 预览并发布。需要从经济事件开始时，先研究并发布宏观传导模型。</p><button type="button" className={`${primaryClass} mt-4`} onClick={newScenario}>从市场冲击开始</button></Empty> : <section className={sectionClass}><div className="overflow-auto"><table className="w-full min-w-[640px] text-sm"><caption className="mb-3 text-left font-semibold">可复用的情景版本</caption><thead className="text-left text-xs text-slate-600"><tr><th scope="col" className="p-2">情景</th><th scope="col" className="p-2">入口 / 期限</th><th scope="col" className="p-2">状态</th><th scope="col" className="p-2">操作</th></tr></thead><tbody className="divide-y divide-slate-100">{releases.map(release => <tr key={release.id}><td className="p-3 font-medium">{release.name}<p className="mt-1 text-xs font-normal text-slate-600">发布：{release.created_at.slice(0, 10)}</p></td><td className="p-3">{entryLabels[release.entry]}<p className="mt-1 text-xs text-slate-600">{frequencyLabels[release.frequency]} · {release.horizon} 期</p></td><td className="p-3">{statusLabels[release.status] ?? release.status}<p className="mt-1 text-xs text-slate-600">{release.reason ?? `到期：${release.expires_at.slice(0, 10)}`}</p></td><td className="p-3"><div className="flex flex-wrap gap-2"><button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={() => void view(release)}>查看路径</button>{release.status === 'active' && <><Link className={buttonClass} to={`/settings/scenario-algorithms/apply?${new URLSearchParams({ scenario_release_id: release.id })}`}>应用到产品或组合</Link><button type="button" className="min-h-11 px-2 text-xs text-rose-700" disabled={Boolean(busy)} onClick={() => void retire(release)}>停用</button></>}</div></td></tr>)}</tbody></table></div></section>)}
      {pane === 'build' && <div className="space-y-4"><Steps labels={['从哪里开始', '设置变化', '预览并发布']} active={step} onChange={setStep} disabled={Boolean(busy)} />
        {step === 0 && <section className={`${sectionClass} space-y-5`}><fieldset disabled={Boolean(busy)} className="min-w-0 space-y-5"><Field label="情景名称"><input className={inputClass} value={draft.name} placeholder="为这次假设起一个清楚的名字" onChange={event => patch({ name: event.target.value })} /></Field><fieldset><legend className="mb-3 text-sm font-semibold">你想从什么问题开始？</legend><div className="grid gap-3 lg:grid-cols-3">{([['event', '发生一件大事', '例如能源成本冲击。需要两段已发布的传导模型。'], ['macro', '经济指标变化', '例如通胀变化。需要宏观变量到市场的传导模型。'], ['market', '市场直接变化', '例如股市下跌或利率上升。不需要先编宏观故事。']] as const).map(([id, label, description]) => <label key={id} className={`flex cursor-pointer items-start gap-3 rounded-lg border p-4 ${draft.entry === id ? 'border-accent-600 bg-accent-50' : 'border-slate-200 hover:border-slate-400'}`}><input type="radio" name="scenario-entry" className="mt-1" value={id} checked={draft.entry === id} onChange={() => patch({ entry: id, event_model_release_id: null, macro_model_release_id: null, input_ids: [], rows: draft.rows.map(() => []) })} /><span><span className="block text-sm font-semibold">{label}</span><span className="mt-2 block text-xs leading-5 text-slate-600">{description}</span></span></label>)}</div></fieldset><Field label="假设说明（可选）"><textarea className={inputClass} rows={2} value={draft.description} onChange={event => patch({ description: event.target.value })} placeholder="说明发生了什么、哪些条件保持不变。" /></Field></fieldset><button type="button" className={primaryClass} disabled={!draft.name.trim()} onClick={() => setStep(1)}>下一步：设置变化</button></section>}
        {step === 1 && <section className={`${sectionClass} space-y-5`}><fieldset disabled={Boolean(busy)} className="min-w-0 space-y-5">
          {draft.entry === 'event' && <><Field label="事件叙事模板" hint="模板只是事件描述，不内置 GDP 或股市跌幅。请确认所选模型的驱动与事件含义一致。"><select className={inputClass} value={draft.event_template} onChange={event => patch({ event_template: event.target.value as ScenarioDraft['event_template'] })}>{catalog.event_templates.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></Field><Field label="第一段：事件驱动 → 宏观变量"><select className={inputClass} value={draft.event_model_release_id ?? ''} onChange={event => chooseModel(event.target.value, 'event')}><option value="">请选择已发布的事件传导模型</option>{activeModels.filter(item => item.stage === 'event_macro').map(item => <option key={item.id} value={item.id}>{item.name} · {frequencyLabels[item.frequency]} · {item.as_of}</option>)}</select></Field></>}
          {draft.entry !== 'market' && <Field label={draft.entry === 'event' ? '第二段：宏观变量 → 市场风险因子' : '宏观变量 → 市场风险因子'}><select className={inputClass} value={draft.macro_model_release_id ?? ''} onChange={event => chooseModel(event.target.value, 'macro')}><option value="">请选择已发布的宏观市场模型</option>{activeModels.filter(item => item.stage === 'macro_market').map(item => { const compatible = draft.entry !== 'event' || Boolean(eventModel && item.frequency === eventModel.frequency && item.inputs.every(input => eventModel.outputs.some(output => output.id === input.id && output.contract_hash === input.contract_hash))); return <option key={item.id} value={item.id} disabled={!compatible}>{item.name} · {frequencyLabels[item.frequency]}{compatible ? '' : '（输入或频率不匹配）'}</option> })}</select></Field>}
          {draft.entry !== 'market' && !canConfigure && <Empty title="先补齐这条传导链"><p>{draft.entry === 'event' ? '需要一份事件→宏观成果，以及能接收其输出的宏观→市场成果。没有可用模型时，不会用固定数字替代。' : '需要先发布一份宏观变量→市场风险因子的研究成果。'}</p><button type="button" className={`${buttonClass} mt-3`} onClick={() => switchPane('transmission')}>去研究传导模型</button></Empty>}
          {draft.entry === 'market' && <fieldset><legend className="text-sm font-semibold">选择要改变的市场风险因子</legend><div className="mt-3 grid max-h-64 gap-2 overflow-y-auto sm:grid-cols-2">{catalog.variables.filter(variable => variable.roles.includes('market')).map(variable => <label className="flex items-start gap-2 rounded-lg border border-slate-200 p-3 text-sm" key={variable.id}><input className="mt-1" type="checkbox" checked={draft.input_ids.includes(variable.id)} disabled={!draft.input_ids.includes(variable.id) && draft.input_ids.length >= 8} onChange={event => selectFactor(variable.id, event.target.checked)} /><span>{variable.name}<span className="mt-1 block text-xs text-slate-600">输入单位：{variable.unit_label}</span></span></label>)}</div><p className="mt-2 text-xs text-slate-600">直接冲击不要求先拟合宏观模型；应用时仍需产品已发布的相容暴露。</p></fieldset>}
          <div className="grid gap-4 sm:grid-cols-2"><Field label="每期代表什么"><select className={inputClass} value={draft.frequency} disabled={draft.entry !== 'market'} onChange={event => patch({ frequency: event.target.value as ScenarioDraft['frequency'] })}>{catalog.frequencies.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></Field><Field label="设置多少期" hint="每格是这一期的变化，不是相对起点的累计变化。"><input className={inputClass} type="number" min="1" max="1200" value={draft.rows.length} onChange={event => { const count = Math.max(1, Math.min(1200, Math.floor(Number(event.target.value)) || 1)); patch({ rows: Array.from({ length: count }, (_, index) => draft.rows[index] ?? inputs.map(() => 0)) }) }} /></Field></div>
          {inputs.length > 0 && <div className="max-h-[420px] overflow-auto rounded-lg border border-slate-200"><table className="w-full min-w-[400px] text-sm"><caption className="p-3 text-left font-medium">填写每期变动；0 表示该期不变</caption><thead className="sticky top-0 bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="p-2 text-left">期数</th>{inputs.map(input => <th scope="col" key={input.id} className="min-w-32 p-2 text-right">{input.name}<br /><span className="font-normal">{input.unit_label}</span></th>)}</tr></thead><tbody>{draft.rows.map((row, index) => <tr key={index}><td className="p-2">{index + 1}</td>{inputs.map((input, column) => <td key={input.id} className="p-1"><NumberInput className={`${inputClass} text-right`} aria-label={`第${index + 1}期${input.name}变化`} value={row[column] ?? 0} onValueChange={number => patch({ rows: draft.rows.map((values, position) => position === index ? inputs.map((_, field) => field === column ? number : values[field] ?? 0) : values) })} /></td>)}</tr>)}</tbody></table></div>}
        </fieldset><div className="flex flex-wrap gap-3"><button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={() => setStep(0)}>上一步</button><button type="button" className={primaryClass} disabled={Boolean(busy) || !canConfigure || !validNumbers} onClick={() => void calculate()}>{busy === 'preview' ? '正在计算传导路径…' : '预览冲击路径'}</button></div></section>}
        {step === 2 && (!preview ? <Empty title="先生成情景路径"><p>在「设置变化」中填写冲击，再预览实际计算结果。</p><button type="button" className={`${primaryClass} mt-3`} onClick={() => setStep(1)}>去设置变化</button></Empty> : <><section className={`${sectionClass} space-y-4`}><div><h3 className="font-semibold">{preview.name} · 市场风险因子路径</h3><p className="mt-1 text-sm text-slate-600">下游产品将使用这组冲击。{preview.entry !== 'market' ? '宏观模型只提供条件统计响应，不证明经济因果。' : '冲击由研究者显式指定。'}</p></div><ScenarioPathTable variables={preview.factors} rows={preview.path} title="最终市场冲击" />{preview.lineage.length > 0 && <details><summary className="cursor-pointer text-sm font-medium text-slate-700">查看中间传导与模型来源</summary><div className="mt-4 space-y-5">{preview.lineage.map(line => <div key={line.release_id}><p className="mb-2 text-sm text-slate-600">{line.stage === 'event_macro' ? '事件驱动 → 宏观变量' : '宏观变量 → 市场风险因子'} · {line.name}</p><ScenarioPathTable variables={line.outputs} rows={line.path} title="本段输出" /><p className="mt-2 break-all text-xs text-slate-600">发布引用：{line.release_id}</p></div>)}</div></details>}<details><summary className="cursor-pointer text-sm font-medium text-slate-700">适用限制</summary><div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">{preview.limitations.map(item => <p key={item}>{item}</p>)}</div></details></section><section className={`${sectionClass} space-y-4`}>{stale ? <p className="text-sm text-amber-800">草稿已变化，当前显示的是旧预览；请重新预览后发布。</p> : publishedId ? <div className="flex flex-wrap items-center gap-3"><p className="text-sm text-accent-900">此情景已经发布；具体可用状态以情景库为准。</p><Link className={buttonClass} to={`/settings/scenario-algorithms/apply?${new URLSearchParams({ scenario_release_id: publishedId })}`}>应用到产品或组合</Link></div> : <><div className="grid gap-4 sm:grid-cols-2"><Field label="情景有效天数" hint="宏观情景不会超过上游模型的到期时间。"><input type="number" className={inputClass} min="1" max="365" value={validDays} disabled={Boolean(busy)} onChange={event => setValidDays(Number(event.target.value))} /></Field><Field label="发布说明"><input className={inputClass} value={note} disabled={Boolean(busy)} onChange={event => setNote(event.target.value)} /></Field></div><label className="flex items-start gap-2 text-sm text-slate-600"><input className="mt-1" type="checkbox" checked={acknowledged} disabled={Boolean(busy)} onChange={event => setAcknowledged(event.target.checked)} />我已核对单位、每期变化和传导模型，理解这是研究假设，不是收益预测。</label><button type="button" className={primaryClass} disabled={Boolean(busy) || !acknowledged || !preview.preview_hash} onClick={() => void publish()}>{busy === 'publish' ? '发布中…' : '确认发布情景'}</button></>}<button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={() => setStep(1)}>返回设置变化</button></section></>)}
      </div>}
    </>}</>}
    <section hidden={pane !== 'transmission'}>{visited.transmission && <ResearchWorkbench domain="transmission" />}</section>
    <section hidden={pane !== 'advanced'}>{visited.advanced && <div className="space-y-4"><p className="rounded-lg bg-amber-50 p-4 text-sm leading-6 text-amber-900">这里保留历史重演、随机路径、状态条件抽样和反向压力的独立实验。手工传导系数是实验假设，不会自动变成已验证的风险暴露；产品应用请使用已发布模型与情景。</p><ScenarioSimulationCenter /></div>}</section>
  </div>
}
