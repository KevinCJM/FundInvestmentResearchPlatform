import { useEffect, useRef, useState } from 'react'
import { Button, Card, Badge, SectionHeader } from '../components/ui'
import { activateLlmProfile, fetchLlmSettings, saveLlmProfile, type LlmSettings, type LlmProfile, type ReasoningEffort } from '../services/llmSettings'

const empty = { name: '', provider: 'openai-compatible', model: '', base_url: '', reasoning_effort: 'default' as ReasoningEffort, timeout_seconds: 60, context_window_tokens: 0 }
const effortLabels: Record<ReasoningEffort, string> = { default: '服务默认', none: '关闭思考（none）', minimal: '最少（minimal）', low: '低（low）', medium: '中（medium）', high: '高（high）', xhigh: '更高（xhigh）', max: '最高（max）' }
const control = 'mt-1 block min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 placeholder:text-slate-600 placeholder:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-500'

export default function LlmSettings() {
  const [settings, setSettings] = useState<LlmSettings | null>(null)
  const [editing, setEditing] = useState<string | null>()
  const editorTrigger = useRef<HTMLButtonElement | null>(null)
  const [form, setForm] = useState(empty)
  const [apiKey, setApiKey] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const edit = (trigger: HTMLButtonElement, profile?: LlmProfile) => {
    editorTrigger.current = trigger
    setEditing(profile?.id ?? null)
    setForm(profile ? { name: profile.name, provider: profile.provider, model: profile.model, base_url: profile.base_url, reasoning_effort: profile.reasoning_effort ?? 'default', timeout_seconds: profile.timeout_seconds ?? 60, context_window_tokens: profile.context_window_tokens ?? 0 } : empty)
    setApiKey(''); setMessage(''); setError('')
  }
  const closeEditor = () => { setEditing(undefined); setForm(empty); setApiKey(''); setError('') }
  useEffect(() => { if (editing === undefined) editorTrigger.current?.focus() }, [editing])
  const load = async () => {
    setLoading(true); setError('')
    try {
      const value = await fetchLlmSettings()
      setSettings(value)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '读取 LLM 配置失败。') }
    finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [])
  const work = async (action: () => Promise<LlmSettings>, notice: string) => {
    setBusy(true); setError(''); setMessage('')
    try { const value = await action(); setSettings(value); setMessage(notice); return value }
    catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败，请重试。') }
    finally { setBusy(false) }
  }
  const save = async () => {
    const value = await work(() => saveLlmProfile(editing ?? null, { ...form, ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}) }), '配置已保存。助手只会使用标记为“正在使用”的 API。')
    if (value) closeEditor()
  }
  const selected = settings?.profiles.find(p => p.id === editing)
  const active = settings?.profiles.find(p => p.id === settings.active_profile_id)
  return <div className="mx-auto max-w-[1100px] space-y-5">
    <SectionHeader eyebrow="Platform capabilities" title="LLM API 配置" description="管理 AI 助手使用的大模型服务。可保存多组兼容 OpenAI Chat Completions 的 API，同一时间只使用指定的一组。密钥仅保存在服务端。" />
    {error && <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}{!settings && <Button onClick={() => void load()} disabled={loading}>重新读取</Button>}</div>}
    {message && <p role="status" className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">{message}</p>}
    {loading ? <p role="status" className="text-sm text-slate-600">正在读取 API 配置…</p> : settings && <>
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-lg font-semibold text-slate-900">已保存的 API</h2><Button disabled={busy} onClick={event => edit(event.currentTarget)}>新增 API</Button></div>
        <p className="mt-2 break-words text-sm text-slate-600">当前使用：{active ? `${active.name} · ${active.model}` : '尚未指定，请选择一组已配置的 API'}。未选中的 API 不会被调用。</p>
        <div className="mt-3 divide-y divide-slate-200">
          {settings.profiles.map(profile => <div key={profile.id} aria-label={`API ${profile.name}`} className="flex min-w-0 flex-col gap-3 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0 flex-1 break-words"><div className="flex flex-wrap items-center gap-2"><h3 className="font-semibold text-slate-900">{profile.name}</h3>{profile.id === settings.active_profile_id && <Badge tone="success">正在使用</Badge>}</div><p className="mt-1 text-sm text-slate-700">{profile.provider} · {profile.model}</p><p className="mt-1 text-xs text-slate-600">思考等级：{effortLabels[profile.reasoning_effort ?? 'default']}</p><p className="mt-1 text-xs text-slate-600">上下文窗口：{profile.context_window_tokens ? `${profile.context_window_tokens.toLocaleString()} tokens` : '自动'}</p><p className="mt-1 break-all text-xs text-slate-600">{profile.base_url}</p><p className="mt-1 text-xs text-slate-600">{profile.configured ? `密钥已配置 ${profile.api_key_masked}` : '未配置密钥，暂不可使用'}</p></div>
            <div className="flex flex-wrap gap-2"><Button disabled={busy} onClick={event => edit(event.currentTarget, profile)}>编辑</Button><Button tone="primary" disabled={busy || !profile.configured || profile.id === settings.active_profile_id} onClick={() => void work(() => activateLlmProfile(profile.id), `已切换为 ${profile.name}，新的对话请求将使用该 API。`)}>{profile.id === settings.active_profile_id ? '已选用' : '设为使用'}</Button></div>
          </div>)}
        </div>
        {settings.profiles.length === 0 && <p className="mt-3 text-sm text-slate-600">还没有 API 配置。点击“新增 API”填写并保存，再点击“设为使用”。</p>}
        {active && <Button disabled={busy} onClick={() => void work(() => activateLlmProfile(null), '已停用 API；选择使用项后可继续对话。')}>停止使用 API</Button>}
      </Card>
      {editing !== undefined && <Card><h2 className="text-lg font-semibold text-slate-900">{editing ? '编辑 API' : '新增 API'}</h2>
        <form className="mt-4 space-y-4" onSubmit={event => { event.preventDefault(); void save() }}>
          <fieldset disabled={busy} className="grid min-w-0 gap-4 sm:grid-cols-2">
            <label className="text-sm font-semibold text-slate-700">配置名称<input autoFocus required maxLength={80} value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} className={control} placeholder="例如：公司模型服务" /></label>
            <label className="text-sm font-semibold text-slate-700">供应商<input required maxLength={300} value={form.provider} onChange={e => setForm({ ...form, provider: e.target.value })} className={control} placeholder="openai-compatible" /></label>
            <label className="sm:col-span-2 text-sm font-semibold text-slate-700">模型<input required maxLength={300} value={form.model} onChange={e => setForm({ ...form, model: e.target.value })} className={control} placeholder="填写服务支持的模型名称" /></label>
            <label className="text-sm font-semibold text-slate-700">思考等级<select aria-describedby="reasoning-help" value={form.reasoning_effort} onChange={e => setForm({ ...form, reasoning_effort: e.target.value as ReasoningEffort })} className={control}>{Object.entries(effortLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="text-sm font-semibold text-slate-700">单次模型请求超时（秒）<input type="number" required min={10} max={1800} step={1} value={form.timeout_seconds || ''} onChange={e => setForm({ ...form, timeout_seconds: Number(e.target.value) })} aria-describedby="reasoning-help" className={control} /></label>
            <p id="reasoning-help" className="sm:col-span-2 text-sm text-slate-600">服务默认不指定思考等级。各模型支持的档位不同，DeepSeek V4 系列可用 low、high、max，none 关闭思考。高等级通常更慢、消耗更多额度；使用 max 时可将超时调为 600 秒。</p>
            <label className="sm:col-span-2 text-sm font-semibold text-slate-700">上下文窗口（tokens）<input type="number" min={8192} max={2000000} step={1} value={form.context_window_tokens || ''} onChange={e => setForm({ ...form, context_window_tokens: Number(e.target.value) })} aria-describedby="context-window-help" className={control} placeholder="自动" /></label>
            <p id="context-window-help" className="sm:col-span-2 text-sm text-slate-600">留空按服务与模型匹配；未知接口保守使用 32,768 tokens。仅在服务商说明了实际窗口时填写。助手会预留输出空间并自动整理历史；此设置不会扩大服务商的容量。</p>
            <label className="sm:col-span-2 text-sm font-semibold text-slate-700">API Base URL<input type="url" required maxLength={300} value={form.base_url} onChange={e => setForm({ ...form, base_url: e.target.value })} className={control} placeholder="https://api.example.com/v1" /></label>
            <label className="sm:col-span-2 text-sm font-semibold text-slate-700">API Key（留空保留此配置的密钥）<input type="password" maxLength={512} value={apiKey} onChange={e => setApiKey(e.target.value)} className={control} placeholder={selected?.api_key_masked || '填写 API Key'} autoComplete="new-password" /></label>
          </fieldset>
          <div className="flex flex-wrap gap-2"><Button type="submit" tone="primary" disabled={busy || !Object.values(form).every(value => String(value).trim())}>{busy ? '处理中…' : '保存配置'}</Button><Button type="button" disabled={busy} onClick={closeEditor}>取消</Button></div>
        </form>
      </Card>}
    </>}
  </div>
}
