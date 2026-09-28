// 全站功能检索。123 条路由挂在 10 个一级入口下，靠逐层点击找不过来，这里直接搜 processRegistry。
// 键盘契约取自 APG：combobox（list autocomplete）把 DOM 焦点留在输入框，用 aria-activedescendant
// 移动视觉焦点；modal dialog 负责 Esc 关闭与焦点归还触发按钮。
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { allStages, statusLabels, type CapabilityStatus } from '../app/processRegistry'
import { localizeStage } from '../i18n/navigation'
import { useI18n } from '../i18n/runtime'
import { Badge } from './ui'

interface Target { path: string; label: string; stage: string; hint: string; status?: CapabilityStatus }

const OPTION_ID = 'command-palette-option-'
// 全局检索层要压过页面级抽屉（最高 z-[200] 的提示气泡），单独占一档。
const OVERLAY_Z = 'z-[300]'

export default function CommandPalette() {
  const { s, version } = useI18n()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // 注册表就是路由的唯一真相源，条目跟着它走，不另维护一份清单。
  const targets = useMemo<Target[]>(() => allStages.map(localizeStage).flatMap(stage => [
    { path: stage.path, label: stage.label, stage: stage.label, hint: stage.description },
    ...stage.nodes.map(node => ({ path: node.path, label: node.label, stage: stage.label, hint: node.description, status: stage.id === 'pre-investment' ? undefined : node.status })),
    ...(stage.tools ?? []).map(tool => ({ path: tool.path, label: tool.label, stage: stage.label, hint: tool.description })),
  ]), [version])

  const keyword = query.trim().toLowerCase()
  const matches = useMemo(
    () => keyword ? targets.filter(item => `${item.label} ${item.stage} ${item.hint} ${item.path}`.toLowerCase().includes(keyword)) : targets,
    [targets, keyword],
  )

  const show = () => { setQuery(''); setActive(0); setOpen(true) }
  const close = () => { setOpen(false); triggerRef.current?.focus() }
  const go = (path: string) => { setOpen(false); navigate(path) }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'k') return
      event.preventDefault()
      if (open) close()
      else show()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => { document.removeEventListener('keydown', onKeyDown) }
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    inputRef.current?.focus()
    // 弹层打开时背景不跟着滚，保持"一个页面一个滚动容器"。
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [open])

  useEffect(() => {
    document.getElementById(`${OPTION_ID}${active}`)?.scrollIntoView?.({ block: 'nearest' })
  }, [active])

  const onInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!matches.length) return
    if (event.key === 'ArrowDown') { event.preventDefault(); setActive(current => (current + 1) % matches.length) }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive(current => (current - 1 + matches.length) % matches.length) }
    else if (event.key === 'Enter' && matches[active]) { event.preventDefault(); go(matches[active].path) }
  }

  const onPanelKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); close(); return }
    if (event.key !== 'Tab') return
    // 模态要困住 Tab：面板里只有输入框和关闭按钮，按顺序循环。
    const focusable = [...(panelRef.current?.querySelectorAll<HTMLElement>('input, button') ?? [])]
    if (!focusable.length) return
    event.preventDefault()
    const index = focusable.indexOf(document.activeElement as HTMLElement)
    focusable[(index + (event.shiftKey ? -1 : 1) + focusable.length) % focusable.length]?.focus()
  }

  const dialog = (
    <div
      className={`fixed inset-0 ${OVERLAY_Z} flex items-start justify-center bg-slate-950/50 px-4 pt-[10vh]`}
      onMouseDown={event => { if (event.target === event.currentTarget) close() }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={s('navigation.search')}
        onKeyDown={onPanelKeyDown}
        className="flex max-h-[70vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
      >
        <div className="flex items-center gap-2 border-b border-slate-200 px-4 py-3">
          <MagnifyingGlassIcon aria-hidden="true" className="h-5 w-5 shrink-0 text-slate-600" />
          {/* 检索框没有可见标题，可访问名由 sr-only 的 label 提供，不靠 placeholder 承担。 */}
          <label className="min-w-0 flex-1">
            <span className="sr-only">{s('navigation.search')}</span>
            <input
              ref={inputRef}
              type="text"
              role="combobox"
              aria-expanded="true"
              aria-controls="command-palette-results"
              aria-autocomplete="list"
              aria-activedescendant={matches[active] ? `${OPTION_ID}${active}` : undefined}
              placeholder={s('navigation.searchPlaceholder')}
              value={query}
              onChange={event => { setQuery(event.target.value); setActive(0) }}
              onKeyDown={onInputKeyDown}
              className="min-h-10 w-full min-w-0 rounded-lg border border-slate-300 px-3 text-sm text-slate-900 placeholder:text-slate-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
            />
          </label>
          <button
            type="button"
            onClick={close}
            aria-label={s('navigation.searchClose')}
            className="inline-flex min-h-10 shrink-0 items-center rounded-lg px-2 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
          >
            <XMarkIcon aria-hidden="true" className="h-5 w-5" />
          </button>
        </div>

        <p role="status" className="px-4 pt-3 text-xs text-slate-600">
          {matches.length ? s('navigation.searchCount', { count: matches.length }) : s('navigation.searchEmpty')}
        </p>

        <ul id="command-palette-results" role="listbox" aria-label={s('navigation.search')} className="min-h-0 flex-1 overflow-y-auto p-2">
          {matches.map((item, index) => (
            <li
              // 会计工作区在两个阶段各挂一次，路径相同名字不同，两条都要能搜到，所以键要带阶段。
              key={`${item.stage}::${item.path}`}
              id={`${OPTION_ID}${index}`}
              role="option"
              aria-selected={index === active}
              onMouseEnter={() => setActive(index)}
              onMouseDown={event => event.preventDefault()}
              onClick={() => go(item.path)}
              className={`flex min-h-11 cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2 ${index === active ? 'bg-accent-50' : ''}`}
            >
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold text-slate-900">{item.label}</span>
                <span className="block truncate text-xs text-slate-600">{item.stage} · {item.path}</span>
              </span>
              {item.status && item.status !== 'available' && <Badge tone="warning">{statusLabels[item.status]}</Badge>}
            </li>
          ))}
        </ul>

        <p className="border-t border-slate-200 px-4 py-2 text-xs text-slate-600">{s('navigation.searchHint')}</p>
      </div>
    </div>
  )

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => { if (open) close(); else show() }}
        aria-haspopup="dialog"
        aria-expanded={open}
        className="inline-flex min-h-10 shrink-0 items-center gap-2 rounded-lg border border-slate-600 bg-slate-900 px-3 text-sm font-medium text-slate-200 transition hover:border-slate-500 hover:bg-slate-800 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
      >
        <MagnifyingGlassIcon aria-hidden="true" className="h-4 w-4" />
        <span>{s('navigation.search')}</span>
        <kbd className="hidden rounded border border-slate-600 px-1.5 text-xs font-medium text-slate-200 sm:inline-block">⌘K</kbd>
      </button>
      {open && createPortal(dialog, document.body)}
    </>
  )
}
