import { useEffect, useRef, useState } from 'react'
import type { ProductPool } from '../../services/productPools'

export interface CreateProductPoolInput {
  name: string
  purpose: string
  owner: string
}

interface ProductPoolHeaderProps {
  pools: ProductPool[]
  selectedPool: ProductPool | null
  selectedPoolId: string
  loading: boolean
  busy: boolean
  onSelectPool: (poolId: string) => void
  onCreatePool: (input: CreateProductPoolInput) => Promise<void>
}

const poolStateLabel: Record<ProductPool['state'], string> = {
  draft: '草稿',
  active: '已发布',
  archived: '已归档',
}

function errorMessage(reason: unknown) {
  return reason instanceof Error && reason.message ? reason.message : '创建产品池失败。'
}

export default function ProductPoolHeader({
  pools,
  selectedPool,
  selectedPoolId,
  loading,
  busy,
  onSelectPool,
  onCreatePool,
}: ProductPoolHeaderProps) {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [name, setName] = useState('')
  const [purpose, setPurpose] = useState('')
  const [owner, setOwner] = useState('')
  const [error, setError] = useState('')
  const nameInputRef = useRef<HTMLInputElement>(null)

  const resetDialog = () => {
    setDialogOpen(false)
    setName('')
    setPurpose('')
    setOwner('')
    setError('')
  }

  const closeDialog = () => {
    if (!busy) resetDialog()
  }

  const openDialog = () => {
    setName('')
    setPurpose('')
    setOwner('')
    setError('')
    setDialogOpen(true)
  }

  useEffect(() => {
    if (!dialogOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const frame = window.requestAnimationFrame(() => nameInputRef.current?.focus())
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) resetDialog()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      window.cancelAnimationFrame(frame)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [busy, dialogOpen])

  const submit = async () => {
    if (!name.trim()) {
      setError('请填写产品池名称。')
      nameInputRef.current?.focus()
      return
    }
    setError('')
    try {
      await onCreatePool({
        name: name.trim(),
        purpose: purpose.trim(),
        owner: owner.trim(),
      })
      closeDialog()
    } catch (reason) {
      setError(errorMessage(reason))
    }
  }

  return <>
    <header className="rounded-2xl bg-slate-900 px-5 py-4 text-white sm:px-7">
      <div className="flex min-w-0 flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div className="min-w-0">
          <p className="text-sm font-medium text-emerald-300">产品研究 · 准入管理</p>
          <h1 className="mt-1 text-2xl font-semibold">产品池构建</h1>
          <p className="mt-2 hidden max-w-4xl text-sm leading-6 text-slate-300 sm:block">
            先复核产品，再发布版本并开展配置研究。规则与评价来源可在下方展开。
          </p>
        </div>

        <div className="min-w-0 w-full rounded-xl bg-white/10 p-4 ring-1 ring-white/15 xl:w-[500px]">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <label className="min-w-0 flex-1 text-xs font-medium text-slate-300">
              当前产品池
              <select
                aria-label="当前产品池"
                value={selectedPoolId}
                disabled={loading || pools.length === 0}
                onChange={(event) => onSelectPool(event.target.value)}
                className="mt-1 min-h-11 w-full rounded-lg border border-white/20 bg-white px-3 text-sm font-medium text-slate-900 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-300/30 disabled:bg-slate-200"
              >
                {pools.length === 0 && <option value="">尚未创建产品池</option>}
                {pools.map((pool) => (
                  <option key={pool.id} value={pool.id}>
                    {pool.name} · {poolStateLabel[pool.state]}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              disabled={busy}
              onClick={openDialog}
              className="min-h-11 shrink-0 rounded-lg bg-emerald-400 px-4 text-sm font-semibold text-slate-950 hover:bg-emerald-300 disabled:bg-slate-500 disabled:text-slate-300"
            >
              ＋ 新建产品池
            </button>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-slate-400">共 {pools.length} 个产品池</span>
            {selectedPool && <>
              <span className="rounded-full bg-white/10 px-2.5 py-1 text-slate-200">{selectedPool.evaluation_plans.length} 套评价方案</span>
              <span className="rounded-full bg-white/10 px-2.5 py-1 text-slate-200">{selectedPool.members.length} 个候选</span>

              <span className={`rounded-full px-2.5 py-1 ${selectedPool.state === 'active' ? 'bg-emerald-400/20 text-emerald-200' : selectedPool.state === 'archived' ? 'bg-slate-500/40 text-slate-300' : 'bg-amber-300/15 text-amber-200'}`}>
                {poolStateLabel[selectedPool.state]}
              </span>
            </>}
          </div>
        </div>
      </div>
    </header>

    {dialogOpen && <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-[1px]"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closeDialog()
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-product-pool-title"
        className="w-full max-w-lg rounded-2xl bg-white p-5 shadow-2xl sm:p-6"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="create-product-pool-title" className="text-xl font-semibold text-slate-900">新建产品池</h2>
            <p className="mt-1 text-sm text-slate-500">先创建草稿，再关联评价方案并复核候选产品。</p>
          </div>
          <button type="button" disabled={busy} onClick={closeDialog} aria-label="关闭新建产品池窗口" className="rounded-lg px-2 py-1 text-xl leading-none text-slate-400 hover:bg-slate-100 hover:text-slate-700 disabled:opacity-40">×</button>
        </div>

        {error && <div role="alert" className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">{error}</div>}

        <form className="mt-5 space-y-4" onSubmit={(event) => { event.preventDefault(); void submit() }}>
          <label className="block text-sm font-medium text-slate-700">
            产品池名称 <span className="text-rose-600">*</span>
            <input
              ref={nameInputRef}
              aria-label="新产品池名称"
              value={name}
              maxLength={80}
              onChange={(event) => setName(event.target.value)}
              placeholder="例如：长期核心产品池"
              className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 px-3 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
            />
          </label>
          <label className="block text-sm font-medium text-slate-700">
            用途
            <input
              aria-label="新产品池用途"
              value={purpose}
              maxLength={200}
              onChange={(event) => setPurpose(event.target.value)}
              placeholder="例如：长期核心配置"
              className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 px-3 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
            />
          </label>
          <label className="block text-sm font-medium text-slate-700">
            负责人
            <input
              aria-label="新产品池负责人"
              value={owner}
              maxLength={80}
              onChange={(event) => setOwner(event.target.value)}
              placeholder="研究负责人"
              className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 px-3 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
            />
          </label>

          <div className="flex justify-end gap-2 border-t border-slate-100 pt-4">
            <button type="button" disabled={busy} onClick={closeDialog} className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40">取消</button>
            <button type="submit" disabled={busy} className="rounded-lg bg-slate-900 px-5 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:bg-slate-400">{busy ? '正在创建…' : '创建草稿'}</button>
          </div>
        </form>
      </section>
    </div>}
  </>
}
