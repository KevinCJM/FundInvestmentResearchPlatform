import { useEffect, useRef, useState } from 'react'
export function useScopeOperation() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const request = useRef<AbortController | null>(null)
  const generation = useRef(0)
  useEffect(() => () => { ++generation.current; request.current?.abort() }, [])
  function invalidate() { ++generation.current; request.current?.abort(); setBusy(false); setError('') }
  async function run<T>(work: (signal: AbortSignal) => Promise<T>, consume: (value: T) => void) {
    invalidate()
    const token = generation.current, controller = new AbortController()
    request.current = controller; setBusy(true)
    try { const value = await work(controller.signal); if (!controller.signal.aborted && token === generation.current) consume(value) }
    catch (reason) { if (!controller.signal.aborted && token === generation.current) setError(reason instanceof Error ? reason.message : '读取或确认失败，请重试。') }
    finally { if (token === generation.current) setBusy(false) }
  }
  return { busy, error, run, invalidate }
}
