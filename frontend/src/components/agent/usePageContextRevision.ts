import { useRef } from 'react'

/**
 * Semantic page conditions get one revision per change; results arriving alone never bump it,
 * so a finished chart update cannot invalidate a question the user already sent. The first
 * render is the baseline: mounting a page must not look like a condition change.
 */
export default function usePageContextRevision(key: string) {
  const current = useRef({ key, revision: 1 })
  if (current.current.key !== key) current.current = { key, revision: current.current.revision + 1 }
  return current.current.revision
}
