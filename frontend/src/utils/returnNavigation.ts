import type { Location, NavigateFunction } from 'react-router-dom'

export interface ReturnNavigationState {
  returnTo: string
  returnLabel: string
}

export function buildReturnNavigationState(
  location: Pick<Location, 'pathname' | 'search' | 'hash'>,
  returnLabel: string,
): ReturnNavigationState {
  return {
    returnTo: `${location.pathname}${location.search}${location.hash}`,
    returnLabel,
  }
}

export function readReturnNavigationState(state: unknown): ReturnNavigationState | null {
  if (!state || typeof state !== 'object') return null
  const candidate = state as Partial<ReturnNavigationState>
  if (
    typeof candidate.returnTo !== 'string'
    || !candidate.returnTo.startsWith('/')
    || candidate.returnTo.startsWith('//')
    || typeof candidate.returnLabel !== 'string'
  ) {
    return null
  }
  return { returnTo: candidate.returnTo, returnLabel: candidate.returnLabel }
}

export function returnToOrigin(
  navigate: NavigateFunction,
  location: Pick<Location, 'key' | 'state'>,
  fallback: string,
): void {
  const origin = readReturnNavigationState(location.state)
  if (location.key !== 'default') {
    navigate(-1)
    return
  }
  navigate(origin?.returnTo ?? fallback, { replace: true })
}
