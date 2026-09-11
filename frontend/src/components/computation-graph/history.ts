/** Immutable editing history extracted from the historical-regime workbench. */
export interface GraphTimeline<T> { past: T[]; present: T; future: T[] }
export type GraphTimelineAction<T> = { type: 'edit'; definition: T } | { type: 'reset'; definition: T } | { type: 'undo' } | { type: 'redo' }
export const sameDocument = <T,>(left: T, right: T) => JSON.stringify(left) === JSON.stringify(right)
export const cloneDocument = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T
export function createTimelineReducer<T>(clone: (value: T) => T = cloneDocument) {
  return (state: GraphTimeline<T>, action: GraphTimelineAction<T>): GraphTimeline<T> => {
    if (action.type === 'reset') return { past: [], present: clone(action.definition), future: [] }
    if (action.type === 'undo') {
      const previous = state.past[state.past.length - 1]
      if (!previous) return state
      return { past: state.past.slice(0, -1), present: previous, future: [state.present, ...state.future] }
    }
    if (action.type === 'redo') {
      const next = state.future[0]
      if (!next) return state
      return { past: [...state.past, state.present], present: next, future: state.future.slice(1) }
    }
    if (sameDocument(state.present, action.definition)) return state
    return { past: [...state.past.slice(-49), state.present], present: action.definition, future: [] }
  }
}
