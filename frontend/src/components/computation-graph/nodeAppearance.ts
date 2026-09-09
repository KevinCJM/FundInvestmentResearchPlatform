/** Optional presentation roles. Omitted roles retain the shared canvas's original style. */
export type NodeAppearance = 'variable' | 'constant' | 'parameter' | 'operator' | 'output'
export const NODE_APPEARANCES = {
  variable: { label: '输入变量', glyph: '▤', className: 'rounded-md border-l-4 border-sky-400 bg-sky-50', color: '#0284c7' },
  constant: { label: '固定常量', glyph: '●', className: 'rounded-[28px] border-amber-400 bg-amber-50', color: '#b45309' },
  parameter: { label: '可变参数', glyph: '↔', className: 'rounded-[28px] border-dashed border-cyan-600 bg-cyan-50', color: '#0e7490' },
  operator: { label: '计算算子', glyph: '◇', className: 'rounded-xl border-violet-400 bg-violet-50', color: '#7c3aed' },
  output: { label: '指标输出', glyph: '◎', className: 'rounded-2xl border-[3px] border-double border-emerald-600 bg-emerald-50', color: '#059669' },
} as const
export const nodeAppearance = (kind?: NodeAppearance) => kind ? NODE_APPEARANCES[kind] : undefined
