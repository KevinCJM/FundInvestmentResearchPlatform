// 姿势与状态的映射只存在这一处。契约与禁区见 docs/frontend-design-guidelines.md 第 15 节。
// 资产是显示宽度的 2 倍，用于高密度屏；因此不需要 srcset。
const POSES = {
  empty: { file: 'mascot-empty-240.webp', width: 120 },
  noresult: { file: 'mascot-noresult-240.webp', width: 120 },
  welcome: { file: 'mascot-welcome-240.webp', width: 120 },
  working: { file: 'mascot-working-160.webp', width: 80 },
  success: { file: 'mascot-success-96.webp', width: 48 },
} as const

export type MascotState = keyof typeof POSES

/**
 * 装饰性品牌形象。状态含义由相邻文字承载，因此对辅助技术隐藏。
 * 不得放在数值、风险提示、PIT 口径警告或核算差错旁边，也不得放进表格与图表。
 * 同一屏幕最多一个实例。
 */
export default function Mascot({ state, className }: { state: MascotState; className?: string }) {
  const pose = POSES[state]
  return (
    <img
      src={`/homepage/images/${pose.file}`}
      width={pose.width}
      height={pose.width}
      alt=""
      aria-hidden="true"
      loading="lazy"
      decoding="async"
      className={className}
      style={{ flexShrink: 0 }}
    />
  )
}
