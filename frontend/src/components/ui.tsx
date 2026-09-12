// 工作台的共用外壳。新页面用这里的组件，不要再复制一份 className 串。
// 取值与首页令牌一致，规则见 docs/frontend-design-guidelines.md。
import type { ReactNode } from 'react'
import Mascot, { type MascotState } from './Mascot'

const cx = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(' ')

/** 卡片外壳。圆角只有一档（12px），描边与底色不接受调用方覆盖。 */
export function Card({ children, className, as: Tag = 'section' }: { children: ReactNode; className?: string; as?: 'section' | 'article' | 'div' }) {
  return <Tag className={cx('rounded-xl border border-slate-200 bg-white p-5 shadow-sm', className)}>{children}</Tag>
}

const BUTTON_TONES = {
  // 主操作：与首页 .home-button-blue 同色。白字 5.12:1，hover 6.57:1，均过 WCAG AA。
  primary: 'bg-accent-600 text-white hover:bg-accent-700',
  // 次操作：描边式，承载"还有别的路可走"。
  secondary: 'border border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:bg-slate-50',
  // 破坏性操作。只有真正不可逆的动作才用。
  danger: 'bg-rose-700 text-white hover:bg-rose-800',
} as const

/** 按钮。min-h-10 是触控下限；focus 环全站同一种。 */
export function Button({ tone = 'secondary', className, ...rest }: { tone?: keyof typeof BUTTON_TONES } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button
    type="button"
    {...rest}
    className={cx(
      'inline-flex min-h-10 items-center justify-center gap-2 whitespace-nowrap rounded-lg px-4 text-sm font-semibold transition',
      'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:ring-offset-2',
      'disabled:cursor-not-allowed disabled:opacity-50',
      BUTTON_TONES[tone], className,
    )}
  />
}

const BADGE_TONES = {
  neutral: 'bg-slate-100 text-slate-700',
  // 语义色只有三种，且只表示状态，不表示分类。分类靠文字，不靠颜色。
  success: 'bg-emerald-100 text-emerald-800',
  warning: 'bg-amber-100 text-amber-900',
  danger: 'bg-rose-100 text-rose-800',
} as const

export function Badge({ tone = 'neutral', children }: { tone?: keyof typeof BADGE_TONES; children: ReactNode }) {
  return <span className={cx('inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold', BADGE_TONES[tone])}>{children}</span>
}

/** 区块标题。眉标不加 uppercase：中文上它是空操作，只留下被拉散的字距。 */
export function SectionHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        {eyebrow && <p className="text-xs font-semibold text-accent-700">{eyebrow}</p>}
        <h2 className="mt-1 text-lg font-semibold text-slate-900">{title}</h2>
        {description && <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

/**
 * 空态。必须说明"为什么空"和"下一步做什么"，只写"暂无数据"不算空态。
 * 吉祥物是装饰，禁区见准则第 15.2 节：不得用在数值、风险、PIT 与核算差错旁。
 */
export function EmptyState({ title, hint, action, mascot = 'empty' }: { title: string; hint?: string; action?: ReactNode; mascot?: MascotState | false }) {
  return (
    <div role="status" className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center">
      {mascot && <Mascot state={mascot} />}
      <p className="text-sm font-semibold text-slate-700">{title}</p>
      {hint && <p className="max-w-md text-sm leading-6 text-slate-600">{hint}</p>}
      {action}
    </div>
  )
}
