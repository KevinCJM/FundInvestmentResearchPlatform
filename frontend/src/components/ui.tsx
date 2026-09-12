// 资产配置页面复用的按钮原语；保持现有主题令牌。
const cx = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(' ')

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
