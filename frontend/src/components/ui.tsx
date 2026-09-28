import { researchMessage } from '../i18n/researchMessages'
import { useI18n } from '../i18n/runtime'
// 工作台的共用外壳。新页面用这里的组件，不要再复制一份 className 串。
// 取值与首页令牌一致，规则见 docs/frontend/README.md。
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

/** 按钮外观。跳转型主操作（`<Link>`）用它取同一套类，不要另抄一串 className。 */
export const actionClass = (tone: keyof typeof BUTTON_TONES = 'secondary', className?: string) => cx(
  'inline-flex min-h-10 items-center justify-center gap-2 whitespace-nowrap rounded-lg px-4 text-sm font-semibold transition',
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:ring-offset-2',
  'disabled:cursor-not-allowed disabled:opacity-50',
  BUTTON_TONES[tone], className,
)

/** 按钮。min-h-10 是触控下限；focus 环全站同一种。 */
export function Button({ tone = 'secondary', className, ...rest }: { tone?: keyof typeof BUTTON_TONES } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" {...rest} className={actionClass(tone, className)} />
}

/**
 * 加载骨架。占住最终内容的形状，数据回来时布局不跳；准则 8.1 要的是骨架屏，
 * 不是居中转圈，也不是一行"正在读取…"——文字行同样不保留布局。
 * 关掉动效时只剩静态灰块，符合 9.3。
 */
export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cx('block h-4 rounded bg-slate-100 motion-safe:animate-pulse', className)} />
}

/**
 * 整块区域在等数据时的公共等待态：居中 `working` 吉祥物加一行说明，按 8.1 的整面板例外
 * 预留高度而不铺灰底色块，数据回来时布局不跳。表格行内、字段内联仍用 `Skeleton`——
 * 15.2 不允许形象进表格与图表内部，也不允许它紧邻已经渲染出来的数值。
 * 同屏另有一处等待中的形象时传 `mascot={false}`，保持 15.3 第 5 条的单实例。
 */
export function LoadingPanel({ text, className, mascot = true }: { text: string; className?: string; mascot?: boolean }) {
  return (
    <div role="status" aria-live="polite" className={cx('grid min-h-56 w-full place-items-center', className)}>
      <div className="space-y-2 text-center">
        {mascot && <Mascot state="working" className="mx-auto" />}
        <p className="text-sm text-slate-600">{text}</p>
      </div>
    </div>
  )
}

/**
 * 整块区域读取失败时的公共错误态：居中 `error` 吉祥物、失败原因和重试入口，与 `LoadingPanel`
 * 互斥占用同一个位置（15.3 第 5 条单实例）。8.3 要「就近展示并提供重试」，`action` 就是那个重试入口。
 * 字段校验、以及旁边已经渲染出业务数值的局部失败仍是纯文字（15.2 第 1 条），不换成这块面板。
 */
export function ErrorPanel({ message, title, action, onRetry, retryLabel, className, mascot = true }: { message?: string; title?: string; action?: ReactNode; onRetry?: () => void; retryLabel?: string; className?: string; mascot?: boolean }) {
  const { s } = useI18n()
  return (
    <div role="alert" className={cx('grid min-h-56 w-full place-items-center', className)}>
      <div className="flex w-full max-w-lg flex-col items-center gap-3 text-center">
        {mascot && <Mascot state="error" />}
        {title && <p className="text-sm font-semibold text-slate-800">{title}</p>}
        <p className="w-full rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm leading-6 text-rose-900">{message ? researchMessage(message) : s('common.dataReadFailed')}</p>
        {action ?? (onRetry && <Button onClick={onRetry}>{retryLabel || s('common.retry')}</Button>)}
      </div>
    </div>
  )
}

export interface TableColumn<Row> {
  /** 表头文字，同时用作列的 key。 */
  header: string
  /** 数字列。右对齐便于逐位比较；等宽数字由 `index.css` 对 `td` / `th` 全局开启。 */
  numeric?: boolean
  /** 日期、代码这类不该断行的列。 */
  nowrap?: boolean
  cell: (row: Row) => ReactNode
}

/**
 * 业务表格。85 个文件各自手写 table 标签，`scope`、`caption`、数字列对齐和空态全靠每页自觉，
 * 已经把 `tracking-wide r` 这种坏类名抄了 6 份。准则 6.6 要的语义在这里默认就对，调用方只描述列。
 * 横向滚动区可聚焦，窄屏靠滚动而不是裁切。
 */
export function DataTable<Row>({ caption, columns, rows, rowKey, minWidth, maxHeight, loading, empty }: {
  /** 表格的可访问名，用 `<caption>` 承载（视觉上隐藏）。必填。 */
  caption: string
  columns: TableColumn<Row>[]
  rows: Row[]
  rowKey: (row: Row, index: number) => string
  /** 列挤不下时的最小宽度，如 `'640px'`。 */
  minWidth?: string
  /** 行数多时限制可视高度，如 `'24rem'`：容器内纵向滚动，表头吸顶。准则 6.3 的例外，只给内联选择器用。 */
  maxHeight?: string
  /** 传文案即进入加载态：表头不动，行位用骨架占住，形状与最终行一致。 */
  loading?: string
  /** 空态文案。准则 8.2：要说明为什么空，只写"暂无数据"不算。 */
  empty: string
}) {
  const body = loading
    ? [0, 1, 2, 3, 4].map(row => (
      <tr key={row} aria-hidden="true">{columns.map(column => <td key={column.header} className="px-4 py-3"><Skeleton /></td>)}</tr>
    ))
    : rows.length
      ? rows.map((row, index) => (
        <tr key={rowKey(row, index)} className="border-t border-slate-100 hover:bg-slate-50">
          {columns.map(column => (
            <td key={column.header} className={cx('px-4 py-3 align-middle text-slate-700', column.numeric && 'text-right', column.nowrap && 'whitespace-nowrap')}>
              {column.cell(row)}
            </td>
          ))}
        </tr>
      ))
      : [<tr key="empty"><td colSpan={columns.length} className="px-4 py-12 text-center text-sm text-slate-600">{empty}</td></tr>]

  return (
    <div tabIndex={0} aria-label={caption} style={maxHeight ? { maxHeight } : undefined} className={cx('max-w-full rounded-xl border border-slate-200 bg-white shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500', maxHeight ? 'overflow-auto' : 'overflow-x-auto')}>
      <p role="status" className="sr-only">{loading ?? ''}</p>
      <table className="w-full text-sm" style={minWidth ? { minWidth } : undefined}>
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-slate-50">
          <tr>{columns.map(column => (
            <th key={column.header} scope="col" className={cx('px-4 py-3 text-xs font-semibold text-slate-600', column.numeric ? 'text-right' : 'text-left', maxHeight && 'sticky top-0 z-10 bg-slate-50')}>{column.header}</th>
          ))}</tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
  )
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
