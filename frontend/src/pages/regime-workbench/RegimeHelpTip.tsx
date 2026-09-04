import { useId, useState, type FocusEvent, type MouseEvent } from 'react'

export default function RegimeHelpTip({ text, label = '查看说明', dark = false }: { text: string; label?: string; dark?: boolean }) {
  const tooltipId = useId()
  const [placement, setPlacement] = useState<'left' | 'center' | 'right'>('right')
  const placeTooltip = (element: HTMLElement) => {
    const box = element.getBoundingClientRect()
    const center = box.left + box.width / 2
    if (center < 144) setPlacement('left')
    else if (center > window.innerWidth - 144) setPlacement('right')
    else setPlacement('center')
  }
  return (
    <span
      className="group relative ml-1 inline-flex align-middle"
      onMouseEnter={(event: MouseEvent<HTMLSpanElement>) => placeTooltip(event.currentTarget)}
      onFocus={(event: FocusEvent<HTMLSpanElement>) => placeTooltip(event.currentTarget)}
    >
      <span
        tabIndex={0}
        role="button"
        aria-label={label}
        aria-describedby={tooltipId}
        className={`inline-grid h-4 w-4 cursor-help place-items-center rounded-full border text-[10px] font-black leading-none outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${dark ? 'border-slate-500 text-slate-300' : 'border-slate-300 bg-white text-slate-500'}`}
      >?</span>
      <span
        id={tooltipId}
        role="tooltip"
        className={`pointer-events-none invisible absolute bottom-[calc(100%+7px)] z-[80] w-[min(16rem,calc(100vw-2rem))] rounded-lg bg-slate-950 px-3 py-2 text-left text-[11px] font-normal leading-5 text-white opacity-0 shadow-xl transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100 ${placement === 'left' ? 'left-0' : placement === 'right' ? 'right-0' : 'left-1/2 -translate-x-1/2'}`}
      >{text}</span>
    </span>
  )
}
