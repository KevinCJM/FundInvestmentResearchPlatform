/** DOM 文字对比度预警：支持纯色、渐变色标及祖先透明度；不采样图片、滤镜或遮挡。 */
export const auditTextContrast = () => {
  type Color = { r: number; g: number; b: number; a: number }
  const clear: Color = { r: 0, g: 0, b: 0, a: 0 }
  const white: Color = { r: 255, g: 255, b: 255, a: 1 }
  const parse = (value: string): Color => {
    const n = value.match(/[\d.]+/g)?.map(Number) ?? []
    return n.length < 3 ? clear : { r: n[0], g: n[1], b: n[2], a: n[3] ?? 1 }
  }
  const over = (top: Color, bottom: Color): Color => {
    const a = top.a + bottom.a * (1 - top.a)
    const channel = (key: 'r' | 'g' | 'b') => a ? (top[key] * top.a + bottom[key] * bottom.a * (1 - top.a)) / a : 0
    return { r: channel('r'), g: channel('g'), b: channel('b'), a }
  }
  const luminance = (color: Color) => {
    const ch = (v: number) => v / 255 <= 0.04045 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4
    return .2126 * ch(color.r) + .7152 * ch(color.g) + .0722 * ch(color.b)
  }
  const failures: { text: string; ratio: number; color: string; path: string }[] = []
  const check = (el: Element, style: CSSStyleDeclaration, label: string, textOpacity = 1) => {
    const color = parse(style.color)
    if (!color.a) return
    let paints = [{ front: { ...color, a: color.a * textOpacity }, back: clear }]
    for (let node: Element | null = el; node; node = node.parentElement) {
      const parent = getComputedStyle(node)
      if (Number(parent.opacity) === 0) return
      if (parent.backgroundImage.includes('url(')) return
      const base = parse(parent.backgroundColor)
      const stops = parent.backgroundImage.match(/rgba?\([\d.,\s]+\)/g)?.map(parse) ?? []
      const backgrounds = stops.length ? stops.map(stop => over(stop, base)) : [base]
      // opacity 作用于整个已绘制的组：文字和该组的背景必须一起合成。
      paints = paints.flatMap(paint => backgrounds.map(bg => {
        const front = over(paint.front, bg), back = over(paint.back, bg)
        const opacity = Number(parent.opacity)
        return { front: { ...front, a: front.a * opacity }, back: { ...back, a: back.a * opacity } }
      }))
    }
    const ratio = Math.min(...paints.map(paint => {
      const [lo, hi] = [luminance(over(paint.front, white)), luminance(over(paint.back, white))].sort((a, b) => a - b)
      return (hi + .05) / (lo + .05)
    }))
    const size = Number.parseFloat(style.fontSize)
    const large = size >= 24 || (size >= 18.6667 && Number(style.fontWeight) >= 700)
    if (ratio < (large ? 3 : 4.5)) failures.push({ text: label.slice(0, 50), ratio: Math.round(ratio * 100) / 100, color: style.color, path: `${el.tagName.toLowerCase()}${el.id ? `#${el.id}` : ''}` })
  }
  for (const el of document.querySelectorAll('body *')) {
    const style = getComputedStyle(el), box = el.getBoundingClientRect()
    if (style.visibility !== 'visible' || box.width < 2 || box.height < 2) continue
    // aria-hidden 仅改变辅助技术读取，不能豁免仍然可见的信息。
    if (el.closest('[disabled],[aria-disabled="true"]')) continue
    const own = Array.from(el.childNodes).filter(n => n.nodeType === Node.TEXT_NODE && n.textContent?.trim()).map(n => n.textContent!.trim()).join(' ')
    if (own) check(el, style, own)
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
      if (['checkbox', 'radio', 'range', 'color', 'file', 'hidden'].includes(el.type)) continue
      if (el.value) check(el, style, `值：${el.value}`)
      else if (el.placeholder) {
        const placeholder = getComputedStyle(el, '::placeholder')
        check(el, placeholder, `占位符：${el.placeholder}`, Number(placeholder.opacity))
      }
    } else if (el instanceof HTMLSelectElement && el.selectedOptions.length) check(el, style, `选中：${el.selectedOptions[0].text}`)
  }
  return failures
}
