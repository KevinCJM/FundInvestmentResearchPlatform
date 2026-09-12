#!/usr/bin/env node
// 前端设计准则的机械检查。规则与阈值的说明见 docs/frontend-design-guidelines.md。
// 棘轮语义：budget 是当前实测值，只允许下降。current > budget 即回归，退出码 1。
// 收敛一批实现后把 budget 改小；target 是准则要求的终值，仅作提示，不参与判定。
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const SRC = join(ROOT, 'frontend/src')
const CJK = /[一-鿿]/

// baseline 实测于 2026-09-12，功能页对齐首页令牌之后。只允许下调。
const BUDGET = {
  'tiny-font': { budget: 0, target: 0 },
  'low-contrast-text': { budget: 0, target: 0 },
  'neutral-ramp-mix': { budget: 0, target: 0 },
  'card-radius-variants': { budget: 1, target: 1 },
  'focus-ring-hues': { budget: 1, target: 1 },
  // 五个语义：accent 主操作 / slate 中性 / emerald 成功 / amber 警告 / rose 危险。
  // 金融界面需要成功与警告两种状态底色，硬压到三种会把状态挤回纯文字。
  'solid-button-hues': { budget: 5, target: 5 },
  'page-bg-declarations': { budget: 1, target: 1 },
  'english-em-dash': { budget: 0, target: 0 },
  // 剩下 11 处是英文眉标（Data governance 等），同一行上另有中文才被计入，属规则的行级近似。
  'uppercase-on-cjk': { budget: 11, target: 0 },
  'homepage-subpixel-font': { budget: 0, target: 0 },
  'homepage-nonstandard-weight': { budget: 0, target: 0 },
  'unscoped-tables': { budget: 0, target: 0 },
  'bare-chart-hex': { budget: 57, target: 0 },
  'homepage-accent-variants': { budget: 3, target: 3 },
  'mascot-in-forbidden-zone': { budget: 0, target: 0 },
  'same-element-contrast': { budget: 0, target: 0 },
  'duplicate-category-color': { budget: 0, target: 0 },
}


// 吉祥物禁区：数值、风险、PIT、核算、交易相关的页面与组件（准则第 15.2 节）。
const MASCOT_FORBIDDEN = /(Accounting|Financial|Ledger|Booking|Reconcil|Statement|Trade|Risk|Pit|Performance|Attribution|Valuation|Allocation|Portfolio|Dashboard)/

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) walk(full, out)
    else out.push(full)
  }
  return out
}

const files = walk(SRC)
  .filter((f) => /\.tsx?$/.test(f) && !f.includes('.test.'))
  .map((f) => ({ path: relative(ROOT, f), text: readFileSync(f, 'utf8') }))
const homepageCss = readFileSync(join(SRC, 'homepage/homepage.css'), 'utf8')

const count = (re) => files.reduce((n, f) => n + (f.text.match(re)?.length ?? 0), 0)
const distinct = (re, group = 1) => {
  const set = new Set()
  for (const f of files) for (const m of f.text.matchAll(re)) set.add(m[group])
  return set
}

const results = {}
const detail = {}

// 1. 小于 12px 的字号。中文在 12px 以下屏显不可靠，任何尺寸都不得低于 text-xs。
// 按实际长度换算，不枚举字面量：原来只认 9/10/11px，text-[8px] 与 text-[0.5rem] 都能溜过去。
const TO_PX = { px: 1, rem: 16, em: 16, pt: 96 / 72 }
// 小数点前可以没有整数位：.5rem 与 0.5rem 都是合法 CSS 长度，只认后者会漏掉一半写法。
export const tinyFont = (text) =>
  [...text.matchAll(/text-\[(\d*\.?\d+)(px|rem|em|pt)\]/g)].filter(
    (m) => Number(m[1]) * TO_PX[m[2]] < 12,
  ).length
results['tiny-font'] = files.reduce((n, f) => n + tinyFont(f.text), 0) + tinyFont(homepageCss)

// 2. 浅色底上对比度不足的文字色（slate/gray 300、400）。
// 深色面板上 slate-300/400 是正确的，跳过那些行，否则这条规则的 target 永远到不了 0，
// 数字也不再表示"有多少处读不清"。
const DARK_SURFACE = /bg-(?:slate|accent)-(?:800|900|950)|bg-gradient|from-[a-z]+-(?:800|900|950)|text-white|bg-black|bg-white\//
// 底色常写在父元素上，往前看几行；看不到深色底才算数。
results['low-contrast-text'] = files.reduce((n, f) => {
  const lines = f.text.split('\n')
  return n + lines.filter(
    (l, i) => /text-(?:slate|gray)-(?:300|400)\b/.test(l) && !lines.slice(Math.max(0, i - 5), i + 1).some((x) => DARK_SURFACE.test(x)),
  ).length
}, 0)

// 3. 同一文件混用 slate 与 gray 两套中性色阶。
results['neutral-ramp-mix'] = files.filter((f) => /-slate-/.test(f.text) && /-gray-/.test(f.text)).length

// 4. 卡片外壳的圆角档数。Shape Consistency Lock：一页一套圆角。
// 卡片是容器（p-N 内边距），带 px-/py- 的是控件，两者不共用一档圆角，也不该放进同一条规则。
const radii = distinct(/rounded-(none|sm|md|lg|xl|2xl|3xl|full)\s+border\s+border-slate-200\s+bg-white(?=[^"'`]*\bp-\d)/g)
results['card-radius-variants'] = radii.size
detail['card-radius-variants'] = [...radii].sort().join(', ')

// 5. 焦点环色相档数。可达性依赖单一可预测的焦点样式。
const rings = distinct(/focus(?:-visible)?:ring-([a-z]+)-\d{2,3}\b/g)
results['focus-ring-hues'] = rings.size
detail['focus-ring-hues'] = [...rings].sort().join(', ')

// 6. 实心按钮底色相档数。Color Consistency Lock：primary / danger / neutral 三个语义。
const solids = distinct(/bg-([a-z]+)-(?:600|700|800|900|950)\b/g)
results['solid-button-hues'] = solids.size
detail['solid-button-hues'] = [...solids].sort().join(', ')

// 7. 页面级背景色的声明处数。多于一处就会互相覆盖。
const indexCss = readFileSync(join(SRC, 'index.css'), 'utf8')
const appTsx = readFileSync(join(SRC, 'App.tsx'), 'utf8')
results['page-bg-declarations'] =
  (/body\s*\{[^}]*bg-/.test(indexCss) ? 1 : 0) + (appTsx.match(/className="min-h-screen bg-\S+"/g)?.length ?? 0)

// 8. 英文界面文案里的破折号（准则第 11 节，来自 skill 第 9.G 条）。三种合法用法不计入：
//    中文 ——（GB/T 15834）、表格空值占位符 '—'、区间分隔符（起止日期、上下界）。
//    代码注释不是界面文案，也不计入。
const RANGE = /(?:\}|[\w.%)\]])\s*[—–]\s*(?:\{|[\w.$'"`])/g
let dashes = 0
for (const f of files) {
  const stripped = f.text.replace(/['"`]\s*[—–]\s*['"`]/g, '').replace(RANGE, '')
  for (const line of stripped.split('\n')) {
    if (/^\s*(?:\/\/|\/?\*)/.test(line)) continue
    for (const m of line.matchAll(/[—–]/g)) {
      if (!CJK.test(line.slice(Math.max(0, m.index - 25), m.index + 25))) dashes += 1
    }
  }
}
results['english-em-dash'] = dashes

// 9. 作用在中文上的 uppercase。对中文是空操作，只留下被拉开的字距。
results['uppercase-on-cjk'] = files.reduce(
  (n, f) => n + f.text.split('\n').filter((l) => l.includes('uppercase') && CJK.test(l)).length,
  0,
)

// 10-11. 首页作用域 CSS 的字号与字重。逐像素还原曾留下 5px/7.3px 与 450/550/710 等非标准值，现已清零，
// 这两条改为守住 12px 下限与 400/500/600/700 四档。
results['homepage-subpixel-font'] = (homepageCss.match(/font-size:\s*\d+(?:\.\d+)?px/g) ?? []).filter(
  (m) => parseFloat(m.replace(/[^\d.]/g, '')) < 12,
).length
results['homepage-nonstandard-weight'] = (homepageCss.match(/font-weight:\s*(\d+)/g) ?? []).filter(
  (m) => ![400, 500, 600, 700].includes(Number(m.replace(/\D/g, ''))),
).length

// 12. 没有 scope 的 <th>。逐个数，不按文件数：文件里只要有一处 scope 就全放过的话，
// 同一张表里漏标的表头永远查不出来。屏幕阅读器无法关联表头与单元格。
export const unscopedTables = (text) =>
  [...text.matchAll(/<th(?![a-z])([^>]*)>/g)].filter((m) => !/scope="(?:col|row)"/.test(m[1])).length
results['unscoped-tables'] = files.reduce((n, f) => n + unscopedTables(f.text), 0)

// 13. 散落的图表十六进制色。应集中到单一 ECharts 主题。
const hexes = distinct(/['"](#[0-9a-fA-F]{6})['"]/g)
results['bare-chart-hex'] = hexes.size

// 14. 首页强调色的饱和蓝变体数。同一意图只允许 3 档令牌（默认 / hover / 深底）。
const hls = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
  const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2
  if (max === min) return [0, 0, l]
  const d = max - min
  const sat = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  const h = max === r ? ((g - b) / d + (g < b ? 6 : 0)) : max === g ? (b - r) / d + 2 : (r - g) / d + 4
  return [h * 60, sat, l]
}
results['homepage-accent-variants'] = [...new Set(homepageCss.match(/#[0-9a-fA-F]{6}\b/g) ?? [])]
  .map((hex) => hls(hex.toLowerCase()))
  .filter(([h, sat, l]) => h >= 200 && h <= 250 && sat > 0.55 && l > 0.35 && l < 0.8).length

// 15. 吉祥物出现在禁区文件中。装饰形象不得靠近数值、风险提示与核算差错。
// EmptyState 默认带吉祥物，所以间接用法也要一起拦，否则禁区规则只挡得住直接引用。
const mascotFiles = files.filter((f) => /<Mascot[\s/>]/.test(f.text) || /<EmptyState(?![^>]*mascot=\{false\})/.test(f.text))
results['mascot-in-forbidden-zone'] = mascotFiles.filter((f) => MASCOT_FORBIDDEN.test(f.path)).length
detail['mascot-in-forbidden-zone'] = mascotFiles.length
  ? `使用处：${mascotFiles.map((f) => f.path.split('/').pop()).join(', ')}`
  : '尚未使用'

// 16. 同一个元素（或同一个三元分支）里，底色与自有文字色的真实对比度。
// 只查元素自身声明的底色：祖先的底色可能写在三元里，静态判定不了，
// 那一类由 frontend/e2e/contrast.spec.ts 在真实渲染结果上量。
// 色值内联在这里，是为了让本脚本保持零依赖、未装 node_modules 也能跑。
const PALETTE = {
  accent: ['#f0f6ff','#e5efff','#c7ddff','#9cc3ff','#5f9aff','#2d7bfa','#1662f5','#0f52d6','#0f43a8','#123a85','#0d2456'],
  slate: ['#f8fafc','#f1f5f9','#e2e8f0','#cbd5e1','#94a3b8','#64748b','#475569','#334155','#1e293b','#0f172a','#020617'],
  emerald: ['#ecfdf5','#d1fae5','#a7f3d0','#6ee7b7','#34d399','#10b981','#059669','#047857','#065f46','#064e3b','#022c22'],
  amber: ['#fffbeb','#fef3c7','#fde68a','#fcd34d','#fbbf24','#f59e0b','#d97706','#b45309','#92400e','#78350f','#451a03'],
  rose: ['#fff1f2','#ffe4e6','#fecdd3','#fda4af','#fb7185','#f43f5e','#e11d48','#be123c','#9f1239','#881337','#4c0519'],
  violet: ['#f5f3ff','#ede9fe','#ddd6fe','#c4b5fd','#a78bfa','#8b5cf6','#7c3aed','#6d28d9','#5b21b6','#4c1d95','#2e1065'],
  teal: ['#f0fdfa','#ccfbf1','#99f6e4','#5eead4','#2dd4bf','#14b8a6','#0d9488','#0f766e','#115e59','#134e4a','#042f2e'],
  orange: ['#fff7ed','#ffedd5','#fed7aa','#fdba74','#fb923c','#f97316','#ea580c','#c2410c','#9a3412','#7c2d12','#431407'],
  pink: ['#fdf2f8','#fce7f3','#fbcfe8','#f9a8d4','#f472b6','#ec4899','#db2777','#be185d','#9d174d','#831843','#500724'],
  red: ['#fef2f2','#fee2e2','#fecaca','#fca5a5','#f87171','#ef4444','#dc2626','#b91c1c','#991b1b','#7f1d1d','#450a0a'],
  green: ['#f0fdf4','#dcfce7','#bbf7d0','#86efac','#4ade80','#22c55e','#16a34a','#15803d','#166534','#14532d','#052e16'],
  yellow: ['#fefce8','#fef9c3','#fef08a','#fde047','#facc15','#eab308','#ca8a04','#a16207','#854d0e','#713f12','#422006'],
}
const STEPS = ['50','100','200','300','400','500','600','700','800','900','950']
const hexOf = (family, value) =>
  family === 'white' ? '#ffffff'
  : family === 'black' ? '#000000'
  : value !== undefined && PALETTE[family] ? PALETTE[family][STEPS.indexOf(value)]
  : undefined
const luminance = (h) => {
  const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}
// 带透明度的底色（bg-x-400/10）压在未知底上，不能当作有效底色。
// disabled: 变体不计：WCAG 1.4.3 把失效控件排除在对比度要求之外。
const BG_TOKEN = /(?<!disabled:)\b(?:bg|from|via|to)-(white|black|[a-z]+)(?:-(\d{2,3}))?(?![\w/-])/g
const TEXT_TOKEN = /(?<!disabled:)\btext-(white|black|[a-z]+)(?:-(\d{2,3}))?(?![\w/-])/g
let sameElement = 0
const sameElementDetail = []
for (const f of files) {
  for (const attr of f.text.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\}|\{([^}]*)\})/g)) {
    const cls = attr[1] ?? attr[2] ?? attr[3] ?? ''
    // 静态部分与每个三元分支各算各的，不互相配对。
    const segments = [cls.replace(/['"`][^'"`]*['"`]/g, ''), ...[...cls.matchAll(/['"`]([^'"`]*)['"`]/g)].map((m) => m[1])]
    for (const seg of segments) {
      const bgs = [...seg.matchAll(BG_TOKEN)].map((m) => hexOf(m[1], m[2])).filter(Boolean)
      if (!bgs.length) continue
      // 渐变取最暗处：最坏情况才是要守住的那个。
      const bg = bgs.reduce((a, b) => (luminance(a) < luminance(b) ? a : b))
      for (const m of seg.matchAll(TEXT_TOKEN)) {
        const fg = hexOf(m[1], m[2])
        if (!fg || contrast(bg, fg) >= 4.5) continue
        sameElement += 1
        const line = f.text.slice(0, attr.index).split('\n').length
        if (sameElementDetail.length < 8) sameElementDetail.push(`${f.path.split('/').pop()}:${line} ${m[0]} on ${bg} = ${contrast(bg, fg).toFixed(2)}`)
      }
    }
  }
}
results['same-element-contrast'] = sameElement
detail['same-element-contrast'] = sameElementDetail.join('\n' + ' '.repeat(32) + '↳ ')

// 17. 同一个字面量数组里重复的分类色。颜色在分类图里承载"是哪一类"，
// 两类同色时堆叠条上相邻的两段会连成一段（52% + 28% 被读成 80%）。
// 只扫数组里直写的颜色串是不够的：颜色一旦提到 ASSET_CLASS_COLORS 这类映射里，
// 数组里剩下的是 ASSET_CLASS_COLORS.equity 这样的引用，重复就查不出来了。
// 所以两头都查：映射自身的取值要互不相同，数组里的引用也要互不相同。
const CATEGORY_KEY = /(?:color|tone|fill|swatch)\s*:\s*/
export const duplicateCategoryColor = (text) => {
  const found = []
  const dupsOf = (used) => [...new Set(used.filter((c, i) => used.indexOf(c) !== i))]
  // a) 分类色映射：属性值恰好是一个 bg-* 类名的对象字面量。
  for (const literal of text.matchAll(/\{[^{}]*\}/g)) {
    const used = [...literal[0].matchAll(/:\s*'(bg-[a-z]+-\d{2,3})'\s*(?=[,}])/g)].map((m) => m[1])
    if (used.length >= 2) found.push(...dupsOf(used))
  }
  // b) 数组元素：直写的颜色串，或对上面那种映射的引用。
  for (const literal of text.matchAll(/\[[^[\]]*\]/g)) {
    for (const re of [
      new RegExp(CATEGORY_KEY.source + "'(bg-[a-z]+-\\d{2,3})'", 'g'),
      new RegExp(CATEGORY_KEY.source + '([A-Z][A-Za-z0-9_]*(?:\\.[A-Za-z0-9_]+)+)', 'g'),
    ]) {
      const used = [...literal[0].matchAll(re)].map((m) => m[1])
      if (used.length >= 2) found.push(...dupsOf(used))
    }
  }
  return found
}
let dupCategory = 0
const dupDetail = []
for (const f of files) {
  const dup = duplicateCategoryColor(f.text)
  dupCategory += dup.length
  if (dup.length && dupDetail.length < 4) dupDetail.push(`${f.path.split('/').pop()} ${dup.join(' ')}`)
}
results['duplicate-category-color'] = dupCategory
detail['duplicate-category-color'] = dupDetail.join('；')

// 规则自检。这三条都曾经"通过"过真实缺陷：tiny-font 漏掉没有整数位的 .5rem，
// duplicate-category-color 只认数组里直写的颜色串，unscoped-tables 曾按文件计数。
// 反例留在这里，规则退化时先在这里炸，而不是等复核时再被人找出来。
const SELFTEST = [
  ['tiny-font 认得没有整数位的小数', () => tinyFont('<p className="text-[.5rem]">过小</p>'), 1],
  ['tiny-font 认得 px 与 rem', () => tinyFont('text-[8px] text-[0.5rem] text-[12px] text-[1rem]'), 2],
  ['duplicate-category-color 认得分类色映射自身的重复', () => duplicateCategoryColor(
    "const ASSET_CLASS_COLORS = { fixedIncome: 'bg-accent-600', equity: 'bg-accent-600' }").length, 1],
  ['duplicate-category-color 认得数组里对映射的重复引用', () => duplicateCategoryColor(
    "const allocation = [{ label: '固收', color: C.fixedIncome }, { label: '权益', color: C.fixedIncome }]").length, 1],
  ['duplicate-category-color 不误报互不相同的映射', () => duplicateCategoryColor(
    "const C = { a: 'bg-accent-600', b: 'bg-violet-500' }").length, 0],
  ['unscoped-tables 逐个 th 计数', () => unscopedTables('<th scope="col">A</th><th>B</th><th>C</th>'), 2],
]
const selftestFailures = SELFTEST.filter(([, run, want]) => run() !== want)
if (selftestFailures.length) {
  console.error('规则自检未通过，检查器本身已失效：')
  for (const [name, run, want] of selftestFailures) console.error(`  ${name}：期望 ${want}，实得 ${run()}`)
  process.exit(2)
}

let failed = 0
const rows = Object.entries(BUDGET).map(([rule, { budget, target }]) => {
  const current = results[rule]
  const state = current > budget ? 'REGRESSION' : current > target ? 'over-target' : 'ok'
  if (state === 'REGRESSION') failed += 1
  return { rule, current, budget, target, state }
})

const pad = (s, n) => String(s).padEnd(n)
console.log(`${pad('rule', 30)}${pad('current', 9)}${pad('budget', 8)}${pad('target', 8)}state`)
for (const r of rows) {
  console.log(`${pad(r.rule, 30)}${pad(r.current, 9)}${pad(r.budget, 8)}${pad(r.target, 8)}${r.state}`)
  if (detail[r.rule]) console.log(`${' '.repeat(30)}↳ ${detail[r.rule]}`)
}

const darkVariants = count(/\bdark:/g)
console.log(`\n参考指标（不判定）：dark: 变体 ${darkVariants} 处，tabular-nums ${count(/tabular-nums/g)} 处，表格 ${count(/<table[\s>]/g)} 处`)

if (failed) {
  console.error(`\n${failed} 项相对 budget 回归。收敛实现或在本脚本中下调 budget，不要上调。`)
  process.exit(1)
}
console.log('\n无回归。')
