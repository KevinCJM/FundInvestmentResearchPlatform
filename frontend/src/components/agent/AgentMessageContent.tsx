import { memo } from 'react'
import Markdown, { type Components } from 'react-markdown'
import { fromMarkdown } from 'mdast-util-from-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './AgentMessageContent.css'

// The model also uses LaTeX's bracket delimiters. Let CommonMark identify code
// and links first, so examples and URLs are never rewritten as live formulas.
function normalizeMath(source: string): string {
  if (!/\\[\[(]/.test(source)) return source
  const ranges: Array<[number, number]> = []
  const tree = fromMarkdown(source)
  const stack: Array<typeof tree | (typeof tree.children)[number]> = [tree]
  while (stack.length) {
    const node = stack.pop()!
    if (['code', 'inlineCode', 'html', 'link', 'image', 'definition'].includes(node.type)) {
      const start = node.position?.start.offset, end = node.position?.end.offset
      if (start !== undefined && end !== undefined) ranges.push([start, end])
    } else if ('children' in node) stack.push(...node.children)
  }
  const convert = (value: string) => value.replace(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g, (match, display, inline, offset: number) => {
    let escapes = 0
    for (let i = offset - 1; i >= 0 && value[i] === '\\'; i--) escapes++
    if (escapes % 2) return match
    if (display !== undefined) {
      // Preserve Markdown container prefixes on existing lines (quotes/lists).
      return display.includes('\n') ? `$$${display}$$` : `\n\n$$\n${display.trim()}\n$$\n\n`
    }
    return `$$${inline.trim()}$$`
  })
  let cursor = 0, result = ''
  for (const [start, end] of ranges.sort((a, b) => a[0] - b[0])) {
    result += convert(source.slice(cursor, start)) + source.slice(start, end)
    cursor = end
  }
  return result + convert(source.slice(cursor))
}

const components: Components = {
  a: ({ node: _node, children, ...props }) => props.href ? <a {...props} target={/^(https?:)?\/\//i.test(props.href) ? '_blank' : undefined} rel="noopener noreferrer" className="text-accent-700 underline underline-offset-4 focus-visible:ring-2 focus-visible:ring-accent-500">{children}</a> : <span>{children}</span>,
  // Markdown text must not start background requests to model-supplied image URLs.
  img: ({ alt }) => alt ? <span className="text-slate-600">{alt}</span> : null,
  pre: ({ node: _node, children, ...props }) => <pre {...props} tabIndex={0} aria-label="代码块" className="max-w-full overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-6 text-slate-200 focus-visible:ring-2 focus-visible:ring-accent-500">{children}</pre>,
  table: ({ node: _node, children, ...props }) => <div tabIndex={0} role="region" aria-label="表格，可横向滚动" className="max-w-full overflow-x-auto focus-visible:ring-2 focus-visible:ring-accent-500"><table {...props} aria-label="回复中的表格" className="w-full border-collapse text-left text-sm">{children}</table></div>,
  th: ({ node: _node, ...props }) => <th {...props} scope="col" className="border-b border-slate-300 bg-slate-100 px-2 py-2 font-semibold" />,
  td: ({ node: _node, ...props }) => <td {...props} className="border-b border-slate-200 px-2 py-2" />,
  span: ({ node: _node, className, children, ...props }) => {
    if (className?.split(' ').includes('katex-error')) return <code className="break-words text-slate-700" title="公式暂无法排版，显示原文">{children}</code>
    if (className?.split(' ').includes('katex-display')) return <span {...props} className={`${className} max-w-full overflow-x-auto focus-visible:ring-2 focus-visible:ring-accent-500`} tabIndex={0} role="group" aria-label="数学公式，可横向滚动">{children}</span>
    return <span {...props} className={className}>{children}</span>
  },
}

const remarkPlugins = [remarkGfm, remarkMath]
const katexOptions = { trust: false, strict: 'ignore' as const, maxExpand: 1000, maxSize: 20 }

export default memo(function AgentMessageContent({ text }: { text: string }) {
  return <div className="agent-message-content min-w-0 max-w-full text-sm leading-6 text-slate-700">
    <Markdown skipHtml remarkPlugins={remarkPlugins} rehypePlugins={[[rehypeKatex, katexOptions]]} components={components}>
      {normalizeMath(text)}
    </Markdown>
  </div>
})
