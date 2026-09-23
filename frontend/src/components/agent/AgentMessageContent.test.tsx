import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import AgentMessageContent from './AgentMessageContent'

afterEach(cleanup)

const rollingFormula = String.raw`\text{Rolling Sharpe}_t = \frac{\operatorname{mean}(r_{t-N+1:t}-r_f)}{\operatorname{std}(r_{t-N+1:t}-r_f)}\times\sqrt{A}`

describe('AgentMessageContent', () => {
  it('渲染截图中的加粗、区块公式、行内公式和列表', () => {
    const { container } = render(<AgentMessageContent text={String.raw`滚动夏普比率用**最近 N 个收益观察期**计算。

\[
${rollingFormula}
\]

其中 \(r_t\) 是收益率。

- 窗口可变
- 保留原始口径`} />)
    expect(container.querySelector('strong')).toHaveTextContent('最近 N 个收益观察期')
    expect(container.querySelectorAll('.katex')).toHaveLength(2)
    expect(container.querySelector('.katex-display annotation')).toHaveTextContent(rollingFormula)
    expect(container.querySelectorAll('li')).toHaveLength(2)
    expect(screen.getByRole('group', { name: '数学公式，可横向滚动' })).toHaveAttribute('tabindex', '0')
  })

  it('支持美元符号定界符，转义后的金额保留原样', () => {
    const { container } = render(<AgentMessageContent text={String.raw`行内 $N=20$ 和 \(A=252\)。金额 \$100 与 \$200。

$$
\sigma=\sqrt{x}
$$`} />)
    expect(container.querySelectorAll('.katex')).toHaveLength(3)
    expect(container.querySelectorAll('.katex-display')).toHaveLength(1)
    expect(container).toHaveTextContent('金额 $100 与 $200。')
  })

  it('代码围栏、引用中的代码及行内代码不转换数学标记', () => {
    const source = String.raw`\[\frac{1}{2}\]`
    const { container } = render(<AgentMessageContent text={[
      '```latex', source, '```', '', '`' + source + '`', '', '> ~~~text', '> ' + source, '> ~~~', '', '```text', String.raw`\(unfinished\)`,
    ].join('\n')} />)
    expect(container.querySelector('.katex')).toBeNull()
    const codes = [...container.querySelectorAll('code')].map(node => node.textContent?.trim())
    expect(codes).toEqual([source, source, source, String.raw`\(unfinished\)`])
  })

  it('表格、标题、引用、链接和行内代码使用 Markdown 结构', () => {
    const { container } = render(<AgentMessageContent text={'### 参数说明\n\n> 示例说明\n\n| 参数 | 含义 |\n| --- | --- |\n| `window` | 窗口 |\n\n[文档](https://example.com/docs)'} />)
    expect(screen.getByRole('heading', { name: '参数说明' })).toBeInTheDocument()
    expect(screen.getAllByRole('columnheader').every(node => node.getAttribute('scope') === 'col')).toBe(true)
    expect(screen.getByRole('table')).toHaveAccessibleName('回复中的表格')
    expect(screen.getByRole('link', { name: '文档' })).toHaveAttribute('rel', 'noopener noreferrer')
    expect(container.querySelector('blockquote')).toHaveTextContent('示例说明')
    expect(container.querySelector('code')).toHaveTextContent('window')
  })

  it('错误公式降级显示源码，其他内容继续渲染', () => {
    const { container } = render(<AgentMessageContent text={String.raw`\[\frac{1}{\]

**仍可阅读后文**`} />)
    expect(screen.getByTitle('公式暂无法排版，显示原文')).toHaveTextContent(String.raw`\frac{1}{`)
    expect(container.querySelector('strong')).toHaveTextContent('仍可阅读后文')
  })

  it('不执行 HTML、危险链接、公式中的受信命令或远程图片', () => {
    const { container } = render(<AgentMessageContent text={String.raw`<script>alert('x')</script>

<img src="https://example.com/track" onerror="alert('x')" />

[危险](javascript:alert%281%29)

![说明图](https://example.com/track.png)

\(\href{javascript:alert(1)}{x}\)`} />)
    expect(container.querySelector('script,img,iframe')).toBeNull()
    expect([...container.querySelectorAll('a')].some(link => link.getAttribute('href')?.toLowerCase().includes('javascript:'))).toBe(false)
    expect(container).toHaveTextContent('说明图')
  })

  it('显式转义的分隔符与链接地址不被改写', () => {
    const { container } = render(<AgentMessageContent text={String.raw`\\[原样显示\\]

[文档](https://example.com/\(manual\))`} />)
    expect(container.querySelector('.katex')).toBeNull()
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com/(manual)')
  })

  it('引用和列表的缩进不会混入公式，改变数学内容', () => {
    const { container } = render(<AgentMessageContent text={String.raw`> 公式：
>
> \[
> x^2
> \]

1. 公式：

   \[
   y^2
   \]`} />)
    expect([...container.querySelectorAll('annotation')].map(node => node.textContent)).toEqual(['x^2', 'y^2'])
    expect(container.querySelector('blockquote .katex-display')).not.toBeNull()
    expect(container.querySelector('li .katex-display')).not.toBeNull()
  })
})
