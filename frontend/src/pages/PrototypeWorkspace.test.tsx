import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import PrototypeWorkspace from './PrototypeWorkspace'

const renderPrototype = (pageKey: string) => render(
  <MemoryRouter><PrototypeWorkspace pageKey={pageKey} /></MemoryRouter>,
)

describe('PrototypeWorkspace', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('明确标识静态演示且不会请求后端', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const user = userEvent.setup()
    renderPrototype('taa')

    expect(screen.getByText('静态功能演示｜未接入真实数据与后端服务')).toBeInTheDocument()
    expect(screen.getByText('预置示例数据')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '记录明细' }))
    await user.type(screen.getByPlaceholderText('输入关键词'), '权益')
    expect(screen.getByText('权益类资产')).toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('职责与口径内容以非功能区展示并默认收起', async () => {
    const user = userEvent.setup()
    renderPrototype('taa')

    const blueprint = screen.getByTestId('non-interactive-blueprint')
    expect(blueprint).not.toHaveAttribute('open')
    expect(screen.getByText('页面职责说明（非功能区）')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '本节点负责' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '本节点负责' })).not.toBeInTheDocument()

    await user.click(screen.getByText('页面职责说明（非功能区）'))
    expect(blueprint).toHaveAttribute('open')
    expect(screen.getByLabelText('研究口径说明（非交互）')).toBeInTheDocument()
  })

  it('会计静态节点展示独立的职责边界', () => {
    renderPrototype('portfolio-ledger')

    expect(screen.getByRole('heading', { name: '组合/基金总账与明细账' })).toBeInTheDocument()
    expect(screen.getAllByText(/借贷平衡/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/表外备查簿/).length).toBeGreaterThan(0)
  })

  it('管理人账页面明确只做公司账映射', () => {
    renderPrototype('manager-ledger')

    expect(screen.getByRole('heading', { name: '管理人公司账映射' })).toBeInTheDocument()
    expect(screen.getAllByText(/不自动写入公司总账/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/完整企业财务系统/).length).toBeGreaterThan(0)
  })

  it('研究口径与参数中心展示跨业务模板及清晰职责边界', () => {
    renderPrototype('research-parameters')

    expect(screen.getByRole('heading', { name: '研究口径与参数中心' })).toBeInTheDocument()
    expect(screen.getByText(/产品研究、资产配置、组合回测和投后分析/)).toBeInTheDocument()
    expect(screen.getByText(/业务页面只选择并引用模板/)).toBeInTheDocument()
  })
})
