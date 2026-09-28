import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import ResumeResearch, { RESUME_IDENTITY_KEYS, resumeLabelKey } from './ResumeResearch'
import { builtinCatalogs } from '../i18n/catalogs'

const at = (entry: string, to: string) => render(<MemoryRouter initialEntries={[entry]}><ResumeResearch to={to} /></MemoryRouter>)

describe('ResumeResearch', () => {
  it('地址栏缺身份时列出上次研究带着什么，并给出显式续接入口', () => {
    at('/pre-investment/saa/policy', '/pre-investment/saa/policy?mandate=m1&cma=cma-7')
    expect(screen.getByText(/投资目标与约束 · LTCMA 假设版本/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '按上次研究继续' })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=m1&cma=cma-7')
  })

  it('地址栏已经带齐同样的身份时不出现', () => {
    const view = at('/pre-investment/saa/policy?mandate=m1', '/pre-investment/saa/policy?mandate=m1')
    expect(view.container).toBeEmptyDOMElement()
  })

  it('recognizes repeated CMA identities without offering a redundant resume action', () => {
    const view = at('/pre-investment/saa/policy?cma=one&cma=two', '/pre-investment/saa/policy?cma=one&cma=two')
    expect(view.container).toBeEmptyDOMElement()
  })

  it('落点不是当前页时不出现，避免在别的步骤上提示续接', () => {
    const view = at('/pre-investment/taa', '/pre-investment/saa/policy?mandate=m1')
    expect(view.container).toBeEmptyDOMElement()
  })

  it('显示开关不算身份，不会为 scope=strategic 这类参数弹提示', () => {
    const view = at('/pre-investment/product-pool', '/pre-investment/product-pool?scope=strategic')
    expect(view.container).toBeEmptyDOMElement()
  })

  it('每个身份参数都有词条，动态取键不会退化成"名称待补充"', () => {
    for (const key of RESUME_IDENTITY_KEYS) expect(builtinCatalogs.system[resumeLabelKey(key)]).toBeTruthy()
  })
})
