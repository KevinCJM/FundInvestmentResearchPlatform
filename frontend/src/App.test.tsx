import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('渲染主导航链接', () => {
    render(<App />)
    expect(screen.getByRole('link', { name: '主界面' })).toBeInTheDocument()
  })
})

