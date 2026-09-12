import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { allStages } from '../src/app/processRegistry'

/**
 * 文字对比度只能在渲染结果上量。祖先的底色常写在三元里，
 * scripts/check_frontend_design.mjs 的 same-element-contrast 静态判定不了那一类，
 * 本用例负责补上：逐个元素向上找到真正生效的底色再算。
 *
 * 覆盖边界（不要把这份用例说成完整的 WCAG AA 验收）：
 * - 只量初始渲染状态。页签切换、展开、悬停、错误态后的颜色不在内。
 * - 渐变按声明出来的色标估算最坏值，不是逐像素采样；背景图与视频底不量。
 * - 本文件将 API 固定为离线错误态，只覆盖该状态的外壳；不能代表有数据的表格。
 * - 指标工作台的正常载入和页签切换另由 indicator-studio.spec.ts 使用固定夹具检查。
 *
 * 路由不手写。手写过一次，把 /portfolio-center/onboarding 写成了不存在的路径，
 * 页面跳回首页而用例照样通过。改成从 processRegistry 派生，并断言落地路由。
 */
const ROUTES = ['/', ...new Set(allStages.flatMap((stage) => [stage.path, ...stage.nodes.map((node) => node.path)]))]


// 所有外壳测试都使用固定离线错误态，禁止请求开发者的真实数据服务。
test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ status: 503, json: { detail: 'Offline contrast fixture' } }))
})

for (const route of ROUTES) {
  test(`${route} 的初始可见文字对比度检查`, async ({ page }) => {
    await page.goto(route)
    await page.waitForLoadState('networkidle')
    // 路由写错时页面会跳回首页，量到的是首页而不是目标页。
    expect(new URL(page.url()).pathname, `${route} 没有落在目标路由上`).toBe(route)
    const failures = await page.evaluate(auditTextContrast)
    expect(failures, failures.map((f) => `${f.ratio}:1 «${f.text}» ${f.color} — ${f.path}`).join('\n')).toEqual([])
  })
}

test('对比度算法覆盖可见 aria-hidden、组透明度及真实显示的占位符', async ({ page }) => {
  await page.setContent(`<style>body{background:white;color:black;font:14px sans-serif}input{background:white;color:black}input::placeholder{color:black;opacity:.5}#filled::placeholder{color:#eee}</style>
    <p aria-hidden="true" style="color:#ddd">可见但不读屏</p>
    <div style="opacity:.5;background:black;color:white"><span>半透明深色组</span></div>
    <input placeholder="需要检查的占位符" />
    <input id="filled" value="已有内容" placeholder="没有显示的占位符" />
    <button disabled style="color:#eee">禁用</button>
    <p style="opacity:0">完全透明</p>`)
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.map(item => item.text).sort()).toEqual(['可见但不读屏', '半透明深色组', '占位符：需要检查的占位符'].sort())
  expect(failures.find(item => item.text === '半透明深色组')!.ratio).toBeCloseTo(3.98, 1)
})
