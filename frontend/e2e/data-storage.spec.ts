import { test, expect } from '@playwright/test'

test('storage settings remain operable at 320/768/1440 and never auto-migrate', async ({ page }, info) => {
  let saves = 0
  const target = '/Volumes/研究 磁盘/FundResearchData'
  const status = { revision: 0, online: true, error: null as string | null, logical_path: '/project/data', actual_path: '/project/data', active: null,
    pending: null as Record<string, unknown> | null, editing_enabled: true, free_bytes: 3 * 1024 ** 3, total_bytes: 500 * 1024 ** 3,
    volumes: [{ name: '研究 磁盘', path: '/Volumes/研究 磁盘', free_bytes: 100 * 1024 ** 3 }] }
  // This is presentation/interaction acceptance, not a real disk migration.
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-storage') return route.fulfill({ json: status })
    if (path === '/api/data-storage/probe') return route.fulfill({ json: {
      target, mount: '/Volumes/研究 磁盘', free_bytes: 100 * 1024 ** 3, total_bytes: 1000 * 1024 ** 3,
      reserve_bytes: 2 * 1024 ** 3, same_device: false, message: '目录能力检查通过',
    } })
    if (path === '/api/data-storage/plan') {
      expect(route.request().postDataJSON()).toEqual({ path: target, expected_revision: 0, confirm: true })
      saves += 1
      status.revision = 1
      status.pending = { id: 'fixture', target, mount: '/Volumes/研究 磁盘', phase: 'PLANNED', files: 0, bytes: 0, message: '计划已保存，未迁移' }
      return route.fulfill({ json: status })
    }
    if (path === '/api/data-sources/etl/runs' || path === '/api/data-sources/etl/workflows') return route.fulfill({ json: [] })
    return route.fulfill({ status: 503, json: { detail: { message: '测试隔离其他业务模块' } } })
  })
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/settings/data-sources')
  const panel = page.getByRole('region', { name: '数据存储位置' })
  await expect(panel.getByText('3.00 GiB / 500.00 GiB')).toBeVisible()
  await panel.getByText('设置外接磁盘或其他目录').click()
  await panel.getByLabel('已挂载磁盘', { exact: true }).selectOption('/Volumes/研究 磁盘')
  await expect(panel.getByLabel('目标绝对目录', { exact: true })).toHaveValue(target)
  await panel.getByRole('button', { name: '检查目录', exact: true }).click()
  await expect(panel.getByRole('button', { name: '保存迁移计划', exact: true })).toBeDisabled()
  expect(saves).toBe(0)
  await panel.getByLabel('确认保存迁移计划', { exact: true }).focus()
  await page.keyboard.press('Space')
  await panel.getByRole('button', { name: '保存迁移计划', exact: true }).click()
  await expect(panel.getByText('./start_services.sh restart', { exact: true })).toBeVisible()
  expect(saves).toBe(1)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await panel.screenshot({ path: info.outputPath('storage-plan.png') })
  status.online = false; status.error = '数据磁盘未挂载'; status.pending = null
  await page.reload()
  await expect(panel.getByRole('alert')).toContainText('不会自动写回本机旧副本')
  await panel.getByText('设置外接磁盘或其他目录').click()
  await expect(panel.getByLabel('目标绝对目录', { exact: true })).toBeDisabled()
  expect(errors).toEqual([])
})

test('attach existing storage requires explicit shared-data confirmation without migration', async ({ page }, info) => {
  let attaches = 0
  const target = '/Volumes/研究 磁盘/TushareData'
  const id = '1234567890abcdef1234567890abcdef'
  const status = { revision: 0, online: true, error: null, logical_path: '/new-project/data', active: null,
    pending: null as Record<string, unknown> | null, editing_enabled: true,
    free_bytes: 50 * 1024 ** 3, total_bytes: 500 * 1024 ** 3, volumes: [] }
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-storage') return route.fulfill({ json: status })
    if (path === '/api/data-storage/existing/probe') return route.fulfill({ json: {
      id, target, mount: '/Volumes/研究 磁盘', free_bytes: 50 * 1024 ** 3, total_bytes: 500 * 1024 ** 3,
      message: '已有数据目录校验通过，不复制、不下载。',
    } })
    if (path === '/api/data-storage/existing/plan') {
      expect(route.request().postDataJSON()).toEqual({ path: target, expected_revision: 0, expected_id: id, confirm: true })
      attaches += 1
      status.revision = 1
      status.pending = { id: 'operation', operation: 'attach', storage_id: id, target, mount: '/Volumes/研究 磁盘', phase: 'PLANNED', files: 0, bytes: 0, message: '等待本项目重启前接入' }
      return route.fulfill({ json: status })
    }
    expect(path).not.toBe('/api/data-storage/plan')
    if (path === '/api/data-sources/etl/runs' || path === '/api/data-sources/etl/workflows') return route.fulfill({ json: [] })
    return route.fulfill({ status: 503, json: { detail: { message: '隔离其他模块' } } })
  })
  await page.goto('/settings/data-sources')
  const panel = page.getByRole('region', { name: '数据存储位置' })
  await panel.getByText('设置外接磁盘或其他目录').click()
  await panel.getByLabel('存储操作方式').selectOption('attach')
  await panel.getByLabel('目标绝对目录', { exact: true }).fill(target)
  await expect(panel.getByText(/共用整个数据区，包括行情、配置、研究记录和本地凭据/)).toBeVisible()
  await panel.getByRole('button', { name: '检查已有目录', exact: true }).click()
  await expect(panel.getByRole('button', { name: '保存接入计划' })).toBeDisabled()
  expect(attaches).toBe(0)
  await panel.getByLabel('确认共用已有数据目录').focus()
  await page.keyboard.press('Space')
  await panel.getByRole('button', { name: '保存接入计划' }).click()
  await expect(panel.getByText('./start_services.sh restart', { exact: true })).toBeVisible()
  expect(attaches).toBe(1)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await panel.screenshot({ path: info.outputPath('storage-attach.png') })
})
