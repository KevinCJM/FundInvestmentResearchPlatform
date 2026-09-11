import { afterEach, describe, expect, it, vi } from 'vitest'
import { getRiskRun, getRiskImpact, publishRiskPreview, publishScenario, riskCatalog, runRiskImpact } from './riskModels'
import { impactFixture, riskFields, riskRunFixture, scenarioDraftFixture } from '../test/publishedRiskFixtures'

afterEach(() => { vi.unstubAllGlobals() })
function response(value: unknown, status = 200, type = 'application/json') {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(typeof value === 'string' ? value : JSON.stringify(value), { status, headers: { 'Content-Type': type } })))
}
describe('已发布研究客户端契约', () => {
  it('历史运行与压测结果读取也检查执行证明', async () => {
    response({ ...riskRunFixture, execution: { ...riskRunFixture.execution, python_fallback: 1 } })
    await expect(getRiskRun('product', riskRunFixture.id)).rejects.toThrow('固定签名 NJIT')
    response({ ...impactFixture, execution: {} })
    await expect(getRiskImpact(impactFixture.id)).rejects.toThrow('固定签名 NJIT')
  })
  it('精确提交两个发布版本，不发送拟合系数或模拟权重', async () => {
    response(impactFixture, 201)
    await runRiskImpact(impactFixture.request)
    expect(fetch).toHaveBeenCalledWith('/api/published-scenarios/impacts', expect.objectContaining({ method: 'POST', body: JSON.stringify(impactFixture.request) }))
  })
  it('确认发布时提交定义与预览哈希，预览本身不是磁盘ID', async () => {
    const hash = '1'.repeat(64)
    response({}, 201)
    await publishRiskPreview('product', riskFields, hash, 30, '')
    expect(fetch).toHaveBeenLastCalledWith('/api/risk-models/releases', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ definition: riskFields, preview_hash: hash, valid_days: 30, note: '', acknowledge_limitations: true }),
    }))
    response({}, 201)
    await publishScenario(scenarioDraftFixture, hash, 90, '')
    expect(fetch).toHaveBeenLastCalledWith('/api/published-scenarios/releases', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ definition: scenarioDraftFixture, preview_hash: hash, valid_days: 90, note: '', acknowledge_limitations: true }),
    }))
  })
  it('HTML错误页面不会伪装为研究数据', async () => {
    response('<html>error</html>', 502, 'text/html')
    await expect(riskCatalog('product')).rejects.toThrow('服务没有返回')
  })
  it('显示后端业务错误，不回退到假数据', async () => {
    response({ detail: { code: 'STORAGE_OFFLINE', message: '数据磁盘离线' } }, 503)
    await expect(riskCatalog('product')).rejects.toThrow('数据磁盘离线')
  })
})
