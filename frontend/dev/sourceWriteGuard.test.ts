// @vitest-environment node
import type { IncomingMessage, ServerResponse } from 'node:http'
import { describe, expect, it, vi } from 'vitest'
import { guardSourceWrites, isLoopbackPeer, localSourceWrites } from './sourceWriteGuard'

function request(peer: string | undefined, method = 'POST', url = '/api/data-sources/etl/runs', headers = {}) {
  const end = vi.fn()
  const writeHead = vi.fn(() => ({ end }))
  const next = vi.fn()
  const req = { socket: { remoteAddress: peer }, method, url, headers } as IncomingMessage
  guardSourceWrites(req, { writeHead } as unknown as ServerResponse, next)
  return { end, writeHead, next }
}

describe('source write proxy boundary', () => {
  it.each(['configureServer', 'configurePreviewServer'] as const)('installs the same guard before the %s proxy', hook => {
    const use = vi.fn()
    const plugin = localSourceWrites()
    const configure = plugin[hook]
    if (typeof configure !== 'function') throw new Error('Expected a direct server hook')
    configure.call({} as never, { middlewares: { use } } as never)
    expect(use).toHaveBeenCalledWith(guardSourceWrites)
  })
  it.each(['127.0.0.1', '127.0.0.2', '::1', '::ffff:127.0.0.1'])('allows a real loopback peer %s', peer => {
    expect(isLoopbackPeer(peer)).toBe(true)
    expect(request(peer).next).toHaveBeenCalledOnce()
  })

  it.each(['192.0.2.1', '::ffff:192.0.2.1', '2001:db8::1', undefined])('rejects a non-local or unknown peer %s', peer => {
    const result = request(peer)
    expect(result.writeHead).toHaveBeenCalledWith(403, expect.any(Object))
    expect(result.next).not.toHaveBeenCalled()
  })

  it('ignores forged origin, forwarded, and client-IP headers', () => {
    const result = request('192.0.2.1', 'PUT', '/api/data-sources/credentials/tushare', {
      origin: 'http://127.0.0.1:5173', 'sec-fetch-site': 'same-origin',
      'x-forwarded-for': '127.0.0.1', 'x-real-ip': '::1', forwarded: 'for=127.0.0.1',
    })
    expect(result.writeHead).toHaveBeenCalledWith(403, expect.any(Object))
    expect(result.next).not.toHaveBeenCalled()
  })

  it.each(['/api/data-sources/etl/runs?confirm=true', '/api/data%2Dsources/etl/runs',
           '/api/data-sources%2Fetl/runs', '/api/data-sources'])('protects query and encoded paths %s', path => {
    expect(request('192.0.2.1', 'POST', path).next).not.toHaveBeenCalled()
  })

  it.each(['GET', 'HEAD', 'OPTIONS'])('preserves read-only source requests %s', method => {
    expect(request('192.0.2.1', method).next).toHaveBeenCalledOnce()
  })

  it('preserves unrelated business proxy routes', () => {
    expect(request('192.0.2.1', 'POST', '/api/indicators/preview').next).toHaveBeenCalledOnce()
  })
})
