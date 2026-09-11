import type { IncomingMessage, ServerResponse } from 'node:http'
import { isIP } from 'node:net'
import type { Plugin } from 'vite'

export function isLoopbackPeer(address: string | undefined): boolean {
  if (address === '::1') return true
  const ipv4 = address?.startsWith('::ffff:') ? address.slice(7) : address
  return !!ipv4 && isIP(ipv4) === 4 && ipv4.split('.')[0] === '127'
}

export function guardSourceWrites(req: IncomingMessage, res: ServerResponse, next: () => void): void {
  let pathname: string
  try {
    pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://localhost').pathname)
  } catch {
    res.writeHead(400).end('Invalid request path')
    return
  }
  const isSourcePath = pathname === '/api/data-sources' || pathname.startsWith('/api/data-sources/')
  if (isSourcePath && !['GET', 'HEAD', 'OPTIONS'].includes(req.method ?? '') && !isLoopbackPeer(req.socket.remoteAddress)) {
    res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8' }).end(JSON.stringify({
      detail: { code: 'SOURCE_LOCAL_ONLY', message: '数据源配置和任务操作仅允许本机访问。' },
    }))
    return
  }
  next()
}

export function localSourceWrites(): Plugin {
  return {
    name: 'local-source-writes',
    configureServer(server) {
      // Runs before Vite's proxy: the backend sees the proxy's loopback socket,
      // so the original peer must be checked here, never from forwarded headers.
      server.middlewares.use(guardSourceWrites)
    },
    configurePreviewServer(server) {
      server.middlewares.use(guardSourceWrites)
    },
  }
}
