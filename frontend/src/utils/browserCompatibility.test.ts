import { afterEach, expect, it, vi } from 'vitest'

const randomBytes = globalThis.crypto.getRandomValues.bind(globalThis.crypto)

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules() })

it('keeps the native UUID implementation in secure contexts', async () => {
  const native = vi.fn(() => 'native-id')
  const random = vi.fn()
  vi.stubGlobal('crypto', { randomUUID: native, getRandomValues: random })
  await import('./browserCompatibility')
  expect(crypto.randomUUID).toBe(native)
  expect(random).not.toHaveBeenCalled()
})

it('supplies distinct UUID v4 values using Web Crypto when LAN HTTP has no randomUUID', async () => {
  const random = vi.fn((bytes: Uint8Array) => randomBytes(bytes))
  vi.stubGlobal('crypto', { getRandomValues: random })
  await import('./browserCompatibility')
  const ids = Array.from({ length: 256 }, () => crypto.randomUUID())
  expect(new Set(ids).size).toBe(ids.length)
  for (const id of ids) expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  expect(random).toHaveBeenCalledTimes(ids.length)
  for (const [bytes] of random.mock.calls) expect(bytes).toHaveLength(16)
})

it('does not replace missing secure randomness with predictable IDs', async () => {
  vi.stubGlobal('crypto', {})
  await import('./browserCompatibility')
  expect(crypto.randomUUID).toBeUndefined()
})
