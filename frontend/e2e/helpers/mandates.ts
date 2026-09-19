import { expect, type APIRequestContext } from '@playwright/test'

/** Supported historical inputs are seeded through the real API; the compact UI has its own suite. */
export async function saveMandateFixture(request: APIRequestContext, api: string, study: unknown) {
  const response = await request.post(`${api}/mandates/preview`, { data: study })
  expect(response.status(), await response.text()).toBe(200)
  const preview = await response.json()
  const saved = await request.post(`${api}/mandates/confirm`, { data: {
    request: study, preview_hash: preview.preview_hash, acknowledge_limits: true,
  } })
  expect(saved.status(), await saved.text()).toBe(201)
  return { preview, version: await saved.json() }
}
