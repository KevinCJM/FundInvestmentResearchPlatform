// Fix for rare npm issue where node_modules/.bin/vite points to ../dist/node/cli.js
// instead of ../vite/dist/node/cli.js, causing ERR_MODULE_NOT_FOUND.
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const binPath = resolve(process.cwd(), 'node_modules/.bin/vite')
try {
  const content = readFileSync(binPath, 'utf-8')
  const wrong = "../dist/node/cli.js"
  const right = "../vite/dist/node/cli.js"
  if (content.includes(wrong) && !content.includes(right)) {
    const fixed = content.replace(wrong, right)
    writeFileSync(binPath, fixed)
    console.log('[fix-vite-bin] patched .bin/vite path -> ../vite/dist/node/cli.js')
  }
} catch (e) {
  // silently ignore if file missing
}

