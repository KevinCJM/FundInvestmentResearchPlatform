import { spawn } from 'node:child_process'
import { homedir, tmpdir } from 'node:os'
import { openSync, closeSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as delay } from 'node:timers/promises'

const root = fileURLToPath(new URL('../', import.meta.url))
const port = Number(process.env.FACTOR_PREVIEW_PORT || '8011')
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error('FACTOR_PREVIEW_PORT must be an integer between 1024 and 65535')
}
const base = 'http://127.0.0.1:' + port

async function probe() {
  try {
    const response = await fetch(base + '/api/factor-research/catalog', {
      signal: AbortSignal.timeout(2_000),
    })
    if (!response.ok) return 'occupied'
    const catalog = await response.json()
    return catalog.ready === true && catalog.execution?.python_fallback === 0 ? 'ready' : 'occupied'
  } catch (error) {
    return error?.cause?.code === 'ECONNREFUSED' ? 'absent' : 'occupied'
  }
}

const initial = await probe()
if (initial === 'ready') {
  console.log('Factor Research Center: ' + base + '/settings/factor-research')
} else {
  if (initial !== 'absent') {
    throw new Error('The preview port is occupied or not ready. Choose a different FACTOR_PREVIEW_PORT.')
  }
  const python = process.env.TEST_PYTHON ||
    join(homedir(), 'Desktop', 'myenv_312', 'bin', 'python3.12')
  const logPath = join(tmpdir(), 'factor-preview-' + process.pid + '.log')
  const logFd = openSync(logPath, 'wx', 0o600)
  const child = spawn(python, ['-m', 'uvicorn', 'backend.app:app',
    '--host', '127.0.0.1', '--port', String(port)], {
    cwd: root, detached: true, stdio: ['ignore', logFd, logFd],
    env: { ...process.env, INDICATOR_PROCESS_WORKERS: process.env.INDICATOR_PROCESS_WORKERS || '2' },
  })
  closeSync(logFd)
  let failure
  child.on('error', error => { failure = error })
  child.on('exit', (code, signal) => { failure = new Error('Preview exited: ' + (signal || code)) })
  child.unref()
  let ready = false
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (failure) throw new Error(failure.message + '\n' + readFileSync(logPath, 'utf8').slice(-8_000))
    await delay(1_000)
    if (await probe() === 'ready') { ready = true; break }
  }
  if (!ready) {
    // Stop only the process this invocation owns; do not disturb an existing server.
    child.kill('SIGTERM')
    throw new Error('Preview did not become ready. Run the documented uvicorn command to inspect startup logs.')
  }
  console.log('Factor Research Center: ' + base + '/settings/factor-research')
  console.log('Local preview PID: ' + child.pid)
  console.log('Startup log: ' + logPath)
}
