import { defineConfig } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
const python=process.env.INDICATOR_TEST_PYTHON||'/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const data=process.env.IMPLEMENTATION_E2E_ROOT||mkdtempSync('/private/tmp/implementation-e2e-')
process.env.IMPLEMENTATION_E2E_ROOT=data
delete process.env.ALL_PROXY
export default defineConfig({
  testDir:'./e2e',testMatch:'implementation.spec.ts',workers:1,fullyParallel:false,timeout:90000,reporter:[['list']],outputDir:data+'/browser',
  use:{baseURL:'http://127.0.0.1:4330',channel:'chrome',headless:true,actionTimeout:15000,trace:'retain-on-failure'},
  projects:[{name:'desktop-1440',use:{viewport:{width:1440,height:1000}}},{name:'tablet-768',use:{viewport:{width:768,height:1000}}},{name:'mobile-320',use:{viewport:{width:320,height:900}}}],
  webServer:[
    {command:`cd .. && IMPLEMENTATION_TEST_DIR="${data}" PYTHONPATH=.:backend "${python}" -m uvicorn backend.tests.implementation_app:app --host 127.0.0.1 --port 8130`,url:'http://127.0.0.1:8130/api/test/implementation-candidate',reuseExistingServer:false,timeout:180000},
    {command:'VITE_API_TARGET=http://127.0.0.1:8130 npm run dev -- --host 127.0.0.1 --port 4330 --strictPort',url:'http://127.0.0.1:4330',reuseExistingServer:false,timeout:120000},
  ],
})
