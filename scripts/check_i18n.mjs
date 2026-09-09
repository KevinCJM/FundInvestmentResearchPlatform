#!/usr/bin/env node
/** Read-only catalog and explicit-reference gate; inventories unmigrated JSX separately. */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from '../frontend/node_modules/typescript/lib/typescript.js'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const errors = []
const load = name => {
  const filename = path.join(root, 'locales', `${name}.json`)
  const text = fs.readFileSync(filename, 'utf8')
  const source = ts.parseJsonText(filename, text)
  const visit = node => {
    if (ts.isObjectLiteralExpression(node)) {
      const names = new Set()
      for (const property of node.properties) {
        const key = property.name?.text
        if (names.has(key)) errors.push(`${name}: duplicate key ${key}`)
        names.add(key)
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return JSON.parse(text)
}
const navigation = load('navigation')
const catalogs = { system: { ...load('system'), ...Object.fromEntries(Object.entries(navigation).map(([key, entry]) => [`navigation.routes.${key}`, entry])) }, business: load('business') }
const placeholders = text => [...new Set([...text.matchAll(/\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}/g)].map(match => match[1]))].sort().join('|')
for (const [scope, entries] of Object.entries(catalogs)) {
  for (const [key, translations] of Object.entries(entries)) {
    if (!translations['zh-CN']) errors.push(`${scope}:${key}: missing default translation`)
    for (const [locale, text] of Object.entries(translations)) {
      if (!['zh-CN', 'en-US'].includes(locale) || typeof text !== 'string') errors.push(`${scope}:${key}: invalid language entry`)
      else if (placeholders(text) !== placeholders(translations['zh-CN'])) errors.push(`${scope}:${key}: placeholder mismatch in ${locale}`)
    }
  }
}
const walk = dir => fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
  const filename = path.join(dir, entry.name)
  return entry.isDirectory() ? walk(filename) : /\.(ts|tsx)$/.test(filename) && !/\.test\./.test(filename) ? [filename] : []
})
const legacy = []
let referenceCount = 0
for (const filename of walk(path.join(root, 'frontend', 'src'))) {
  const relative = path.relative(root, filename)
  const source = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true, filename.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS)
  let unconverted = 0
  const visit = node => {
    if (ts.isJsxText(node) && /[\u3400-\u9fff]/u.test(node.text)) unconverted++
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      const name = node.expression.text
      const scope = ['s', 'systemText'].includes(name) ? 'system' : ['b', 'businessText'].includes(name) ? 'business' : null
      const argument = node.arguments[0]
      if (scope && argument && ts.isStringLiteral(argument) && /^[A-Za-z][A-Za-z0-9_.-]*$/.test(argument.text)) {
        referenceCount++
        if (!Object.hasOwn(catalogs[scope], argument.text)) errors.push(`${relative}:${source.getLineAndCharacterOfPosition(node.getStart()).line + 1}: missing ${scope}:${argument.text}`)
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  if (unconverted) legacy.push({ file: relative, jsx_text_fragments: unconverted })
}
console.log(JSON.stringify({
  valid: errors.length === 0,
  catalogs: Object.fromEntries(Object.entries(catalogs).map(([scope, values]) => [scope, Object.keys(values).length])),
  checked_literal_references: referenceCount,
  legacy_jsx_inventory: { files: legacy.length, fragments: legacy.reduce((sum, item) => sum + item.jsx_text_fragments, 0), largest: legacy.sort((a, b) => b.jsx_text_fragments - a.jsx_text_fragments).slice(0, 12), note: 'Source inventory only; user text, prototypes and untranslated legacy UI require separate review.' },
  errors,
}, null, 2))
process.exitCode = errors.length ? 1 : 0
