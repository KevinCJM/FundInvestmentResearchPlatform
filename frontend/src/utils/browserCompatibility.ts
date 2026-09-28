// LAN HTTP lacks randomUUID, while getRandomValues remains available.
// Install before the app imports so draft IDs and operation keys share UUID v4.
const browserCrypto = globalThis.crypto
if (browserCrypto && typeof browserCrypto.randomUUID !== 'function'
    && typeof browserCrypto.getRandomValues === 'function') {
  browserCrypto.randomUUID = () => {
    const bytes = browserCrypto.getRandomValues(new Uint8Array(16))
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  }
}

export {}
