/** @type {import('tailwindcss').Config} */
// 令牌层。准则见 docs/frontend-design-guidelines.md，回归检查见 scripts/check_frontend_design.mjs。
// 这里只放跨页面共享的语义令牌；不要把一次性数值写进来。
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  // 深色模式尚未实现（dark: 变体 0 处）。先声明策略，避免后续两种方案并存。
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        // 中文不用 webfont：一份 CJK 字体 3-8MB，会直接毁掉 LCP。
        // 与 homepage.css 的 .quant-homepage 声明保持同一份字体栈。
        sans: [
          'Avenir Next', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI',
          'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', 'sans-serif',
        ],
      },
      colors: {
        // 全站唯一强调色，围绕首页 --home-blue #1662f5 展开的完整色阶。
        // 白底正文只用 600 及更深：600 = 5.12:1，700 = 6.57:1，均过 WCAG AA。
        // 500 = 4.03:1 只可用于 18px 以上大字与非文字元素；400 及更浅仅作描边与底色。
        accent: {
          50: '#f0f6ff',
          100: '#e5efff',
          200: '#c7ddff',
          300: '#9cc3ff',
          400: '#5f9aff',
          500: '#2d7bfa',
          600: '#1662f5',
          700: '#0f52d6',
          800: '#0f43a8',
          900: '#123a85',
          950: '#0d2456',
          // DEFAULT 供 index.css 的全局焦点环使用；其余一律走数字档。
          DEFAULT: '#1662f5',
        },
      },
    },
  },
  plugins: [],
}
