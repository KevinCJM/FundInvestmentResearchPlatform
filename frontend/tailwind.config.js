/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      // SAA/TAA controls share the existing homepage brand palette.
      colors: {
        accent: {
          50: '#f0f6ff', 100: '#e5efff', 200: '#c7ddff', 300: '#9cc3ff',
          400: '#5f9aff', 500: '#2d7bfa', 600: '#1662f5', 700: '#0f52d6',
          800: '#0f43a8', 900: '#123a85', 950: '#0d2456', DEFAULT: '#1662f5',
        },
      },
    },
  },
  plugins: [],
}
