/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        slate: {
          50: '#f8f6f2',
          100: '#24231f',
          200: '#302e29',
          300: '#45413a',
          400: '#615c53',
          500: '#726c62',
          600: '#81796d',
          700: '#d8d2c8',
          800: '#e9e5dd',
          900: '#ffffff',
          950: '#f8f6f2',
        },
        blue: {
          300: '#8f4e39',
          400: '#a25840',
          500: '#b76449',
          600: '#a9553c',
        },
        cyan: {
          400: '#a25840',
        },
      },
    },
  },
  plugins: [],
}

