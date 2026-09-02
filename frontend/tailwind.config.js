/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: { fontFamily: { sans: ['Nunito', 'ui-sans-serif', 'system-ui', 'sans-serif'] } },
  },
  plugins: [],
};