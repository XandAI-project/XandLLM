/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            color: "inherit",
            a: { color: "inherit" },
            strong: { color: "inherit" },
            code: { color: "inherit" },
          },
        },
      },
    },
  },
  plugins: [],
};
