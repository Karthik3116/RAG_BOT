import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react' // if using React

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})
