// Use react 19's create Root API 
// import global tailwind styles (index.css)




import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Older versions saved an API key in plain text under this key; it is no longer used.
localStorage.removeItem('mara_settings')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
