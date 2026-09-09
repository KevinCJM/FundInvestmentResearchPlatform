import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import App from './App'
import { installPitOverrideFetch } from './services/pitOverride'

// Before the first render: every data request has to carry this tab's viewing
//口径, including the ones fired on mount.
installPitOverrideFetch()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

