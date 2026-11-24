import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Layout from './Layout'
import Overview from './screens/Overview'
import PatientDetail from './screens/PatientDetail'
import AlertCentre from './screens/AlertCentre'
import ModelPerformance from './screens/ModelPerformance'
import SystemAdmin from './screens/SystemAdmin'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="patients/:pid" element={<PatientDetail />} />
          <Route path="alerts" element={<AlertCentre />} />
          <Route path="model" element={<ModelPerformance />} />
          <Route path="system" element={<SystemAdmin />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  </StrictMode>,
)