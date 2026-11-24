import { NavLink, Outlet } from 'react-router-dom'
import { useLiveData } from './useLiveData'

const link = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded-lg text-sm font-medium transition ${
    isActive ? 'bg-sky-500/20 text-sky-300' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`

export default function Layout() {
  const { connected } = useLiveData()
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-7xl px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-sky-400 to-blue-600 grid place-items-center font-black text-slate-950">
              SG
            </div>
            <div>
              <div className="font-bold leading-tight">SepsiGuard</div>
              <div className="text-[11px] text-slate-400 leading-tight">
                Early Sepsis Prediction · Clinical Decision Support
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1 ml-auto">
            <NavLink to="/" end className={link}>ICU Overview</NavLink>
            <NavLink to="/alerts" className={link}>Alert Centre</NavLink>
            <NavLink to="/model" className={link}>Model Performance</NavLink>
            <NavLink to="/system" className={link}>System Admin</NavLink>
          </nav>
          <div className="flex items-center gap-2 text-xs">
            <span className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
            <span className="text-slate-400">{connected ? 'live' : 'offline'}</span>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-7xl px-6 py-8">
        <Outlet />
      </div>
      <footer className="border-t border-slate-800 mt-12">
        <div className="mx-auto max-w-7xl px-6 py-4 text-[11px] text-slate-500 flex flex-wrap gap-x-6">
          <span>Research prototype — not a diagnostic or autonomous treatment system.</span>
          <span>Data source: historical PhysioNet/CinC 2019 test-split replay (simulated live data).</span>
        </div>
      </footer>
    </div>
  )
}