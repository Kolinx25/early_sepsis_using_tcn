import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Bell,
  BarChart3,
  Settings,
  Menu,
  Search,
  Moon,
  Globe,
  MessageSquare,
  ChevronDown,
  Activity,
} from 'lucide-react'
import { useLiveData } from './useLiveData'
import { useState } from 'react'

const navItems = [
  { to: '/', label: 'ICU Overview', icon: LayoutDashboard },
  { to: '/alerts', label: 'Alert Centre', icon: Bell },
  { to: '/model', label: 'Model Performance', icon: BarChart3 },
  { to: '/system', label: 'System Admin', icon: Settings },
]

export default function Layout() {
  const { connected } = useLiveData()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()
  const activeLabel = navItems.find((n) => n.to === location.pathname)?.label ?? 'ICU Overview'

  return (
    <div className="min-h-screen bg-spike-bg flex">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'w-64' : 'w-20'
        } bg-white rounded-r-[30px] shadow-sm flex flex-col transition-all duration-300 fixed lg:static h-screen z-20`}
      >
        {/* Logo */}
        <div className="h-20 flex items-center gap-3 px-6">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 grid place-items-center text-white shrink-0">
            <Activity size={20} />
          </div>
          {sidebarOpen && (
            <div>
              <div className="font-bold text-lg text-spike-text leading-tight">SepsiGuard</div>
              <div className="text-[10px] text-spike-muted leading-tight">Clinical Decision Support</div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-4 space-y-1">
          <div className={`text-[10px] uppercase font-bold text-spike-muted mb-3 px-3 ${!sidebarOpen && 'hidden'}`}>
            Main
          </div>
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition ${
                    isActive
                      ? 'bg-spike-primary-light text-spike-primary'
                      : 'text-spike-muted hover:bg-spike-bg hover:text-spike-text'
                  } ${!sidebarOpen && 'justify-center'}`
                }
              >
                <Icon size={20} />
                {sidebarOpen && <span>{item.label}</span>}
              </NavLink>
            )
          })}
        </nav>

        {/* User card */}
        <div className="p-4">
          <div className="bg-spike-bg rounded-2xl p-3 flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 grid place-items-center text-white font-bold text-sm shrink-0">
              SG
            </div>
            {sidebarOpen && (
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-spike-text truncate">SepsiGuard</div>
                <div className="text-[10px] text-spike-muted truncate">Research Prototype</div>
              </div>
            )}
            {sidebarOpen && <ChevronDown size={16} className="text-spike-muted" />}
          </div>
        </div>
      </aside>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-10 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top header */}
        <header className="h-20 px-6 flex items-center justify-between bg-spike-bg sticky top-0 z-10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="h-10 w-10 rounded-xl bg-white shadow-sm grid place-items-center text-spike-text hover:bg-spike-primary-light hover:text-spike-primary transition"
            >
              <Menu size={20} />
            </button>
            <div>
              <h1 className="text-xl font-bold text-spike-text">{activeLabel}</h1>
              <p className="text-xs text-spike-muted">Continuous sepsis-risk monitoring</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="hidden md:flex items-center bg-white rounded-xl px-4 py-2.5 shadow-sm w-64">
              <Search size={18} className="text-spike-muted" />
              <input
                type="text"
                placeholder="Try to searching..."
                className="ml-2 text-sm outline-none w-full text-spike-text placeholder:text-spike-muted"
              />
            </div>

            {/* Theme toggle */}
            <button className="h-10 w-10 rounded-xl bg-white shadow-sm grid place-items-center text-spike-muted hover:text-spike-text transition">
              <Moon size={18} />
            </button>

            {/* Language */}
            <button className="h-10 w-10 rounded-xl bg-white shadow-sm grid place-items-center text-spike-muted hover:text-spike-text transition">
              <Globe size={18} />
            </button>

            {/* Messages */}
            <button className="h-10 w-10 rounded-xl bg-white shadow-sm grid place-items-center text-spike-muted hover:text-spike-text transition relative">
              <MessageSquare size={18} />
              <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-red-500" />
            </button>

            {/* Notifications */}
            <button className="h-10 w-10 rounded-xl bg-white shadow-sm grid place-items-center text-spike-muted hover:text-spike-text transition relative">
              <Bell size={18} />
              <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-red-500" />
            </button>

            {/* Live status */}
            <div className="hidden sm:flex items-center gap-2 bg-white rounded-xl px-3 py-2 shadow-sm">
              <span className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`} />
              <span className="text-xs font-medium text-spike-text">{connected ? 'Live' : 'Offline'}</span>
            </div>

            {/* User profile */}
            <div className="flex items-center gap-3 bg-white rounded-2xl pl-3 pr-4 py-2 shadow-sm cursor-pointer hover:shadow-md transition">
              <div className="h-9 w-9 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 grid place-items-center text-white font-bold text-xs">
                SG
              </div>
              <div className="hidden md:block">
                <div className="text-sm font-semibold text-spike-text leading-tight">SepsiGuard</div>
                <div className="text-[10px] text-spike-muted leading-tight">Admin</div>
              </div>
              <ChevronDown size={16} className="text-spike-muted" />
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 px-6 pb-8">
          <Outlet />
        </main>

        <footer className="px-6 py-4 text-[11px] text-spike-muted">
          Research prototype — not a diagnostic or autonomous treatment system. Data source: historical PhysioNet/CinC
          2019 test-split replay.
        </footer>
      </div>
    </div>
  )
}
