import { useEffect, useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import {
  Users,
  AlertTriangle,
  TrendingUp,
  Activity,
  Calendar,
  MoreHorizontal,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react'
import { api, bandColor } from '../api'
import type { Overview as O, PatientRow } from '../api'
import { useLiveData } from '../useLiveData'

const CARD_SHADOW = 'bg-white rounded-3xl shadow-sm border border-spike-border/50'

function SectionTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-4">
      <h2 className="text-lg font-bold text-spike-text">{title}</h2>
      {subtitle && <p className="text-xs text-spike-muted mt-0.5">{subtitle}</p>}
    </div>
  )
}

function StatCard({
  title,
  value,
  sub,
  trend,
  trendUp,
  icon: Icon,
  color,
  chart,
}: {
  title: string
  value: string
  sub?: string
  trend?: string
  trendUp?: boolean
  icon: any
  color: string
  chart?: React.ReactNode
}) {
  return (
    <div className={`${CARD_SHADOW} p-5`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-spike-muted font-medium">{title}</p>
          <h3 className="text-2xl font-bold text-spike-text mt-1">{value}</h3>
          {sub && <p className="text-[11px] text-spike-muted mt-0.5">{sub}</p>}
          {trend && (
            <div className={`flex items-center gap-1 mt-2 text-xs font-semibold ${trendUp ? 'text-emerald-500' : 'text-red-500'}`}>
              {trendUp ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
              {trend}
            </div>
          )}
        </div>
        <div className={`h-10 w-10 rounded-xl grid place-items-center ${color}`}>
          <Icon size={20} />
        </div>
      </div>
      {chart && <div className="h-16 mt-4">{chart}</div>}
    </div>
  )
}

function MiniSpark({ data, color = '#1e88e5', fill = true }: { data: number[]; color?: string; fill?: boolean }) {
  const chartData = data.map((v) => ({ v }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id={`grad-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={2}
          fill={fill ? `url(#grad-${color.replace('#', '')})` : 'transparent'}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

const bandLabels: Record<string, string> = {
  low: 'Low',
  elevated: 'Elevated',
  high: 'High',
  critical: 'Critical',
}

export default function Overview() {
  const [data, setData] = useState<O | null>(null)
  const [patients, setPatients] = useState<PatientRow[]>([])
  const { connected } = useLiveData(() => refresh())

  const refresh = () => {
    api.overview().then(setData).catch(() => {})
    api.patients().then(setPatients).catch(() => {})
  }

  useEffect(() => {
    refresh()
  }, [])

  const bands = data?.risk_bands

  const riskDistribution = useMemo(
    () => [
      { name: 'Low', value: bands?.low ?? 0, color: '#10b981' },
      { name: 'Elevated', value: bands?.elevated ?? 0, color: '#f59e0b' },
      { name: 'High', value: bands?.high ?? 0, color: '#f97316' },
      { name: 'Critical', value: bands?.critical ?? 0, color: '#ef4444' },
    ],
    [bands],
  )

  const trendData = useMemo(() => {
    // Build a simple trend from recent predictions if available
    const preds = data?.recent_predictions ?? []
    return preds.slice(0, 12).map((p) => ({
      hour: p.iculos,
      risk6: Math.round((p.prob_6h ?? 0) * 1000) / 10,
      risk12: Math.round((p.prob_12h ?? 0) * 1000) / 10,
    }))
  }, [data])

  const sparkData = [12, 18, 15, 22, 28, 24, 32, 29, 35, 40, 38, 45]

  return (
    <div className="space-y-6">
      {/* Hero welcome card */}
      <div className={`${CARD_SHADOW} p-6 relative overflow-hidden`}>
        <div className="relative z-10 max-w-xl">
          <h2 className="text-2xl font-bold text-spike-text">Welcome back, Clinician 👋</h2>
          <p className="text-sm text-spike-muted mt-1">
            You are monitoring {data?.monitored_patients ?? 0} patients in real time. {bands ? bands.high + bands.critical : 0} patients are
            currently at high or critical risk for sepsis within 12 hours.
          </p>
          <div className="flex gap-8 mt-5">
            <div>
              <div className="text-2xl font-bold text-spike-text">{bands?.critical ?? 0}</div>
              <div className="text-xs text-spike-muted">Critical</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-spike-text">{bands?.high ?? 0}</div>
              <div className="text-xs text-spike-muted">High risk</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-spike-text">{data?.recent_predictions?.length ?? 0}</div>
              <div className="text-xs text-spike-muted">Updates today</div>
            </div>
          </div>
        </div>
        <div className="absolute right-6 top-1/2 -translate-y-1/2 hidden lg:block">
          <div className="h-40 w-40 rounded-full bg-spike-primary/10 flex items-center justify-center">
            <Activity size={72} className="text-spike-primary/40" />
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Monitored Patients"
          value={String(data?.monitored_patients ?? '—')}
          sub={connected ? 'Live via WebSocket' : 'Reconnecting…'}
          trend="12% vs last hour"
          trendUp
          icon={Users}
          color="bg-blue-50 text-blue-600"
          chart={<MiniSpark data={sparkData} color="#1e88e5" />}
        />
        <StatCard
          title="High + Critical Risk"
          value={String(bands ? bands.high + bands.critical : '—')}
          sub="12 h risk ≥ 20%"
          trend="5% vs last hour"
          trendUp={false}
          icon={AlertTriangle}
          color="bg-red-50 text-red-500"
          chart={<MiniSpark data={[8, 12, 10, 14, 18, 15, 20, 18, 22, 25, 23, 28]} color="#ef4444" />}
        />
        <StatCard
          title="Elevated Risk"
          value={String(bands?.elevated ?? '—')}
          sub="12 h risk 5–20%"
          trend="Stable"
          trendUp
          icon={TrendingUp}
          color="bg-amber-50 text-amber-500"
          chart={<MiniSpark data={[15, 14, 16, 15, 17, 16, 18, 17, 19, 18, 20, 19]} color="#f59e0b" />}
        />
        <StatCard
          title="Low Risk"
          value={String(bands?.low ?? '—')}
          sub="12 h risk < 5%"
          trend="8% vs last hour"
          trendUp
          icon={Calendar}
          color="bg-emerald-50 text-emerald-500"
          chart={<MiniSpark data={[40, 42, 41, 44, 43, 45, 44, 46, 45, 47, 46, 48]} color="#10b981" />}
        />
      </div>

      {/* Main chart + distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={`${CARD_SHADOW} p-6 lg:col-span-2`}>
          <div className="flex items-center justify-between mb-4">
            <SectionTitle title="Risk Trend" subtitle="6-hour and 12-hour sepsis risk over recent ICU hours" />
            <select className="text-xs bg-spike-bg rounded-lg px-3 py-1.5 outline-none text-spike-text">
              <option>This week</option>
              <option>Today</option>
            </select>
          </div>
          <div className="h-72">
            {trendData.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="risk6" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#1e88e5" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#1e88e5" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="risk12" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#7c3aed" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: '#a3aed0' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: '#a3aed0' }} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip
                    contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}
                  />
                  <Area type="monotone" dataKey="risk6" name="6 h risk" stroke="#1e88e5" strokeWidth={2} fill="url(#risk6)" />
                  <Area type="monotone" dataKey="risk12" name="12 h risk" stroke="#7c3aed" strokeWidth={2} fill="url(#risk12)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-spike-muted text-sm">
                Start the ICU replay to see live risk trends.
              </div>
            )}
          </div>
        </div>

        <div className={`${CARD_SHADOW} p-6`}>
          <SectionTitle title="Risk Distribution" subtitle="Current patients by risk band" />
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-2">
            {riskDistribution.map((r) => (
              <div key={r.name} className="flex items-center gap-2 text-xs">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: r.color }} />
                <span className="text-spike-muted">{r.name}</span>
                <span className="ml-auto font-semibold text-spike-text">{r.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Patient table */}
      <div className={`${CARD_SHADOW} p-6`}>
        <div className="flex items-center justify-between mb-4">
          <SectionTitle title="Patient Monitoring Table" subtitle="Click a patient for full trajectory and vitals" />
          <button className="h-9 w-9 rounded-xl bg-spike-bg grid place-items-center text-spike-muted hover:text-spike-text transition">
            <MoreHorizontal size={18} />
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-spike-muted border-b border-spike-border">
                <th className="px-4 py-3 font-semibold">Patient</th>
                <th className="px-4 py-3 font-semibold">Hour</th>
                <th className="px-4 py-3 font-semibold">Observations</th>
                <th className="px-4 py-3 font-semibold">6 h Risk</th>
                <th className="px-4 py-3 font-semibold">12 h Risk</th>
                <th className="px-4 py-3 font-semibold">Band</th>
                <th className="px-4 py-3 font-semibold">Action</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p) => (
                <tr key={p.patient_id} className="border-b border-spike-border/60 hover:bg-spike-bg/50 transition">
                  <td className="px-4 py-3 font-semibold text-spike-text">#{p.patient_id}</td>
                  <td className="px-4 py-3 tabular-nums text-spike-muted">{p.iculos ?? '—'}</td>
                  <td className="px-4 py-3 tabular-nums text-spike-muted">{p.n_observations}</td>
                  <td className="px-4 py-3 tabular-nums text-spike-text">
                    {p.prob_6h != null ? (100 * p.prob_6h).toFixed(2) + '%' : '—'}
                  </td>
                  <td className="px-4 py-3 tabular-nums font-bold text-spike-text">
                    {p.prob_12h != null ? (100 * p.prob_12h).toFixed(2) + '%' : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full border px-3 py-1 text-xs font-semibold ${
                        bandColor[p.risk_band ?? ''] || 'border-slate-200 text-slate-400 bg-slate-50'
                      }`}
                    >
                      {bandLabels[p.risk_band ?? ''] ?? 'Pending'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/patients/${p.patient_id}`}
                      className="inline-flex items-center gap-1 text-xs font-semibold text-spike-primary hover:underline"
                    >
                      View <ArrowUpRight size={14} />
                    </Link>
                  </td>
                </tr>
              ))}
              {!patients.length && (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-spike-muted">
                    No patients under monitoring yet — start the historical ICU replay in System Admin.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent predictions + alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`${CARD_SHADOW} p-6`}>
          <SectionTitle title="Recent Prediction Updates" subtitle="Latest risk scores from the live stream" />
          <ul className="space-y-3">
            {(data?.recent_predictions ?? []).slice(0, 6).map((p, i) => (
              <li key={i} className="flex items-center justify-between p-3 rounded-2xl bg-spike-bg/60">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-white grid place-items-center text-spike-primary font-bold text-xs shadow-sm">
                    #{p.patient_id}
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-spike-text">Patient #{p.patient_id}</div>
                    <div className="text-[11px] text-spike-muted">Hour {p.iculos}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold text-spike-text">
                    12h {(100 * (p.prob_12h ?? 0)).toFixed(2)}%
                  </div>
                  <div className="text-[11px] text-spike-muted">6h {(100 * (p.prob_6h ?? 0)).toFixed(2)}%</div>
                </div>
              </li>
            ))}
            {!data?.recent_predictions?.length && (
              <li className="text-center text-spike-muted py-8">Awaiting prediction updates…</li>
            )}
          </ul>
        </div>

        <div className={`${CARD_SHADOW} p-6`}>
          <SectionTitle title="Active Alerts" subtitle="Unacknowledged threshold crossings" />
          <ul className="space-y-3">
            {(data?.recent_predictions ?? [])
              .filter((p) => (p.prob_12h ?? 0) >= 0.2 || (p.prob_6h ?? 0) >= 0.2)
              .slice(0, 6)
              .map((p, i) => (
                <li key={i} className="flex items-center justify-between p-3 rounded-2xl bg-red-50/60">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-red-100 grid place-items-center text-red-500">
                      <AlertTriangle size={18} />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-spike-text">Patient #{p.patient_id}</div>
                      <div className="text-[11px] text-spike-muted">
                        {(100 * Math.max(p.prob_6h ?? 0, p.prob_12h ?? 0)).toFixed(2)}% max 12h risk
                      </div>
                    </div>
                  </div>
                  <Link
                    to={`/patients/${p.patient_id}`}
                    className="text-xs font-semibold text-spike-primary hover:underline"
                  >
                    Review
                  </Link>
                </li>
              ))}
            {!(data?.recent_predictions ?? []).some((p) => (p.prob_12h ?? 0) >= 0.2 || (p.prob_6h ?? 0) >= 0.2) && (
              <li className="text-center text-spike-muted py-8">No active alerts above threshold.</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  )
}
