import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { Alert } from '../api'
import { useLiveData } from '../useLiveData'

export default function AlertCentre() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filter, setFilter] = useState<'all' | 'unacked'>('unacked')
  const refresh = () => api.alerts().then(setAlerts).catch(() => {})
  useLiveData(refresh)
  useEffect(() => { refresh() }, [])

  const shown = filter === 'unacked' ? alerts.filter(a => !a.acknowledged) : alerts

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Prediction &amp; Alert Centre</h1>
        <p className="text-sm text-slate-400 mt-1">
          Threshold-crossing events from the 6 h / 12 h prediction heads, with acknowledgement.
        </p>
      </div>

      <div className="flex gap-2">
        {(['unacked', 'all'] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition ${
              filter === f ? 'bg-sky-500/20 text-sky-300 border-sky-400/40' : 'text-slate-400 border-slate-700 hover:bg-slate-800'}`}>
            {f === 'unacked' ? `Unacknowledged (${alerts.filter(a => !a.acknowledged).length})` : `All (${alerts.length})`}
          </button>
        ))}
      </div>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
        <ul className="divide-y divide-slate-800/60">
          {shown.map(a => (
            <li key={a.id} className="px-5 py-4 flex flex-wrap items-center gap-4">
              <div className={`h-2.5 w-2.5 rounded-full ${a.acknowledged ? 'bg-slate-600' : 'bg-red-400 animate-pulse'}`} />
              <Link to={`/patients/${a.patient_id}`} className="text-sky-300 hover:underline font-medium w-24">
                #{a.patient_id}
              </Link>
              <div className="flex-1 min-w-64">
                <div className="text-sm">
                  <span className={`inline-block rounded border px-2 py-0.5 text-[10px] font-bold uppercase mr-2 ${
                    a.horizon === 'y12' ? 'text-orange-300 border-orange-400/40' : 'text-sky-300 border-sky-400/40'}`}>
                    {a.horizon === 'y12' ? '12 h' : '6 h'}
                  </span>
                  <span className="text-slate-300">{a.message}</span>
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {new Date(a.created_at).toLocaleTimeString()} · probability {(100 * a.probability).toFixed(2)}%
                </div>
              </div>
              <div>
                {a.acknowledged ? (
                  <span className="text-xs text-slate-500 border border-slate-700 rounded-lg px-3 py-1.5">Acknowledged</span>
                ) : (
                  <button onClick={async () => { await api.ackAlert(a.id); refresh() }}
                    className="text-xs font-semibold text-emerald-300 border border-emerald-400/40 rounded-lg px-3 py-1.5 hover:bg-emerald-400/10 transition">
                    Acknowledge
                  </button>
                )}
              </div>
            </li>
          ))}
          {!shown.length && (
            <li className="px-5 py-12 text-center text-slate-500 text-sm">
              No {filter === 'unacked' ? 'unacknowledged ' : ''}alerts. Alerts appear when predicted risk crosses the
              validation-derived alert threshold.
            </li>
          )}
        </ul>
      </section>
    </div>
  )
}