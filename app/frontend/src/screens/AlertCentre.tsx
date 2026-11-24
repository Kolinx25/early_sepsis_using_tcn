import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bell, CheckCircle2, AlertTriangle, Clock } from 'lucide-react'
import { api } from '../api'
import type { Alert } from '../api'
import { useLiveData } from '../useLiveData'

const CARD_SHADOW = 'bg-white rounded-3xl shadow-sm border border-spike-border/50'

export default function AlertCentre() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filter, setFilter] = useState<'all' | 'unacked'>('unacked')
  const refresh = () => api.alerts().then(setAlerts).catch(() => {})
  useLiveData(refresh)
  useEffect(() => {
    refresh()
  }, [])

  const shown = filter === 'unacked' ? alerts.filter((a) => !a.acknowledged) : alerts

  return (
    <div className="space-y-6">
      <div className={`${CARD_SHADOW} p-6`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-spike-text">Prediction &amp; Alert Centre</h2>
            <p className="text-xs text-spike-muted mt-1">
              Threshold-crossing events from the 6 h / 12 h prediction heads, with acknowledgement.
            </p>
          </div>
          <div className="flex gap-2">
            {(['unacked', 'all'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-xl text-xs font-semibold transition ${
                  filter === f
                    ? 'bg-spike-primary-light text-spike-primary'
                    : 'bg-spike-bg text-spike-muted hover:text-spike-text'
                }`}
              >
                {f === 'unacked'
                  ? `Unacknowledged (${alerts.filter((a) => !a.acknowledged).length})`
                  : `All (${alerts.length})`}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className={`${CARD_SHADOW} p-6`}>
        <ul className="divide-y divide-spike-border/60">
          {shown.map((a) => (
            <li key={a.id} className="px-4 py-4 flex flex-wrap items-center gap-4">
              <div
                className={`h-10 w-10 rounded-full grid place-items-center ${
                  a.acknowledged ? 'bg-spike-bg text-spike-muted' : 'bg-red-50 text-red-500'
                }`}
              >
                {a.acknowledged ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
              </div>
              <Link
                to={`/patients/${a.patient_id}`}
                className="text-spike-primary hover:underline font-semibold w-20"
              >
                #{a.patient_id}
              </Link>
              <div className="flex-1 min-w-64">
                <div className="text-sm">
                  <span
                    className={`inline-block rounded-lg border px-2.5 py-0.5 text-[10px] font-bold uppercase mr-2 ${
                      a.horizon === 'y12'
                        ? 'text-orange-500 bg-orange-50 border-orange-100'
                        : 'text-blue-500 bg-blue-50 border-blue-100'
                    }`}
                  >
                    {a.horizon === 'y12' ? '12 h' : '6 h'}
                  </span>
                  <span className="text-spike-text">{a.message}</span>
                </div>
                <div className="text-[11px] text-spike-muted mt-1 flex items-center gap-1">
                  <Clock size={12} />
                  {new Date(a.created_at).toLocaleTimeString()} · probability{' '}
                  {(100 * a.probability).toFixed(2)}%
                </div>
              </div>
              <div>
                {a.acknowledged ? (
                  <span className="text-xs text-spike-muted bg-spike-bg rounded-lg px-3 py-1.5 font-medium">
                    Acknowledged
                  </span>
                ) : (
                  <button
                    onClick={async () => {
                      await api.ackAlert(a.id)
                      refresh()
                    }}
                    className="text-xs font-semibold text-emerald-600 bg-emerald-50 rounded-lg px-3 py-1.5 hover:bg-emerald-100 transition"
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </li>
          ))}
          {!shown.length && (
            <li className="px-4 py-12 text-center text-spike-muted text-sm">
              <Bell size={40} className="mx-auto mb-3 text-spike-border" />
              No {filter === 'unacked' ? 'unacknowledged ' : ''}alerts. Alerts appear when predicted risk crosses
              the validation-derived alert threshold.
            </li>
          )}
        </ul>
      </div>
    </div>
  )
}
