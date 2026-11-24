import { useEffect, useState } from 'react'
import { api } from '../api'
import type { HealthStatus, ReplayStatus } from '../api'
import { useLiveData } from '../useLiveData'
import { Play, Pause, Square, RefreshCw, Activity, Database, Cpu, FileText } from 'lucide-react'

const CARD_SHADOW = 'bg-white rounded-3xl shadow-sm border border-spike-border/50'

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs font-semibold rounded-full px-3 py-1 ${
        ok ? 'text-emerald-600 bg-emerald-50' : 'text-red-500 bg-red-50'
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-500' : 'bg-red-500'}`} />
      {label}
    </span>
  )
}

export default function SystemAdmin() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [logs, setLogs] = useState<any[]>([])
  const [speed, setSpeed] = useState(10)
  const [nPatients, setNPatients] = useState(25)
  const [busy, setBusy] = useState(false)
  const refresh = () => {
    api.health().then(setHealth).catch(() => {})
    api.logs().then(setLogs).catch(() => {})
  }
  useLiveData(refresh)
  useEffect(refresh, [])

  const replay: ReplayStatus | undefined = health?.replay
  const run = async (action: string, opts: any = {}) => {
    setBusy(true)
    try {
      await api.replay(action, opts)
    } finally {
      setBusy(false)
      refresh()
    }
  }

  return (
    <div className="space-y-6">
      <div className={`${CARD_SHADOW} p-6`}>
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-2xl bg-spike-primary-light grid place-items-center text-spike-primary shrink-0">
            <Activity size={24} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-spike-text">System Administration</h2>
            <p className="text-xs text-spike-muted mt-1">
              Historical ICU replay controls, ingestion status and service health.
            </p>
          </div>
        </div>
      </div>

      <div className={`${CARD_SHADOW} p-6 space-y-5`}>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h2 className="text-lg font-bold text-spike-text">Historical ICU replay</h2>
            <p className="text-xs text-spike-muted">
              Streams genuine PhysioNet 2019 test-split observations chronologically through the inference pipeline.
            </p>
          </div>
          <StatusPill ok={replay?.status === 'running'} label={`replay: ${replay?.status ?? '…'}`} />
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs text-spike-muted">
            <div className="mb-1 font-semibold">Speed (simulated hours / real second)</div>
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
              className="bg-spike-bg border border-spike-border rounded-xl px-3 py-2 w-32 text-sm tabular-nums outline-none"
            />
          </label>
          <label className="text-xs text-spike-muted">
            <div className="mb-1 font-semibold">Patients</div>
            <input
              type="number"
              min={1}
              max={100}
              value={nPatients}
              onChange={(e) => setNPatients(+e.target.value)}
              className="bg-spike-bg border border-spike-border rounded-xl px-3 py-2 w-24 text-sm tabular-nums outline-none"
            />
          </label>
          <div className="flex gap-2">
            <button
              disabled={busy}
              onClick={() => run('start', { speed, n_patients: nPatients })}
              className="flex items-center gap-1 px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-50 text-emerald-600 hover:bg-emerald-100 disabled:opacity-40 transition"
            >
              <Play size={14} /> Start
            </button>
            <button
              disabled={busy}
              onClick={() => run('pause')}
              className="flex items-center gap-1 px-4 py-2 rounded-xl text-xs font-semibold bg-amber-50 text-amber-600 hover:bg-amber-100 disabled:opacity-40 transition"
            >
              <Pause size={14} /> Pause
            </button>
            <button
              disabled={busy}
              onClick={() => run('resume')}
              className="flex items-center gap-1 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-50 text-blue-600 hover:bg-blue-100 disabled:opacity-40 transition"
            >
              <RefreshCw size={14} /> Resume
            </button>
            <button
              disabled={busy}
              onClick={() => run('stop')}
              className="flex items-center gap-1 px-4 py-2 rounded-xl text-xs font-semibold bg-red-50 text-red-600 hover:bg-red-100 disabled:opacity-40 transition"
            >
              <Square size={14} /> Stop
            </button>
          </div>
        </div>
        {replay?.session_id != null && (
          <div className="text-xs text-spike-muted bg-spike-bg rounded-xl px-4 py-2">
            Session #{replay.session_id} · speed {replay.speed} sim-h/s · {replay.n_patients} patients
          </div>
        )}
      </div>

      <section className="grid md:grid-cols-3 gap-4">
        <div className={`${CARD_SHADOW} p-5 space-y-3`}>
          <div className="flex items-center gap-2 mb-2">
            <Activity size={18} className="text-spike-primary" />
            <h2 className="font-bold text-spike-text">Service health</h2>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-spike-muted">API</span>
              <StatusPill ok={health?.api === 'healthy'} label={health?.api ?? '…'} />
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-spike-muted">PostgreSQL</span>
              <StatusPill ok={health?.database === 'healthy'} label={health?.database ?? '…'} />
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-spike-muted">Model</span>
              <StatusPill ok={!!health?.model_version} label={health?.model_version ?? '…'} />
            </div>
          </div>
        </div>
        <div className={`${CARD_SHADOW} p-5`}>
          <div className="flex items-center gap-2 mb-2">
            <Database size={18} className="text-spike-primary" />
            <h2 className="font-bold text-spike-text">Data ingestion</h2>
          </div>
          <div className="text-sm text-spike-muted space-y-2">
            <p>
              Every ingested hour is persisted to PostgreSQL (observations, predictions, alerts) before
              broadcasting to connected dashboards over WebSocket.
            </p>
            <p className="text-xs">
              Pending predictions are never reported as zero risk — they appear as <em>pending_history</em> until a
              full 12-hour window exists.
            </p>
          </div>
        </div>
        <div className={`${CARD_SHADOW} p-5`}>
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={18} className="text-spike-primary" />
            <h2 className="font-bold text-spike-text">Model information</h2>
          </div>
          <div className="text-sm text-spike-muted space-y-1">
            <div>
              Version: <span className="text-spike-text font-mono text-xs">sepsiguard_v1.0</span>
            </div>
            <div>
              Variant: <span className="text-spike-text font-mono text-xs">{health?.model_version}</span>
            </div>
            <div>Input: 69 channels × 12 h causal window</div>
            <div>Heads: 6 h &amp; 12 h probability (Platt-calibrated)</div>
          </div>
        </div>
      </section>

      <section className={`${CARD_SHADOW} overflow-hidden`}>
        <div className="flex items-center gap-2 px-5 py-3 border-b border-spike-border/60">
          <FileText size={18} className="text-spike-primary" />
          <h2 className="font-bold text-spike-text">Error &amp; event logs</h2>
        </div>
        <ul className="divide-y divide-spike-border/60 text-sm max-h-72 overflow-y-auto">
          {logs.map((l: any, i) => (
            <li key={i} className="px-5 py-2.5 flex gap-3">
              <span
                className={`text-[10px] font-bold uppercase rounded-lg px-2 py-0.5 h-fit mt-0.5 ${
                  l.level === 'error' ? 'text-red-600 bg-red-50' : 'text-spike-muted bg-spike-bg'
                }`}
              >
                {l.level}
              </span>
              <span className="text-spike-text flex-1">{l.message}</span>
              <span className="text-[11px] text-spike-muted tabular-nums">
                {new Date(l.created_at).toLocaleTimeString()}
              </span>
            </li>
          ))}
          {!logs.length && <li className="px-5 py-8 text-center text-spike-muted">No events logged.</li>}
        </ul>
      </section>
    </div>
  )
}
