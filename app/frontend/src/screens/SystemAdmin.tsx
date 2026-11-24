import { useEffect, useState } from 'react'
import { api } from '../api'
import type { HealthStatus, ReplayStatus } from '../api'
import { useLiveData } from '../useLiveData'

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold rounded-full border px-3 py-1 ${
      ok ? 'text-emerald-300 border-emerald-400/40 bg-emerald-400/10' : 'text-red-300 border-red-400/40 bg-red-400/10'}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-400'}`} />
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
    try { await api.replay(action, opts) } finally { setBusy(false); refresh() }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">System Administration</h1>
        <p className="text-sm text-slate-400 mt-1">
          Historical ICU replay controls, ingestion status and service health.
        </p>
      </div>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="font-semibold">Historical ICU replay <span className="text-xs text-amber-400/80 font-medium ml-2 border border-amber-400/30 rounded px-2 py-0.5">simulated live data</span></h2>
          <StatusPill ok={replay?.status === 'running'} label={`replay: ${replay?.status ?? '…'}`} />
        </div>
        <p className="text-xs text-slate-500 -mt-3">
          Streams genuine PhysioNet 2019 test-split observations chronologically through the same inference
          pipeline as a prospective hospital feed. No data are fabricated.
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs text-slate-400">
            <div className="mb-1">Speed (simulated hours / real second)</div>
            <input type="number" min={0.1} step={0.5} value={speed} onChange={e => setSpeed(+e.target.value)}
              className="bg-slate-800/70 border border-slate-700 rounded-lg px-3 py-1.5 w-32 text-sm tabular-nums" />
          </label>
          <label className="text-xs text-slate-400">
            <div className="mb-1">Patients</div>
            <input type="number" min={1} max={100} value={nPatients} onChange={e => setNPatients(+e.target.value)}
              className="bg-slate-800/70 border border-slate-700 rounded-lg px-3 py-1.5 w-24 text-sm tabular-nums" />
          </label>
          <div className="flex gap-2">
            <button disabled={busy} onClick={() => run('start', { speed, n_patients: nPatients })}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-400/40 hover:bg-emerald-400/10 disabled:opacity-40">
              ▶ Start
            </button>
            <button disabled={busy} onClick={() => run('pause')}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-amber-500/20 text-amber-300 border border-amber-400/40 hover:bg-amber-400/10 disabled:opacity-40">
              ⏸ Pause
            </button>
            <button disabled={busy} onClick={() => run('resume')}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-sky-500/20 text-sky-300 border border-sky-400/40 hover:bg-sky-400/10 disabled:opacity-40">
              ⏵ Resume
            </button>
            <button disabled={busy} onClick={() => run('stop')}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-red-500/20 text-red-300 border border-red-400/40 hover:bg-red-400/10 disabled:opacity-40">
              ⏹ Stop
            </button>
          </div>
        </div>
        {replay?.session_id != null && (
          <div className="text-xs text-slate-400">
            Session #{replay.session_id} · speed {replay.speed} sim-h/s · {replay.n_patients} patients
          </div>
        )}
      </section>

      <section className="grid md:grid-cols-3 gap-4">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 space-y-3">
          <h2 className="font-semibold">Service health</h2>
          <div className="space-y-2">
            <div className="flex justify-between items-center"><span className="text-sm text-slate-400">API</span>
              <StatusPill ok={health?.api === 'healthy'} label={health?.api ?? '…'} /></div>
            <div className="flex justify-between items-center"><span className="text-sm text-slate-400">PostgreSQL</span>
              <StatusPill ok={health?.database === 'healthy'} label={health?.database ?? '…'} /></div>
            <div className="flex justify-between items-center"><span className="text-sm text-slate-400">Model</span>
              <StatusPill ok={!!health?.model_version} label={health?.model_version ?? '…'} /></div>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="font-semibold mb-3">Data ingestion</h2>
          <div className="text-sm text-slate-400 space-y-2">
            <p>Every ingested hour is persisted to PostgreSQL (observations,
            predictions, alerts) before broadcasting to connected dashboards
            over WebSocket.</p>
            <p className="text-xs text-slate-500">Pending predictions are never reported as zero risk —
            they appear as <em>pending_history</em> until a full 12-hour window exists.</p>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="font-semibold mb-3">Model information</h2>
          <div className="text-sm text-slate-400 space-y-1">
            <div>Version: <span className="text-slate-200 font-mono text-xs">sepsiguard_v1.0</span></div>
            <div>Variant: <span className="text-slate-200 font-mono text-xs">{health?.model_version}</span></div>
            <div>Input: 69 channels × 12 h causal window</div>
            <div>Heads: 6 h &amp; 12 h sepsis-onset probability (Platt-calibrated)</div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
        <h2 className="font-semibold px-5 py-3 border-b border-slate-800">Error &amp; event logs</h2>
        <ul className="divide-y divide-slate-800/60 text-sm max-h-72 overflow-y-auto">
          {logs.map((l: any, i) => (
            <li key={i} className="px-5 py-2 flex gap-3">
              <span className={`text-[10px] font-bold uppercase rounded px-1.5 py-0.5 h-fit mt-0.5 ${
                l.level === 'error' ? 'text-red-300 bg-red-400/10' : 'text-slate-400 bg-slate-800'}`}>{l.level}</span>
              <span className="text-slate-300 flex-1">{l.message}</span>
              <span className="text-[11px] text-slate-500 tabular-nums">{new Date(l.created_at).toLocaleTimeString()}</span>
            </li>
          ))}
          {!logs.length && <li className="px-5 py-8 text-center text-slate-500">No events logged.</li>}
        </ul>
      </section>
    </div>
  )
}