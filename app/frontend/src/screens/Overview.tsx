import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, bandColor } from '../api'
import type { Overview as O, PatientRow } from '../api'
import { useLiveData } from '../useLiveData'

function Card({ title, value, sub, accent }: any) {
  return (
    <div className={`rounded-2xl border p-5 ${accent || 'border-slate-800 bg-slate-900/50'}`}>
      <div className="text-xs uppercase tracking-wide text-slate-400">{title}</div>
      <div className="mt-2 text-3xl font-bold tabular-nums">{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export default function Overview() {
  const [data, setData] = useState<O | null>(null)
  const [patients, setPatients] = useState<PatientRow[]>([])
  const refresh = () => {
    api.overview().then(setData).catch(() => {})
    api.patients().then(setPatients).catch(() => {})
  }
  const { connected } = useLiveData(refresh)
  useEffect(() => { refresh() }, [])

  const bands = data?.risk_bands
  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">ICU Overview</h1>
          <p className="text-sm text-slate-400 mt-1">
            Continuous sepsis-risk monitoring · 6 h and 12 h prediction horizons
          </p>
        </div>
        <div className="text-xs text-slate-400 border border-slate-800 rounded-lg px-3 py-2 bg-slate-900/50">
          <span className="font-semibold text-slate-300">Data source:</span>{' '}
          {data?.data_source.label ?? 'connecting…'}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card title="Monitored patients" value={data?.monitored_patients ?? '—'}
              sub={connected ? 'updates live via WebSocket' : 'reconnecting…'} />
        <Card title="High risk" value={bands ? bands.high + bands.critical : '—'}
              accent="border-orange-400/30 bg-orange-400/5" sub="12 h risk ≥ 20%" />
        <Card title="Elevated risk" value={bands?.elevated ?? '—'}
              accent="border-amber-400/30 bg-amber-400/5" sub="12 h risk 5–20%" />
        <Card title="Low risk" value={bands?.low ?? '—'}
              accent="border-emerald-400/30 bg-emerald-400/5" sub="12 h risk < 5%" />
        <Card title="Alerts (unacked)" value={patients.length ? 'see Alert Centre' : '—'}
              sub="threshold crossing events" />
      </div>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
          <h2 className="font-semibold">Patient monitoring table</h2>
          <span className="text-xs text-slate-500">click a patient for full trajectory</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-400 border-b border-slate-800">
                <th className="px-5 py-3">Patient</th>
                <th className="px-5 py-3">Hour (ICULOS)</th>
                <th className="px-5 py-3">Observations</th>
                <th className="px-5 py-3">6 h risk</th>
                <th className="px-5 py-3">12 h risk</th>
                <th className="px-5 py-3">Risk band</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.patient_id} className="border-b border-slate-800/60 hover:bg-slate-800/40 transition">
                  <td className="px-5 py-3">
                    <Link to={`/patients/${p.patient_id}`} className="text-sky-300 hover:underline font-medium">
                      #{p.patient_id}
                    </Link>
                  </td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">{p.iculos ?? '—'}</td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">{p.n_observations}</td>
                  <td className="px-5 py-3 tabular-nums">
                    {p.prob_6h != null ? (100 * p.prob_6h).toFixed(2) + '%' : '—'}
                  </td>
                  <td className="px-5 py-3 tabular-nums font-semibold">
                    {p.prob_12h != null ? (100 * p.prob_12h).toFixed(2) + '%' : '—'}
                  </td>
                  <td className="px-5 py-3">
                    <span className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-semibold ${bandColor[p.risk_band ?? ''] || 'border-slate-700 text-slate-400'}`}>
                      {p.risk_band ?? 'pending'}
                    </span>
                  </td>
                </tr>
              ))}
              {!patients.length && (
                <tr><td colSpan={6} className="px-5 py-10 text-center text-slate-500">
                  No patients under monitoring yet — start the historical ICU replay in System Admin.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-800">
          <h2 className="font-semibold">Recent prediction updates</h2>
        </div>
        <ul className="divide-y divide-slate-800/60 text-sm">
          {(data?.recent_predictions ?? []).map((p, i) => (
            <li key={i} className="px-5 py-2.5 flex items-center gap-4">
              <Link to={`/patients/${p.patient_id}`} className="text-sky-300 hover:underline w-28">
                #{p.patient_id}
              </Link>
              <span className="text-slate-400 w-24">hour {p.iculos}</span>
              <span className="tabular-nums text-slate-300">6h {(100 * (p.prob_6h ?? 0)).toFixed(2)}%</span>
              <span className="tabular-nums font-semibold">12h {(100 * (p.prob_12h ?? 0)).toFixed(2)}%</span>
              <span className={`ml-auto rounded-full border px-2.5 py-0.5 text-xs font-semibold ${bandColor[p.risk_band ?? ''] || ''}`}>
                {p.risk_band}
              </span>
            </li>
          ))}
          {!data?.recent_predictions?.length && (
            <li className="px-5 py-10 text-center text-slate-500">Awaiting prediction updates…</li>
          )}
        </ul>
      </section>
    </div>
  )
}