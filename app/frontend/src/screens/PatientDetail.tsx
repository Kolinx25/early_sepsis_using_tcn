import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, bandColor } from '../api'
import { useLiveData } from '../useLiveData'

const VITALS: [string, string, string][] = [
  ['HR', 'Heart rate', '#f87171'],
  ['SBP', 'Systolic BP', '#60a5fa'],
  ['DBP', 'Diastolic BP', '#93c5fd'],
  ['Resp', 'Respiratory rate', '#fbbf24'],
  ['Temp', 'Temperature', '#fb923c'],
  ['O2Sat', 'O₂ saturation', '#34d399'],
]
const LABS: string[] = ['Lactate', 'WBC', 'BUN', 'Creatinine', 'Glucose', 'Platelets', 'Hgb', 'pH']

export default function PatientDetail() {
  const { pid } = useParams()
  const patientId = Number(pid)
  const [detail, setDetail] = useState<any>(null)
  const [vitals, setVitals] = useState<any[]>([])
  const refresh = () => {
    api.patient(patientId).then(setDetail).catch(() => {})
    api.vitals(patientId).then(setVitals).catch(() => {})
  }
  useLiveData((e: any) => { if (e.patient_id === patientId) refresh() })
  useEffect(() => { refresh() }, [patientId])

  const riskHistory = (detail?.risk_history ?? [])
    .filter((r: any) => r.prediction_status === 'ok')
    .map((r: any) => ({
      iculos: r.iculos,
      p6: 100 * r.prob_6h, p12: 100 * r.prob_12h,
    }))
  const latest = detail?.risk_history?.filter((r: any) => r.prediction_status === 'ok').at(-1)
  const missing: [string, boolean][] = detail?.missingness
    ? Object.entries(detail.missingness) : []

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Patient #{patientId} <span className="text-slate-500 font-normal text-lg">· anonymised</span></h1>
          <p className="text-sm text-slate-400 mt-1">
            Streaming risk assessment from the 12-hour observation window (simulated live data)
          </p>
        </div>
        {latest && (
          <div className="flex gap-4">
            <div className="rounded-2xl border border-sky-400/30 bg-sky-400/10 px-5 py-3 text-right">
              <div className="text-xs uppercase text-sky-300">6 h risk</div>
              <div className="text-2xl font-bold tabular-nums">{(100 * latest.prob_6h).toFixed(2)}%</div>
            </div>
            <div className="rounded-2xl border border-orange-400/30 bg-orange-400/10 px-5 py-3 text-right">
              <div className="text-xs uppercase text-orange-300">12 h risk</div>
              <div className="text-2xl font-bold tabular-nums">{(100 * latest.prob_12h).toFixed(2)}%</div>
            </div>
            <div className="rounded-2xl border border-slate-700 bg-slate-900 px-5 py-3 grid place-items-center">
              <span className={`rounded-full border px-3 py-0.5 text-xs font-semibold ${bandColor[latest.risk_band] ?? ''}`}>
                {latest.risk_band}
              </span>
            </div>
          </div>
        )}
      </div>

      <section className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="font-semibold mb-4">Vital-sign trajectories (observed hours)</h2>
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={vitals}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="iculos" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {VITALS.map(([k, label, color]) => (
                  <Line key={k} dataKey={k} name={label} stroke={color} dot={false}
                        strokeWidth={1.6} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500 mt-2">
            Gaps = unmeasured hours (missingness preserved; model receives explicit missingness indicators).
          </p>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="font-semibold mb-3">Latest measurements</h2>
          <ul className="text-sm space-y-1.5">
            {VITALS.slice(0, 6).map(([k, label]) => {
              const v = vitals.at(-1)?.[k]
              return (
                <li key={k} className="flex justify-between border-b border-slate-800/60 pb-1.5">
                  <span className="text-slate-400">{label}</span>
                  <span className="tabular-nums">{v != null ? v : <em className="text-slate-600 not-italic">not measured</em>}</span>
                </li>
              )
            })}
          </ul>
        </div>
      </section>

      <section className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="font-semibold mb-4">Historical risk trajectory</h2>
          <div style={{ height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={riskHistory}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="iculos" stroke="#64748b" tick={{ fontSize: 11 }}
                       label={{ value: 'ICU hour', fill: '#64748b', fontSize: 10, position: 'insideBottom' }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit="%" domain={[0, 40]} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="p6" name="6 h risk" stroke="#38bdf8" dot={false} strokeWidth={2} unit="%" />
                <Line dataKey="p12" name="12 h risk" stroke="#fb923c" dot={false} strokeWidth={2} unit="%" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
            <h2 className="font-semibold mb-3">Laboratory measurements (last)</h2>
            <ul className="text-sm space-y-1.5">
              {LABS.map(k => {
                const v = vitals.at(-1)?.[k]
                return (
                  <li key={k} className="flex justify-between border-b border-slate-800/60 pb-1.5">
                    <span className="text-slate-400">{k}</span>
                    <span className="tabular-nums">{v != null ? v : <em className="text-slate-600 not-italic">not measured</em>}</span>
                  </li>
                )
              })}
            </ul>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
            <h2 className="font-semibold mb-3">Missingness (latest hour)</h2>
            <div className="flex flex-wrap gap-1.5">
              {missing.map(([k, isMissing]: any) => (
                <span key={k} className={`text-[10px] rounded px-1.5 py-0.5 border font-mono ${
                  isMissing ? 'text-slate-400 border-slate-700 bg-slate-800/40' : 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10'}`}>
                  {k}
                </span>
              ))}
              {!missing.length && <span className="text-sm text-slate-500">No observations yet.</span>}
            </div>
            <p className="text-[11px] text-slate-500 mt-3">
              Green = measured this hour. Missingness patterns are informative and gate the model.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}