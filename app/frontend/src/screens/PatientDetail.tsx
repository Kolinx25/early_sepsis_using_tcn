import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart,
} from 'recharts'
import { ActivitySquare } from 'lucide-react'
import { api, bandColor } from '../api'
import { useLiveData } from '../useLiveData'

const CARD_SHADOW = 'bg-white rounded-3xl shadow-sm border border-spike-border/50'

const VITALS: [string, string, string][] = [
  ['HR', 'Heart rate', '#ef4444'],
  ['SBP', 'Systolic BP', '#1e88e5'],
  ['DBP', 'Diastolic BP', '#42a5f5'],
  ['Resp', 'Respiratory rate', '#f59e0b'],
  ['Temp', 'Temperature', '#f97316'],
  ['O2Sat', 'O₂ saturation', '#10b981'],
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
  useLiveData((e: any) => {
    if (e.patient_id === patientId) refresh()
  })
  useEffect(() => {
    refresh()
  }, [patientId])

  const riskHistory = (detail?.risk_history ?? [])
    .filter((r: any) => r.prediction_status === 'ok')
    .map((r: any) => ({
      iculos: r.iculos,
      p6: 100 * r.prob_6h,
      p12: 100 * r.prob_12h,
    }))
  const latest = detail?.risk_history?.filter((r: any) => r.prediction_status === 'ok').at(-1)
  const missing: [string, boolean][] = detail?.missingness ? Object.entries(detail.missingness) : []

  return (
    <div className="space-y-6">
      <div className={`${CARD_SHADOW} p-6`}>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="h-12 w-12 rounded-2xl bg-spike-primary-light grid place-items-center text-spike-primary shrink-0">
              <ActivitySquare size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-spike-text">
                Patient #{patientId}{' '}
                <span className="text-spike-muted font-normal text-base">· anonymised</span>
              </h2>
              <p className="text-xs text-spike-muted mt-1">
                Streaming risk assessment from the 12-hour observation window (simulated live data)
              </p>
            </div>
          </div>
          {latest && (
            <div className="flex gap-3">
              <div className="bg-blue-50 rounded-2xl px-5 py-3 text-right min-w-[110px]">
                <div className="text-xs uppercase font-semibold text-blue-600">6 h risk</div>
                <div className="text-2xl font-bold tabular-nums text-spike-text">
                  {(100 * latest.prob_6h).toFixed(2)}%
                </div>
              </div>
              <div className="bg-orange-50 rounded-2xl px-5 py-3 text-right min-w-[110px]">
                <div className="text-xs uppercase font-semibold text-orange-500">12 h risk</div>
                <div className="text-2xl font-bold tabular-nums text-spike-text">
                  {(100 * latest.prob_12h).toFixed(2)}%
                </div>
              </div>
              <div className="bg-spike-bg rounded-2xl px-5 py-3 grid place-items-center">
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                    bandColor[latest.risk_band] ?? 'border-slate-200 text-slate-400 bg-slate-50'
                  }`}
                >
                  {latest.risk_band}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      <section className="grid lg:grid-cols-3 gap-4">
        <div className={`${CARD_SHADOW} p-5 lg:col-span-2`}>
          <h2 className="font-bold text-spike-text mb-4">Vital-sign trajectories (observed hours)</h2>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={vitals}>
                <CartesianGrid stroke="#e0e5f2" vertical={false} />
                <XAxis dataKey="iculos" stroke="#a3aed0" tick={{ fontSize: 11 }} />
                <YAxis stroke="#a3aed0" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {VITALS.map(([k, label, color]) => (
                  <Line
                    key={k}
                    dataKey={k}
                    name={label}
                    stroke={color}
                    dot={false}
                    strokeWidth={1.8}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-spike-muted mt-2">
            Gaps = unmeasured hours (missingness preserved; model receives explicit missingness indicators).
          </p>
        </div>
        <div className={`${CARD_SHADOW} p-5`}>
          <h2 className="font-bold text-spike-text mb-3">Latest measurements</h2>
          <ul className="text-sm space-y-2">
            {VITALS.slice(0, 6).map(([k, label]) => {
              const v = vitals.at(-1)?.[k]
              return (
                <li key={k} className="flex justify-between border-b border-spike-border/60 pb-2">
                  <span className="text-spike-muted">{label}</span>
                  <span className="tabular-nums font-medium text-spike-text">
                    {v != null ? v : <em className="text-spike-muted not-italic">not measured</em>}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      </section>

      <section className="grid lg:grid-cols-3 gap-4">
        <div className={`${CARD_SHADOW} p-5 lg:col-span-2`}>
          <h2 className="font-bold text-spike-text mb-4">Historical risk trajectory</h2>
          <div style={{ height: 240 }}>
            <ResponsiveContainer>
              <AreaChart data={riskHistory}>
                <defs>
                  <linearGradient id="p6" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1e88e5" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#1e88e5" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="p12" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#e0e5f2" vertical={false} />
                <XAxis
                  dataKey="iculos"
                  stroke="#a3aed0"
                  tick={{ fontSize: 11 }}
                  label={{ value: 'ICU hour', fill: '#a3aed0', fontSize: 10, position: 'insideBottom' }}
                />
                <YAxis stroke="#a3aed0" tick={{ fontSize: 11 }} unit="%" domain={[0, 40]} />
                <Tooltip
                  contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area
                  type="monotone"
                  dataKey="p6"
                  name="6 h risk"
                  stroke="#1e88e5"
                  strokeWidth={2}
                  fill="url(#p6)"
                  unit="%"
                />
                <Area
                  type="monotone"
                  dataKey="p12"
                  name="12 h risk"
                  stroke="#f97316"
                  strokeWidth={2}
                  fill="url(#p12)"
                  unit="%"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-4">
          <div className={`${CARD_SHADOW} p-5`}>
            <h2 className="font-bold text-spike-text mb-3">Laboratory measurements (last)</h2>
            <ul className="text-sm space-y-2">
              {LABS.map((k) => {
                const v = vitals.at(-1)?.[k]
                return (
                  <li key={k} className="flex justify-between border-b border-spike-border/60 pb-2">
                    <span className="text-spike-muted">{k}</span>
                    <span className="tabular-nums font-medium text-spike-text">
                      {v != null ? v : <em className="text-spike-muted not-italic">not measured</em>}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
          <div className={`${CARD_SHADOW} p-5`}>
            <h2 className="font-bold text-spike-text mb-3">Missingness (latest hour)</h2>
            <div className="flex flex-wrap gap-1.5">
              {missing.map(([k, isMissing]: any) => (
                <span
                  key={k}
                  className={`text-[10px] rounded-lg px-2 py-1 border font-mono ${
                    isMissing
                      ? 'text-spike-muted border-spike-border bg-spike-bg'
                      : 'text-emerald-600 border-emerald-100 bg-emerald-50'
                  }`}
                >
                  {k}
                </span>
              ))}
              {!missing.length && <span className="text-sm text-spike-muted">No observations yet.</span>}
            </div>
            <p className="text-[11px] text-spike-muted mt-3">
              Green = measured this hour. Missingness patterns are informative and gate the model.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
