import { useEffect, useState } from 'react'
import { api } from '../api'
import type { FinalEvaluation } from '../api'

function Metric({ label, value, ci }: { label: string; value?: any; ci?: any }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1.5 text-2xl font-bold tabular-nums">
        {value != null ? value : '—'}
      </div>
      {ci && (
        <div className="text-[11px] text-slate-500 mt-1 tabular-nums">
          95% CI [{ci[0]?.toFixed(3)}, {ci[1]?.toFixed(3)}] (patient bootstrap)
        </div>
      )}
    </div>
  )
}

function ConfusionMatrix({ cm }: { cm: any }) {
  if (!cm) return null
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
      <h3 className="font-semibold mb-3">Confusion matrix (test, Youden threshold)</h3>
      <table className="text-sm tabular-nums">
        <thead>
          <tr className="text-xs text-slate-400">
            <th className="px-4 py-2 text-left"></th>
            <th className="px-4 py-2">Predicted 0</th>
            <th className="px-4 py-2">Predicted 1</th>
          </tr>
        </thead>
        <tbody>
          <tr><th className="px-4 py-2 text-xs text-slate-400 text-left font-normal">Actual 0</th>
            <td className="px-4 py-2 text-emerald-300">{cm.tn.toLocaleString()}</td>
            <td className="px-4 py-2 text-red-300">{cm.fp.toLocaleString()}</td></tr>
          <tr><th className="px-4 py-2 text-xs text-slate-400 text-left font-normal">Actual 1</th>
            <td className="px-4 py-2 text-red-300">{cm.fn.toLocaleString()}</td>
            <td className="px-4 py-2 text-emerald-300">{cm.tp.toLocaleString()}</td></tr>
        </tbody>
      </table>
    </div>
  )
}

export default function ModelPerformance() {
  const [ev, setEv] = useState<FinalEvaluation | null>(null)
  useEffect(() => { api.modelPerformance().then(setEv).catch(() => {}) }, [])

  const horizons = ev ? [
    { key: 'test_y6' as const, title: 'Sepsis within 6 hours' },
    { key: 'test_y12' as const, title: 'Sepsis within 12 hours' },
  ] : []

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Model Performance</h1>
        <p className="text-sm text-slate-400 mt-1">
          Genuine held-out evaluation results (test split, never used for training,
          threshold selection or calibration).
        </p>
      </div>

      <div className="rounded-2xl border border-sky-400/20 bg-sky-400/5 px-5 py-3 text-sm text-slate-300">
        Deployed model variant: <span className="font-semibold text-sky-300">
          {ev?.deployed_variant ?? '…'}</span> · model version {ev && 'sepsiguard_v1.0'} ·
        evaluation dataset: <span className="font-semibold">PhysioNet 2019 test split</span>{' '}
        (4,600 patients, 108,207 prediction windows)
      </div>

      {horizons.map(({ key, title }) => {
        const m = ev?.[key] ?? {}
        return (
          <section key={key} className="space-y-4">
            <h2 className="text-lg font-semibold">{title}</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Metric label="AUROC" value={m.auroc?.toFixed(4)} ci={m.auroc_ci95_patient_bootstrap} />
              <Metric label="AUPRC" value={m.auprc?.toFixed(4)} ci={m.auprc_ci95_patient_bootstrap} />
              <Metric label="Sensitivity" value={m.sensitivity?.toFixed(4)} />
              <Metric label="Specificity" value={m.specificity?.toFixed(4)} />
              <Metric label="Precision" value={m.precision?.toFixed(4)} />
              <Metric label="Recall" value={m.recall?.toFixed(4)} />
              <Metric label="F1 score" value={m.f1?.toFixed(4)} />
              <Metric label="Brier score" value={m.brier?.toFixed(4)} />
            </div>
            <ConfusionMatrix cm={m.confusion} />
          </section>
        )
      })}

      {ev && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 text-sm text-slate-300 space-y-1.5">
          <h3 className="font-semibold mb-2">Clinical utility</h3>
          <div>Net benefit at Youden threshold (12 h): <b>{ev.test_y12.net_benefit_at_youden_threshold?.toFixed(5)}</b></div>
          <div>Alert threshold (validation-derived, sensitivity ≥ 80%): 12 h risk ≥ <b>{(100 * ev.test_y12.alert_threshold_sensitivity80).toFixed(2)}%</b></div>
          <div>Median early-warning lead time when alerting (12 h head): <b>{ev.test_y12.median_lead_time_hours_when_alerting} h</b></div>
        </section>
      )}
    </div>
  )
}