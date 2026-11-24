import { useEffect, useState } from 'react'
import { api } from '../api'
import type { FinalEvaluation } from '../api'
import { BarChart3, Target } from 'lucide-react'

const CARD_SHADOW = 'bg-white rounded-3xl shadow-sm border border-spike-border/50'

function Metric({ label, value, ci }: { label: string; value?: any; ci?: any }) {
  return (
    <div className={`${CARD_SHADOW} p-5`}>
      <div className="text-xs uppercase font-semibold text-spike-muted">{label}</div>
      <div className="mt-1.5 text-2xl font-bold tabular-nums text-spike-text">{value != null ? value : '—'}</div>
      {ci && (
        <div className="text-[11px] text-spike-muted mt-1 tabular-nums">
          95% CI [{ci[0]?.toFixed(3)}, {ci[1]?.toFixed(3)}]
        </div>
      )}
    </div>
  )
}

function ConfusionMatrix({ cm }: { cm: any }) {
  if (!cm) return null
  return (
    <div className={`${CARD_SHADOW} p-5`}>
      <h3 className="font-bold text-spike-text mb-3">Confusion matrix (test, Youden threshold)</h3>
      <table className="text-sm tabular-nums w-full">
        <thead>
          <tr className="text-xs text-spike-muted">
            <th className="px-4 py-2 text-left"></th>
            <th className="px-4 py-2">Predicted 0</th>
            <th className="px-4 py-2">Predicted 1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className="px-4 py-2 text-xs text-spike-muted text-left font-normal">Actual 0</th>
            <td className="px-4 py-2 text-emerald-600 font-semibold">{cm.tn.toLocaleString()}</td>
            <td className="px-4 py-2 text-red-500 font-semibold">{cm.fp.toLocaleString()}</td>
          </tr>
          <tr>
            <th className="px-4 py-2 text-xs text-spike-muted text-left font-normal">Actual 1</th>
            <td className="px-4 py-2 text-red-500 font-semibold">{cm.fn.toLocaleString()}</td>
            <td className="px-4 py-2 text-emerald-600 font-semibold">{cm.tp.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

export default function ModelPerformance() {
  const [ev, setEv] = useState<FinalEvaluation | null>(null)
  useEffect(() => {
    api.modelPerformance().then(setEv).catch(() => {})
  }, [])

  const horizons = ev
    ? [
        { key: 'test_y6' as const, title: 'Sepsis within 6 hours' },
        { key: 'test_y12' as const, title: 'Sepsis within 12 hours' },
      ]
    : []

  return (
    <div className="space-y-6">
      <div className={`${CARD_SHADOW} p-6`}>
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-2xl bg-spike-primary-light grid place-items-center text-spike-primary shrink-0">
            <BarChart3 size={24} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-spike-text">Model Performance</h2>
            <p className="text-xs text-spike-muted mt-1">
              Genuine held-out evaluation results (test split, never used for training, threshold selection or
              calibration).
            </p>
          </div>
        </div>
        <div className="mt-4 bg-spike-bg rounded-2xl px-4 py-3 text-xs text-spike-text">
          Deployed model variant: <span className="font-bold text-spike-primary">{ev?.deployed_variant ?? '…'}</span> ·
          model version {ev && 'sepsiguard_v1.0'} · evaluation dataset:{' '}
          <span className="font-semibold">PhysioNet 2019 test split</span> (4,600 patients, 108,207 prediction windows)
        </div>
      </div>

      {horizons.map(({ key, title }) => {
        const m = ev?.[key] ?? {}
        return (
          <section key={key} className="space-y-4">
            <div className={`${CARD_SHADOW} p-6`}>
              <h2 className="text-lg font-bold text-spike-text mb-4">{title}</h2>
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
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <ConfusionMatrix cm={m.confusion} />
              <div className={`${CARD_SHADOW} p-5 space-y-3`}>
                <h3 className="font-bold text-spike-text flex items-center gap-2">
                  <Target size={18} className="text-spike-primary" /> Clinical utility
                </h3>
                <div className="text-sm text-spike-text">
                  Net benefit at Youden threshold:{' '}
                  <b>{m.net_benefit_at_youden_threshold?.toFixed(5)}</b>
                </div>
                <div className="text-sm text-spike-text">
                  Alert threshold (sensitivity ≥ 80%): risk ≥{' '}
                  <b>{(100 * m.alert_threshold_sensitivity80).toFixed(2)}%</b>
                </div>
                <div className="text-sm text-spike-text">
                  Median early-warning lead time: <b>{m.median_lead_time_hours_when_alerting} h</b>
                </div>
              </div>
            </div>
          </section>
        )
      })}
    </div>
  )
}
