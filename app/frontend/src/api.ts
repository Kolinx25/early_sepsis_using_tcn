// SepsiGuard API client + shared types
export interface RiskBand { critical: number; high: number; elevated: number; low: number }

export interface Overview {
  monitored_patients: number
  risk_bands: RiskBand
  recent_predictions: Prediction[]
  data_source: { kind: string; label: string; replay: ReplayStatus }
}

export interface Prediction {
  id?: number
  patient_id: number
  iculos: number
  prob_6h: number | null
  prob_12h: number | null
  risk_band: string | null
  alert_6h?: boolean
  alert_12h?: boolean
  prediction_status?: string
  predicted_at?: string
  model_version?: string
  new_alerts?: Alert[]
}

export interface PatientRow {
  patient_id: number
  first_seen_at?: string
  last_seen_at?: string
  prob_6h?: number
  prob_12h?: number
  risk_band?: string
  iculos?: number
  n_observations: number
}

export interface ReplayStatus {
  status: string
  session_id: number | null
  speed: number
  n_patients: number
}

export interface Alert {
  id: number
  patient_id: number
  horizon: string
  kind: string
  probability: number
  message: string
  acknowledged: boolean
  created_at: string
}

export interface HealthStatus {
  api: string
  database: string
  model_version: string
  replay: ReplayStatus
}

export interface FinalEvaluation {
  deployed_variant: string
  test_y6: Record<string, any>
  test_y12: Record<string, any>
}

const base = ''
export const api = {
  overview: () => fetch(`${base}/api/overview`).then(r => r.json()) as Promise<Overview>,
  patients: () => fetch(`${base}/api/patients`).then(r => r.json()) as Promise<PatientRow[]>,
  patient: (pid: number) => fetch(`${base}/api/patients/${pid}`).then(r => r.json()),
  vitals: (pid: number) => fetch(`${base}/api/patients/${pid}/vitals`).then(r => r.json()),
  alerts: () => fetch(`${base}/api/alerts`).then(r => r.json()) as Promise<Alert[]>,
  ackAlert: (id: number) =>
    fetch(`${base}/api/alerts/${id}/ack`, { method: 'POST' }).then(r => r.json()),
  modelPerformance: () => fetch(`${base}/api/model/performance`).then(r => r.json()) as Promise<FinalEvaluation>,
  health: () => fetch(`${base}/api/system/health`).then(r => r.json()) as Promise<HealthStatus>,
  logs: () => fetch(`${base}/api/system/logs`).then(r => r.json()),
  replay: (action: string, opts: { speed?: number; n_patients?: number } = {}) =>
    fetch(`${base}/api/replay`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action, ...opts }),
    }).then(r => r.json()) as Promise<ReplayStatus>,
}

export const bandColor: Record<string, string> = {
  low: 'text-emerald-600 bg-emerald-50 border-emerald-100',
  elevated: 'text-amber-600 bg-amber-50 border-amber-100',
  high: 'text-orange-600 bg-orange-50 border-orange-100',
  critical: 'text-red-600 bg-red-50 border-red-100',
}