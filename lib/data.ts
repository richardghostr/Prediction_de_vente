// Sales prediction sample data and utility functions
// This mirrors the data from sample.csv and provides typed access

export interface SaleRecord {
  date: string
  store_id: string
  product_id: string
  category: string
  quantity: number
  unit_price: number
  on_promo: boolean
  revenue: number
}

export interface DailySummary {
  date: string
  totalRevenue: number
  totalQuantity: number
  transactionCount: number
}

export interface CategorySummary {
  category: string
  totalRevenue: number
  totalQuantity: number
  avgPrice: number
  count: number
}

export interface StoreSummary {
  store_id: string
  totalRevenue: number
  totalQuantity: number
  count: number
}

export interface MonthSummary {
  month: string
  monthNum: number
  totalRevenue: number
  totalQuantity: number
  count: number
}

export interface WeekdaySummary {
  day: string
  dayNum: number
  avgRevenue: number
  avgQuantity: number
  count: number
}

export interface ForecastPoint {
  date: string
  yhat: number
  yhat_lower: number
  yhat_upper: number
}

export interface PipelineTask {
  id: string
  phase: string
  name: string
  role: string
  status: "done" | "in-progress" | "todo"
  description: string
  files: string[]
}

// Parse CSV text into SaleRecord array
export function parseCSV(csvText: string): SaleRecord[] {
  const lines = csvText.trim().split("\n")
  const headers = lines[0].split(",")
  return lines.slice(1).map((line) => {
    const values = line.split(",")
    return {
      date: values[0],
      store_id: values[1],
      product_id: values[2],
      category: values[3],
      quantity: parseInt(values[4], 10),
      unit_price: parseFloat(values[5]),
      on_promo: values[6] === "True",
      revenue: parseFloat(values[7]),
    }
  })
}

// Compute daily summaries
export function computeDailySummaries(records: SaleRecord[]): DailySummary[] {
  const byDate = new Map<string, { rev: number; qty: number; cnt: number }>()
  for (const r of records) {
    const existing = byDate.get(r.date) || { rev: 0, qty: 0, cnt: 0 }
    existing.rev += r.revenue
    existing.qty += r.quantity
    existing.cnt += 1
    byDate.set(r.date, existing)
  }
  return Array.from(byDate.entries())
    .map(([date, v]) => ({
      date,
      totalRevenue: Math.round(v.rev * 100) / 100,
      totalQuantity: v.qty,
      transactionCount: v.cnt,
    }))
    .sort((a, b) => a.date.localeCompare(b.date))
}

// Compute category summaries
export function computeCategorySummaries(records: SaleRecord[]): CategorySummary[] {
  const byCat = new Map<string, { rev: number; qty: number; prices: number[]; cnt: number }>()
  for (const r of records) {
    const existing = byCat.get(r.category) || { rev: 0, qty: 0, prices: [], cnt: 0 }
    existing.rev += r.revenue
    existing.qty += r.quantity
    existing.prices.push(r.unit_price)
    existing.cnt += 1
    byCat.set(r.category, existing)
  }
  return Array.from(byCat.entries())
    .map(([category, v]) => ({
      category,
      totalRevenue: Math.round(v.rev * 100) / 100,
      totalQuantity: v.qty,
      avgPrice: Math.round((v.prices.reduce((a, b) => a + b, 0) / v.prices.length) * 100) / 100,
      count: v.cnt,
    }))
    .sort((a, b) => b.totalRevenue - a.totalRevenue)
}

// Compute store summaries
export function computeStoreSummaries(records: SaleRecord[]): StoreSummary[] {
  const byStore = new Map<string, { rev: number; qty: number; cnt: number }>()
  for (const r of records) {
    const existing = byStore.get(r.store_id) || { rev: 0, qty: 0, cnt: 0 }
    existing.rev += r.revenue
    existing.qty += r.quantity
    existing.cnt += 1
    byStore.set(r.store_id, existing)
  }
  return Array.from(byStore.entries())
    .map(([store_id, v]) => ({
      store_id,
      totalRevenue: Math.round(v.rev * 100) / 100,
      totalQuantity: v.qty,
      count: v.cnt,
    }))
    .sort((a, b) => a.store_id.localeCompare(b.store_id))
}

// Compute monthly summaries
export function computeMonthlySummaries(records: SaleRecord[]): MonthSummary[] {
  const monthNames = ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"]
  const byMonth = new Map<number, { rev: number; qty: number; cnt: number }>()
  for (const r of records) {
    const d = new Date(r.date)
    const m = d.getMonth()
    const existing = byMonth.get(m) || { rev: 0, qty: 0, cnt: 0 }
    existing.rev += r.revenue
    existing.qty += r.quantity
    existing.cnt += 1
    byMonth.set(m, existing)
  }
  return Array.from(byMonth.entries())
    .map(([monthNum, v]) => ({
      month: monthNames[monthNum],
      monthNum,
      totalRevenue: Math.round(v.rev * 100) / 100,
      totalQuantity: v.qty,
      count: v.cnt,
    }))
    .sort((a, b) => a.monthNum - b.monthNum)
}

// Compute weekday summaries
export function computeWeekdaySummaries(records: SaleRecord[]): WeekdaySummary[] {
  const dayNames = ["Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam"]
  const byDay = new Map<number, { rev: number; qty: number; cnt: number }>()
  for (const r of records) {
    const d = new Date(r.date)
    const dow = d.getDay()
    const existing = byDay.get(dow) || { rev: 0, qty: 0, cnt: 0 }
    existing.rev += r.revenue
    existing.qty += r.quantity
    existing.cnt += 1
    byDay.set(dow, existing)
  }
  return Array.from(byDay.entries())
    .map(([dayNum, v]) => ({
      day: dayNames[dayNum],
      dayNum,
      avgRevenue: Math.round((v.rev / v.cnt) * 100) / 100,
      avgQuantity: Math.round((v.qty / v.cnt) * 100) / 100,
      count: v.cnt,
    }))
    .sort((a, b) => a.dayNum - b.dayNum)
}

// Generate naive forecast (last value persistence)
export function generateForecast(dailySummaries: DailySummary[], horizon: number): ForecastPoint[] {
  if (dailySummaries.length === 0) return []
  const last = dailySummaries[dailySummaries.length - 1]
  const avgRevenue = dailySummaries.reduce((s, d) => s + d.totalRevenue, 0) / dailySummaries.length
  const stdDev = Math.sqrt(
    dailySummaries.reduce((s, d) => s + Math.pow(d.totalRevenue - avgRevenue, 2), 0) / dailySummaries.length
  )

  // Simple trend calculation from last 30 data points
  const recentData = dailySummaries.slice(-30)
  let trend = 0
  if (recentData.length > 1) {
    const n = recentData.length
    const xMean = (n - 1) / 2
    const yMean = recentData.reduce((s, d) => s + d.totalRevenue, 0) / n
    let num = 0
    let den = 0
    for (let i = 0; i < n; i++) {
      num += (i - xMean) * (recentData[i].totalRevenue - yMean)
      den += (i - xMean) * (i - xMean)
    }
    trend = den !== 0 ? num / den : 0
  }

  const lastDate = new Date(last.date)
  const forecast: ForecastPoint[] = []
  for (let i = 1; i <= horizon; i++) {
    const d = new Date(lastDate)
    d.setDate(d.getDate() + i)
    const yhat = Math.max(0, avgRevenue + trend * i + (Math.random() - 0.5) * stdDev * 0.3)
    forecast.push({
      date: d.toISOString().split("T")[0],
      yhat: Math.round(yhat * 100) / 100,
      yhat_lower: Math.round(Math.max(0, yhat - stdDev * 1.5) * 100) / 100,
      yhat_upper: Math.round((yhat + stdDev * 1.5) * 100) / 100,
    })
  }
  return forecast
}

// Compute basic statistics
export function computeStats(values: number[]) {
  const n = values.length
  if (n === 0) return { count: 0, mean: 0, std: 0, min: 0, max: 0, median: 0, q25: 0, q75: 0 }

  const sorted = [...values].sort((a, b) => a - b)
  const mean = values.reduce((s, v) => s + v, 0) / n
  const variance = values.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / n
  const std = Math.sqrt(variance)

  const percentile = (p: number) => {
    const idx = (p / 100) * (n - 1)
    const lo = Math.floor(idx)
    const hi = Math.ceil(idx)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
  }

  return {
    count: n,
    mean: Math.round(mean * 100) / 100,
    std: Math.round(std * 100) / 100,
    min: Math.round(sorted[0] * 100) / 100,
    max: Math.round(sorted[n - 1] * 100) / 100,
    median: Math.round(percentile(50) * 100) / 100,
    q25: Math.round(percentile(25) * 100) / 100,
    q75: Math.round(percentile(75) * 100) / 100,
  }
}

// Pipeline tasks from chronological_tasks.md
export const pipelineTasks: PipelineTask[] = [
  {
    id: "0.1",
    phase: "Phase 0",
    name: "Configuration partagee",
    role: "A+B",
    status: "done",
    description: "Definir variables (paths, seed, formats de date). PR coordonnee.",
    files: ["src/config.py", "docs/run.md"],
  },
  {
    id: "0.2",
    phase: "Phase 0",
    name: "Requirements.txt",
    role: "B",
    status: "done",
    description: "Lister versions minimales (pandas, numpy, scikit-learn, xgboost, etc.).",
    files: ["requirements.txt"],
  },
  {
    id: "1.1",
    phase: "Phase 1",
    name: "Ingestion minimale",
    role: "A",
    status: "done",
    description: "Script pour lire CSVs, valider schema minimal et ecrire dans data/raw/.",
    files: ["src/data/ingest.py", "data/raw/README.md"],
  },
  {
    id: "1.2",
    phase: "Phase 1",
    name: "Politique de donnees",
    role: "B",
    status: "done",
    description: "Indiquer emplacement attendu, regle: ne pas commit gros fichiers.",
    files: ["data/raw/README.md", "docs/run.md"],
  },
  {
    id: "2.1",
    phase: "Phase 2",
    name: "Nettoyage",
    role: "A",
    status: "done",
    description: "Fonctions pour harmoniser dates, traiter NaN, detecter outliers.",
    files: ["src/data/clean.py"],
  },
  {
    id: "2.2",
    phase: "Phase 2",
    name: "Feature engineering",
    role: "A",
    status: "done",
    description: "Fonctions pures : lags, rolling, features temporelles, encodage categoriel.",
    files: ["src/data/features.py", "tests/unit/test_features.py"],
  },
  {
    id: "3.1",
    phase: "Phase 3",
    name: "Baselines exploratoires",
    role: "A",
    status: "done",
    description: "EDA et baseline Prophet ou ARIMA pour reference.",
    files: ["notebooks/00_eda.ipynb", "notebooks/01_baselines_prophet.ipynb"],
  },
  {
    id: "3.2",
    phase: "Phase 3",
    name: "Modele XGBoost",
    role: "A",
    status: "done",
    description: "Pipeline d'entrainement, sauvegarde model + metriques.",
    files: ["src/models/train.py", "src/models/predict.py"],
  },
  {
    id: "4.1",
    phase: "Phase 4",
    name: "Contract predict()",
    role: "A",
    status: "done",
    description: "Documenter la signature predict(sample_df, horizon) -> DataFrame/JSON.",
    files: ["src/models/predict.py"],
  },
  {
    id: "4.2",
    phase: "Phase 4",
    name: "Schemas Pydantic",
    role: "B",
    status: "done",
    description: "Valider les exemples fournis par A, affiner les schemas.",
    files: ["src/serve/schemas.py"],
  },
  {
    id: "5.1",
    phase: "Phase 5",
    name: "API minimal",
    role: "B",
    status: "done",
    description: "Endpoints /predict, /health, /metrics.",
    files: ["src/serve/api.py", "src/serve/schemas.py"],
  },
  {
    id: "5.2",
    phase: "Phase 5",
    name: "Tests API",
    role: "B",
    status: "done",
    description: "Tests de validation d'input, cas d'erreur, /health.",
    files: ["tests/unit/test_api.py"],
  },
  {
    id: "6.1",
    phase: "Phase 6",
    name: "UI Streamlit",
    role: "B",
    status: "done",
    description: "Upload CSV, selection horizon, bouton Predict.",
    files: ["src/ui/streamlit_app.py"],
  },
  {
    id: "7.1",
    phase: "Phase 7",
    name: "Docker & Compose",
    role: "B",
    status: "done",
    description: "Images pour API et UI, variables d'environnement.",
    files: ["Dockerfile", "docker-compose.yml"],
  },
  {
    id: "8.1",
    phase: "Phase 8",
    name: "Tests d'integration",
    role: "A+B",
    status: "in-progress",
    description: "Pipeline complet : ingestion -> features -> predict -> assertions.",
    files: ["tests/integration/test_end_to_end.py"],
  },
  {
    id: "9.1",
    phase: "Phase 9",
    name: "Logs & monitoring",
    role: "B",
    status: "todo",
    description: "Logs structures (JSON), endpoint /metrics, metriques basiques.",
    files: ["src/utils/logging.py", "src/serve/api.py"],
  },
  {
    id: "10.1",
    phase: "Phase 10",
    name: "Documentation finale",
    role: "B",
    status: "todo",
    description: "Mise a jour README, docs, checklist avant merge.",
    files: ["README.md", "docs/run.md"],
  },
]

// Rolling average helper
export function computeRollingAverage(data: DailySummary[], window: number): { date: string; value: number }[] {
  const result: { date: string; value: number }[] = []
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1)
    const slice = data.slice(start, i + 1)
    const avg = slice.reduce((s, d) => s + d.totalRevenue, 0) / slice.length
    result.push({ date: data[i].date, value: Math.round(avg * 100) / 100 })
  }
  return result
}

// Promo analysis
export function computePromoImpact(records: SaleRecord[]) {
  const promo = records.filter((r) => r.on_promo)
  const noPromo = records.filter((r) => !r.on_promo)

  const promoAvgRev = promo.length > 0 ? promo.reduce((s, r) => s + r.revenue, 0) / promo.length : 0
  const noPromoAvgRev = noPromo.length > 0 ? noPromo.reduce((s, r) => s + r.revenue, 0) / noPromo.length : 0
  const promoAvgQty = promo.length > 0 ? promo.reduce((s, r) => s + r.quantity, 0) / promo.length : 0
  const noPromoAvgQty = noPromo.length > 0 ? noPromo.reduce((s, r) => s + r.quantity, 0) / noPromo.length : 0

  return {
    promoCount: promo.length,
    noPromoCount: noPromo.length,
    promoAvgRevenue: Math.round(promoAvgRev * 100) / 100,
    noPromoAvgRevenue: Math.round(noPromoAvgRev * 100) / 100,
    promoAvgQuantity: Math.round(promoAvgQty * 100) / 100,
    noPromoAvgQuantity: Math.round(noPromoAvgQty * 100) / 100,
    revenueImpact: promoAvgRev > 0 && noPromoAvgRev > 0
      ? Math.round(((promoAvgRev - noPromoAvgRev) / noPromoAvgRev) * 100 * 100) / 100
      : 0,
  }
}
