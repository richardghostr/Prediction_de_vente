import { NextRequest, NextResponse } from "next/server"

interface DataPoint {
  date: string
  value: number
  [key: string]: unknown
}

interface ForecastPoint {
  ds: string
  yhat: number
  yhat_lower: number
  yhat_upper: number
}

/**
 * Compute a simple forecast using:
 * 1. Linear trend from last N observations
 * 2. Day-of-week seasonality if enough data
 * 3. Confidence intervals based on historical variance
 *
 * This is a JS-based fallback since the Python model can't run on Vercel serverless.
 * For production, point this at the FastAPI backend.
 */
function computeForecast(
  series: DataPoint[],
  horizon: number
): ForecastPoint[] {
  if (series.length === 0) return []

  // Sort by date
  const sorted = [...series]
    .filter((d) => d.value != null && !isNaN(d.value))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

  if (sorted.length === 0) return []

  const values = sorted.map((d) => d.value)
  const dates = sorted.map((d) => new Date(d.date))

  // Compute day-of-week seasonality
  const dowSum: number[] = new Array(7).fill(0)
  const dowCount: number[] = new Array(7).fill(0)
  for (let i = 0; i < sorted.length; i++) {
    const dow = dates[i].getDay()
    dowSum[dow] += values[i]
    dowCount[dow] += 1
  }
  const dowAvg = dowSum.map((s, i) => (dowCount[i] > 0 ? s / dowCount[i] : 0))
  const overallMean =
    values.reduce((a, b) => a + b, 0) / values.length
  const dowSeason = dowAvg.map((avg) =>
    overallMean > 0 ? avg / overallMean : 1
  )

  // Linear regression on last min(60, len) points for trend
  const trendWindow = Math.min(60, values.length)
  const recentValues = values.slice(-trendWindow)
  const n = recentValues.length
  const xMean = (n - 1) / 2
  const yMean = recentValues.reduce((a, b) => a + b, 0) / n
  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (recentValues[i] - yMean)
    den += (i - xMean) * (i - xMean)
  }
  const slope = den !== 0 ? num / den : 0
  const intercept = yMean - slope * xMean

  // Variance for confidence intervals
  const residuals = recentValues.map(
    (v, i) => v - (intercept + slope * i)
  )
  const variance =
    residuals.reduce((a, r) => a + r * r, 0) / Math.max(1, n - 2)
  const stdDev = Math.sqrt(variance)

  // Generate forecast
  const lastDate = dates[dates.length - 1]
  const forecast: ForecastPoint[] = []

  for (let h = 1; h <= horizon; h++) {
    const futureDate = new Date(lastDate)
    futureDate.setDate(futureDate.getDate() + h)

    // Trend projection
    const trendValue = intercept + slope * (n - 1 + h)

    // Apply day-of-week seasonality
    const dow = futureDate.getDay()
    const seasonFactor =
      dowSeason[dow] !== 0 ? dowSeason[dow] : 1

    let yhat = trendValue * seasonFactor

    // Ensure non-negative for quantities
    yhat = Math.max(0, yhat)

    // Confidence interval widens with horizon
    const ci = 1.96 * stdDev * Math.sqrt(1 + h / n)
    const yhat_lower = Math.max(0, yhat - ci)
    const yhat_upper = yhat + ci

    forecast.push({
      ds: futureDate.toISOString().split("T")[0],
      yhat: Math.round(yhat * 100) / 100,
      yhat_lower: Math.round(yhat_lower * 100) / 100,
      yhat_upper: Math.round(yhat_upper * 100) / 100,
    })
  }

  return forecast
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { series, horizon = 30 } = body as {
      series: DataPoint[]
      horizon: number
    }

    if (!series || !Array.isArray(series) || series.length === 0) {
      return NextResponse.json(
        { error: "Le champ 'series' est requis et doit contenir des donnees." },
        { status: 400 }
      )
    }

    const forecast = computeForecast(series, Math.min(horizon, 365))

    return NextResponse.json({
      id: "forecast",
      forecast,
      metrics: {
        horizon,
        input_points: series.length,
        method: "linear_trend_dow_seasonality",
      },
    })
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Erreur interne du serveur",
      },
      { status: 500 }
    )
  }
}
