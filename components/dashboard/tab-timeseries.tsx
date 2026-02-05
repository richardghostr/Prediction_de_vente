"use client"

import { useMemo, useState } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
  BarChart,
  Bar,
  Legend,
  ComposedChart,
} from "recharts"
import { ChartCard } from "./chart-card"
import { KpiCard } from "./kpi-card"
import { computeRollingAverage } from "@/lib/data"
import type { SaleRecord, DailySummary, WeekdaySummary } from "@/lib/data"
import { TrendingUp, TrendingDown, Activity, Calendar } from "lucide-react"

interface TimeSeriesTabProps {
  records: SaleRecord[]
  dailySummaries: DailySummary[]
  weekdaySummaries: WeekdaySummary[]
}

export function TimeSeriesTab({ records, dailySummaries, weekdaySummaries }: TimeSeriesTabProps) {
  const [aggregation, setAggregation] = useState<"daily" | "weekly" | "monthly">("daily")
  const [rollingWindow, setRollingWindow] = useState(7)

  // Aggregated data
  const aggregatedData = useMemo(() => {
    if (aggregation === "daily") return dailySummaries

    const grouped = new Map<string, { rev: number; qty: number; cnt: number }>()
    for (const d of dailySummaries) {
      const date = new Date(d.date)
      let key: string
      if (aggregation === "weekly") {
        const weekStart = new Date(date)
        weekStart.setDate(weekStart.getDate() - weekStart.getDay() + 1)
        key = weekStart.toISOString().split("T")[0]
      } else {
        key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-01`
      }
      const existing = grouped.get(key) || { rev: 0, qty: 0, cnt: 0 }
      existing.rev += d.totalRevenue
      existing.qty += d.totalQuantity
      existing.cnt += d.transactionCount
      grouped.set(key, existing)
    }
    return Array.from(grouped.entries())
      .map(([date, v]) => ({
        date,
        totalRevenue: Math.round(v.rev * 100) / 100,
        totalQuantity: v.qty,
        transactionCount: v.cnt,
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
  }, [dailySummaries, aggregation])

  // Rolling average
  const rollingAvg = useMemo(() => computeRollingAverage(dailySummaries, rollingWindow), [dailySummaries, rollingWindow])

  // Combined data for overlay chart
  const overlayData = useMemo(() => {
    const map = new Map(dailySummaries.map((d) => [d.date, { ...d, rollingAvg: 0 }]))
    for (const r of rollingAvg) {
      const existing = map.get(r.date)
      if (existing) existing.rollingAvg = r.value
    }
    return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date))
  }, [dailySummaries, rollingAvg])

  // Heatmap data: week x day
  const heatmapData = useMemo(() => {
    const weeks = new Map<number, Map<number, number>>()
    for (const d of dailySummaries) {
      const date = new Date(d.date)
      const weekNum = Math.ceil(
        ((date.getTime() - new Date(date.getFullYear(), 0, 1).getTime()) / 86400000 + 1) / 7
      )
      const dow = date.getDay()
      if (!weeks.has(weekNum)) weeks.set(weekNum, new Map())
      const week = weeks.get(weekNum)!
      week.set(dow, (week.get(dow) || 0) + d.totalRevenue)
    }
    const result: { week: number; day: number; value: number }[] = []
    for (const [weekNum, days] of weeks) {
      for (const [day, value] of days) {
        result.push({ week: weekNum, day, value: Math.round(value) })
      }
    }
    return result
  }, [dailySummaries])

  // Seasonality decomposition (simplified)
  const monthlySeasonality = useMemo(() => {
    const byMonth = new Map<number, number[]>()
    for (const d of dailySummaries) {
      const month = new Date(d.date).getMonth()
      if (!byMonth.has(month)) byMonth.set(month, [])
      byMonth.get(month)!.push(d.totalRevenue)
    }
    const monthNames = ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aout", "Sep", "Oct", "Nov", "Dec"]
    return Array.from(byMonth.entries())
      .map(([month, values]) => ({
        month: monthNames[month],
        monthNum: month,
        avgRevenue: Math.round((values.reduce((s, v) => s + v, 0) / values.length) * 100) / 100,
        count: values.length,
      }))
      .sort((a, b) => a.monthNum - b.monthNum)
  }, [dailySummaries])

  // KPI calculations
  const totalDays = dailySummaries.length
  const avgDailyRevenue = dailySummaries.reduce((s, d) => s + d.totalRevenue, 0) / totalDays
  const maxDay = dailySummaries.reduce((max, d) => (d.totalRevenue > max.totalRevenue ? d : max), dailySummaries[0])
  const minDay = dailySummaries.reduce((min, d) => (d.totalRevenue < min.totalRevenue ? d : min), dailySummaries[0])

  // Trend
  const firstHalf = dailySummaries.slice(0, Math.floor(totalDays / 2))
  const secondHalf = dailySummaries.slice(Math.floor(totalDays / 2))
  const firstAvg = firstHalf.reduce((s, d) => s + d.totalRevenue, 0) / firstHalf.length
  const secondAvg = secondHalf.reduce((s, d) => s + d.totalRevenue, 0) / secondHalf.length
  const trendPct = firstAvg > 0 ? ((secondAvg - firstAvg) / firstAvg) * 100 : 0

  return (
    <div className="flex flex-col gap-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Jours de donnees"
          value={totalDays}
          icon={Calendar}
          subtitle={`${dailySummaries[0]?.date ?? ""} au ${dailySummaries[totalDays - 1]?.date ?? ""}`}
        />
        <KpiCard
          title="Rev. quotidien moyen"
          value={`${avgDailyRevenue.toFixed(0)} EUR`}
          icon={Activity}
          variant="primary"
        />
        <KpiCard
          title="Meilleur jour"
          value={`${maxDay?.totalRevenue.toFixed(0) ?? 0} EUR`}
          icon={TrendingUp}
          subtitle={maxDay?.date}
          variant="success"
        />
        <KpiCard
          title="Tendance globale"
          value={`${trendPct >= 0 ? "+" : ""}${trendPct.toFixed(1)}%`}
          icon={trendPct >= 0 ? TrendingUp : TrendingDown}
          subtitle="2e moitie vs 1ere"
          variant={trendPct >= 0 ? "success" : "destructive"}
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 bg-card rounded-lg border shadow-sm px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Agregation :</span>
          {(["daily", "weekly", "monthly"] as const).map((agg) => (
            <button
              key={agg}
              onClick={() => setAggregation(agg)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                aggregation === agg
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {agg === "daily" ? "Jour" : agg === "weekly" ? "Semaine" : "Mois"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Moyenne mobile :</span>
          {[3, 7, 14, 30].map((w) => (
            <button
              key={w}
              onClick={() => setRollingWindow(w)}
              className={`px-2 py-1 text-xs font-medium rounded-md transition-colors ${
                rollingWindow === w
                  ? "bg-secondary text-secondary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {w}j
            </button>
          ))}
        </div>
      </div>

      {/* Main chart */}
      <ChartCard title="Evolution temporelle avec moyenne mobile" subtitle={`Agregation: ${aggregation === "daily" ? "Jour" : aggregation === "weekly" ? "Semaine" : "Mois"} | Moyenne: ${rollingWindow}j`}>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={overlayData}>
              <defs>
                <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#1B6B93" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#1B6B93" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "#6b7280" }}
                tickFormatter={(v) => v.slice(5)}
                interval="preserveStartEnd"
              />
              <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                labelFormatter={(v) => `Date: ${v}`}
                formatter={(v: number, name: string) => [
                  `${v.toFixed(2)} EUR`,
                  name === "totalRevenue" ? "Revenu" : `Moyenne ${rollingWindow}j`,
                ]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} formatter={(v) => (v === "totalRevenue" ? "Revenu" : `Moy. ${rollingWindow}j`)} />
              <Area type="monotone" dataKey="totalRevenue" stroke="#1B6B93" fill="url(#colorArea)" strokeWidth={1.5} />
              <Line type="monotone" dataKey="rollingAvg" stroke="#F39C12" strokeWidth={2.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Aggregated bar chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title={`Revenus agreges (${aggregation === "daily" ? "Jour" : aggregation === "weekly" ? "Semaine" : "Mois"})`} subtitle="Total par periode">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={aggregatedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 9, fill: "#6b7280" }}
                  tickFormatter={(v) => v.slice(5)}
                  interval={aggregation === "daily" ? "preserveStartEnd" : 0}
                />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v.toFixed(0)} EUR`, "Revenu"]}
                />
                <Bar dataKey="totalRevenue" fill="#1B6B93" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Seasonality */}
        <ChartCard title="Saisonnalite mensuelle" subtitle="Revenu moyen par mois">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlySeasonality}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v.toFixed(2)} EUR`, "Rev. moy."]}
                />
                <Bar dataKey="avgRevenue" fill="#F39C12" radius={[4, 4, 0, 0]} name="Rev. moy." />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Weekday analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Analyse par jour de la semaine" subtitle="Revenu et quantite moyens">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weekdaySummaries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar yAxisId="left" dataKey="avgRevenue" fill="#1B6B93" radius={[4, 4, 0, 0]} name="Rev. moy." />
                <Bar yAxisId="right" dataKey="avgQuantity" fill="#27AE60" radius={[4, 4, 0, 0]} name="Qte moy." />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Transactions par jour" subtitle="Volume de ventes quotidien">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={dailySummaries}>
                <defs>
                  <linearGradient id="colorTx" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#27AE60" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#27AE60" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "#6b7280" }}
                  tickFormatter={(v) => v.slice(5)}
                  interval="preserveStartEnd"
                />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [v, "Transactions"]}
                />
                <Area type="monotone" dataKey="transactionCount" stroke="#27AE60" fill="url(#colorTx)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>
    </div>
  )
}
