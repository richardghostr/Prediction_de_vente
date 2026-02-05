"use client"

import { useMemo, useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  Area,
} from "recharts"
import { ChartCard } from "./chart-card"
import type {
  SaleRecord,
  CategorySummary,
  StoreSummary,
  WeekdaySummary,
  MonthSummary,
} from "@/lib/data"
import { computePromoImpact } from "@/lib/data"

const COLORS = ["#1B6B93", "#F39C12", "#27AE60", "#E74C3C", "#8E44AD"]

interface ChartsTabProps {
  records: SaleRecord[]
  categorySummaries: CategorySummary[]
  storeSummaries: StoreSummary[]
  weekdaySummaries: WeekdaySummary[]
  monthlySummaries: MonthSummary[]
}

export function ChartsTab({
  records,
  categorySummaries,
  storeSummaries,
  weekdaySummaries,
  monthlySummaries,
}: ChartsTabProps) {
  const [selectedMetric, setSelectedMetric] = useState<"revenue" | "quantity">("revenue")
  const promoImpact = useMemo(() => computePromoImpact(records), [records])

  // Cross-analysis: category x store
  const catStoreData = useMemo(() => {
    const map = new Map<string, Record<string, number>>()
    for (const r of records) {
      const existing = map.get(r.category) || {}
      existing[r.store_id] = (existing[r.store_id] || 0) + r.revenue
      map.set(r.category, existing)
    }
    return Array.from(map.entries()).map(([category, stores]) => ({
      category,
      ...stores,
    }))
  }, [records])

  const storeIds = useMemo(() => [...new Set(records.map((r) => r.store_id))].sort(), [records])

  // Product ranking
  const productRanking = useMemo(() => {
    const byProduct = new Map<string, { rev: number; qty: number; category: string }>()
    for (const r of records) {
      const existing = byProduct.get(r.product_id) || { rev: 0, qty: 0, category: r.category }
      existing.rev += r.revenue
      existing.qty += r.quantity
      byProduct.set(r.product_id, existing)
    }
    return Array.from(byProduct.entries())
      .map(([id, v]) => ({ product_id: id, revenue: Math.round(v.rev), quantity: v.qty, category: v.category }))
      .sort((a, b) => b.revenue - a.revenue)
  }, [records])

  // Promo comparison data
  const promoData = [
    { name: "Avec promo", revenue: promoImpact.promoAvgRevenue, quantity: promoImpact.promoAvgQuantity },
    { name: "Sans promo", revenue: promoImpact.noPromoAvgRevenue, quantity: promoImpact.noPromoAvgQuantity },
  ]

  // Radar data for categories
  const radarData = categorySummaries.map((c) => ({
    category: c.category,
    revenue: c.totalRevenue,
    quantity: c.totalQuantity,
    avgPrice: c.avgPrice * 100,
  }))

  return (
    <div className="flex flex-col gap-6">
      {/* Metric selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Metrique :</span>
        <button
          onClick={() => setSelectedMetric("revenue")}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
            selectedMetric === "revenue"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
        >
          Revenu
        </button>
        <button
          onClick={() => setSelectedMetric("quantity")}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
            selectedMetric === "quantity"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
        >
          Quantite
        </button>
      </div>

      {/* Top row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category x Store stacked bar */}
        <ChartCard title="Revenus par categorie et magasin" subtitle="Analyse croisee">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={catStoreData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="category" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {storeIds.map((store, i) => (
                  <Bar key={store} dataKey={store} stackId="a" fill={COLORS[i % COLORS.length]} radius={i === storeIds.length - 1 ? [4, 4, 0, 0] : undefined} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Radar chart */}
        <ChartCard title="Profil des categories" subtitle="Radar multi-dimensions">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} outerRadius={90}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <PolarRadiusAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
                <Radar name="Revenu" dataKey="revenue" stroke="#1B6B93" fill="#1B6B93" fillOpacity={0.3} />
                <Radar name="Quantite" dataKey="quantity" stroke="#F39C12" fill="#F39C12" fillOpacity={0.2} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Middle Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Weekday analysis */}
        <ChartCard title="Moyenne par jour de la semaine" subtitle={selectedMetric === "revenue" ? "Revenu moyen" : "Quantite moyenne"}>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weekdaySummaries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar
                  dataKey={selectedMetric === "revenue" ? "avgRevenue" : "avgQuantity"}
                  fill="#27AE60"
                  radius={[4, 4, 0, 0]}
                  name={selectedMetric === "revenue" ? "Rev. moy." : "Qte moy."}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Promo impact */}
        <ChartCard title="Impact des promotions" subtitle={`Effet: ${promoImpact.revenueImpact > 0 ? "+" : ""}${promoImpact.revenueImpact}% sur le revenu`}>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={promoData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar dataKey="revenue" fill="#F39C12" radius={[4, 4, 0, 0]} name="Rev. moy." />
                <Bar dataKey="quantity" fill="#1B6B93" radius={[4, 4, 0, 0]} name="Qte moy." />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
            <span>Promo: {promoImpact.promoCount} transactions</span>
            <span>Hors promo: {promoImpact.noPromoCount} transactions</span>
          </div>
        </ChartCard>

        {/* Monthly trend line */}
        <ChartCard title="Tendance mensuelle" subtitle="Revenus et quantites">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={monthlySummaries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area yAxisId="left" type="monotone" dataKey="totalRevenue" fill="#1B6B93" fillOpacity={0.1} stroke="#1B6B93" name="Revenu" />
                <Line yAxisId="right" type="monotone" dataKey="totalQuantity" stroke="#F39C12" strokeWidth={2} dot={false} name="Quantite" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Product ranking */}
      <ChartCard title="Classement des produits" subtitle="Top produits par revenu total">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={productRanking} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" tick={{ fontSize: 10, fill: "#6b7280" }} />
              <YAxis dataKey="product_id" type="category" tick={{ fontSize: 11, fill: "#6b7280" }} width={50} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                formatter={(v: number, name: string) => [`${v.toLocaleString("fr-FR")} ${name === "revenue" ? "EUR" : "unites"}`, name === "revenue" ? "Revenu" : "Quantite"]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="revenue" fill="#1B6B93" radius={[0, 4, 4, 0]} name="Revenu" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  )
}
