"use client"

import {
  ShoppingCart,
  DollarSign,
  CalendarDays,
  Store,
  Tag,
  TrendingUp,
} from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  CartesianGrid,
  Legend,
} from "recharts"
import { KpiCard } from "./kpi-card"
import { ChartCard } from "./chart-card"
import type {
  SaleRecord,
  DailySummary,
  CategorySummary,
  StoreSummary,
  MonthSummary,
} from "@/lib/data"

const COLORS = ["#1B6B93", "#F39C12", "#27AE60", "#E74C3C", "#8E44AD"]

interface OverviewTabProps {
  records: SaleRecord[]
  dailySummaries: DailySummary[]
  categorySummaries: CategorySummary[]
  storeSummaries: StoreSummary[]
  monthlySummaries: MonthSummary[]
}

export function OverviewTab({
  records,
  dailySummaries,
  categorySummaries,
  storeSummaries,
  monthlySummaries,
}: OverviewTabProps) {
  const totalRevenue = records.reduce((s, r) => s + r.revenue, 0)
  const totalQuantity = records.reduce((s, r) => s + r.quantity, 0)
  const avgRevenuePerTransaction = totalRevenue / records.length
  const dateRange = dailySummaries.length > 0
    ? `${dailySummaries[0].date} - ${dailySummaries[dailySummaries.length - 1].date}`
    : "N/A"
  const uniqueProducts = new Set(records.map((r) => r.product_id)).size
  const promoRate = Math.round((records.filter((r) => r.on_promo).length / records.length) * 100)

  // Recent 7 days trend
  const recent7 = dailySummaries.slice(-7)
  const prev7 = dailySummaries.slice(-14, -7)
  const recent7Rev = recent7.reduce((s, d) => s + d.totalRevenue, 0)
  const prev7Rev = prev7.reduce((s, d) => s + d.totalRevenue, 0)
  const weekTrend = prev7Rev > 0 ? ((recent7Rev - prev7Rev) / prev7Rev) * 100 : 0

  return (
    <div className="flex flex-col gap-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <KpiCard
          title="Revenu Total"
          value={`${totalRevenue.toLocaleString("fr-FR", { minimumFractionDigits: 0 })} EUR`}
          icon={DollarSign}
          variant="primary"
          trend={weekTrend}
          subtitle="vs 7 jours precedents"
        />
        <KpiCard
          title="Quantite Totale"
          value={totalQuantity.toLocaleString("fr-FR")}
          icon={ShoppingCart}
          variant="secondary"
        />
        <KpiCard
          title="Transactions"
          value={records.length.toLocaleString("fr-FR")}
          icon={Tag}
          subtitle={`${uniqueProducts} produits`}
        />
        <KpiCard
          title="Revenu Moyen"
          value={`${avgRevenuePerTransaction.toFixed(2)} EUR`}
          icon={TrendingUp}
          subtitle="par transaction"
        />
        <KpiCard
          title="Magasins"
          value={storeSummaries.length}
          icon={Store}
          subtitle={dateRange}
        />
        <KpiCard
          title="Taux Promo"
          value={`${promoRate}%`}
          icon={CalendarDays}
          subtitle={`${records.filter((r) => r.on_promo).length} en promo`}
          variant="success"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue trend */}
        <ChartCard title="Tendance du revenu journalier" subtitle="Somme quotidienne des revenus">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={dailySummaries}>
                <defs>
                  <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1B6B93" stopOpacity={0.3} />
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
                  formatter={(v: number) => [`${v.toFixed(2)} EUR`, "Revenu"]}
                />
                <Area
                  type="monotone"
                  dataKey="totalRevenue"
                  stroke="#1B6B93"
                  strokeWidth={2}
                  fill="url(#colorRevenue)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Revenue by category pie */}
        <ChartCard title="Repartition par categorie" subtitle="Part du revenu total">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categorySummaries}
                  dataKey="totalRevenue"
                  nameKey="category"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  innerRadius={55}
                  paddingAngle={2}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {categorySummaries.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v.toLocaleString("fr-FR")} EUR`, "Revenu"]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Monthly revenue */}
        <ChartCard title="Revenu mensuel" subtitle="Total par mois" className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlySummaries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v.toLocaleString("fr-FR")} EUR`, "Revenu"]}
                />
                <Bar dataKey="totalRevenue" fill="#1B6B93" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Store breakdown */}
        <ChartCard title="Revenu par magasin" subtitle="Comparaison des points de vente">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={storeSummaries} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis
                  dataKey="store_id"
                  type="category"
                  tick={{ fontSize: 11, fill: "#6b7280" }}
                  width={70}
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v.toLocaleString("fr-FR")} EUR`, "Revenu"]}
                />
                <Bar dataKey="totalRevenue" radius={[0, 4, 4, 0]}>
                  {storeSummaries.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>
    </div>
  )
}
