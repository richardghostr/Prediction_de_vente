"use client"

import { useState, useMemo } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts"
import { ChartCard } from "./chart-card"
import { KpiCard } from "./kpi-card"
import { computeStats } from "@/lib/data"
import type { SaleRecord } from "@/lib/data"
import { Search, Filter } from "lucide-react"

interface ExplorerTabProps {
  records: SaleRecord[]
}

export function ExplorerTab({ records }: ExplorerTabProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [categoryFilter, setCategoryFilter] = useState("all")
  const [storeFilter, setStoreFilter] = useState("all")
  const [sortColumn, setSortColumn] = useState<keyof SaleRecord>("date")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc")

  const categories = useMemo(() => [...new Set(records.map((r) => r.category))].sort(), [records])
  const stores = useMemo(() => [...new Set(records.map((r) => r.store_id))].sort(), [records])

  const filtered = useMemo(() => {
    return records.filter((r) => {
      if (categoryFilter !== "all" && r.category !== categoryFilter) return false
      if (storeFilter !== "all" && r.store_id !== storeFilter) return false
      if (searchTerm) {
        const term = searchTerm.toLowerCase()
        return (
          r.date.includes(term) ||
          r.store_id.toLowerCase().includes(term) ||
          r.product_id.toLowerCase().includes(term) ||
          r.category.toLowerCase().includes(term)
        )
      }
      return true
    })
  }, [records, categoryFilter, storeFilter, searchTerm])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const av = a[sortColumn]
      const bv = b[sortColumn]
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av
      }
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av))
    })
  }, [filtered, sortColumn, sortDir])

  const revenueStats = useMemo(() => computeStats(filtered.map((r) => r.revenue)), [filtered])
  const quantityStats = useMemo(() => computeStats(filtered.map((r) => r.quantity)), [filtered])

  // Distribution data
  const revBins = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      range: `${Math.round(revenueStats.min + (i * (revenueStats.max - revenueStats.min)) / 10)}-${Math.round(revenueStats.min + ((i + 1) * (revenueStats.max - revenueStats.min)) / 10)}`,
      count: 0,
    }))
    const binWidth = (revenueStats.max - revenueStats.min) / 10 || 1
    for (const r of filtered) {
      const idx = Math.min(9, Math.floor((r.revenue - revenueStats.min) / binWidth))
      bins[idx].count++
    }
    return bins
  }, [filtered, revenueStats])

  // Scatter plot data: quantity vs revenue
  const scatterData = useMemo(
    () => filtered.map((r) => ({ x: r.quantity, y: r.revenue, z: r.unit_price, category: r.category })),
    [filtered]
  )

  const handleSort = (col: keyof SaleRecord) => {
    if (sortColumn === col) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setSortColumn(col)
      setSortDir("desc")
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Filters */}
      <div className="bg-card rounded-lg border shadow-sm p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 flex-1 min-w-[200px] max-w-sm">
            <Search className="h-4 w-4 text-muted-foreground shrink-0" />
            <input
              type="text"
              placeholder="Rechercher..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-muted rounded-md px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="bg-muted rounded-md px-3 py-1.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="all">Toutes categories</option>
              {categories.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <select
              value={storeFilter}
              onChange={(e) => setStoreFilter(e.target.value)}
              className="bg-muted rounded-md px-3 py-1.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="all">Tous magasins</option>
              {stores.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <p className="text-xs text-muted-foreground ml-auto">
            {filtered.length} / {records.length} enregistrements
          </p>
        </div>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard title="Revenu moyen" value={`${revenueStats.mean} EUR`} subtitle={`Ecart-type: ${revenueStats.std}`} variant="primary" />
        <KpiCard title="Revenu median" value={`${revenueStats.median} EUR`} subtitle={`Q25: ${revenueStats.q25} | Q75: ${revenueStats.q75}`} />
        <KpiCard title="Quantite moyenne" value={quantityStats.mean} subtitle={`Min: ${quantityStats.min} | Max: ${quantityStats.max}`} variant="secondary" />
        <KpiCard title="Plage revenus" value={`${revenueStats.min} - ${revenueStats.max}`} subtitle="Min - Max (EUR)" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Distribution des revenus" subtitle="Histogramme">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={revBins}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#6b7280" }} angle={-20} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <Bar dataKey="count" fill="#1B6B93" radius={[4, 4, 0, 0]} name="Nb transactions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Quantite vs Revenu" subtitle="Nuage de points">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="x" name="Quantite" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis dataKey="y" name="Revenu" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <ZAxis dataKey="z" range={[20, 80]} name="Prix unitaire" />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number, name: string) => [v.toFixed(2), name]}
                />
                <Scatter data={scatterData} fill="#F39C12" opacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Data Table */}
      <div className="bg-card rounded-lg border shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-border">
          <h3 className="text-sm font-semibold text-foreground">Donnees detaillees</h3>
        </div>
        <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted sticky top-0">
              <tr>
                {(["date", "store_id", "product_id", "category", "quantity", "unit_price", "on_promo", "revenue"] as const).map((col) => (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wide cursor-pointer hover:text-foreground transition-colors whitespace-nowrap"
                  >
                    {col.replace("_", " ")}
                    {sortColumn === col && (sortDir === "asc" ? " ▲" : " ▼")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {sorted.slice(0, 100).map((r, i) => (
                <tr key={i} className="hover:bg-muted/50 transition-colors">
                  <td className="px-4 py-2 text-xs font-mono text-foreground">{r.date}</td>
                  <td className="px-4 py-2 text-xs text-foreground">{r.store_id}</td>
                  <td className="px-4 py-2 text-xs text-foreground">{r.product_id}</td>
                  <td className="px-4 py-2 text-xs">
                    <span className="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground">
                      {r.category}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-xs font-mono text-foreground text-right">{r.quantity}</td>
                  <td className="px-4 py-2 text-xs font-mono text-foreground text-right">{r.unit_price.toFixed(2)}</td>
                  <td className="px-4 py-2 text-xs text-center">
                    <span className={`inline-block w-2 h-2 rounded-full ${r.on_promo ? "bg-success" : "bg-muted-foreground/30"}`} />
                  </td>
                  <td className="px-4 py-2 text-xs font-mono font-semibold text-foreground text-right">{r.revenue.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {sorted.length > 100 && (
          <div className="px-5 py-2 border-t border-border text-xs text-muted-foreground">
            Affichage des 100 premiers resultats sur {sorted.length} total
          </div>
        )}
      </div>
    </div>
  )
}
