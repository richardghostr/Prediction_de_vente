"use client"

import {
  LayoutDashboard,
  Database,
  BarChart3,
  TrendingUp,
  BrainCircuit,
  ListChecks,
  Download,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"

export type TabKey =
  | "overview"
  | "explorer"
  | "charts"
  | "timeseries"
  | "prediction"
  | "pipeline"
  | "export"

interface SidebarProps {
  activeTab: TabKey
  onTabChange: (tab: TabKey) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

const navItems: { key: TabKey; label: string; icon: React.ElementType; group: string }[] = [
  { key: "overview", label: "Vue d'ensemble", icon: LayoutDashboard, group: "ANALYSE" },
  { key: "explorer", label: "Explorateur", icon: Database, group: "ANALYSE" },
  { key: "charts", label: "Graphiques", icon: BarChart3, group: "ANALYSE" },
  { key: "timeseries", label: "Series temporelles", icon: TrendingUp, group: "MODELISATION" },
  { key: "prediction", label: "Prediction", icon: BrainCircuit, group: "MODELISATION" },
  { key: "pipeline", label: "Pipeline", icon: ListChecks, group: "PROJET" },
  { key: "export", label: "Export", icon: Download, group: "PROJET" },
]

export function Sidebar({ activeTab, onTabChange, collapsed, onToggleCollapse }: SidebarProps) {
  const groups = [...new Set(navItems.map((item) => item.group))]

  return (
    <aside
      className={cn(
        "flex flex-col bg-sidebar-bg text-sidebar-foreground transition-all duration-300 ease-in-out h-screen sticky top-0",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-5 border-b border-sidebar-accent">
        {!collapsed && (
          <div>
            <h1 className="text-sm font-semibold text-sidebar-primary tracking-tight">
              Prediction de Vente
            </h1>
            <p className="text-xs text-sidebar-muted mt-0.5">Tableau de bord</p>
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded-md hover:bg-sidebar-accent transition-colors text-sidebar-muted hover:text-sidebar-foreground"
          aria-label={collapsed ? "Ouvrir le menu" : "Fermer le menu"}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 overflow-y-auto">
        {groups.map((group) => (
          <div key={group} className="mb-4">
            {!collapsed && (
              <p className="px-4 mb-2 text-[10px] font-semibold tracking-widest text-sidebar-muted uppercase">
                {group}
              </p>
            )}
            <ul className="flex flex-col gap-0.5 px-2">
              {navItems
                .filter((item) => item.group === group)
                .map((item) => {
                  const Icon = item.icon
                  const isActive = activeTab === item.key
                  return (
                    <li key={item.key}>
                      <button
                        onClick={() => onTabChange(item.key)}
                        className={cn(
                          "flex items-center gap-3 w-full rounded-md px-3 py-2 text-sm font-medium transition-all",
                          isActive
                            ? "bg-sidebar-accent text-sidebar-primary"
                            : "text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-primary"
                        )}
                        title={collapsed ? item.label : undefined}
                      >
                        <Icon className={cn("h-4 w-4 shrink-0", isActive && "text-sidebar-primary")} />
                        {!collapsed && <span className="truncate">{item.label}</span>}
                      </button>
                    </li>
                  )
                })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-sidebar-accent">
          <p className="text-[10px] text-sidebar-muted">
            Donnees: sample.csv
          </p>
          <p className="text-[10px] text-sidebar-muted">
            1 000 enregistrements
          </p>
        </div>
      )}
    </aside>
  )
}
