"use client"

import { cn } from "@/lib/utils"
import type { LucideIcon } from "lucide-react"

interface KpiCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: LucideIcon
  trend?: number
  className?: string
  variant?: "default" | "primary" | "secondary" | "success" | "destructive"
}

const variantStyles = {
  default: "border-border",
  primary: "border-l-4 border-l-primary border-t-0 border-r-0 border-b-0",
  secondary: "border-l-4 border-l-secondary border-t-0 border-r-0 border-b-0",
  success: "border-l-4 border-l-success border-t-0 border-r-0 border-b-0",
  destructive: "border-l-4 border-l-destructive border-t-0 border-r-0 border-b-0",
}

export function KpiCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  className,
  variant = "default",
}: KpiCardProps) {
  return (
    <div
      className={cn(
        "bg-card text-card-foreground rounded-lg border p-4 shadow-sm",
        variantStyles[variant],
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide truncate">
            {title}
          </p>
          <p className="mt-1 text-2xl font-bold tracking-tight text-foreground">{value}</p>
          {subtitle && (
            <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
          )}
          {trend !== undefined && (
            <p
              className={cn(
                "mt-1 text-xs font-medium",
                trend >= 0 ? "text-success" : "text-destructive"
              )}
            >
              {trend >= 0 ? "+" : ""}
              {trend.toFixed(1)}%
            </p>
          )}
        </div>
        {Icon && (
          <div className="ml-3 p-2 rounded-md bg-muted">
            <Icon className="h-5 w-5 text-muted-foreground" />
          </div>
        )}
      </div>
    </div>
  )
}
