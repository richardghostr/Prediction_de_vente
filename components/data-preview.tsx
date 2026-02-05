"use client"

interface DataPreviewProps {
  headers: string[]
  rows: Record<string, string | number | null>[]
  dateColumn: string | null
  valueColumn: string | null
  maxRows?: number
}

export function DataPreview({
  headers,
  rows,
  dateColumn,
  valueColumn,
  maxRows = 10,
}: DataPreviewProps) {
  const displayRows = rows.slice(0, maxRows)

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/50">
            {headers.map((h) => (
              <th
                key={h}
                className={`px-4 py-2.5 text-left font-medium whitespace-nowrap ${
                  h === dateColumn
                    ? "text-primary"
                    : h === valueColumn
                      ? "text-accent"
                      : "text-muted-foreground"
                }`}
              >
                {h}
                {h === dateColumn && (
                  <span className="ml-1.5 text-[10px] font-normal rounded-sm bg-primary/10 text-primary px-1.5 py-0.5">
                    date
                  </span>
                )}
                {h === valueColumn && (
                  <span className="ml-1.5 text-[10px] font-normal rounded-sm bg-accent/10 text-accent px-1.5 py-0.5">
                    valeur
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-border/50 last:border-0 hover:bg-secondary/30 transition-colors"
            >
              {headers.map((h) => (
                <td
                  key={h}
                  className={`px-4 py-2 whitespace-nowrap font-mono text-xs ${
                    h === dateColumn
                      ? "text-primary/80"
                      : h === valueColumn
                        ? "text-accent/80"
                        : "text-muted-foreground"
                  }`}
                >
                  {row[h] == null ? (
                    <span className="text-muted-foreground/40 italic">null</span>
                  ) : (
                    String(row[h])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <div className="px-4 py-2 text-xs text-muted-foreground border-t border-border bg-secondary/30">
          Affichage de {maxRows} sur {rows.length} lignes
        </div>
      )}
    </div>
  )
}
