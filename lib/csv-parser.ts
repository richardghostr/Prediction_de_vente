export interface ParsedData {
  headers: string[]
  rows: Record<string, string | number | null>[]
  dateColumn: string | null
  valueColumn: string | null
}

const DATE_CANDIDATES = [
  "date",
  "Date",
  "ds",
  "timestamp",
  "datetime",
  "time",
  "sale_date",
  "saledate",
]

const VALUE_CANDIDATES = [
  "quantity",
  "qty",
  "value",
  "y",
  "sales",
  "amount",
  "revenue",
  "unit_price",
]

function isDateLike(value: string): boolean {
  if (!value || value.trim() === "") return false
  const d = new Date(value)
  return !isNaN(d.getTime())
}

function isNumeric(value: string): boolean {
  if (!value || value.trim() === "") return false
  return !isNaN(Number(value))
}

export function parseCSV(text: string): ParsedData {
  const lines = text.trim().split("\n")
  if (lines.length < 2) {
    return { headers: [], rows: [], dateColumn: null, valueColumn: null }
  }

  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""))

  const rows: Record<string, string | number | null>[] = []
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",").map((v) => v.trim().replace(/^"|"$/g, ""))
    if (values.length !== headers.length) continue

    const row: Record<string, string | number | null> = {}
    for (let j = 0; j < headers.length; j++) {
      const val = values[j]
      if (val === "" || val === "null" || val === "None") {
        row[headers[j]] = null
      } else if (isNumeric(val) && !DATE_CANDIDATES.includes(headers[j].toLowerCase())) {
        row[headers[j]] = Number(val)
      } else {
        row[headers[j]] = val
      }
    }
    rows.push(row)
  }

  // Auto-detect date column
  let dateColumn: string | null = null
  for (const candidate of DATE_CANDIDATES) {
    if (headers.includes(candidate)) {
      dateColumn = candidate
      break
    }
  }
  if (!dateColumn) {
    for (const h of headers) {
      const sample = rows.slice(0, 5).map((r) => String(r[h] ?? ""))
      if (sample.filter(isDateLike).length >= 3) {
        dateColumn = h
        break
      }
    }
  }

  // Auto-detect value column
  let valueColumn: string | null = null
  for (const candidate of VALUE_CANDIDATES) {
    const match = headers.find((h) => h.toLowerCase() === candidate)
    if (match) {
      valueColumn = match
      break
    }
  }
  if (!valueColumn) {
    for (const h of headers) {
      if (h === dateColumn) continue
      const sample = rows.slice(0, 10).map((r) => r[h])
      if (sample.filter((v) => typeof v === "number").length >= 5) {
        valueColumn = h
        break
      }
    }
  }

  return { headers, rows, dateColumn, valueColumn }
}

export interface AggregatedPoint {
  date: string
  value: number
}

export function aggregateByDate(
  rows: Record<string, string | number | null>[],
  dateCol: string,
  valueCol: string
): AggregatedPoint[] {
  const grouped: Record<string, number[]> = {}

  for (const row of rows) {
    const dateVal = row[dateCol]
    const numVal = row[valueCol]
    if (dateVal == null || numVal == null) continue

    const dateStr = new Date(String(dateVal)).toISOString().split("T")[0]
    if (dateStr === "Invalid Date") continue

    if (!grouped[dateStr]) grouped[dateStr] = []
    grouped[dateStr].push(Number(numVal))
  }

  return Object.entries(grouped)
    .map(([date, vals]) => ({
      date,
      value: vals.reduce((a, b) => a + b, 0),
    }))
    .sort((a, b) => a.date.localeCompare(b.date))
}
