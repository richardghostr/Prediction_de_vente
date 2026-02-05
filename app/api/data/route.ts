import { NextResponse } from "next/server"
import { readFileSync } from "fs"
import { join } from "path"

export async function GET() {
  try {
    const csvPath = join(process.cwd(), "sample.csv")
    const csvText = readFileSync(csvPath, "utf-8")
    return NextResponse.json({ csv: csvText })
  } catch {
    return NextResponse.json({ csv: "" }, { status: 500 })
  }
}
