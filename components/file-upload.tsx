"use client"

import { useCallback, useState, useRef } from "react"
import { Upload, FileSpreadsheet } from "lucide-react"

interface FileUploadProps {
  onFileLoaded: (content: string, filename: string) => void
}

export function FileUpload({ onFileLoaded }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith(".csv")) {
        alert("Veuillez selectionner un fichier CSV.")
        return
      }
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        onFileLoaded(text, file.name)
      }
      reader.readAsText(file)
    },
    [onFileLoaded]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 cursor-pointer transition-all ${
        isDragging
          ? "border-primary bg-primary/5"
          : fileName
            ? "border-chart-3/50 bg-chart-3/5"
            : "border-border hover:border-muted-foreground/50 hover:bg-secondary/50"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
        }}
      />
      {fileName ? (
        <>
          <FileSpreadsheet className="h-10 w-10 text-chart-3" />
          <p className="text-sm text-foreground font-medium">{fileName}</p>
          <p className="text-xs text-muted-foreground">
            Cliquez pour changer de fichier
          </p>
        </>
      ) : (
        <>
          <Upload className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm text-foreground font-medium">
            Glissez-deposez un fichier CSV ici
          </p>
          <p className="text-xs text-muted-foreground">
            ou cliquez pour parcourir
          </p>
        </>
      )}
    </div>
  )
}
