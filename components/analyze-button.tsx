"use client"

import { Button } from "@/components/ui/button"

interface AnalyzeButtonProps {
  onClick: () => void
  disabled: boolean
}

export default function AnalyzeButton({ onClick, disabled }: AnalyzeButtonProps) {
  return (
    <div className="relative group">
      <div className="absolute -inset-1 bg-gradient-to-r from-accent via-accent to-accent/60 rounded-lg blur opacity-60 group-hover:opacity-100 transition duration-500 animate-pulse"></div>
      <Button
        onClick={onClick}
        disabled={disabled}
        className="relative w-full py-6 text-lg bg-accent text-background hover:bg-accent/90 border-0 rounded-lg transition-all duration-200"
      >
        {disabled ? "Analyzing..." : "Analyze Memory Usage"}
      </Button>
    </div>
  )
}
