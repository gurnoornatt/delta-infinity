"use client"

import { Button } from "@/components/ui/button"
import MemoryGauge from "@/components/memory-gauge"
import MemoryChart from "@/components/memory-chart"
import type { AnalysisResult, Model } from "@/lib/constants"

interface ResultsDisplayProps {
  result: AnalysisResult
  model: Model
  onAnalyzeAnother: () => void
}

export default function ResultsDisplay({ result, model, onAnalyzeAnother }: ResultsDisplayProps) {
  // Generate chart data from actual backend results
  // Sample every 4th batch size for cleaner visualization (or take last 8 data points)
  const allResults = result.results || []
  const sampledResults = allResults.filter((_, idx) => idx % 4 === 0 || idx === allResults.length - 1).slice(-8)

  // Current memory usage - show the baseline (current batch size usage)
  const currentBatchMemory = result.currentMemoryUsage
  const memoryOverTime = sampledResults.map((r, idx) => ({
    time: `Batch ${r.batchSize}`,
    value: currentBatchMemory
  }))

  // Optimal memory usage - show the progression toward optimal
  const optimalOverTime = sampledResults.map((r) => ({
    time: `Batch ${r.batchSize}`,
    value: r.memoryGb
  }))

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Header */}
      <div className="border-b border-border/30 pb-6">
        <h2 className="text-3xl text-foreground mb-2">Memory Analysis Results</h2>
        <p className="text-sm text-muted-foreground">{model.name} • NVIDIA A10 - 24GB</p>
      </div>

      {/* Key Metrics Gauges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-red-500/10 to-red-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <MemoryGauge
              label="Current Memory"
              value={result.currentMemoryUsage}
              maxValue={24}
              color="red"
              unit=" GB"
            />
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <MemoryGauge
              label="Optimal Memory"
              value={result.optimalMemoryUsage}
              maxValue={24}
              color="green"
              unit=" GB"
            />
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-red-500/10 to-red-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <MemoryGauge label="Memory Waste" value={result.wastePercentage} maxValue={100} color="red" unit="%" />
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <MemoryGauge
              label="Efficiency Gain"
              value={100 - result.wastePercentage}
              maxValue={100}
              color="green"
              unit="%"
            />
          </div>
        </div>
      </div>

      {/* Memory Usage Over Time */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MemoryChart title="Current Memory Usage" data={memoryOverTime} color="red" />
        <MemoryChart title="Optimal Memory Usage" data={optimalOverTime} color="green" />
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-blue-500/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
          <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md text-center">
            <p className="text-sm text-muted-foreground mb-2">Performance Speedup</p>
            <p className="text-3xl text-blue-400 mb-1">{result.speedup}x</p>
            <p className="text-xs text-muted-foreground">Faster inference time</p>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-purple-500/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
          <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md text-center">
            <p className="text-sm text-muted-foreground mb-2">Cost Per Run</p>
            <p className="text-3xl text-purple-400 mb-1">${result.costPerRun.toFixed(3)}</p>
            <p className="text-xs text-muted-foreground">Reduced GPU cost</p>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
          <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md text-center">
            <p className="text-sm text-muted-foreground mb-2">Annual Savings</p>
            <p className="text-3xl text-accent mb-1">${result.annualSavings}</p>
            <p className="text-xs text-muted-foreground">For 1M runs/year</p>
          </div>
        </div>
      </div>

      {/* Analyze Another Button */}
      <div className="relative group">
        <div className="absolute -inset-1 bg-gradient-to-r from-accent via-accent to-accent/60 rounded-lg blur opacity-60 group-hover:opacity-100 transition duration-500"></div>
        <Button
          onClick={onAnalyzeAnother}
          className="relative w-full py-6 text-lg bg-accent text-background hover:bg-accent/90 border-0 rounded-lg transition-all duration-200"
        >
          Analyze Another Model
        </Button>
      </div>
    </div>
  )
}
