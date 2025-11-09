"use client"

import { Button } from "@/components/ui/button"
import { Tooltip } from "@/components/ui/tooltip"
import MemoryGauge from "@/components/memory-gauge"
import MemoryChart from "@/components/memory-chart"
import type { AnalysisResult, Model } from "@/lib/constants"
import { HelpCircle } from "lucide-react"

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

      {/* Explanatory Summary */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-50"></div>
        <div className="relative bg-card/80 backdrop-blur-md border border-accent/30 rounded-lg p-6">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center">
              <svg className="w-6 h-6 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-foreground mb-2">
                You're wasting {result.wastePercentage.toFixed(1)}% of your GPU memory
              </h3>
              <p className="text-sm text-muted-foreground mb-3">
                Your current batch size ({result.currentBatchSize}) uses only {result.currentMemoryUsage.toFixed(1)} GB of {result.gpuMemoryTotal} GB available.
                By increasing to batch size {result.optimalBatchSize}, you could achieve {result.speedup}x faster training with {(100 - result.wastePercentage).toFixed(1)}% GPU efficiency.
              </p>
              <div className="flex items-center gap-2 text-xs text-accent">
                <span className="font-mono">💰 Annual Savings: ${result.annualSavings}</span>
                <span className="text-muted-foreground">•</span>
                <span className="font-mono">⚡ {result.speedup}x Speedup</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics Gauges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Current Memory - Your Setup */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-red-500/10 to-red-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">Your Setup</span>
                <Tooltip content={`Memory used with your current batch size (${result.currentBatchSize}). This is what you're using now before optimization.`}>
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge
                label={`Batch ${result.currentBatchSize}`}
                value={result.currentMemoryUsage}
                maxValue={24}
                color="red"
                unit=" GB"
              />
            </div>
          </div>
        </div>

        {/* Optimal Memory - Recommended Setup */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">Recommended</span>
                <Tooltip content={`Memory used with optimal batch size (${result.optimalBatchSize}). This maximizes GPU utilization for ${result.speedup}x faster training.`}>
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge
                label={`Batch ${result.optimalBatchSize}`}
                value={result.optimalMemoryUsage}
                maxValue={24}
                color="green"
                unit=" GB"
              />
            </div>
          </div>
        </div>

        {/* Memory Waste */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-red-500/10 to-red-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">GPU Waste</span>
                <Tooltip content="Percentage of GPU memory currently being wasted. Lower is better - it means you're using more of your available GPU.">
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge label="Wasted" value={result.wastePercentage} maxValue={100} color="red" unit="%" />
            </div>
          </div>
        </div>

        {/* Efficiency Gain */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">GPU Efficiency</span>
                <Tooltip content="Percentage of GPU memory that will be utilized with the optimal batch size. Higher is better - aiming for 90%+.">
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge
                label="Utilized"
                value={100 - result.wastePercentage}
                maxValue={100}
                color="green"
                unit="%"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Memory Usage Over Time */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-sm font-medium text-muted-foreground">Your Current Setup (Batch {result.currentBatchSize})</h3>
            <Tooltip content={`Memory usage with your current batch size of ${result.currentBatchSize}. This shows the baseline memory consumption before optimization.`}>
              <HelpCircle className="w-4 h-4 text-muted-foreground/50 hover:text-accent transition-colors" />
            </Tooltip>
          </div>
          <MemoryChart title="" data={memoryOverTime} color="red" />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-sm font-medium text-muted-foreground">Memory Progression to Optimal (Batch {result.optimalBatchSize})</h3>
            <Tooltip content={`How memory usage scales as batch size increases from ${result.currentBatchSize} to the optimal ${result.optimalBatchSize}. Shows real GPU measurements from testing.`}>
              <HelpCircle className="w-4 h-4 text-muted-foreground/50 hover:text-accent transition-colors" />
            </Tooltip>
          </div>
          <MemoryChart title="" data={optimalOverTime} color="green" />
        </div>
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
