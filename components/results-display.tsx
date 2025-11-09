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

  const wastePercentage = ((result.currentMemoryUsage - result.optimalMemoryUsage) / result.currentMemoryUsage) * 100
  const improvementPercentage = ((result.optimalBatchSize - result.currentBatchSize) / result.currentBatchSize) * 100

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Header */}
      <div className="border-b border-border/30 pb-6">
        <h2 className="text-3xl text-foreground mb-2">Memory Analysis Results</h2>
        <p className="text-sm text-muted-foreground">{model.name} • NVIDIA A10 - 24GB</p>
      </div>

      {/* Summary Card - Main Recommendation */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/30 via-accent/20 to-accent/10 rounded-xl blur-xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
        <div className="relative bg-card border-2 border-accent/50 rounded-xl p-8 backdrop-blur-md">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div>
              <h3 className="text-2xl font-bold mb-4 text-accent">🎯 Key Recommendation</h3>
              <div className="space-y-3">
                <div className="flex items-baseline gap-3">
                  <span className="text-5xl font-bold text-foreground">{result.optimalBatchSize}</span>
                  <span className="text-xl text-muted-foreground">batch size</span>
                </div>
                <p className="text-muted-foreground">
                  Increase from <span className="text-red-400 font-semibold">{result.currentBatchSize}</span> to{" "}
                  <span className="text-accent font-semibold">{result.optimalBatchSize}</span> to save{" "}
                  <span className="text-accent font-semibold">{result.memorySaved.toFixed(1)} GB</span> of memory
                </p>
                <div className="flex items-center gap-4 pt-2">
                  <div className="px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg">
                    <p className="text-xs text-muted-foreground">Current Waste</p>
                    <p className="text-xl font-bold text-red-400">{wastePercentage.toFixed(0)}%</p>
                  </div>
                  <div className="px-4 py-2 bg-accent/10 border border-accent/30 rounded-lg">
                    <p className="text-xs text-muted-foreground">Improvement</p>
                    <p className="text-xl font-bold text-accent">+{improvementPercentage.toFixed(0)}%</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-center">
                  <p className="text-sm text-muted-foreground mb-1">Before</p>
                  <p className="text-3xl font-bold text-red-400">{result.currentMemoryUsage.toFixed(1)} GB</p>
                  <p className="text-xs text-muted-foreground mt-1">Batch: {result.currentBatchSize}</p>
                </div>
                <div className="bg-accent/10 border border-accent/30 rounded-lg p-4 text-center">
                  <p className="text-sm text-muted-foreground mb-1">After</p>
                  <p className="text-3xl font-bold text-accent">{result.optimalMemoryUsage.toFixed(1)} GB</p>
                  <p className="text-xs text-muted-foreground mt-1">Batch: {result.optimalBatchSize}</p>
                </div>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 text-center">
                <p className="text-sm text-muted-foreground mb-1">Performance Gain</p>
                <p className="text-3xl font-bold text-blue-400">{result.speedup.toFixed(2)}x</p>
                <p className="text-xs text-muted-foreground mt-1">Faster training</p>
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

        {/* Memory Saved */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">Memory Saved</span>
                <Tooltip content={`Amount of GPU memory you can save by switching to optimal batch size. This is the difference between current and optimal memory usage.`}>
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge
                label="Saved"
                value={result.memorySaved}
                maxValue={24}
                color="green"
                unit=" GB"
              />
            </div>
          </div>
        </div>

        {/* GPU Utilization */}
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 to-blue-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border border-border/30 rounded-lg backdrop-blur-md">
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">GPU Utilization</span>
                <Tooltip content={`Percentage of GPU memory that will be utilized with the optimal batch size. Higher is better - aiming for 90%+.`}>
                  <HelpCircle className="w-3 h-3 text-muted-foreground/50 hover:text-accent transition-colors" />
                </Tooltip>
              </div>
              <MemoryGauge
                label="Utilized"
                value={result.gpuUtilization}
                maxValue={100}
                color="blue"
                unit="%"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Memory Charts */}
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

      {/* Before/After Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-br from-red-500/20 to-red-500/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border-2 border-red-500/30 rounded-lg p-6 backdrop-blur-md">
            <h3 className="text-lg font-semibold mb-4 text-red-400">❌ Current Configuration</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Memory Usage</span>
                <span className="text-2xl font-bold text-red-400">{result.currentMemoryUsage.toFixed(1)} GB</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Batch Size</span>
                <span className="text-2xl font-bold text-red-400">{result.currentBatchSize}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">GPU Utilization</span>
                <span className="text-2xl font-bold text-red-400">{result.gpuUtilization}%</span>
              </div>
              <div className="pt-3 border-t border-border/30">
                <span className="text-sm text-muted-foreground">Waste: </span>
                <span className="text-lg font-bold text-red-400">{wastePercentage.toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-br from-accent/20 to-accent/5 rounded-lg blur opacity-50 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative bg-card border-2 border-accent/30 rounded-lg p-6 backdrop-blur-md">
            <h3 className="text-lg font-semibold mb-4 text-accent">✅ Optimal Configuration</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Memory Usage</span>
                <span className="text-2xl font-bold text-accent">{result.optimalMemoryUsage.toFixed(1)} GB</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Batch Size</span>
                <span className="text-2xl font-bold text-accent">{result.optimalBatchSize}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">GPU Utilization</span>
                <span className="text-2xl font-bold text-accent">
                  {Math.round((result.optimalMemoryUsage / 24) * 100)}%
                </span>
              </div>
              <div className="pt-3 border-t border-border/30">
                <span className="text-sm text-muted-foreground">Savings: </span>
                <span className="text-lg font-bold text-accent">{result.memorySaved.toFixed(1)} GB</span>
              </div>
            </div>
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
