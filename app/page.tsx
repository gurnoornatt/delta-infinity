"use client"

import { useState } from "react"
import Header from "@/components/header"
import ModelSelector from "@/components/model-selector"
import GPUInfo from "@/components/gpu-info"
import AnalyzeButton from "@/components/analyze-button"
import LoadingState from "@/components/loading-state"
import ResultsDisplay from "@/components/results-display"
import { MODELS, type AnalysisResult } from "@/lib/constants"

export default function Home() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0])
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const handleAnalyze = async () => {
    setIsLoading(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2500))

    setResult({
      currentMemoryUsage: 18.5,
      optimalMemoryUsage: 12.2,
      memorySaved: 6.3,
      gpuUtilization: 51,
      currentBatchSize: 16,
      optimalBatchSize: 32,
      speedup: 1.24,
    })

    setIsLoading(false)
  }

  const handleAnalyzeAnother = () => {
    setResult(null)
    setIsLoading(false)
  }

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden">
      {/* Background gradient effect */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl"></div>
      </div>

      {/* Content */}
      <div className="relative z-10">
        <Header />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {!result ? (
            <div className="space-y-12">
              {/* Hero Section */}
              <div className="text-center space-y-6">
                <div className="space-y-4">
                  <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
                    You're Wasting{" "}
                    <span className="text-red-400">60-70%</span> of Your{" "}
                    <span className="text-accent">GPU Memory</span>
                  </h1>
                  <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
                    Find optimal batch sizes with full training simulation
                    <br />
                    <span className="text-sm text-muted-foreground/80">
                      Unlike naive tools, we simulate forward + backward pass for accurate results
                    </span>
                  </p>
                </div>

                {/* Impact Numbers */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto pt-8">
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-green-500/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
                    <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md">
                      <p className="text-4xl font-bold text-green-400 mb-2">$47,000</p>
                      <p className="text-sm text-muted-foreground">Annual Savings</p>
                    </div>
                  </div>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-blue-500/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
                    <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md">
                      <p className="text-4xl font-bold text-blue-400 mb-2">3x</p>
                      <p className="text-sm text-muted-foreground">Speedup</p>
                    </div>
                  </div>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-purple-500/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
                    <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md">
                      <p className="text-4xl font-bold text-purple-400 mb-2">100%</p>
                      <p className="text-sm text-muted-foreground">Full Backward Pass</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Demo Flow Indicator */}
              <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-accent text-background flex items-center justify-center font-bold">1</div>
                  <span>Select Model</span>
                </div>
                <div className="w-12 h-0.5 bg-border"></div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-border text-foreground flex items-center justify-center font-bold">2</div>
                  <span>Analyze</span>
                </div>
                <div className="w-12 h-0.5 bg-border"></div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-border text-foreground flex items-center justify-center font-bold">3</div>
                  <span>See Results</span>
                </div>
              </div>

              {/* Control Section */}
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ModelSelector models={MODELS} selected={selectedModel} onSelect={setSelectedModel} />
                  <GPUInfo />
                </div>

                <AnalyzeButton onClick={handleAnalyze} disabled={isLoading} />
              </div>

              {/* Loading State */}
              {isLoading && <LoadingState />}
            </div>
          ) : (
            <ResultsDisplay result={result} model={selectedModel} onAnalyzeAnother={handleAnalyzeAnother} />
          )}
        </main>
      </div>
    </div>
  )
}
