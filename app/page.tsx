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
      wastePercentage: 34,
      speedup: 1.24,
      costPerRun: 0.045,
      annualSavings: 1240,
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
            <div className="space-y-8">
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
