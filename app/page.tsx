"use client"

import { useState } from "react"
import Header from "@/components/header"
import ModelSelector from "@/components/model-selector"
import GPUInfo from "@/components/gpu-info"
import AnalyzeButton from "@/components/analyze-button"
import TerminalView from "@/components/terminal-view"
import ResultsDisplay from "@/components/results-display"
import { MODELS, type AnalysisResult } from "@/lib/constants"
import { analyzeModel, mapBackendToFrontend, APIError } from "@/lib/api"

export default function Home() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0])
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Call real Flask backend API (runs in parallel with terminal animation)
      const backendData = await analyzeModel(selectedModel.id)

      // Map backend response to frontend interface
      const frontendResult = mapBackendToFrontend(backendData)

      setResult(frontendResult)
    } catch (err) {
      // Handle errors with user-friendly messages
      if (err instanceof APIError) {
        if (err.statusCode === 408) {
          setError("Analysis timeout - GPU analysis took too long. Please try again or contact support.")
        } else if (err.statusCode === 0) {
          setError("Cannot connect to backend server. Please ensure the Flask server is running on port 5001.")
        } else {
          setError(`Error: ${err.message}`)
        }
      } else {
        setError("An unexpected error occurred. Please try again.")
      }
      console.error("Analysis error:", err)
      setIsLoading(false)
    }
  }

  const handleTerminalComplete = () => {
    // Terminal animation completed - if API call finished, show results
    // If API still running, results will show when it completes
    if (!isLoading) {
      // API already finished, results are ready
      return
    }
  }

  const handleAnalyzeAnother = () => {
    setResult(null)
    setIsLoading(false)
    setError(null)
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
          <div className="space-y-8">
            {/* Hero Section - Only show when no result */}
            {!result && !isLoading && (
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
              </div>
            )}

            {/* Control Section - Always visible when no result */}
            {!result && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ModelSelector models={MODELS} selected={selectedModel} onSelect={setSelectedModel} />
                  <GPUInfo />
                </div>

                <AnalyzeButton onClick={handleAnalyze} disabled={isLoading} />
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="rounded-lg border border-red-500/50 bg-red-500/10 p-4">
                <div className="flex items-start gap-3">
                  <svg
                    className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-red-500">Error</h3>
                    <p className="mt-1 text-sm text-red-400">{error}</p>
                    <button
                      onClick={() => setError(null)}
                      className="mt-2 text-sm text-red-400 hover:text-red-300 underline"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Terminal View - Live Analysis (shown during loading AND after completion if result exists) */}
            {(isLoading || result) && (
              <div>
                {result && (
                  <div className="mb-4">
                    <h2 className="text-xl font-semibold text-foreground mb-2">Analysis Process</h2>
                    <p className="text-sm text-muted-foreground">Real-time GPU memory testing on Lambda Labs A10</p>
                  </div>
                )}
                <TerminalView
                  modelName={selectedModel.name}
                  modelId={selectedModel.id}
                  onComplete={handleTerminalComplete}
                />
              </div>
            )}

            {/* Transition Divider */}
            {result && (
              <div className="relative py-8">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-accent/20"></div>
                </div>
                <div className="relative flex justify-center">
                  <span className="bg-background px-4 text-sm text-muted-foreground">Analysis Complete</span>
                </div>
              </div>
            )}

            {/* Results Display - Shown after completion */}
            {result && (
              <ResultsDisplay result={result} model={selectedModel} onAnalyzeAnother={handleAnalyzeAnother} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
